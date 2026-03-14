#!/usr/bin/env python3
"""Test questions on the test founders dataset.

Questions mode supports a canonical single-list input:
- test_questions/selected_questions_all_anonymised.csv

Questions mode writes:
- predictions_test_all_anonymised.csv

Optional prepare stage (`--prepare-questions`) performs the former 04 behavior
inside this script:
- validate canonical questions on validation founders
- write validation predictions + stats artifacts
- write canonical selected questions file
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.questions.question_io import load_questions
from src.questions.testing import (
    test_question_anonymised,
    update_founder_index,
    update_results_file,
    update_success_proportion,
)
from src.questions.validation_postprocess import run_stats_postprocess
from src.questions.vanilla_testing import (
    test_vanilla_predictions_anonymised,
    test_vanilla_predictions_few_shot_anonymised,
    update_vanilla_results,
)
from src.utils.dataset_utils import load_dataset
from src.utils.file_utils import setup_logging

_thread_local = threading.local()

# Keep explicit prepare-stage defaults for reproducibility.
POSTPROCESS_M = 100
POSTPROCESS_ALPHA = 0.05
POSTPROCESS_B_NULL = 1000
POSTPROCESS_PERC_SUCCESS = 0.10
POSTPROCESS_JACCARD_VALUE = 1.00
POSTPROCESS_WRITE_PLOTS = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and test evaluation questions")
    parser.add_argument("--provider", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--results_dir_extension", type=str, default="precomputed")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument(
        "--max_concurrent_batches",
        type=int,
        default=6,
        help="Concurrency for question-mode founder batches.",
    )
    parser.add_argument(
        "--prepare-questions",
        action="store_true",
        help=(
            "Run validation-stage preparation before test inference (legacy 04 behavior): "
            "validation scoring + stats/filtering + selected_questions_all write."
        ),
    )
    parser.add_argument(
        "--prepare_max_concurrent_batches",
        type=int,
        default=6,
        help="Concurrency for prepare stage founder batches.",
    )
    parser.add_argument(
        "--prepare_skip_stats_postprocess",
        action="store_true",
        help="Skip stats/filtering postprocess during prepare stage.",
    )
    parser.add_argument(
        "--questions_file",
        type=str,
        default=None,
        help="Optional explicit question CSV path; defaults to selected_questions_all.",
    )
    parser.add_argument(
        "--mode",
        choices=["questions", "vanilla", "vanilla_few_shot"],
        default="questions",
        help="'questions' runs question prompts; vanilla modes run baseline prompting.",
    )
    parser.add_argument("--rep", type=int, default=1)
    return parser.parse_args()


def load_questions_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {path}")
    df = pd.read_csv(path)
    if "Question" not in df.columns:
        raise ValueError(f"Missing 'Question' column in {path}")
    df = df.copy()
    df["Question"] = df["Question"].astype(str).str.strip()
    df = df[df["Question"] != ""].drop_duplicates(subset=["Question"]).reset_index(drop=True)
    return df


def resolve_questions_file(args: argparse.Namespace, suffix: str) -> Path:
    if args.questions_file:
        return Path(args.questions_file)

    base_dir = (
        Path(project_root)
        / args.results_dir_extension
        / args.model.replace("-", "_")
        / "test_questions"
    )
    return base_dir / f"selected_questions_all{suffix}.csv"


def get_thread_llm_client(provider: str, model: str, temperature: float):
    client = getattr(_thread_local, "llm_client", None)
    if client is None:
        from src.llms.utils import get_llm_client

        client = get_llm_client(provider, model, temperature)
        _thread_local.llm_client = client
    return client


def score_batch(
    provider: str,
    model: str,
    temperature: float,
    question: str,
    batch_df: pd.DataFrame,
) -> list[dict]:
    client = get_thread_llm_client(provider, model, temperature)
    return test_question_anonymised(client, question, batch_df)


def run_parallel_question_batches(
    *,
    provider: str,
    model: str,
    temperature: float,
    question: str,
    data: pd.DataFrame,
    batch_size: int,
    num_batches: int,
    max_concurrent_batches: int,
) -> list[dict]:
    if max_concurrent_batches <= 1:
        combined_predictions: list[dict] = []
        for i in range(num_batches):
            batch_start = i * batch_size
            batch_end = batch_start + batch_size
            batch = data.iloc[batch_start:batch_end]
            preds = score_batch(provider, model, temperature, question, batch)
            combined_predictions.extend(preds)
        return combined_predictions

    preds_by_batch: dict[int, list[dict]] = {}

    with ThreadPoolExecutor(max_workers=max_concurrent_batches) as executor:
        futures = {}
        for i in range(num_batches):
            batch_start = i * batch_size
            batch_end = batch_start + batch_size
            batch = data.iloc[batch_start:batch_end]
            fut = executor.submit(
                score_batch,
                provider,
                model,
                temperature,
                question,
                batch,
            )
            futures[fut] = i

        for fut in as_completed(futures):
            batch_idx = futures[fut]
            preds_by_batch[batch_idx] = fut.result()

    # Preserve deterministic order to keep output stable.
    combined_predictions: list[dict] = []
    for i in range(num_batches):
        combined_predictions.extend(preds_by_batch.get(i, []))
    return combined_predictions


def run_prepare_stage(args: argparse.Namespace) -> None:
    suffix = "_anonymised"

    model_dir = Path(project_root) / args.results_dir_extension / args.model.replace("-", "_")
    output_dir = model_dir / "validation_predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    canonical_questions_file = model_dir / f"initial_questions/generated_questions_semantic_dedup{suffix}.csv"

    val_file = f"02_question_validation_data{suffix}.csv"
    data_path = Path(project_root) / args.results_dir_extension / val_file
    validation_data = load_dataset(data_path, "validation")

    questions = load_questions(canonical_questions_file)
    logger.info(
        "Prepare stage: loaded %d canonical questions from %s",
        len(questions),
        canonical_questions_file,
    )
    logger.info(
        "Prepare stage concurrency (founder batches per question): %d",
        args.prepare_max_concurrent_batches,
    )

    prediction_file = output_dir / f"predictions_val_set_all_question{suffix}.csv"

    # Resume support to avoid repeated calls.
    special_rows = {'Dataset', 'Success', 'SUCCESS_PROPORTION', 'Founder Index'}
    if prediction_file.exists():
        existing_df = pd.read_csv(prediction_file)
        answered_questions = set(existing_df["Question"].dropna().astype(str).tolist()) - special_rows
        logger.info(
            "Prepare stage: found %d already processed question rows in %s",
            len(answered_questions),
            prediction_file.name,
        )
    else:
        answered_questions = set()

    # Keep existing behavior for reproducibility (floor division, no tail batch).
    num_batches = len(validation_data) // args.batch_size
    dataset_assignments = {
        founder["founder_uuid"]: "Validation" for _, founder in validation_data.iterrows()
    }

    for idx, question in enumerate(questions, start=1):
        if question in answered_questions:
            logger.info(
                "Prepare stage: ⏭️ skipping question %d/%d – already processed.",
                idx,
                len(questions),
            )
            continue

        start = time.perf_counter()

        combined_predictions = run_parallel_question_batches(
            provider=args.provider,
            model=args.model,
            temperature=args.temperature,
            question=question,
            data=validation_data,
            batch_size=args.batch_size,
            num_batches=num_batches,
            max_concurrent_batches=args.prepare_max_concurrent_batches,
        )

        update_results_file(
            results_file=prediction_file,
            question=question,
            responses=combined_predictions,
            test_founders=validation_data,
            logger=logger,
            dataset_assignments=dataset_assignments,
        )

        update_founder_index(prediction_file)
        update_success_proportion(prediction_file)

        elapsed = time.perf_counter() - start
        logger.info(
            "Prepare stage: ⏱️ question %d/%d '%s…' took %.2fs",
            idx,
            len(questions),
            question[:50],
            elapsed,
        )

    if not args.prepare_skip_stats_postprocess:
        run_stats_postprocess(
            model_dir=model_dir,
            prediction_file=prediction_file,
            suffix=suffix,
            model_name=args.model.replace("-", "_"),
            m=POSTPROCESS_M,
            alpha=POSTPROCESS_ALPHA,
            b_null=POSTPROCESS_B_NULL,
            perc_success=POSTPROCESS_PERC_SUCCESS,
            jaccard_value=POSTPROCESS_JACCARD_VALUE,
            write_plots=POSTPROCESS_WRITE_PLOTS,
            logger=logger,
        )

    logger.info("Prepare stage complete.")


def run_questions_mode(args: argparse.Namespace, test_data: pd.DataFrame, num_batches: int) -> None:
    suffix = "_anonymised"

    questions_file = resolve_questions_file(args, suffix)
    logger.info(f"Question file: {questions_file}")
    questions_df = load_questions_df(questions_file)
    questions = questions_df["Question"].tolist()
    logger.info(f"Loaded {len(questions)} questions")
    logger.info(
        "Question mode concurrency (founder batches per question): %d",
        args.max_concurrent_batches,
    )

    output_dir = (
        Path(project_root) / args.results_dir_extension / args.model.replace("-", "_") / "test_predictions"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_file = output_dir / f"predictions_test_all{suffix}.csv"

    special_rows = {'Dataset', 'Success', 'SUCCESS_PROPORTION', 'Founder Index'}
    if prediction_file.exists():
        df_existing = pd.read_csv(prediction_file)
        answered_questions = set(df_existing["Question"].dropna().astype(str).tolist()) - special_rows
        logger.info(f"Found {len(answered_questions)} already-processed rows in {prediction_file.name}")
    else:
        answered_questions = set()

    dataset_assignments = {founder["founder_uuid"]: "Validation" for _, founder in test_data.iterrows()}

    for idx, question in enumerate(questions, start=1):
        if question in answered_questions:
            logger.info(f"⏭️ Skipping question {idx}/{len(questions)} – already processed.")
            continue

        start = time.perf_counter()

        combined_predictions = run_parallel_question_batches(
            provider=args.provider,
            model=args.model,
            temperature=args.temperature,
            question=question,
            data=test_data,
            batch_size=args.batch_size,
            num_batches=num_batches,
            max_concurrent_batches=args.max_concurrent_batches,
        )

        update_results_file(
            results_file=prediction_file,
            question=question,
            responses=combined_predictions,
            test_founders=test_data,
            logger=logger,
            dataset_assignments=dataset_assignments,
        )

        update_founder_index(prediction_file)
        update_success_proportion(prediction_file)

        elapsed = time.perf_counter() - start
        logger.info(f"⏱️ Question {idx}/{len(questions)} '{question[:50]}…' took {elapsed:.2f}s")

    logger.info("Question testing completed.")


def run_vanilla_mode(
    args: argparse.Namespace,
    llm_client,
    test_data: pd.DataFrame,
    train_data: pd.DataFrame,
    num_batches: int,
) -> None:
    suffix = "_anonymised"
    logger.info(f"Beginning vanilla loop with {len(test_data)} founders in {num_batches} batches.")
    start = time.perf_counter()

    results_dir = Path(project_root) / args.results_dir_extension / args.model.replace("-", "_") / "vanilla"
    prefix = "vanilla_few_shot_" if args.mode == "vanilla_few_shot" else "vanilla_zero_shot_"
    summary_file = results_dir / f"{prefix}{args.model.replace('-', '_')}{suffix}_rep_{args.rep}.csv"

    dataset_assignments = {row["founder_uuid"]: "Validation" for _, row in test_data.iterrows()}

    for i in range(num_batches):
        logger.info(f"Running vanilla batch {i + 1}/{num_batches}…")
        batch_start = i * args.batch_size
        batch_end = batch_start + args.batch_size
        batch = test_data.iloc[batch_start:batch_end]

        if args.mode == "vanilla_few_shot":
            preds = test_vanilla_predictions_few_shot_anonymised(llm_client, batch, train_data)
        elif args.mode == "vanilla":
            preds = test_vanilla_predictions_anonymised(llm_client, batch)
        else:
            raise ValueError(f"Unexpected mode: {args.mode}")

        update_vanilla_results(
            results_file=summary_file,
            predictions=preds,
            test_founders=test_data,
            dataset_assignments=dataset_assignments,
            logger=logger,
        )

    logger.info(f"  ✓ Updated summary with batch {i + 1}")
    elapsed = time.perf_counter() - start
    logger.info(f"⏱️ Vanilla baseline complete in {elapsed:.2f}s; summary at {summary_file}")


def main() -> None:
    args = parse_args()

    # Setup logging
    global logger
    logger = setup_logging("question_testing")

    suffix = "_anonymised"
    base_dir = Path(project_root) / args.results_dir_extension
    train_data_path = base_dir / f"01_question_training_data{suffix}.csv"
    test_data_path = base_dir / f"03_full_cross_validation_test_data{suffix}.csv"

    test_data = load_dataset(str(test_data_path), "test")
    train_data = load_dataset(str(train_data_path), "train")

    if args.mode == "questions" and args.prepare_questions:
        run_prepare_stage(args)

    # Keep existing batching behavior for reproducibility.
    num_batches = len(test_data) // args.batch_size

    if args.mode == "questions":
        run_questions_mode(args, test_data, num_batches)
    elif args.mode in ["vanilla", "vanilla_few_shot"]:
        from src.llms.utils import get_llm_client

        llm_client = get_llm_client(args.provider, args.model, temperature=args.temperature)
        run_vanilla_mode(args, llm_client, test_data, train_data, num_batches)
    else:
        raise ValueError(f"Unexpected mode: {args.mode}")


if __name__ == "__main__":
    main()
