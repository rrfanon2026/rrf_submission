#!/usr/bin/env python3
"""Generate questions and write canonical all/unique/semantic-dedup artifacts.

Outputs:
- initial_questions/generated_questions_all_anonymised.csv
- initial_questions/generated_questions_unique_anonymised.csv
- initial_questions/generated_questions_semantic_dedup_anonymised.csv
- similarity/questions_dedup_0_80_anonymised.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Setup
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.llms.utils import get_llm_client
from src.questions.anonymised_generation import (
    generate_questions_anonymised,
    generate_updated_question_set_anonymised,
)
from src.questions.question_io import dedupe_preserve_order, write_question_list
from src.questions.similarity_utils import deduplicate_questions, extract_similar_pairs
from src.utils.dataset_utils import split_into_sets
from src.utils.plotting import plot_similarity_heatmap

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate canonical question artifacts")
    parser.add_argument("--provider", required=True, help="LLM provider (e.g., openai)")
    parser.add_argument("--model", required=True, help="LLM model (e.g., gpt-4o-mini)")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--results_dir_extension", type=str, default="precomputed")
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.8,
        help="Semantic similarity threshold for deduplication stage.",
    )
    return parser.parse_args()


def compute_similarity_matrix(model: SentenceTransformer, questions: list[str]):
    embeddings = model.encode(questions, convert_to_tensor=True)
    return util.pytorch_cos_sim(embeddings, embeddings).cpu().numpy()


def main() -> None:
    args = parse_args()
    suffix = "_anonymised"
    model_clean = args.model.replace("-", "_")

    model_dir = project_root / args.results_dir_extension / model_clean
    initial_dir = model_dir / "initial_questions"
    similarity_dir = model_dir / "similarity"
    initial_dir.mkdir(parents=True, exist_ok=True)
    similarity_dir.mkdir(parents=True, exist_ok=True)

    all_file = initial_dir / f"generated_questions_all{suffix}.csv"
    unique_file = initial_dir / f"generated_questions_unique{suffix}.csv"
    semantic_file = initial_dir / f"generated_questions_semantic_dedup{suffix}.csv"

    # Load training data
    data_path = project_root / args.results_dir_extension / f"01_question_training_data{suffix}.csv"
    logger.info(f"Loading training data from: {data_path}")
    train_data = pd.read_csv(data_path)

    # Generate questions across splits
    sets = split_into_sets(train_data)
    llm_client = get_llm_client(args.provider, args.model, args.temperature)

    all_questions: list[str] = []
    for idx, founder_set in enumerate(sets):
        logger.info(f"Generating questions for set {idx + 1}/{len(sets)}")
        questions = (
            generate_questions_anonymised(llm_client, founder_set)
            if idx == 0
            else generate_updated_question_set_anonymised(founder_set, llm_client, logger)
        )
        all_questions.extend([str(q).strip() for q in questions if str(q).strip()])

    # 1) All questions (keep duplicates/order)
    write_question_list(all_file, all_questions)
    logger.info(f"✓ Saved all questions ({len(all_questions)}) -> {all_file}")

    # 2) Exact-string unique questions
    unique_questions = dedupe_preserve_order(all_questions)
    write_question_list(unique_file, unique_questions)
    logger.info(f"✓ Saved unique questions ({len(unique_questions)}) -> {unique_file}")

    # 3) Semantic dedup on unique questions
    embedding_model_path = project_root / "results" / "all-MiniLM-L6-v2"
    embed_model = SentenceTransformer(str(embedding_model_path), local_files_only=True)

    sim_matrix = compute_similarity_matrix(embed_model, unique_questions)
    pairs_df = extract_similar_pairs(sim_matrix, unique_questions, args.similarity_threshold)

    pd.DataFrame(sim_matrix).to_csv(
        similarity_dir / f"question_similarity_matrix{suffix}.csv", index=False
    )
    pairs_df.to_csv(similarity_dir / f"high_similarity_pairs{suffix}.csv", index=False)
    plot_similarity_heatmap(sim_matrix, similarity_dir, suffix)

    semantic_questions = deduplicate_questions(unique_questions, sim_matrix, args.similarity_threshold)

    embeddings_dedup = embed_model.encode(semantic_questions, convert_to_tensor=True)
    sim_matrix_dedup = util.pytorch_cos_sim(embeddings_dedup, embeddings_dedup).cpu().numpy()
    pd.DataFrame(sim_matrix_dedup).to_csv(
        similarity_dir / f"question_similarity_matrix_dedup{suffix}.csv", index=False
    )
    pairs_dedup_df = extract_similar_pairs(
        sim_matrix_dedup, semantic_questions, args.similarity_threshold
    )
    pairs_dedup_df.to_csv(
        similarity_dir / f"high_similarity_pairs_dedup_0_8{suffix}.csv", index=False
    )

    # Keep historical/canonical names used elsewhere
    write_question_list(similarity_dir / f"questions_dedup_0_80{suffix}.csv", semantic_questions)
    write_question_list(semantic_file, semantic_questions)

    logger.info(f"✓ Saved semantic-dedup questions ({len(semantic_questions)}) -> {semantic_file}")
    logger.info("🎉 Done.")


if __name__ == "__main__":
    main()
