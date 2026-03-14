from __future__ import annotations

import glob
import logging
from pathlib import Path

import pandas as pd

from src.questions.question_io import write_canonical_selected_questions
from src.questions.stats import (
    compute_question_matrix,
    drop_duplicates,
    drop_similar,
    find_similar_questions,
    normalize_questions,
    run_permutation_evaluation,
)


def build_and_clean(
    predictions_dir: Path,
    pattern: str,
    metric_columns: set[str],
    stats_dir: Path,
    suffix: str,
    jaccard_value: float,
    results_df: pd.DataFrame,
    logger: logging.Logger,
    perc_success: float,
) -> Path:
    cols_keep = ["Question", "Precision", "Recall", "p_value", "significant"]
    float_cols = ["Precision", "Recall", "p_value"]
    results_base = results_df[results_df["Precision"] >= perc_success].copy()
    results_base = results_base[results_base["Recall"] >= 0.1]
    results_base = results_base[cols_keep]
    results_base[float_cols] = results_base[float_cols].round(3)

    rows = []
    for fn in sorted(glob.glob(str(predictions_dir / pattern))):
        df = pd.read_csv(fn)
        founder_cols = [c for c in df.columns if c not in metric_columns]
        sub = df[df["Question"].isin(results_base["Question"].tolist())].copy()
        if not sub.empty:
            sub = sub[["Question"] + founder_cols]
            sub["Source_File"] = Path(fn).name
            rows.append(sub)

    if not rows:
        raise RuntimeError(
            f"No prediction rows found for baseline question set using pattern={pattern}."
        )

    pred_df = pd.concat(rows, ignore_index=True).drop_duplicates(subset="Question").reset_index(
        drop=True
    )
    prec_map = results_df.set_index("Question")["Precision"].to_dict()
    pred_df["Prec"] = pred_df["Question"].map(prec_map)
    pred_df["__set"] = 0

    norm = normalize_questions(pred_df)
    deduped, _ = drop_duplicates(norm)
    mat = compute_question_matrix(deduped)
    names = deduped["Question"].tolist()
    pairs = find_similar_questions(mat, names, thresh=jaccard_value)
    clean = drop_similar(deduped, pairs)

    clean["Question"] = (
        clean["Question"]
        .str.strip()
        .str.lower()
        .apply(lambda x: "".join(ch for ch in x if ch.isprintable()))
    )
    results_base["Question"] = (
        results_base["Question"]
        .str.strip()
        .str.lower()
        .apply(lambda x: "".join(ch for ch in x if ch.isprintable()))
    )

    merged = clean.merge(results_base, on="Question", how="left")
    non_pred = ["Source_File", "Question", "Precision", "Recall", "p_value", "significant"]
    pred_cols = [c for c in merged.columns if c not in non_pred]
    merged = merged[non_pred + pred_cols]

    out_file = stats_dir / (
        f"high_precision_questions{suffix}_baseline_j{str(jaccard_value).replace('.', '_')}.csv"
    )
    merged.to_csv(out_file, index=False)
    logger.info(f"Saved cleaned baseline results to: {out_file}")
    return out_file


def run_stats_postprocess(
    *,
    model_dir: Path,
    prediction_file: Path,
    suffix: str,
    model_name: str,
    m: int,
    alpha: float,
    b_null: int,
    perc_success: float,
    jaccard_value: float,
    write_plots: bool,
    logger: logging.Logger,
) -> None:
    if not prediction_file.exists():
        raise FileNotFoundError(f"Validation prediction file missing: {prediction_file}")

    stats_dir = model_dir / "validation_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = model_dir / "validation_predictions"
    pattern = prediction_file.name

    metric_columns = {
        "Index",
        "Question",
        "Pass Rate",
        "Prec",
        "TP",
        "FP",
        "TN",
        "FN",
        "Rec",
        "F1",
        "F0.5",
        "Prec_Train",
        "Prec_Validation",
        "Prec_Test",
        "Prec_Mean",
    }
    excluded_questions = {"Founder Index", "Dataset", "Success", "SUCCESS_PROPORTION"}

    perm_out = stats_dir / f"permutation_test_results_m{m}{suffix}.csv"
    results_df = run_permutation_evaluation(
        predictions_dir=predictions_dir,
        output_path=perm_out,
        model=model_name,
        suffix=suffix,
        metric_cols=metric_columns,
        m=m,
        B_null=b_null,
        alpha=alpha,
        excluded=excluded_questions,
        pattern=pattern,
    )

    expected_high_precision = stats_dir / (
        f"high_precision_questions{suffix}_baseline_j{str(jaccard_value).replace('.', '_')}.csv"
    )
    expected_selected_questions = model_dir / "test_questions" / f"selected_questions_all{suffix}.csv"
    if expected_high_precision.exists() and expected_selected_questions.exists():
        logger.info(
            "Postprocess artifacts already exist; skipping rewrite."
        )
        return

    if write_plots:
        from src.utils.plotting import (
            plot_precision_colored_bar_chart,
            plot_precision_recall_f05_bars,
        )

        plot_precision_recall_f05_bars(results_df, perc_success, stats_dir, m, suffix)
        plot_precision_colored_bar_chart(results_df, perc_success, alpha, stats_dir, m, suffix)

    high_precision_path = build_and_clean(
        predictions_dir=predictions_dir,
        pattern=pattern,
        metric_columns=metric_columns,
        stats_dir=stats_dir,
        suffix=suffix,
        jaccard_value=jaccard_value,
        results_df=results_df,
        logger=logger,
        perc_success=perc_success,
    )

    write_canonical_selected_questions(
        high_precision_file=high_precision_path,
        test_question_dir=model_dir / "test_questions",
        suffix=suffix,
        logger=logger,
    )
