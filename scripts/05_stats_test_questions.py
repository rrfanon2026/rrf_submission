### Currently stats testing not applied so this is likely to be simplified quite a bit when final decisions made

import logging
import pandas as pd
from pathlib import Path
import argparse
import glob
import sys

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.plotting import plot_precision_recall_f05_bars, plot_precision_colored_bar_chart
from src.questions.stats import (
    normalize_questions,
    drop_duplicates,
    compute_question_matrix,
    find_similar_questions,
    drop_similar,
    run_permutation_evaluation
)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate predictions with permutation testing.")
    parser.add_argument("--model", type=str, default="gpt_4o_mini")
    parser.add_argument('--results_dir_extension', type=str, default='precomputed')
    parser.add_argument("--m", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--B_null", type=int, default=1000)
    parser.add_argument("--perc_success", type=float, default=0.10)
    parser.add_argument("--jaccard_value", type=float, default=1.00)

    return parser.parse_args()

# --- Helper to build, clean, and merge ---
def build_and_clean(questions, results_subset, out_name, jacc_thresh, pattern):
    rows = []
    for fn in sorted(glob.glob(str(predictions_dir / pattern))):
        df = pd.read_csv(fn)
        founder_cols = [c for c in df.columns if c not in metric_columns]
        sub = df[df["Question"].isin(questions)].copy()
        if not sub.empty:
            sub = sub[["Question"] + founder_cols]
            sub["Source_File"] = Path(fn).name
            rows.append(sub)

    if not rows:
        logger.warning(f"No prediction rows found for '{out_name}' at jacc={jacc_thresh}, skipping.")
        return

    pred_df = pd.concat(rows, ignore_index=True).drop_duplicates(subset="Question").reset_index(drop=True)

    # attach metrics
    prec_map = results_df.set_index("Question")["Precision"].to_dict()
    pred_df["Prec"] = pred_df["Question"].map(prec_map)
    pred_df["__set"] = 0

    # normalize, dedup, drop-similar
    norm = normalize_questions(pred_df)
    deduped, _ = drop_duplicates(norm)
    mat = compute_question_matrix(deduped)
    names = deduped["Question"].tolist()
    pairs = find_similar_questions(mat, names, thresh=jacc_thresh)
    clean = drop_similar(deduped, pairs)

    # normalize text for merge
    clean["Question"] = (
        clean["Question"].str.strip().str.lower()
        .apply(lambda x: ''.join(ch for ch in x if ch.isprintable()))
    )
    results_subset["Question"] = (
        results_subset["Question"].str.strip().str.lower()
        .apply(lambda x: ''.join(ch for ch in x if ch.isprintable()))
    )

    merged = clean.merge(results_subset, on="Question", how="left")
    non_pred = ['Source_File', 'Question', 'Precision', 'Recall', 'p_value', 'significant']
    pred_cols = [c for c in merged.columns if c not in non_pred]
    merged = merged[non_pred + pred_cols]

    out_file = stats_dir / f"high_precision_questions{suffix}_{out_name}_j{str(jacc_thresh).replace('.','_')}.csv"
    merged.to_csv(out_file, index=False)
    logger.info(f"Saved cleaned '{out_name}' results at jacc={jacc_thresh} to: {out_file}")

if __name__ == "__main__":

    # --- Argument Parser ---
    args = parse_args()

    # --- Config ---
    B_null = args.B_null
    alpha = args.alpha
    m = args.m
    perc_success = args.perc_success
    model = args.model.replace('-','_')
    suffix = "_anonymised"
    pattern = f"predictions_val_set_*_question{suffix}.csv"

    # --- Paths ---
    base_dir = Path(__file__).resolve().parent.parent
    results_dir = base_dir / args.results_dir_extension / model
    predictions_dir = results_dir / "validation_predictions/"

    stats_dir = results_dir / "validation_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    print(f"Prediction dir is: {predictions_dir}")
    print(f"Suffix is: {suffix}")

    # --- Logging setup ---
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    # --- Constants ---
    metric_columns = {
        "Index", "Question", "Pass Rate", "Prec", "TP", "FP", "TN", "FN",
        "Rec", "F1", "F0.5", "Prec_Train", "Prec_Validation", "Prec_Test", "Prec_Mean"
    }
    excluded_questions = {'Founder Index', 'Dataset', 'Success', 'SUCCESS_PROPORTION'}

    # --- Run permutation evaluation ---
    perm_out = stats_dir / f"permutation_test_results_m{m}{suffix}.csv"
    results_df = run_permutation_evaluation(
        predictions_dir=predictions_dir,
        output_path=perm_out,
        model=model,
        suffix=suffix,
        metric_cols=metric_columns,
        m=m,
        B_null=B_null,
        alpha=alpha,
        excluded=excluded_questions,
        pattern=pattern
    )

    # --- 1) Significant-only filter & save ---
    cols_keep = ['Question', 'Precision', 'Recall', 'p_value', 'significant']
    float_cols = ['Precision', 'Recall', 'p_value']

    results_sig = results_df[results_df['significant']].copy()
    # drop any with recall < 0.1
    results_sig = results_sig[results_sig['Recall'] >= 0.1]
    results_sig = results_sig[cols_keep]
    results_sig[float_cols] = results_sig[float_cols].round(3)
    # sig_csv = stats_dir / f"permutation_test_results_m{m}_sig.csv"
    # results_sig.to_csv(sig_csv, index=False)
    # logger.info(f"Saved significant-only results to: {sig_csv} (after excluding recall<0.1)")

    # --- 2) Baseline (precision ≥ random chance) filter & save ---
    results_base = results_df[results_df['Precision'] >= perc_success].copy()
    # drop any with recall < 0.1
    results_base = results_base[results_base['Recall'] >= 0.1]
    results_base = results_base[cols_keep]
    results_base[float_cols] = results_base[float_cols].round(3)
    # base_csv = stats_dir / f"permutation_test_results_m{m}_baseline.csv"
    # results_base.to_csv(base_csv, index=False)
    # logger.info(f"Saved baseline-threshold results to: {base_csv} (after excluding recall<0.1)")

    # --- Shared plots on full results ---
    plot_precision_recall_f05_bars(results_df, perc_success, stats_dir, m, suffix)
    plot_precision_colored_bar_chart(results_df, perc_success, alpha, stats_dir, m, suffix)

    print(results_base.head())

    # --- Use no threshold to keep all questions for now ---
    logger.info(f"Running de-dup & de-similarize at Jaccard ≥ {args.jaccard_value}")
    build_and_clean(results_base['Question'].tolist(), results_base.copy(deep=True), "baseline", args.jaccard_value, pattern)
