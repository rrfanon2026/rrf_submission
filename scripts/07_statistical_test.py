#!/usr/bin/env python3
"""Paired bootstrap significance test: RRF vs vanilla baseline.

Replays the primary-model nested CV to reconstruct per-founder predictions,
then runs a paired bootstrap on F0.5 difference for each CV repeat.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.questions.question_filtering import (  # noqa: E402
    construct_predictions_file_list,
    filter_questions,
)
from src.utils.scoring_utils import (  # noqa: E402
    compute_weights,
    evaluate_test_fold,
    select_best_hyperparams,
)

# ---------------------------------------------------------------------------
# Config — must match configs/primary_model_locked.json exactly
# ---------------------------------------------------------------------------
MODE = "llm"
SIMILARITY_METRIC = "hamming"
SIMILARITY_THRESHOLD = 0.15
OPTIMISE_FOR = "f0.5"
SORT_BY = "f0.5"
N_SPLITS = 10
N_REPEATS = 10
SUFFIX = "_anonymised"

PREDICTIONS_DIR = PROJECT_ROOT / "precomputed" / "gpt_4o_mini" / "test_predictions"
SAVED_FOLD_METRICS = (
    PROJECT_ROOT / "precomputed" / "gpt_4o_mini" / "primary_results"
    / "primary_model_fold_metrics.csv"
)
VANILLA_CSV = (
    PROJECT_ROOT / "precomputed" / "gpt_4o_mini" / "vanilla"
    / "vanilla_few_shot_o3_anonymised_rep_1.csv"
)

B_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def f05(tp: int, fp: int, fn: int) -> float:
    """F0.5 from confusion counts — same definition as scoring_utils."""
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 0.0 if (p + r) == 0 else 1.25 * p * r / (0.25 * p + r)


def f05_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute F0.5 from binary prediction and label arrays."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return f05(tp, fp, fn)


# ---------------------------------------------------------------------------
# Step 1: Replay nested CV → per-founder predictions
# ---------------------------------------------------------------------------
def replay_nested_cv() -> tuple[pd.DataFrame, list[str]]:
    """Re-run the primary-model nested CV and return per-founder predictions.

    Returns:
        (DataFrame with columns founder_idx/repeat/fold/y_true/y_rrf,
         list of founder column names indexed by founder_idx)
    """
    # Build an args namespace that filter_questions / construct_predictions_file_list expect
    args = argparse.Namespace(
        mode=MODE,
        similarity_metric=SIMILARITY_METRIC,
        similarity_threshold=SIMILARITY_THRESHOLD,
        optimise_for=OPTIMISE_FOR,
        sort_by=SORT_BY,
    )

    predictions_files = construct_predictions_file_list(args, SUFFIX, predictions_dir=PREDICTIONS_DIR)
    df = filter_questions(PREDICTIONS_DIR, predictions_files, SUFFIX, args, sort_by=SORT_BY)

    # Separate special rows from question rows
    special_rows = df[df["Question"].isin(["Founder Index", "Dataset", "Success"])]
    filtered_df = df[~df["Question"].isin(["Founder Index", "Dataset", "Success"])].copy()

    metric_cols = [
        "Index", "Question", "Pass Rate", "Prec", "TP", "FP", "TN", "FN",
        "Rec", "F1", "F0.5", "Prec_Train", "Prec_Validation", "Prec_Test", "Prec_Mean",
    ]
    founder_cols = [c for c in filtered_df.columns if c not in metric_cols]
    filtered_df[founder_cols] = (
        filtered_df[founder_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    )

    X = filtered_df[founder_cols].values  # shape (Q, F)
    q_count, founder_count = X.shape

    success_series = special_rows[special_rows["Question"] == "Success"][founder_cols].iloc[0]
    success_values = pd.to_numeric(success_series, errors="coerce").fillna(0).astype(int).values

    founder_indices = np.arange(founder_count)
    score_thresholds = list(range(1, 71))
    exponents = [0.0]
    weighting = 0

    rows: list[dict] = []

    for repeat in range(1, N_REPEATS + 1):
        outer_skf = StratifiedKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=42 + (repeat - 1)
        )

        for outer_idx, (train_idx, test_idx) in enumerate(
            outer_skf.split(founder_indices, success_values), start=1
        ):
            train_ids = founder_indices[train_idx]
            test_ids = founder_indices[test_idx]
            y_train = success_values[train_ids]
            X_train = X[:, train_ids]

            best_combo, prec_array, ratio_array, _ = select_best_hyperparams(
                X_train, y_train, train_ids, weighting, exponents,
                score_thresholds, q_count, success_values, N_SPLITS,
                outer_idx + repeat, optimise_for=OPTIMISE_FOR,
            )

            exp_opt, nq_opt, t_opt = best_combo
            W_full, _ = compute_weights(weighting, exp_opt, prec_array, ratio_array)
            preds_test, y_test, tp, fp, tn, fn, *_ = evaluate_test_fold(
                X, W_full, nq_opt, t_opt, test_ids, success_values
            )

            # Collect per-founder predictions
            for i, fid in enumerate(test_ids):
                rows.append({
                    "founder_idx": int(fid),
                    "repeat": repeat,
                    "fold": outer_idx,
                    "y_true": int(y_test[i]),
                    "y_rrf": int(preds_test[i]),
                    # For sanity checking:
                    "_tp": int(tp), "_fp": int(fp), "_tn": int(tn), "_fn": int(fn),
                })

    return pd.DataFrame(rows), founder_cols


# ---------------------------------------------------------------------------
# Step 2: Verify reconstructed predictions match saved fold metrics
# ---------------------------------------------------------------------------
def verify_fold_metrics(rrf_df: pd.DataFrame) -> None:
    """Hard-fail if reconstructed fold-level confusion counts don't match saved file."""
    saved = pd.read_csv(SAVED_FOLD_METRICS)
    mismatches = 0

    for _, saved_row in saved.iterrows():
        rep = int(saved_row["Repeat"])
        fold = int(saved_row["Outer_Fold"])
        mask = (rrf_df["repeat"] == rep) & (rrf_df["fold"] == fold)
        fold_rows = rrf_df[mask]

        # Reconstruct confusion counts from per-founder predictions
        tp = int(((fold_rows["y_rrf"] == 1) & (fold_rows["y_true"] == 1)).sum())
        fp = int(((fold_rows["y_rrf"] == 1) & (fold_rows["y_true"] == 0)).sum())
        tn = int(((fold_rows["y_rrf"] == 0) & (fold_rows["y_true"] == 0)).sum())
        fn = int(((fold_rows["y_rrf"] == 0) & (fold_rows["y_true"] == 1)).sum())

        if (tp != int(saved_row["TP"]) or fp != int(saved_row["FP"])
                or tn != int(saved_row["TN"]) or fn != int(saved_row["FN"])):
            print(
                f"  MISMATCH repeat={rep} fold={fold}: "
                f"got TP={tp} FP={fp} TN={tn} FN={fn}, "
                f"expected TP={int(saved_row['TP'])} FP={int(saved_row['FP'])} "
                f"TN={int(saved_row['TN'])} FN={int(saved_row['FN'])}"
            )
            mismatches += 1

    if mismatches > 0:
        raise RuntimeError(
            f"FATAL: {mismatches} fold-level mismatches with saved metrics. "
            "Reconstruction does not match the paper's reported results."
        )
    print("  All 100 fold-level TP/FP/TN/FN match saved metrics exactly.")


# ---------------------------------------------------------------------------
# Step 3: Load vanilla predictions
# ---------------------------------------------------------------------------
def load_vanilla_predictions() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load per-founder binary predictions from the vanilla CSV.

    Returns (y_true, y_pred, founder_col_names).
    """
    df = pd.read_csv(VANILLA_CSV)
    founder_cols = list(df.columns[15:])

    # Row 0 = Dataset, Row 1 = Success (ground truth), Row 2 = SUCCESS_PROPORTION, Row 3 = predictions
    y_true = pd.to_numeric(df.iloc[1, 15:], errors="coerce").fillna(0).astype(int).values
    y_pred = pd.to_numeric(df.iloc[3, 15:], errors="coerce").fillna(0).astype(int).values

    return y_true, y_pred, founder_cols


# ---------------------------------------------------------------------------
# Step 4: Paired bootstrap
# ---------------------------------------------------------------------------
def paired_bootstrap_f05(
    y_true: np.ndarray,
    rrf_pred: np.ndarray,
    vanilla_pred: np.ndarray,
    n_bootstrap: int = B_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Paired bootstrap test on F0.5 difference (RRF - vanilla).

    Resamples founders with replacement, recomputes F0.5 for both methods
    on each bootstrap sample, and computes the distribution of deltas.
    """
    n = len(y_true)
    observed_rrf = f05_from_arrays(y_true, rrf_pred)
    observed_van = f05_from_arrays(y_true, vanilla_pred)
    observed_delta = observed_rrf - observed_van

    rng = np.random.RandomState(seed)
    deltas = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        rrf_b = rrf_pred[idx]
        van_b = vanilla_pred[idx]
        deltas[b] = f05_from_arrays(yt, rrf_b) - f05_from_arrays(yt, van_b)

    ci_lo = float(np.percentile(deltas, 2.5))
    ci_hi = float(np.percentile(deltas, 97.5))
    p_value = float(np.mean(deltas <= 0))  # one-sided: proportion where RRF is not better

    return {
        "f05_rrf": observed_rrf,
        "f05_vanilla": observed_van,
        "delta": observed_delta,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_value": p_value,
        "n_bootstrap": n_bootstrap,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 72)
    print("PAIRED BOOTSTRAP SIGNIFICANCE TEST: RRF vs Few-shot o3 (vanilla)")
    print("=" * 72)

    # -- Step 1: Replay nested CV --
    print("\n[1/5] Replaying nested CV to reconstruct per-founder predictions ...")
    rrf_df, rrf_founder_cols = replay_nested_cv()
    print(f"  Reconstructed {len(rrf_df)} per-founder predictions "
          f"({rrf_df['founder_idx'].nunique()} founders, {N_REPEATS} repeats).")

    # -- Step 2: Verify --
    print("\n[2/5] Verifying fold-level metrics match saved results ...")
    verify_fold_metrics(rrf_df)

    # -- Step 3: Load vanilla --
    print("\n[3/5] Loading vanilla baseline predictions ...")
    van_y_true, van_y_pred, van_founder_cols = load_vanilla_predictions()
    van_f05 = f05_from_arrays(van_y_true, van_y_pred)
    print(f"  Vanilla F0.5 = {van_f05:.3f}  (expected 0.088)")
    if abs(van_f05 - 0.088) > 0.001:
        print(f"  WARNING: vanilla F0.5 = {van_f05:.4f}, expected ~0.088")

    # -- Step 4: Paired bootstrap per repeat --
    print(f"\n[4/5] Running paired bootstrap (B={B_BOOTSTRAP}) for each of {N_REPEATS} repeats ...")

    # Align RRF founder indices with vanilla founder columns.
    # RRF founder_idx maps to rrf_founder_cols[idx]. The last column is "Source"
    # (a metadata column treated as a founder by the original pipeline).
    # Vanilla has the same 8,173 real founder columns but in a different order.
    # Build a mapping: rrf_founder_idx -> vanilla_col_position (or -1 for Source).
    van_col_to_pos = {col: i for i, col in enumerate(van_founder_cols)}
    rrf_to_van = []
    real_founder_mask = []  # True for real founders, False for Source
    for idx, col_name in enumerate(rrf_founder_cols):
        if col_name in van_col_to_pos:
            rrf_to_van.append(van_col_to_pos[col_name])
            real_founder_mask.append(True)
        else:
            rrf_to_van.append(-1)
            real_founder_mask.append(False)
    real_founder_mask = np.array(real_founder_mask)
    n_real_founders = int(real_founder_mask.sum())
    print(f"  Aligned {n_real_founders} real founders (excluded {len(rrf_founder_cols) - n_real_founders} metadata columns).")

    bootstrap_results: list[dict] = []

    for rep in range(1, N_REPEATS + 1):
        rep_df = rrf_df[rrf_df["repeat"] == rep].sort_values("founder_idx")

        # Filter to real founders only (exclude Source column)
        rep_real = rep_df[rep_df["founder_idx"].isin(
            np.where(real_founder_mask)[0]
        )].sort_values("founder_idx")

        rrf_pred_rep = rep_real["y_rrf"].values
        y_true_rep = rep_real["y_true"].values

        # Map RRF founder indices to vanilla positions and extract vanilla preds
        rrf_idxs = rep_real["founder_idx"].values
        van_positions = np.array([rrf_to_van[i] for i in rrf_idxs])
        van_pred_aligned = van_y_pred[van_positions]
        van_y_true_aligned = van_y_true[van_positions]

        # Sanity: labels must match between vanilla and RRF
        assert np.array_equal(y_true_rep, van_y_true_aligned), (
            f"Repeat {rep}: ground truth labels differ between RRF and vanilla"
        )

        result = paired_bootstrap_f05(
            y_true_rep, rrf_pred_rep, van_pred_aligned,
            seed=BOOTSTRAP_SEED + rep,
        )
        result["repeat"] = rep
        bootstrap_results.append(result)

        print(
            f"  Repeat {rep:2d}: RRF F0.5={result['f05_rrf']:.4f}  "
            f"delta={result['delta']:+.4f}  "
            f"95% CI [{result['ci_lo']:+.4f}, {result['ci_hi']:+.4f}]  "
            f"p={result['p_value']:.4f}"
        )

    # -- Summarise across repeats --
    deltas = [r["delta"] for r in bootstrap_results]
    p_values = [r["p_value"] for r in bootstrap_results]
    ci_los = [r["ci_lo"] for r in bootstrap_results]
    ci_his = [r["ci_hi"] for r in bootstrap_results]

    median_delta = float(np.median(deltas))
    median_ci_lo = float(np.median(ci_los))
    median_ci_hi = float(np.median(ci_his))
    worst_p = max(p_values)
    n_significant = sum(1 for p in p_values if p < 0.05)

    print("\n  --- Summary across repeats ---")
    print(f"  Median delta:  {median_delta:+.4f}")
    print(f"  Median 95% CI: [{median_ci_lo:+.4f}, {median_ci_hi:+.4f}]")
    print(f"  Worst p-value: {worst_p:.4f}")
    print(f"  Significant at alpha=0.05: {n_significant}/{N_REPEATS} repeats")

    # -- Step 5: Supplementary one-sample t-test --
    print("\n[5/5] Supplementary: one-sample t-test (10 repeat-level F0.5 vs 0.088) ...")
    repeat_f05_values = np.array([r["f05_rrf"] for r in bootstrap_results])
    t_stat, t_p_two = stats.ttest_1samp(repeat_f05_values, van_f05)
    t_p_one = t_p_two / 2 if t_stat > 0 else 1 - t_p_two / 2  # one-sided
    print(f"  Repeat F0.5 scores: {repeat_f05_values.round(4)}")
    print(f"  Mean = {repeat_f05_values.mean():.4f}, SD = {repeat_f05_values.std(ddof=1):.4f}")
    print(f"  t({N_REPEATS - 1}) = {t_stat:.3f}, two-sided p = {t_p_two:.4f}, one-sided p = {t_p_one:.4f}")

    # -- Paper-ready output --
    print("\n" + "=" * 72)
    print("PAPER-READY OUTPUT")
    print("=" * 72)

    print("\n--- Methods sentence ---")
    print(
        "To assess statistical significance, we performed a paired bootstrap test "
        "(B = 10,000) comparing per-founder RRF out-of-fold predictions against the "
        "best vanilla baseline (Few-shot o3) on the same held-out founders. "
        "For each of the 10 CV repeats, we resampled founders with replacement "
        "and recomputed F_{0.5} for both methods, yielding a distribution of "
        "F_{0.5} differences. We report the 95\\% percentile confidence interval "
        "and the bootstrap p-value (proportion of resampled deltas <= 0)."
    )

    print("\n--- Results sentence ---")
    if n_significant == N_REPEATS:
        print(
            f"The RRF's F_{{0.5}} improvement over the Few-shot o3 baseline was "
            f"statistically significant across all 10 CV repeats "
            f"(median $\\Delta F_{{0.5}}$ = {median_delta:.3f}, "
            f"95\\% CI [{median_ci_lo:.3f}, {median_ci_hi:.3f}]; "
            f"all bootstrap $p < {worst_p:.3f}$)."
        )
    else:
        print(
            f"The RRF's F_{{0.5}} improvement over the Few-shot o3 baseline was "
            f"significant in {n_significant}/{N_REPEATS} CV repeats "
            f"(median $\\Delta F_{{0.5}}$ = {median_delta:.3f}; "
            f"worst-case $p = {worst_p:.3f}$)."
        )

    print("\n--- Table-note sentence ---")
    print(
        f"$\\dagger$ Paired bootstrap test (B = 10,000) on per-founder predictions; "
        f"significant at $\\alpha = 0.05$ in all 10 CV repeats "
        f"(worst-case $p = {worst_p:.3f}$)."
        if n_significant == N_REPEATS else
        f"$\\dagger$ Paired bootstrap test (B = 10,000) on per-founder predictions; "
        f"significant at $\\alpha = 0.05$ in {n_significant}/10 CV repeats."
    )

    print("\n--- Supplementary: one-sample t-test ---")
    print(
        f"One-sample t-test of 10 repeat-level F_{{0.5}} scores against the vanilla "
        f"baseline (F_{{0.5}} = {van_f05:.3f}): "
        f"t(9) = {t_stat:.2f}, one-sided p = {t_p_one:.4f}."
    )

    print()


if __name__ == "__main__":
    main()
