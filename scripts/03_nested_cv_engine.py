#!/usr/bin/env python3
"""Nested-CV ablation analysis on precomputed predictions.

This script performs a fully offline (no API) nested CV evaluation over the
already-computed founder-level prediction matrices in `precomputed/`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.questions.question_filtering import (  # noqa: E402
    construct_predictions_file_list,
    filter_questions,
)
from src.utils.scoring_utils import (  # noqa: E402
    compute_weights,
    evaluate_test_fold,
    select_best_hyperparams,
)


def _summarise_block(df_block: pd.DataFrame, label: str) -> None:
    """Print mean/std of key metrics + pooled confusion totals."""
    if df_block.empty:
        print(f"\n{label}: (no rows)")
        return

    agg = df_block[["Precision_Outer", "Recall_Outer", "F05_Outer"]].agg(["mean", "std"])
    p_mean, p_std = agg.loc["mean", "Precision_Outer"], agg.loc["std", "Precision_Outer"]
    r_mean, r_std = agg.loc["mean", "Recall_Outer"], agg.loc["std", "Recall_Outer"]
    f_mean, f_std = agg.loc["mean", "F05_Outer"], agg.loc["std", "F05_Outer"]

    tp = int(df_block["TP"].sum())
    fp = int(df_block["FP"].sum())
    tn = int(df_block["TN"].sum())
    fn = int(df_block["FN"].sum())

    pooled_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    pooled_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    print(f"\n📊 {label}")
    print(
        "  Fold-avg: "
        f"prec={p_mean:.3f} ± {p_std:.3f} | "
        f"rec={r_mean:.3f} ± {r_std:.3f} | "
        f"F0.5={f_mean:.3f} ± {f_std:.3f}"
    )
    print(
        "  Pooled:   "
        f"prec={pooled_prec:.3f} | rec={pooled_rec:.3f} | "
        f"TP={tp} FP={fp} TN={tn} FN={fn}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run nested CV on precomputed RRF predictions.")
    parser.add_argument("--results_dir_extension", type=str, default="precomputed")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["llm", "llm_expert", "expert_only"],
        required=True,
        help="Which question mode to load",
    )
    parser.add_argument(
        "--similarity-metric",
        type=str,
        choices=["jaccard", "hamming", "cosine-cluster"],
        required=True,
        help="Similarity metric used to remove redundant questions",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        required=True,
        help="Threshold for filtering similar questions",
    )
    parser.add_argument(
        "--optimise-for",
        type=str,
        choices=["precision", "f0.5", "f1", "f2"],
        default="f0.5",
        help="Metric to optimise",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        choices=["precision", "f0.5"],
        default="precision",
        help="Sort final questions by this metric",
    )
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Exit immediately if output CSV already exists (default: true)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation even if output CSV exists",
    )
    return parser.parse_args()


def build_output_path(args: argparse.Namespace) -> Path:
    suffix = "_anonymised"
    similarity_str = f"{args.similarity_threshold:.2f}".replace(".", "_")
    opt_str = f"optimise{args.optimise_for.replace('.', '_').upper()}"
    sort_str = f"sortby{args.sort_by.replace('.', '_')}"

    out_dir = project_root / args.results_dir_extension / "gpt_4o_mini" / "ablation_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / (
        f"nested_cv_{args.mode}_{suffix}"
        f"similarity_{args.similarity_metric}_{similarity_str}_{opt_str}_{sort_str}.csv"
    )


def main() -> None:
    args = parse_args()

    n_splits = args.n_splits
    n_repeats = args.n_repeats
    weighting = 0
    exponents = [float(weighting)]

    suffix = "_anonymised"
    predictions_dir = project_root / args.results_dir_extension / "gpt_4o_mini" / "test_predictions"

    out_path = build_output_path(args)
    print(f"Output path: {out_path}")
    if out_path.exists() and args.skip_existing and not args.force:
        print(f"🛑 Skipping analysis — results already exist at:\n{out_path}")
        return

    predictions_files = construct_predictions_file_list(args, suffix, predictions_dir=predictions_dir)
    df = filter_questions(predictions_dir, predictions_files, suffix, args, sort_by=args.sort_by)

    special_rows = df[df["Question"].isin(["Founder Index", "Dataset", "Success"])]
    filtered_df = df[~df["Question"].isin(["Founder Index", "Dataset", "Success"])].copy()

    metric_cols = [
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
    ]
    founder_cols = [c for c in filtered_df.columns if c not in metric_cols]
    filtered_df[founder_cols] = (
        filtered_df[founder_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    )

    X = filtered_df[founder_cols].values
    q_count, founder_count = X.shape

    success_series = special_rows[special_rows["Question"] == "Success"][founder_cols].iloc[0]
    success_values = pd.to_numeric(success_series, errors="coerce").fillna(0).astype(int).values

    results: list[dict] = []
    founder_indices = np.arange(founder_count)
    score_thresholds = list(range(1, 71))

    for repeat in range(1, n_repeats + 1):
        print(f"\n🔁 Repeat {repeat}/{n_repeats}")
        outer_skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=42 + (repeat - 1)
        )
        repeat_rows: list[dict] = []

        for outer_idx, (train_idx, test_idx) in enumerate(
            outer_skf.split(founder_indices, success_values), start=1
        ):
            print(f"  🧪 Fold {outer_idx}/{n_splits}")
            train_ids = founder_indices[train_idx]
            test_ids = founder_indices[test_idx]
            y_train = success_values[train_ids]
            y_test = success_values[test_ids]
            X_train = X[:, train_ids]

            best_combo, prec_array, ratio_array, best_mean_f05 = select_best_hyperparams(
                X_train,
                y_train,
                train_ids,
                weighting,
                exponents,
                score_thresholds,
                q_count,
                success_values,
                n_splits,
                outer_idx + repeat,
                optimise_for=args.optimise_for,
            )

            exp_opt, nq_opt, t_opt = best_combo
            print(
                f"    ✔ best(inner): exp={exp_opt}, n_q={nq_opt}, "
                f"t={t_opt}, F0.5={best_mean_f05:.3f}"
            )

            W_full, _ = compute_weights(weighting, exp_opt, prec_array, ratio_array)
            (
                preds_test,
                yt,
                tp,
                fp,
                tn,
                fn,
                p_out,
                r_out,
                f05_out,
                f05_out_sk,
                f1_out,
                f2_out,
                mcc_out,
            ) = evaluate_test_fold(X, W_full, nq_opt, t_opt, test_ids, success_values)

            print(
                f"    ✔ outer scores: prec={p_out:.3f}, rec={r_out:.3f}, F0.5={f05_out:.3f}"
            )

            row = {
                "Repeat": repeat,
                "Outer_Fold": outer_idx,
                "Best_Exp": exp_opt,
                "Best_n_q": nq_opt,
                "Best_Thr": t_opt,
                "Precision_Outer": round(p_out, 3),
                "Recall_Outer": round(r_out, 3),
                "F05_Outer": round(f05_out, 3),
                "TP": int(tp),
                "FP": int(fp),
                "TN": int(tn),
                "FN": int(fn),
                "Pred_Pos": int(preds_test.sum()),
                "Actual_Pos": int(yt.sum()),
                "f05_out_sk": round(f05_out_sk, 3),
                "f1_out": round(f1_out, 3),
                "f2_out": round(f2_out, 3),
                "mcc_out": round(mcc_out, 3),
            }
            results.append(row)
            repeat_rows.append(row)

        repeat_df = pd.DataFrame(repeat_rows)
        _summarise_block(repeat_df, label=f"Repeat {repeat}/{n_repeats}")

    out_df = pd.DataFrame(results)
    _summarise_block(
        out_df,
        label=f"Overall across all repeats (n_repeats={n_repeats}, n_splits={n_splits})",
    )

    out_df.to_csv(out_path, index=False)
    print(f"\n✅ Saved repeated results to {out_path}")


if __name__ == "__main__":
    main()
