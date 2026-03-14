#!/usr/bin/env python3
"""Run ElasticNet LogReg benchmark with 10 repeats × 10 folds.

Trains an elastic-net logistic regression on the same binary question-response
matrix used by RRF, with threshold optimisation on the training set.

Usage:
    python scripts/06_logreg_benchmark.py --results-dir precomputed/gpt_4o_mini
    python scripts/06_logreg_benchmark.py --results-dir precomputed/gemini_2_5_pro
    python scripts/06_logreg_benchmark.py --results-dir precomputed/claude_sonnet_4_5
    python scripts/06_logreg_benchmark.py --results-dir precomputed/gpt_5_2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

project_root = Path(__file__).resolve().parent.parent


def load_data(results_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Load binary prediction matrix and success labels."""
    base_dir = project_root / results_dir / "test_predictions"
    all_files = sorted(base_dir.glob("*anonymised.csv"))

    combined = []
    success_row = None
    for f in all_files:
        df = pd.read_csv(f)
        if "Source" not in df.columns:
            df["Source"] = "llm"
        df = df[df["Source"] == "llm"]
        if success_row is None:
            s = df[df["Question"] == "Success"]
            if not s.empty:
                success_row = s
        combined.append(
            df[~df["Question"].isin(["Success", "Founder Index", "Dataset", "SUCCESS_PROPORTION"])]
        )

    df_all = pd.concat(combined, ignore_index=True)
    metric_cols = [
        "Index", "Question", "Source", "Pass Rate", "Prec", "Rec", "F1",
        "TP", "FP", "TN", "FN", "F0.5",
        "Prec_Train", "Prec_Validation", "Prec_Test", "Prec_Mean",
    ]
    valid_cols = [c for c in df_all.columns if c not in metric_cols]
    y = success_row[valid_cols].iloc[0].infer_objects(copy=False).fillna(0).astype(int).values
    X = df_all[valid_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.T
    print(f"Loaded data: X={X.shape}, y={y.shape}, prevalence={y.mean():.3f}")
    return X, y


def optimize_threshold_f05(y: np.ndarray, proba: np.ndarray) -> float:
    """Find threshold that maximises F0.5 on training data."""
    best_t, best_s = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 50):
        pred = (proba >= t).astype(int)
        s = fbeta_score(y, pred, beta=0.5, zero_division=0)
        if s > best_s:
            best_s = s
            best_t = t
    return best_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="precomputed/gpt_4o_mini")
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--n-splits", type=int, default=10)
    args = parser.parse_args()

    X, y = load_data(args.results_dir)

    results = []
    for repeat in range(1, args.n_repeats + 1):
        print(f"\n🔁 Repeat {repeat}/{args.n_repeats}")
        skf = StratifiedKFold(
            n_splits=args.n_splits, shuffle=True, random_state=42 + (repeat - 1)
        )

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te, y_te = X[test_idx], y[test_idx]

            model = LogisticRegressionCV(
                cv=5, penalty="elasticnet", solver="saga",
                l1_ratios=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
                max_iter=5000, scoring="roc_auc", n_jobs=1, random_state=42,
            )
            model.fit(X_tr, y_tr)

            # Optimise threshold on train
            tr_probs = model.predict_proba(X_tr)[:, 1]
            t_opt = optimize_threshold_f05(y_tr, tr_probs)

            # Evaluate on test
            te_probs = model.predict_proba(X_te)[:, 1]
            te_pred = (te_probs >= t_opt).astype(int)

            prec = precision_score(y_te, te_pred, zero_division=0)
            rec = recall_score(y_te, te_pred, zero_division=0)
            f05 = fbeta_score(y_te, te_pred, beta=0.5, zero_division=0)

            results.append({
                "Repeat": repeat,
                "Fold": fold_idx,
                "Model": "ElasticNet LogReg",
                "Test Prec": prec,
                "Test Rec": rec,
                "Test F0.5": f05,
            })
            print(f"  Fold {fold_idx}: Prec={prec:.3f}, Rec={rec:.3f}, F0.5={f05:.3f}")

    df = pd.DataFrame(results)

    # Summary per repeat
    print("\n📊 Per-repeat summary:")
    for rep, g in df.groupby("Repeat"):
        print(f"  Repeat {rep}: F0.5={g['Test F0.5'].mean():.4f}, "
              f"Prec={g['Test Prec'].mean():.4f}, Rec={g['Test Rec'].mean():.4f}")

    # Overall
    repeat_means = df.groupby("Repeat")[["Test Prec", "Test Rec", "Test F0.5"]].mean()
    print(f"\n📊 Overall (mean ± SD across {args.n_repeats} repeats):")
    for col in ["Test Prec", "Test Rec", "Test F0.5"]:
        print(f"  {col}: {repeat_means[col].mean():.3f} ± {repeat_means[col].std():.3f}")

    # Save
    out_dir = project_root / args.results_dir / "benchmark_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Derive model name from results_dir
    model_name = Path(args.results_dir).name
    out_path = out_dir / f"benchmark_{model_name}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n✅ Saved to {out_path}")


if __name__ == "__main__":
    main()
