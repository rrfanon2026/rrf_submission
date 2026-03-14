#!/usr/bin/env python3
"""Run locked ElasticNet logistic-regression baseline for clinical Phase I."""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from _rrf_eval import load_question_matrix


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"sklearn\.linear_model\._logistic",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "clinical_trials" / "configs" / "logreg_baseline_locked.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run locked ElasticNet logistic baseline for clinical-trials Phase I."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Locked config path (default: clinical_trials/configs/logreg_baseline_locked.json)",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=None,
        help="Optional repeat override for smoke tests.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation even if output exists.",
    )
    return parser.parse_args()


def load_locked_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("locked") is not True:
        raise ValueError("Baseline config must set locked=true.")
    if cfg.get("analysis_role") != "baseline_logreg":
        raise ValueError("Baseline config must set analysis_role='baseline_logreg'.")
    return cfg


def resolve_paths(cfg: dict) -> tuple[Path, Path, Path]:
    input_dir = (REPO_ROOT / cfg["input_dir"]).resolve()
    out_dir = (REPO_ROOT / cfg["output_dir"]).resolve()
    val_path = input_dir / cfg["val_file"]
    test_path = input_dir / cfg["test_file"]
    return val_path, test_path, out_dir


def metric_from_predictions(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    metric = metric.lower()
    if metric == "precision":
        return float(precision_score(y_true, y_pred, zero_division=0))
    if metric == "recall":
        return float(recall_score(y_true, y_pred, zero_division=0))
    if metric == "f0.5":
        return float(fbeta_score(y_true, y_pred, beta=0.5, zero_division=0))
    if metric == "f1":
        return float(f1_score(y_true, y_pred, zero_division=0))
    if metric == "f2":
        return float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0))
    if metric == "mcc":
        return float(matthews_corrcoef(y_true, y_pred))
    raise ValueError(f"Unsupported optimise_for metric: {metric}")


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(tp), int(fp), int(tn), int(fn)


def evaluate_all_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> dict:
    y_pred = (y_prob >= thr).astype(int)
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    roc_auc = (
        float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    )
    pr_auc = (
        float(average_precision_score(y_true, y_prob))
        if len(np.unique(y_true)) > 1
        else float("nan")
    )
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f05 = float(fbeta_score(y_true, y_pred, beta=0.5, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    f2 = float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0))
    specificity = (tn / (tn + fp)) if (tn + fp) else 0.0
    youden_j = recall + specificity - 1.0
    bal_acc = 0.5 * (recall + specificity)
    return {
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "Precision": precision,
        "Recall": recall,
        "F0_5": f05,
        "F1": f1,
        "F2": f2,
        "Specificity": float(specificity),
        "Youden_J": float(youden_j),
        "Bal_Acc": float(bal_acc),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "Pred_Pos_Rate": float(np.mean(y_pred)),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
    }


def fit_single_repeat(
    *,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: dict,
    seed: int,
) -> dict:
    inner_cv = StratifiedKFold(
        n_splits=int(cfg["inner_cv_splits"]),
        shuffle=True,
        random_state=seed,
    )
    clf = LogisticRegressionCV(
        Cs=int(cfg["cs"]),
        cv=inner_cv,
        penalty="elasticnet",
        solver="saga",
        l1_ratios=list(cfg["l1_ratios"]),
        class_weight=cfg.get("class_weight", None),
        max_iter=int(cfg["max_iter"]),
        scoring="neg_log_loss",
        random_state=seed,
    )
    clf.fit(X_val, y_val)

    y_prob_val = clf.predict_proba(X_val)[:, 1]
    y_prob_test = clf.predict_proba(X_test)[:, 1]

    optimise_for = str(cfg["optimise_for"]).lower()
    best_thr = 0.5
    best_score = -1.0
    for thr in np.linspace(0.01, 0.99, 99):
        y_pred = (y_prob_val >= thr).astype(int)
        s = metric_from_predictions(y_val, y_pred, optimise_for)
        if s > best_score:
            best_score = s
            best_thr = float(thr)

    metrics = evaluate_all_metrics(y_test, y_prob_test, best_thr)
    c_selected = float(np.atleast_1d(clf.C_)[0])
    l1_ratio_selected = float(np.atleast_1d(clf.l1_ratio_)[0])
    return {
        "Val_Score": float(best_score),
        "Thr": float(best_thr),
        "C_selected": c_selected,
        "l1_ratio_selected": l1_ratio_selected,
        **metrics,
    }


def summarise_repeat_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n_repeats_observed": 0}
    n_rep = int(df["Repeat"].nunique()) if "Repeat" in df.columns else int(len(df))

    has_roc = "ROC_AUC" in df.columns
    has_pr = "PR_AUC" in df.columns
    summary = {
        "n_repeats_observed": n_rep,
        "roc_auc_repeat_mean": float(df["ROC_AUC"].mean()) if has_roc else None,
        "roc_auc_repeat_sd": (
            float(df["ROC_AUC"].std(ddof=1)) if has_roc and n_rep > 1 else 0.0
        ),
        "pr_auc_repeat_mean": float(df["PR_AUC"].mean()) if has_pr else None,
        "pr_auc_repeat_sd": (
            float(df["PR_AUC"].std(ddof=1)) if has_pr and n_rep > 1 else 0.0
        ),
        "precision_repeat_mean": float(df["Precision"].mean()),
        "precision_repeat_sd": float(df["Precision"].std(ddof=1)) if n_rep > 1 else 0.0,
        "recall_repeat_mean": float(df["Recall"].mean()),
        "recall_repeat_sd": float(df["Recall"].std(ddof=1)) if n_rep > 1 else 0.0,
        "f05_repeat_mean": float(df["F0_5"].mean()),
        "f05_repeat_sd": float(df["F0_5"].std(ddof=1)) if n_rep > 1 else 0.0,
        "f1_repeat_mean": float(df["F1"].mean()),
        "f1_repeat_sd": float(df["F1"].std(ddof=1)) if n_rep > 1 else 0.0,
        "f2_repeat_mean": float(df["F2"].mean()),
        "f2_repeat_sd": float(df["F2"].std(ddof=1)) if n_rep > 1 else 0.0,
        "tp_total": int(df["TP"].sum()),
        "fp_total": int(df["FP"].sum()),
        "tn_total": int(df["TN"].sum()),
        "fn_total": int(df["FN"].sum()),
    }
    tp = summary["tp_total"]
    fp = summary["fp_total"]
    fn = summary["fn_total"]
    summary["pooled_precision"] = tp / (tp + fp) if (tp + fp) else 0.0
    summary["pooled_recall"] = tp / (tp + fn) if (tp + fn) else 0.0
    return summary


def print_table_metrics(summary: dict) -> None:
    def fm(mean_v: float | None, sd_v: float | None) -> str:
        if mean_v is None or sd_v is None:
            return "n/a"
        return f"{mean_v:.3f} +/- {sd_v:.3f}"

    print("\nClinical-trials LogReg baseline metrics (mean +/- SD across repeats):")
    print(
        "LogReg (EN) | "
        f"F1={fm(summary.get('f1_repeat_mean'), summary.get('f1_repeat_sd'))} | "
        f"PR-AUC={fm(summary.get('pr_auc_repeat_mean'), summary.get('pr_auc_repeat_sd'))} | "
        f"ROC-AUC={fm(summary.get('roc_auc_repeat_mean'), summary.get('roc_auc_repeat_sd'))} | "
        f"N_repeats={summary['n_repeats_observed']}"
    )


def main() -> None:
    args = parse_args()
    cfg = load_locked_config(args.config)
    if args.n_repeats is not None:
        cfg = dict(cfg)
        cfg["n_repeats"] = int(args.n_repeats)

    val_path, test_path, out_dir = resolve_paths(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    repeat_csv = out_dir / "logreg_repeat_metrics.csv"
    summary_json = out_dir / "logreg_summary.json"
    manifest_json = out_dir / "logreg_manifest.json"

    if repeat_csv.exists() and not args.force:
        print(f"LogReg output already exists at: {repeat_csv}")
        print("Use --force to recompute.")
        required = {"f1_repeat_mean", "pr_auc_repeat_mean", "roc_auc_repeat_mean"}
        if summary_json.exists():
            with summary_json.open("r", encoding="utf-8") as f:
                summary = json.load(f)
            if not required.issubset(summary.keys()):
                df_existing = pd.read_csv(repeat_csv)
                summary = summarise_repeat_metrics(df_existing)
        else:
            df_existing = pd.read_csv(repeat_csv)
            summary = summarise_repeat_metrics(df_existing)
        print_table_metrics(summary)
        return

    print("=" * 72)
    print("CLINICAL TRIALS LOGREG BASELINE RUN (LOCKED)")
    print("=" * 72)
    print("Guardrail: baseline config is fixed for paper comparison.")
    print(f"Validation matrix: {val_path}")
    print(f"Test matrix:       {test_path}")

    M_val, y_val = load_question_matrix(val_path)
    M_test, y_test = load_question_matrix(test_path)
    X_val = M_val.T
    X_test = M_test.T

    rows: list[dict] = []
    for repeat in range(int(cfg["n_repeats"])):
        seed = int(cfg["seed_start"]) + repeat
        row = fit_single_repeat(
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            cfg=cfg,
            seed=seed,
        )
        rows.append({"Repeat": repeat + 1, "Seed": seed, **row})

    df = pd.DataFrame(rows)
    df.to_csv(repeat_csv, index=False)

    summary = summarise_repeat_metrics(df)
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print_table_metrics(summary)

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": "baseline_logreg",
        "locked": True,
        "config_path": str(args.config),
        "config": cfg,
        "inputs": {"val_csv": str(val_path), "test_csv": str(test_path)},
        "outputs": {
            "repeat_metrics_csv": str(repeat_csv),
            "summary_json": str(summary_json),
            "manifest_json": str(manifest_json),
        },
        "guardrail_note": (
            "LogReg baseline is fixed for comparison. "
            "Treat ablations as secondary analysis."
        ),
    }
    with manifest_json.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nClinical LogReg baseline run complete.")
    print(f"Repeat metrics CSV: {repeat_csv}")
    print(f"Summary JSON:       {summary_json}")
    print(f"Manifest JSON:      {manifest_json}")


if __name__ == "__main__":
    main()
