#!/usr/bin/env python3
"""Shared RRF evaluation helpers for clinical-trials Phase I scripts."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
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


METADATA_ROWS = {
    "Success",
    "SUCCESS_PROPORTION",
    "Dataset",
    "Fold",
    "Trial Index",
    "Question",
}


def load_question_matrix(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path, dtype=str)
    if "Question" not in df.columns:
        raise ValueError(f"Missing 'Question' column in {csv_path}")

    trial_cols = [c for c in df.columns if re.fullmatch(r"NCT\d+", str(c))]
    if not trial_cols:
        raise ValueError(f"No trial columns (NCT...) found in {csv_path}")

    success_rows = df[df["Question"].astype(str).str.strip() == "Success"]
    if success_rows.empty:
        raise ValueError(f"No Success row found in {csv_path}")

    y = (
        pd.to_numeric(success_rows.iloc[0][trial_cols], errors="coerce")
        .fillna(0)
        .astype(int)
        .values
    )

    q_df = df[~df["Question"].astype(str).isin(METADATA_ROWS)].copy()
    q_df = q_df[q_df["Question"].notna()]
    M = (
        q_df[trial_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype(int)
        .values
    )
    return M, y


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(tp), int(fp), int(tn), int(fn)


def metric_from_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, metric: str, beta: float | None = None
) -> float:
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
    if metric == "f_beta":
        beta_val = 0.7 if beta is None else float(beta)
        return float(fbeta_score(y_true, y_pred, beta=beta_val, zero_division=0))
    if metric == "mcc":
        return float(matthews_corrcoef(y_true, y_pred))
    if metric == "youden_j":
        tp, fp, tn, fn = confusion_counts(y_true, y_pred)
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        return float(rec + spec - 1.0)
    raise ValueError(f"Unsupported optimise_for metric: {metric}")


def evaluate_all_metrics(y_true: np.ndarray, votes: np.ndarray, thr: int) -> dict:
    y_pred = (votes >= thr).astype(int)
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    roc_auc = (
        float(roc_auc_score(y_true, votes)) if len(np.unique(y_true)) > 1 else float("nan")
    )
    pr_auc = (
        float(average_precision_score(y_true, votes))
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


def tune_hyperparameters(
    M_val: np.ndarray,
    y_val: np.ndarray,
    n_splits: int,
    min_q: int,
    optimise_for: str,
    seed: int,
    beta: float | None = None,
) -> tuple[int, int, float]:
    q_count = M_val.shape[0]
    min_q = min(max(1, min_q), q_count)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    scores: dict[tuple[int, int], list[float]] = {}
    for _, fold_idx in cv.split(M_val.T, y_val):
        M_fold = M_val[:, fold_idx]
        y_fold = y_val[fold_idx]
        cumsum_votes = M_fold.cumsum(axis=0)
        for n_q in range(min_q, q_count + 1):
            votes = cumsum_votes[n_q - 1, :]
            for thr in range(1, n_q + 1):
                y_pred = (votes >= thr).astype(int)
                score = metric_from_predictions(y_fold, y_pred, optimise_for, beta=beta)
                scores.setdefault((n_q, thr), []).append(score)

    best_nq, best_thr, best_score = min_q, 1, -1.0
    ranked = sorted(
        ((cfg, float(np.mean(vals))) for cfg, vals in scores.items()),
        key=lambda item: (-item[1], item[0][0], item[0][1]),
    )
    if ranked:
        (best_nq, best_thr), best_score = ranked[0]
    return int(best_nq), int(best_thr), float(best_score)


def run_rrf_repeats(
    *,
    M_val: np.ndarray,
    y_val: np.ndarray,
    M_test: np.ndarray,
    y_test: np.ndarray,
    n_splits: int,
    min_q: int,
    optimise_for: str,
    n_repeats: int,
    seed_start: int,
    beta: float | None = None,
) -> pd.DataFrame:
    q_count = M_val.shape[0]
    min_q = min(max(1, min_q), q_count)

    rows: list[dict] = []
    for repeat in range(n_repeats):
        seed = seed_start + repeat
        best_nq, best_thr, val_score = tune_hyperparameters(
            M_val=M_val,
            y_val=y_val,
            n_splits=n_splits,
            min_q=min_q,
            optimise_for=optimise_for,
            seed=seed,
            beta=beta,
        )
        best_nq = min(best_nq, M_test.shape[0])
        votes_test = M_test[:best_nq, :].sum(axis=0)
        metrics = evaluate_all_metrics(y_test, votes_test, best_thr)
        rows.append(
            {
                "Repeat": repeat + 1,
                "Seed": seed,
                "n_q": best_nq,
                "thr": best_thr,
                "Val_Score": val_score,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def summarise_repeat_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n_repeats_observed": 0}

    n_rep = int(df["Repeat"].nunique()) if "Repeat" in df.columns else int(len(df))
    summary = {
        "n_repeats_observed": n_rep,
        "roc_auc_repeat_mean": float(df["ROC_AUC"].mean()) if "ROC_AUC" in df.columns else None,
        "roc_auc_repeat_sd": (
            float(df["ROC_AUC"].std(ddof=1))
            if "ROC_AUC" in df.columns and n_rep > 1
            else 0.0
        ),
        "pr_auc_repeat_mean": float(df["PR_AUC"].mean()) if "PR_AUC" in df.columns else None,
        "pr_auc_repeat_sd": (
            float(df["PR_AUC"].std(ddof=1))
            if "PR_AUC" in df.columns and n_rep > 1
            else 0.0
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
