#!/usr/bin/env python3
"""Run structured secondary ablations for clinical-trials Phase I RRF."""

from __future__ import annotations

import argparse
import json
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Keep matplotlib cache writable in restricted environments.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from _rrf_eval import load_question_matrix, run_rrf_repeats, summarise_repeat_metrics


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRID = REPO_ROOT / "clinical_trials" / "configs" / "secondary_ablation_grid.json"
PRIMARY_CONFIG = REPO_ROOT / "clinical_trials" / "configs" / "primary_model_locked.json"
LOGREG_CONFIG = REPO_ROOT / "clinical_trials" / "configs" / "logreg_baseline_locked.json"
DEFAULT_BETA_SWEEP = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"sklearn\.linear_model\._logistic",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run structured clinical-trials RRF secondary ablations."
    )
    parser.add_argument(
        "--grid",
        type=Path,
        default=DEFAULT_GRID,
        help="Ablation grid path (default: clinical_trials/configs/secondary_ablation_grid.json)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Optional comma-separated subset of experiment labels to run.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=None,
        help="Optional repeat override for all ablations (useful for smoke tests).",
    )
    parser.add_argument(
        "--skip-beta-sweep",
        action="store_true",
        help="Skip the F_beta beta sweep section.",
    )
    parser.add_argument(
        "--skip-logreg-beta-sweep",
        action="store_true",
        help="Skip secondary ElasticNet logreg beta sweep figure/table.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation even if output files exist.",
    )
    return parser.parse_args()


def load_grid(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        grid = json.load(f)
    if grid.get("analysis_role") != "secondary_ablation":
        raise ValueError("Grid must set analysis_role='secondary_ablation'.")
    if not grid.get("experiments"):
        raise ValueError("Grid has no experiments.")
    return grid


def load_primary_cfg() -> dict:
    with PRIMARY_CONFIG.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def load_logreg_cfg() -> dict:
    with LOGREG_CONFIG.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def resolve_paths(grid: dict) -> tuple[Path, Path, Path]:
    input_dir = (REPO_ROOT / grid["input_dir"]).resolve()
    out_dir = (REPO_ROOT / grid["output_dir"]).resolve()
    val_path = input_dir / grid["val_file"]
    test_path = input_dir / grid["test_file"]
    return val_path, test_path, out_dir


def fmt_metric(mean_v: float, sd_v: float) -> str:
    return f"{mean_v:.3f} +/- {sd_v:.3f}"


def print_ablation_table(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        print("\nNo ablation results to display.")
        return

    table = pd.DataFrame(
        {
            "label": summary_df["label"],
            "optimise_for": summary_df["optimise_for"],
            "precision": summary_df.apply(
                lambda r: fmt_metric(r["precision_repeat_mean"], r["precision_repeat_sd"]),
                axis=1,
            ),
            "recall": summary_df.apply(
                lambda r: fmt_metric(r["recall_repeat_mean"], r["recall_repeat_sd"]), axis=1
            ),
            "f0.5": summary_df.apply(
                lambda r: fmt_metric(r["f05_repeat_mean"], r["f05_repeat_sd"]), axis=1
            ),
            "f1": summary_df.apply(
                lambda r: fmt_metric(r["f1_repeat_mean"], r["f1_repeat_sd"]), axis=1
            ),
            "f2": summary_df.apply(
                lambda r: fmt_metric(r["f2_repeat_mean"], r["f2_repeat_sd"]), axis=1
            ),
        }
    )
    print("\nClinical-trials ablation table (mean +/- SD across repeats):")
    print(table.to_string(index=False))


def summarise_beta_sweep(beta_repeat_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for beta, grp in beta_repeat_df.groupby("beta", sort=True):
        n_rep = int(grp["Repeat"].nunique())
        modal_n_q = int(grp["n_q"].mode().iloc[0]) if "n_q" in grp.columns else None
        modal_thr = int(grp["thr"].mode().iloc[0]) if "thr" in grp.columns else None
        rows.append(
            {
                "beta": float(beta),
                "n_repeats_observed": n_rep,
                "f1_mean": float(grp["F1"].mean()),
                "f1_sd": float(grp["F1"].std(ddof=1)) if n_rep > 1 else 0.0,
                "pr_auc_mean": float(grp["PR_AUC"].mean()),
                "pr_auc_sd": float(grp["PR_AUC"].std(ddof=1)) if n_rep > 1 else 0.0,
                "roc_auc_mean": float(grp["ROC_AUC"].mean()),
                "roc_auc_sd": float(grp["ROC_AUC"].std(ddof=1)) if n_rep > 1 else 0.0,
                "specificity_mean": float(grp["Specificity"].mean()),
                "specificity_sd": (
                    float(grp["Specificity"].std(ddof=1)) if n_rep > 1 else 0.0
                ),
                "recall_mean": float(grp["Recall"].mean()),
                "recall_sd": float(grp["Recall"].std(ddof=1)) if n_rep > 1 else 0.0,
                "modal_n_q": modal_n_q,
                "modal_thr": modal_thr,
            }
        )
    return pd.DataFrame(rows).sort_values("beta").reset_index(drop=True)


def print_beta_table(beta_summary_df: pd.DataFrame) -> None:
    if beta_summary_df.empty:
        print("\nNo beta-sweep results to display.")
        return

    table = pd.DataFrame(
        {
            "beta": beta_summary_df["beta"].map(lambda x: f"{x:.2f}"),
            "f1": beta_summary_df.apply(
                lambda r: fmt_metric(r["f1_mean"], r["f1_sd"]), axis=1
            ),
            "pr_auc": beta_summary_df.apply(
                lambda r: fmt_metric(r["pr_auc_mean"], r["pr_auc_sd"]), axis=1
            ),
            "roc_auc": beta_summary_df.apply(
                lambda r: fmt_metric(r["roc_auc_mean"], r["roc_auc_sd"]), axis=1
            ),
            "modal_n_q": beta_summary_df["modal_n_q"].astype(int),
            "modal_thr": beta_summary_df["modal_thr"].astype(int),
        }
    )
    print("\nBeta sweep table (mean +/- SD across repeats):")
    print(table.to_string(index=False))


def plot_beta_figure(
    beta_summary_df: pd.DataFrame, out_path: Path, highlight_beta: float = 0.7
) -> None:
    if beta_summary_df.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = beta_summary_df["beta"].astype(float)
    ax.plot(x, beta_summary_df["f1_mean"], marker="o", label="F1")
    ax.plot(x, beta_summary_df["pr_auc_mean"], marker="s", label="PR-AUC")
    ax.plot(x, beta_summary_df["roc_auc_mean"], marker="^", label="ROC-AUC")
    ax.set_xlabel("Beta (optimisation target: F_beta)")
    ax.set_ylabel("Score")
    ax.set_title("Clinical Phase I: Metrics vs Beta")
    ax.grid(True, alpha=0.3)

    closest_idx = (beta_summary_df["beta"] - highlight_beta).abs().idxmin()
    row = beta_summary_df.loc[closest_idx]
    b = float(row["beta"])
    ax.axvline(
        b,
        color="black",
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
        label=f"Chosen beta={b:.2f}",
    )
    ax.scatter([b], [row["f1_mean"]], color="C0", zorder=4)
    ax.scatter([b], [row["pr_auc_mean"]], color="C1", zorder=4)
    ax.scatter([b], [row["roc_auc_mean"]], color="C2", zorder=4)
    ann = (
        f"beta={b:.2f}\n"
        f"F1={row['f1_mean']:.3f}\n"
        f"PR-AUC={row['pr_auc_mean']:.3f}\n"
        f"ROC-AUC={row['roc_auc_mean']:.3f}"
    )
    ax.text(
        0.99,
        0.02,
        ann,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_specificity_recall(
    beta_summary_df: pd.DataFrame, out_path: Path, highlight_beta: float = 0.7
) -> None:
    if beta_summary_df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    x = beta_summary_df["specificity_mean"].astype(float)
    y = beta_summary_df["recall_mean"].astype(float)
    ax.plot(x, y, marker="o", linewidth=1.5)
    for _, row in beta_summary_df.iterrows():
        ax.annotate(
            f"{float(row['beta']):.2f}",
            (float(row["specificity_mean"]), float(row["recall_mean"])),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    closest_idx = (beta_summary_df["beta"] - highlight_beta).abs().idxmin()
    row = beta_summary_df.loc[closest_idx]
    ax.scatter(
        [float(row["specificity_mean"])],
        [float(row["recall_mean"])],
        marker="*",
        s=180,
        color="crimson",
        label=f"Chosen beta={float(row['beta']):.2f}",
        zorder=4,
    )
    ax.set_xlabel("Specificity")
    ax.set_ylabel("Recall")
    ax.set_title("Clinical Phase I: Recall vs Specificity Across Beta")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _logreg_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    return (tn / (tn + fp)) if (tn + fp) else 0.0


def _best_threshold_for_beta(y_true: np.ndarray, y_prob: np.ndarray, beta: float) -> float:
    best_thr = 0.5
    best_score = -1.0
    for thr in np.linspace(0.01, 0.99, 99):
        y_pred = (y_prob >= thr).astype(int)
        s = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
        if s > best_score:
            best_score = float(s)
            best_thr = float(thr)
    return best_thr


def run_logreg_beta_sweep(
    *,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    logreg_cfg: dict,
    beta_values: list[float],
    n_repeats: int,
    seed_start: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    for repeat in range(n_repeats):
        seed = seed_start + repeat
        inner_cv = StratifiedKFold(
            n_splits=int(logreg_cfg["inner_cv_splits"]),
            shuffle=True,
            random_state=seed,
        )
        clf = LogisticRegressionCV(
            Cs=int(logreg_cfg["cs"]),
            cv=inner_cv,
            penalty="elasticnet",
            solver="saga",
            l1_ratios=list(logreg_cfg["l1_ratios"]),
            class_weight=logreg_cfg.get("class_weight", None),
            max_iter=int(logreg_cfg["max_iter"]),
            scoring="neg_log_loss",
            random_state=seed,
        )
        clf.fit(X_val, y_val)
        y_prob_val = clf.predict_proba(X_val)[:, 1]
        y_prob_test = clf.predict_proba(X_test)[:, 1]
        pr_auc = (
            float(average_precision_score(y_test, y_prob_test))
            if len(np.unique(y_test)) > 1
            else float("nan")
        )
        roc_auc = (
            float(roc_auc_score(y_test, y_prob_test))
            if len(np.unique(y_test)) > 1
            else float("nan")
        )

        for beta in beta_values:
            thr = _best_threshold_for_beta(y_val, y_prob_val, beta)
            y_pred_test = (y_prob_test >= thr).astype(int)
            rows.append(
                {
                    "beta": float(beta),
                    "Repeat": repeat + 1,
                    "Seed": seed,
                    "Thr": float(thr),
                    "F1": float(f1_score(y_test, y_pred_test, zero_division=0)),
                    "PR_AUC": pr_auc,
                    "ROC_AUC": roc_auc,
                    "Recall": float(recall_score(y_test, y_pred_test, zero_division=0)),
                    "Specificity": _logreg_specificity(y_test, y_pred_test),
                    "Precision": float(
                        precision_score(y_test, y_pred_test, zero_division=0)
                    ),
                }
            )

    repeat_df = pd.DataFrame(rows)
    summary_df = summarise_beta_sweep(repeat_df)
    return repeat_df, summary_df


def plot_logreg_beta_figure(
    beta_summary_df: pd.DataFrame, out_path: Path, highlight_beta: float = 0.7
) -> None:
    if beta_summary_df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = beta_summary_df["beta"].astype(float)
    ax.plot(x, beta_summary_df["f1_mean"], marker="o", label="F1")
    ax.plot(x, beta_summary_df["pr_auc_mean"], marker="s", label="PR-AUC")
    ax.plot(x, beta_summary_df["roc_auc_mean"], marker="^", label="ROC-AUC")

    closest_idx = (beta_summary_df["beta"] - highlight_beta).abs().idxmin()
    row = beta_summary_df.loc[closest_idx]
    b = float(row["beta"])
    ax.axvline(
        b,
        color="black",
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
        label=f"Chosen beta={b:.2f}",
    )
    ann = (
        f"beta={b:.2f}\n"
        f"F1={row['f1_mean']:.3f}\n"
        f"PR-AUC={row['pr_auc_mean']:.3f}\n"
        f"ROC-AUC={row['roc_auc_mean']:.3f}"
    )
    ax.text(
        0.99,
        0.02,
        ann,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )
    ax.set_xlabel("Beta (threshold optimisation target for ElasticNet)")
    ax.set_ylabel("Score")
    ax.set_title("Clinical Phase I (LogReg EN): Metrics vs Beta")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def print_logreg_beta_table(beta_summary_df: pd.DataFrame) -> None:
    if beta_summary_df.empty:
        print("\nNo ElasticNet beta-sweep results to display.")
        return
    table = pd.DataFrame(
        {
            "beta": beta_summary_df["beta"].map(lambda x: f"{x:.2f}"),
            "f1": beta_summary_df.apply(
                lambda r: fmt_metric(r["f1_mean"], r["f1_sd"]), axis=1
            ),
            "pr_auc": beta_summary_df.apply(
                lambda r: fmt_metric(r["pr_auc_mean"], r["pr_auc_sd"]), axis=1
            ),
            "roc_auc": beta_summary_df.apply(
                lambda r: fmt_metric(r["roc_auc_mean"], r["roc_auc_sd"]), axis=1
            ),
        }
    )
    print("\nElasticNet beta sweep table (mean +/- SD across repeats):")
    print(table.to_string(index=False))


def main() -> None:
    args = parse_args()
    grid = load_grid(args.grid)
    primary_cfg = load_primary_cfg()
    logreg_cfg = load_logreg_cfg()
    highlight_beta = float(grid.get("highlight_beta", 0.7))

    selected_labels: set[str] | None = None
    if args.labels:
        selected_labels = {s.strip() for s in args.labels.split(",") if s.strip()}

    experiments = [dict(e) for e in grid["experiments"]]
    if selected_labels is not None:
        experiments = [e for e in experiments if e["label"] in selected_labels]
        if not experiments:
            raise ValueError("No experiments matched --labels filter.")

    val_path, test_path, out_dir = resolve_paths(grid)
    out_dir.mkdir(parents=True, exist_ok=True)
    repeat_csv = out_dir / "ablation_repeat_metrics.csv"
    summary_csv = out_dir / "ablation_summary.csv"
    beta_repeat_csv = out_dir / "beta_sweep_repeat_metrics.csv"
    beta_summary_csv = out_dir / "beta_sweep_summary.csv"
    beta_plot_png = out_dir / "beta_sweep_f1_pr_auc_roc_auc.png"
    beta_spec_rec_png = out_dir / "beta_sweep_specificity_vs_recall.png"
    logreg_beta_repeat_csv = out_dir / "logreg_beta_sweep_repeat_metrics.csv"
    logreg_beta_summary_csv = out_dir / "logreg_beta_sweep_summary.csv"
    logreg_beta_plot_png = out_dir / "logreg_beta_sweep_f1_pr_auc_roc_auc.png"
    manifest_json = out_dir / "ablation_manifest.json"

    cached_ok = repeat_csv.exists() and summary_csv.exists()
    if not args.skip_beta_sweep:
        cached_ok = cached_ok and beta_summary_csv.exists()
    if not args.skip_logreg_beta_sweep:
        cached_ok = cached_ok and logreg_beta_summary_csv.exists()

    if cached_ok and not args.force:
        print(f"Ablation outputs already exist in: {out_dir}")
        print("Use --force to recompute.")
        summary_df = pd.read_csv(summary_csv)
        print_ablation_table(summary_df)
        if not args.skip_beta_sweep and beta_summary_csv.exists():
            beta_summary_df = pd.read_csv(beta_summary_csv)
            print_beta_table(beta_summary_df)
            if beta_plot_png.exists():
                print(f"Beta figure: {beta_plot_png}")
            if beta_spec_rec_png.exists():
                print(f"Specificity/Recall figure: {beta_spec_rec_png}")
        if not args.skip_logreg_beta_sweep and logreg_beta_summary_csv.exists():
            logreg_beta_summary_df = pd.read_csv(logreg_beta_summary_csv)
            print_logreg_beta_table(logreg_beta_summary_df)
            if logreg_beta_plot_png.exists():
                print(f"ElasticNet beta figure: {logreg_beta_plot_png}")
        return

    print("=" * 72)
    print("CLINICAL TRIALS SECONDARY ABLATION RUN")
    print("=" * 72)
    print("Guardrail: secondary analysis only; do not replace locked primary headline.")
    print(f"Validation matrix: {val_path}")
    print(f"Test matrix:       {test_path}")

    M_val, y_val = load_question_matrix(val_path)
    M_test, y_test = load_question_matrix(test_path)

    repeat_rows: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    completed: list[dict] = []

    for exp in experiments:
        if args.n_repeats is not None:
            exp["n_repeats"] = int(args.n_repeats)
        label = exp["label"]
        print(f"\nRunning ablation: {label} ({exp['optimise_for']})")
        df = run_rrf_repeats(
            M_val=M_val,
            y_val=y_val,
            M_test=M_test,
            y_test=y_test,
            n_splits=int(exp["n_splits"]),
            min_q=int(exp["min_q"]),
            optimise_for=str(exp["optimise_for"]),
            n_repeats=int(exp["n_repeats"]),
            seed_start=int(exp["seed_start"]),
            beta=exp.get("beta"),
        )
        df.insert(0, "label", label)
        df.insert(1, "optimise_for", str(exp["optimise_for"]))
        repeat_rows.append(df)

        summary = summarise_repeat_metrics(df)
        summary_row = {
            "label": label,
            "description": exp.get("description", ""),
            "optimise_for": str(exp["optimise_for"]),
            "beta": exp.get("beta"),
            "n_splits": int(exp["n_splits"]),
            "min_q": int(exp["min_q"]),
            "n_repeats": int(exp["n_repeats"]),
            "seed_start": int(exp["seed_start"]),
            **summary,
        }
        summary_rows.append(summary_row)
        completed.append({"experiment": exp, "summary": summary})

    repeat_df = pd.concat(repeat_rows, ignore_index=True) if repeat_rows else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values("f05_repeat_mean", ascending=False).reset_index(
            drop=True
        )

    repeat_df.to_csv(repeat_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    print_ablation_table(summary_df)

    if not summary_df.empty:
        winner_label = str(summary_df.iloc[0]["label"])
        winner_f05 = float(summary_df.iloc[0]["f05_repeat_mean"])
        print(f"\nBest ablation by mean F0.5: {winner_label} ({winner_f05:.3f})")
        if winner_label != "primary_reference":
            print(
                "NOTE: primary_reference did not win this ablation set. "
                "If you want, promote winner manually into primary locked config."
            )

    beta_summary_df = pd.DataFrame()
    beta_values = []
    if not args.skip_beta_sweep:
        beta_values = [float(v) for v in grid.get("beta_sweep_values", DEFAULT_BETA_SWEEP)]
        # Anchor sweep settings to the primary reference experiment.
        primary_exp = next((e for e in experiments if e["label"] == "primary_reference"), experiments[0])
        sweep_repeats = int(args.n_repeats) if args.n_repeats is not None else int(primary_exp["n_repeats"])

        beta_repeat_rows: list[pd.DataFrame] = []
        for beta in beta_values:
            print(f"\nRunning beta sweep: beta={beta:.2f}")
            df_beta = run_rrf_repeats(
                M_val=M_val,
                y_val=y_val,
                M_test=M_test,
                y_test=y_test,
                n_splits=int(primary_exp["n_splits"]),
                min_q=int(primary_exp["min_q"]),
                optimise_for="f_beta",
                n_repeats=sweep_repeats,
                seed_start=int(primary_exp["seed_start"]),
                beta=beta,
            )
            df_beta.insert(0, "beta", beta)
            beta_repeat_rows.append(df_beta)

        beta_repeat_df = pd.concat(beta_repeat_rows, ignore_index=True)
        beta_repeat_df.to_csv(beta_repeat_csv, index=False)
        beta_summary_df = summarise_beta_sweep(beta_repeat_df)
        beta_summary_df.to_csv(beta_summary_csv, index=False)
        print_beta_table(beta_summary_df)
        plot_beta_figure(beta_summary_df, beta_plot_png, highlight_beta=highlight_beta)
        plot_specificity_recall(
            beta_summary_df, beta_spec_rec_png, highlight_beta=highlight_beta
        )
        print(f"\nBeta sweep figure: {beta_plot_png}")
        print(f"Specificity/Recall figure: {beta_spec_rec_png}")

    logreg_beta_values: list[float] = []
    if not args.skip_logreg_beta_sweep:
        logreg_beta_values = [float(v) for v in grid.get("beta_sweep_values", DEFAULT_BETA_SWEEP)]
        logreg_repeats = int(args.n_repeats) if args.n_repeats is not None else int(
            logreg_cfg["n_repeats"]
        )
        M_val_logreg, y_val_logreg = load_question_matrix(val_path)
        M_test_logreg, y_test_logreg = load_question_matrix(test_path)
        X_val = M_val_logreg.T
        X_test = M_test_logreg.T
        print("\nRunning ElasticNet beta sweep (secondary figure)...")
        logreg_beta_repeat_df, logreg_beta_summary_df = run_logreg_beta_sweep(
            X_val=X_val,
            y_val=y_val_logreg,
            X_test=X_test,
            y_test=y_test_logreg,
            logreg_cfg=logreg_cfg,
            beta_values=logreg_beta_values,
            n_repeats=logreg_repeats,
            seed_start=int(logreg_cfg["seed_start"]),
        )
        logreg_beta_repeat_df.to_csv(logreg_beta_repeat_csv, index=False)
        logreg_beta_summary_df.to_csv(logreg_beta_summary_csv, index=False)
        print_logreg_beta_table(logreg_beta_summary_df)
        plot_logreg_beta_figure(
            logreg_beta_summary_df,
            logreg_beta_plot_png,
            highlight_beta=highlight_beta,
        )
        print(f"ElasticNet beta figure: {logreg_beta_plot_png}")

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": "secondary_ablation",
        "grid_path": str(args.grid),
        "primary_config_path": str(PRIMARY_CONFIG),
        "primary_config_snapshot": primary_cfg,
        "inputs": {"val_csv": str(val_path), "test_csv": str(test_path)},
        "outputs": {
            "repeat_metrics_csv": str(repeat_csv),
            "summary_csv": str(summary_csv),
            "beta_sweep_repeat_csv": str(beta_repeat_csv) if not args.skip_beta_sweep else None,
            "beta_sweep_summary_csv": str(beta_summary_csv) if not args.skip_beta_sweep else None,
            "beta_sweep_plot_png": str(beta_plot_png) if not args.skip_beta_sweep else None,
            "beta_sweep_specificity_recall_png": (
                str(beta_spec_rec_png) if not args.skip_beta_sweep else None
            ),
            "logreg_beta_sweep_repeat_csv": (
                str(logreg_beta_repeat_csv) if not args.skip_logreg_beta_sweep else None
            ),
            "logreg_beta_sweep_summary_csv": (
                str(logreg_beta_summary_csv) if not args.skip_logreg_beta_sweep else None
            ),
            "logreg_beta_sweep_plot_png": (
                str(logreg_beta_plot_png) if not args.skip_logreg_beta_sweep else None
            ),
            "manifest_json": str(manifest_json),
        },
        "experiments_run": completed,
        "beta_sweep_values": beta_values,
        "logreg_beta_sweep_values": logreg_beta_values,
        "highlight_beta": highlight_beta,
        "guardrail_note": (
            "Secondary ablations are exploratory and sensitivity-focused. "
            "Primary headline remains locked to clinical_trials/pipeline/09_primary_model.py."
        ),
    }
    with manifest_json.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nClinical ablation run complete.")
    print(f"Repeat metrics CSV: {repeat_csv}")
    print(f"Summary CSV:        {summary_csv}")
    if not args.skip_beta_sweep:
        print(f"Beta repeat CSV:    {beta_repeat_csv}")
        print(f"Beta summary CSV:   {beta_summary_csv}")
        print(f"Beta plot PNG:      {beta_plot_png}")
        print(f"Spec/Recall PNG:    {beta_spec_rec_png}")
    if not args.skip_logreg_beta_sweep:
        print(f"LogReg beta CSV:    {logreg_beta_summary_csv}")
        print(f"LogReg beta PNG:    {logreg_beta_plot_png}")
    print(f"Manifest JSON:      {manifest_json}")


if __name__ == "__main__":
    main()
