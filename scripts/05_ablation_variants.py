#!/usr/bin/env python3
"""Run structured secondary ablations for offline RRF evaluation.

This script is intentionally separate from the locked primary model path.
Guardrail: ablation outputs are secondary analysis and must not replace
headline metrics from scripts/04_primary_model.py.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINE_SCRIPT = REPO_ROOT / "scripts" / "03_nested_cv_engine.py"
DEFAULT_GRID = REPO_ROOT / "configs" / "secondary_ablation_grid.json"
PRIMARY_CONFIG = REPO_ROOT / "configs" / "primary_model_locked.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run structured secondary ablations on precomputed predictions."
    )
    parser.add_argument(
        "--grid",
        type=Path,
        default=DEFAULT_GRID,
        help="Path to ablation grid JSON (default: configs/secondary_ablation_grid.json)",
    )
    parser.add_argument(
        "--results_dir_extension",
        type=str,
        default=None,
        help="Optional override for results root directory (e.g., precomputed).",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Optional comma-separated subset of experiment labels to run.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation for each experiment.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=None,
        help="Optional override for evaluation repeats across all ablations.",
    )
    return parser.parse_args()


def load_grid(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        grid = json.load(f)

    experiments = grid.get("experiments", [])
    if not experiments:
        raise ValueError(f"No experiments found in grid: {path}")
    return grid


def _col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _mean_sd_from_repeats(
    df: pd.DataFrame, metric_col: str, repeat_col: str = "Repeat"
) -> tuple[float, float]:
    if repeat_col in df.columns and df[repeat_col].nunique() > 0:
        rep_vals = (
            df.groupby(repeat_col, as_index=False)[metric_col]
            .mean()[metric_col]
            .astype(float)
        )
    else:
        rep_vals = pd.Series([float(df[metric_col].mean())])

    n_rep = int(len(rep_vals))
    mean_v = float(rep_vals.mean())
    sd_v = float(rep_vals.std(ddof=1)) if n_rep > 1 else 0.0
    return mean_v, sd_v


def _fmt_metric(mean_v: float | None, sd_v: float | None) -> str:
    if mean_v is None or sd_v is None:
        return "n/a"
    return f"{mean_v:.3f} +/- {sd_v:.3f}"


def expected_output_path(results_dir_extension: str, model: str, exp: dict) -> Path:
    suffix = "_anonymised"
    similarity_str = f"{exp['similarity_threshold']:.2f}".replace(".", "_")
    opt_str = f"optimise{exp['optimise_for'].replace('.', '_').upper()}"
    sort_str = f"sortby{exp['sort_by'].replace('.', '_')}"

    out_dir = REPO_ROOT / results_dir_extension / model / "ablation_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / (
        f"nested_cv_{exp['mode']}_{suffix}"
        f"similarity_{exp['similarity_metric']}_{similarity_str}_{opt_str}_{sort_str}.csv"
    )


def summarise_csv(path: Path) -> dict:
    df = pd.read_csv(path)
    if df.empty:
        return {"n_rows": 0}

    p_col = _col(df, ["Precision_Outer"])
    r_col = _col(df, ["Recall_Outer"])
    f05_col = _col(df, ["F05_Outer"])
    f1_col = _col(df, ["f1_out", "F1_Outer", "F1"])
    f2_col = _col(df, ["f2_out", "F2_Outer", "F2"])
    tp_col = _col(df, ["TP"])
    fp_col = _col(df, ["FP"])
    tn_col = _col(df, ["TN"])
    fn_col = _col(df, ["FN"])

    tp = int(df[tp_col].sum()) if tp_col else 0
    fp = int(df[fp_col].sum()) if fp_col else 0
    fn = int(df[fn_col].sum()) if fn_col else 0

    pooled_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    pooled_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    p_rep_mean, p_rep_sd = _mean_sd_from_repeats(df, p_col) if p_col else (None, None)
    r_rep_mean, r_rep_sd = _mean_sd_from_repeats(df, r_col) if r_col else (None, None)
    f05_rep_mean, f05_rep_sd = (
        _mean_sd_from_repeats(df, f05_col) if f05_col else (None, None)
    )
    f1_rep_mean, f1_rep_sd = _mean_sd_from_repeats(df, f1_col) if f1_col else (None, None)
    f2_rep_mean, f2_rep_sd = _mean_sd_from_repeats(df, f2_col) if f2_col else (None, None)
    n_rep = int(df["Repeat"].nunique()) if "Repeat" in df.columns else 1

    return {
        "n_rows": int(len(df)),
        "precision_mean": float(df[p_col].mean()) if p_col else None,
        "precision_std": float(df[p_col].std()) if p_col else None,
        "recall_mean": float(df[r_col].mean()) if r_col else None,
        "recall_std": float(df[r_col].std()) if r_col else None,
        "f05_mean": float(df[f05_col].mean()) if f05_col else None,
        "f05_std": float(df[f05_col].std()) if f05_col else None,
        "f1_mean": float(df[f1_col].mean()) if f1_col else None,
        "f1_std": float(df[f1_col].std()) if f1_col else None,
        "f2_mean": float(df[f2_col].mean()) if f2_col else None,
        "f2_std": float(df[f2_col].std()) if f2_col else None,
        "tp_total": tp,
        "fp_total": fp,
        "tn_total": int(df[tn_col].sum()) if tn_col else 0,
        "fn_total": int(df[fn_col].sum()) if fn_col else 0,
        "pooled_precision": pooled_precision,
        "pooled_recall": pooled_recall,
        "n_repeats_observed": n_rep,
        "f05_repeat_mean": f05_rep_mean,
        "f05_repeat_sd": f05_rep_sd,
        "precision_repeat_mean": p_rep_mean,
        "precision_repeat_sd": p_rep_sd,
        "recall_repeat_mean": r_rep_mean,
        "recall_repeat_sd": r_rep_sd,
        "f1_repeat_mean": f1_rep_mean,
        "f1_repeat_sd": f1_rep_sd,
        "f2_repeat_mean": f2_rep_mean,
        "f2_repeat_sd": f2_rep_sd,
    }


def print_optimise_target_table(summary_df: pd.DataFrame, primary_cfg: dict) -> None:
    target_df = summary_df[
        (summary_df["mode"] == primary_cfg["mode"])
        & (summary_df["similarity_metric"] == primary_cfg["similarity_metric"])
        & (
            pd.to_numeric(summary_df["similarity_threshold"], errors="coerce")
            == float(primary_cfg["similarity_threshold"])
        )
        & (summary_df["sort_by"] == primary_cfg["sort_by"])
        & (summary_df["optimise_for"].isin(["f0.5", "precision", "f1", "f2"]))
    ].copy()

    if target_df.empty:
        print("\nNo optimisation-target ablation rows found for target table.")
        return

    order = {"f0.5": 0, "precision": 1, "f1": 2, "f2": 3}
    target_df["__ord"] = target_df["optimise_for"].map(order).fillna(999)
    target_df = target_df.sort_values("__ord").reset_index(drop=True)

    table_rows: list[dict[str, str]] = []
    for _, row in target_df.iterrows():
        table_rows.append(
            {
                "optimise_for": str(row["optimise_for"]),
                "precision": _fmt_metric(
                    row.get("precision_repeat_mean"), row.get("precision_repeat_sd")
                ),
                "recall": _fmt_metric(
                    row.get("recall_repeat_mean"), row.get("recall_repeat_sd")
                ),
                "f0.5": _fmt_metric(row.get("f05_repeat_mean"), row.get("f05_repeat_sd")),
                "f1": _fmt_metric(row.get("f1_repeat_mean"), row.get("f1_repeat_sd")),
                "f2": _fmt_metric(row.get("f2_repeat_mean"), row.get("f2_repeat_sd")),
            }
        )

    print("\nOptimisation-target ablation table (mean +/- SD across repeats):")
    print(pd.DataFrame(table_rows).to_string(index=False))


def run_experiment(results_dir_extension: str, model: str, exp: dict, force: bool) -> Path:
    out_csv = expected_output_path(results_dir_extension, model, exp)

    cmd = [
        sys.executable,
        str(ENGINE_SCRIPT),
        "--results_dir_extension",
        results_dir_extension,
        "--mode",
        exp["mode"],
        "--similarity-metric",
        exp["similarity_metric"],
        "--similarity-threshold",
        str(exp["similarity_threshold"]),
        "--optimise-for",
        exp["optimise_for"],
        "--sort-by",
        exp["sort_by"],
        "--n-splits",
        str(exp["n_splits"]),
        "--n-repeats",
        str(exp["n_repeats"]),
    ]
    if force:
        cmd.append("--force")

    print("\nRunning secondary ablation experiment:")
    print(f"  label: {exp['label']}")
    print(f"  description: {exp.get('description', '(none)')}")
    print("  cmd:")
    print("   " + " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    if not out_csv.exists():
        raise RuntimeError(f"Expected experiment output not found: {out_csv}")
    return out_csv


def main() -> None:
    args = parse_args()
    grid = load_grid(args.grid)

    with PRIMARY_CONFIG.open("r", encoding="utf-8") as f:
        primary_cfg = json.load(f)

    results_dir_extension = args.results_dir_extension or primary_cfg["results_dir_extension"]
    model = primary_cfg["model"]

    selected_labels: set[str] | None = None
    if args.labels:
        selected_labels = {label.strip() for label in args.labels.split(",") if label.strip()}

    experiments = grid["experiments"]
    if selected_labels is not None:
        experiments = [e for e in experiments if e["label"] in selected_labels]
        if not experiments:
            raise ValueError("No experiments matched --labels filter")

    out_dir = REPO_ROOT / results_dir_extension / model / "ablation_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_summary = out_dir / "secondary_ablation_summary.csv"
    out_manifest = out_dir / "secondary_ablation_manifest.json"

    print("=" * 72)
    print("SECONDARY ABLATION RUN")
    print("=" * 72)
    print(
        "Guardrail: This is secondary analysis only. "
        "Do not replace headline metrics from scripts/04_primary_model.py."
    )

    summary_rows: list[dict] = []
    completed: list[dict] = []

    for exp in experiments:
        exp = dict(exp)
        if args.results_dir_extension:
            exp["results_dir_extension"] = args.results_dir_extension
        if args.n_repeats is not None:
            exp["n_repeats"] = args.n_repeats

        out_csv = run_experiment(results_dir_extension, model, exp, force=args.force)
        summary = summarise_csv(out_csv)

        row = {
            "label": exp["label"],
            "description": exp.get("description", ""),
            "is_primary_reference": exp["label"] == "primary_reference",
            "mode": exp["mode"],
            "similarity_metric": exp["similarity_metric"],
            "similarity_threshold": exp["similarity_threshold"],
            "optimise_for": exp["optimise_for"],
            "sort_by": exp["sort_by"],
            "n_splits": exp["n_splits"],
            "n_repeats": exp["n_repeats"],
            "output_csv": str(out_csv),
            **summary,
        }
        summary_rows.append(row)
        completed.append({"experiment": exp, "output_csv": str(out_csv), "summary": summary})

    summary_df = pd.DataFrame(summary_rows)
    sort_col = "f05_repeat_mean" if "f05_repeat_mean" in summary_df.columns else "f05_mean"
    summary_df = summary_df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    summary_df.to_csv(out_summary, index=False)

    print("\nPaper Table 2 metrics (ablation mean F0.5 across repeats):")
    for _, row in summary_df.iterrows():
        if "f05_repeat_mean" in row and "f05_repeat_sd" in row:
            metric_str = f"{row['f05_repeat_mean']:.3f} +/- {row['f05_repeat_sd']:.3f}"
        else:
            metric_str = f"{row['f05_mean']:.3f}"
        print(f"{row['label']}: mean F0.5={metric_str}")

    winner_label = str(summary_df.iloc[0]["label"])
    winner_f05 = float(summary_df.iloc[0][sort_col])
    print(f"\nBest ablation by {sort_col}: {winner_label} ({winner_f05:.3f})")
    if "primary_reference" in summary_df["label"].values and winner_label != "primary_reference":
        print(
            "NOTE: primary_reference is not the best ablation. "
            "If desired, promote winner manually to primary config."
        )

    print_optimise_target_table(summary_df, primary_cfg)

    manifest = {
        "analysis_role": "secondary_ablation",
        "grid_path": str(args.grid),
        "primary_config_path": str(PRIMARY_CONFIG),
        "guardrail_note": (
            "Secondary ablations are exploratory. "
            "Primary headline reporting remains locked to scripts/04_primary_model.py."
        ),
        "experiments_run": completed,
        "summary_csv": str(out_summary),
    }
    with out_manifest.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nSecondary ablations complete.")
    print(f"Summary CSV: {out_summary}")
    print(f"Manifest:    {out_manifest}")


if __name__ == "__main__":
    main()
