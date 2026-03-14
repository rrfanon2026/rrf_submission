#!/usr/bin/env python3
"""Run the fixed primary offline RRF model (precomputed predictions only).

This script is the publication-facing entry point for headline metrics.
Guardrail: model settings are locked in configs/primary_model_locked.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINE_SCRIPT = REPO_ROOT / "scripts" / "03_nested_cv_engine.py"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "primary_model_locked.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fixed primary offline RRF model (locked config)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to locked primary config (default: configs/primary_model_locked.json)",
    )
    parser.add_argument(
        "--results_dir_extension",
        type=str,
        default=None,
        help=(
            "Optional override for results root directory (e.g., precomputed). "
            "Model settings stay fixed."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation even if primary output exists.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=None,
        help="Optional override for evaluation repeats (useful for smoke tests).",
    )
    return parser.parse_args()


def load_locked_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    if not cfg.get("locked", False):
        raise ValueError(
            f"Primary config must be locked=true. Invalid config: {path}"
        )
    if cfg.get("analysis_role") != "primary":
        raise ValueError(
            f"Primary config must declare analysis_role='primary'. Invalid config: {path}"
        )
    return cfg


def expected_engine_output_path(cfg: dict) -> Path:
    suffix = "_anonymised"
    similarity_str = f"{cfg['similarity_threshold']:.2f}".replace(".", "_")
    opt_str = f"optimise{cfg['optimise_for'].replace('.', '_').upper()}"
    sort_str = f"sortby{cfg['sort_by'].replace('.', '_')}"

    out_dir = REPO_ROOT / cfg["results_dir_extension"] / cfg["model"] / "ablation_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / (
        f"nested_cv_{cfg['mode']}_{suffix}"
        f"similarity_{cfg['similarity_metric']}_{similarity_str}_{opt_str}_{sort_str}.csv"
    )


def primary_output_paths(cfg: dict) -> tuple[Path, Path, Path]:
    out_dir = REPO_ROOT / cfg["results_dir_extension"] / cfg["model"] / "primary_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    folds_csv = out_dir / "primary_model_fold_metrics.csv"
    summary_json = out_dir / "primary_model_summary.json"
    manifest_json = out_dir / "primary_model_manifest.json"
    return folds_csv, summary_json, manifest_json


def run_engine(cfg: dict, force: bool) -> Path:
    out_path = expected_engine_output_path(cfg)
    cmd = [
        sys.executable,
        str(ENGINE_SCRIPT),
        "--results_dir_extension",
        cfg["results_dir_extension"],
        "--mode",
        cfg["mode"],
        "--similarity-metric",
        cfg["similarity_metric"],
        "--similarity-threshold",
        str(cfg["similarity_threshold"]),
        "--optimise-for",
        cfg["optimise_for"],
        "--sort-by",
        cfg["sort_by"],
        "--n-splits",
        str(cfg["n_splits"]),
        "--n-repeats",
        str(cfg["n_repeats"]),
    ]
    if force:
        cmd.append("--force")

    print("Running locked primary model via offline engine:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    return out_path


def summarise(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    if df.empty:
        return {"n_rows": 0}

    tp = int(df["TP"].sum()) if "TP" in df.columns else 0
    fp = int(df["FP"].sum()) if "FP" in df.columns else 0
    fn = int(df["FN"].sum()) if "FN" in df.columns else 0
    pooled_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    pooled_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if "Repeat" in df.columns and df["Repeat"].nunique() > 0:
        repeat_df = (
            df.groupby("Repeat", as_index=False)
            .agg(
                Precision_Repeat=("Precision_Outer", "mean"),
                Recall_Repeat=("Recall_Outer", "mean"),
                F05_Repeat=("F05_Outer", "mean"),
            )
            .sort_values("Repeat")
        )
    else:
        repeat_df = pd.DataFrame(
            [
                {
                    "Repeat": 1,
                    "Precision_Repeat": float(df["Precision_Outer"].mean()),
                    "Recall_Repeat": float(df["Recall_Outer"].mean()),
                    "F05_Repeat": float(df["F05_Outer"].mean()),
                }
            ]
        )

    n_rep = int(len(repeat_df))
    p_rep_mean = float(repeat_df["Precision_Repeat"].mean())
    r_rep_mean = float(repeat_df["Recall_Repeat"].mean())
    f05_rep_mean = float(repeat_df["F05_Repeat"].mean())
    p_rep_sd = (
        float(repeat_df["Precision_Repeat"].std(ddof=1)) if n_rep > 1 else 0.0
    )
    r_rep_sd = float(repeat_df["Recall_Repeat"].std(ddof=1)) if n_rep > 1 else 0.0
    f05_rep_sd = float(repeat_df["F05_Repeat"].std(ddof=1)) if n_rep > 1 else 0.0

    return {
        "n_rows": int(len(df)),
        "precision_mean": float(df["Precision_Outer"].mean()),
        "precision_std": float(df["Precision_Outer"].std()),
        "recall_mean": float(df["Recall_Outer"].mean()),
        "recall_std": float(df["Recall_Outer"].std()),
        "f05_mean": float(df["F05_Outer"].mean()),
        "f05_std": float(df["F05_Outer"].std()),
        "tp_total": tp,
        "fp_total": fp,
        "tn_total": int(df["TN"].sum()) if "TN" in df.columns else 0,
        "fn_total": int(df["FN"].sum()) if "FN" in df.columns else 0,
        "pooled_precision": pooled_precision,
        "pooled_recall": pooled_recall,
        "n_repeats_observed": n_rep,
        "precision_repeat_mean": p_rep_mean,
        "precision_repeat_sd": p_rep_sd,
        "recall_repeat_mean": r_rep_mean,
        "recall_repeat_sd": r_rep_sd,
        "f05_repeat_mean": f05_rep_mean,
        "f05_repeat_sd": f05_rep_sd,
    }


def print_paper_metrics(summary: dict) -> None:
    print("\nPaper Table 1 metrics (mean +/- SD across repeats):")
    print(
        "RRF (primary) | "
        f"Precision={summary['precision_repeat_mean']:.3f} +/- "
        f"{summary['precision_repeat_sd']:.3f} | "
        f"Recall={summary['recall_repeat_mean']:.3f} +/- "
        f"{summary['recall_repeat_sd']:.3f} | "
        f"F0.5={summary['f05_repeat_mean']:.3f} +/- "
        f"{summary['f05_repeat_sd']:.3f} | "
        f"N_repeats={summary['n_repeats_observed']}"
    )


def main() -> None:
    args = parse_args()
    cfg = load_locked_config(args.config)

    if args.results_dir_extension:
        cfg = dict(cfg)
        cfg["results_dir_extension"] = args.results_dir_extension
    if args.n_repeats is not None:
        cfg = dict(cfg)
        cfg["n_repeats"] = args.n_repeats

    folds_csv, summary_json, manifest_json = primary_output_paths(cfg)
    if folds_csv.exists() and not args.force:
        print(f"Primary output already exists at: {folds_csv}")
        print("Use --force to recompute.")
        summary = summarise(folds_csv)
        with summary_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print_paper_metrics(summary)
        return

    print("=" * 72)
    print("PRIMARY MODEL RUN (LOCKED)")
    print("=" * 72)
    print("Guardrail: This config is fixed for headline reporting.")
    print("Secondary analyses must be run separately via scripts/05_ablation_variants.py.")

    engine_output = run_engine(cfg, force=args.force)
    if not engine_output.exists():
        raise RuntimeError(f"Engine output not found: {engine_output}")

    shutil.copy2(engine_output, folds_csv)
    summary = summarise(folds_csv)
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print_paper_metrics(summary)

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": "primary",
        "locked": True,
        "config_path": str(args.config),
        "config": cfg,
        "engine_script": str(ENGINE_SCRIPT),
        "engine_output_csv": str(engine_output),
        "primary_output_csv": str(folds_csv),
        "primary_summary_json": str(summary_json),
        "guardrail_note": (
            "Primary model config is fixed before analysis. "
            "Do not replace headline metrics using secondary ablation results."
        ),
    }
    with manifest_json.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nPrimary run complete.")
    print(f"Primary folds CSV: {folds_csv}")
    print(f"Primary summary:   {summary_json}")
    print(f"Primary manifest:  {manifest_json}")


if __name__ == "__main__":
    main()
