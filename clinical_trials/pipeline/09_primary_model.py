#!/usr/bin/env python3
"""Run locked primary RRF evaluation for clinical-trials Phase I."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from _rrf_eval import load_question_matrix, run_rrf_repeats, summarise_repeat_metrics


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "clinical_trials" / "configs" / "primary_model_locked.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run locked primary model for clinical-trials Phase I."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Locked config path (default: clinical_trials/configs/primary_model_locked.json)",
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
        raise ValueError("Primary config must set locked=true.")
    if cfg.get("analysis_role") != "primary":
        raise ValueError("Primary config must set analysis_role='primary'.")
    return cfg


def resolve_paths(cfg: dict) -> tuple[Path, Path, Path]:
    input_dir = (REPO_ROOT / cfg["input_dir"]).resolve()
    out_dir = (REPO_ROOT / cfg["output_dir"]).resolve()
    val_path = input_dir / cfg["val_file"]
    test_path = input_dir / cfg["test_file"]
    return val_path, test_path, out_dir


def print_table_metrics(summary: dict) -> None:
    def fm(mean_v: float | None, sd_v: float | None) -> str:
        if mean_v is None or sd_v is None:
            return "n/a"
        return f"{mean_v:.3f} +/- {sd_v:.3f}"

    print("\nClinical-trials primary metrics (mean +/- SD across repeats):")
    print(
        "RRF | "
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
    repeat_csv = out_dir / "primary_model_repeat_metrics.csv"
    summary_json = out_dir / "primary_model_summary.json"
    manifest_json = out_dir / "primary_model_manifest.json"

    if repeat_csv.exists() and not args.force:
        print(f"Primary output already exists at: {repeat_csv}")
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
    print("CLINICAL TRIALS PRIMARY MODEL RUN (LOCKED)")
    print("=" * 72)
    print("Guardrail: config is fixed before analysis.")
    print("Ablations must be run separately and treated as secondary analysis.")
    print(f"Validation matrix: {val_path}")
    print(f"Test matrix:       {test_path}")

    M_val, y_val = load_question_matrix(val_path)
    M_test, y_test = load_question_matrix(test_path)

    df = run_rrf_repeats(
        M_val=M_val,
        y_val=y_val,
        M_test=M_test,
        y_test=y_test,
        n_splits=int(cfg["n_splits"]),
        min_q=int(cfg["min_q"]),
        optimise_for=str(cfg["optimise_for"]),
        n_repeats=int(cfg["n_repeats"]),
        seed_start=int(cfg["seed_start"]),
    )
    df.to_csv(repeat_csv, index=False)

    summary = summarise_repeat_metrics(df)
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print_table_metrics(summary)

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": "primary",
        "locked": True,
        "config_path": str(args.config),
        "config": cfg,
        "inputs": {
            "val_csv": str(val_path),
            "test_csv": str(test_path),
        },
        "outputs": {
            "repeat_metrics_csv": str(repeat_csv),
            "summary_json": str(summary_json),
            "manifest_json": str(manifest_json),
        },
        "guardrail_note": (
            "Primary model configuration is fixed up front. "
            "Do not overwrite headline reporting with secondary ablations."
        ),
    }
    with manifest_json.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nClinical primary run complete.")
    print(f"Repeat metrics CSV: {repeat_csv}")
    print(f"Summary JSON:       {summary_json}")
    print(f"Manifest JSON:      {manifest_json}")


if __name__ == "__main__":
    main()
