#!/usr/bin/env python3
"""One-command offline reproduction for RRF submission.

Default behavior runs the locked primary model from precomputed artifacts (no API
calls), then writes a concise run manifest + summary under `repro/`.
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
DEFAULT_CONFIG = REPO_ROOT / "configs" / "primary_model_locked.json"
PRIMARY_SCRIPT = REPO_ROOT / "scripts" / "04_primary_model.py"
SECONDARY_ABLATION_SCRIPT = REPO_ROOT / "scripts" / "05_ablation_variants.py"
DEFAULT_ABLATION_GRID = REPO_ROOT / "configs" / "secondary_ablation_grid.json"


REQUIRED_COMMON_ARTIFACTS = [
    "01_question_training_data_anonymised.csv",
    "02_question_validation_data_anonymised.csv",
    "03_full_cross_validation_test_data_anonymised.csv",
    "gpt_4o_mini/test_predictions/predictions_test_all_anonymised.csv",
]


def required_artifacts_for_mode(mode: str) -> list[str]:
    files = list(REQUIRED_COMMON_ARTIFACTS)
    if mode == "llm_expert":
        files.extend(
            [
                "gpt_4o_mini/test_predictions/predictions_test_set_7_anonymised_EXPERT.csv",
                "gpt_4o_mini/test_predictions/predictions_test_set_8_anonymised_EXPERT.csv",
            ]
        )
    elif mode == "expert_only":
        files = [
            "01_question_training_data_anonymised.csv",
            "02_question_validation_data_anonymised.csv",
            "03_full_cross_validation_test_data_anonymised.csv",
            "gpt_4o_mini/test_predictions/predictions_test_set_7_anonymised_EXPERT.csv",
            "gpt_4o_mini/test_predictions/predictions_test_set_8_anonymised_EXPERT.csv",
        ]
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline one-command paper reproduction")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to primary config (default: configs/primary_model_locked.json)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name for repro outputs",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation even if output already exists",
    )
    parser.add_argument(
        "--run-secondary-ablations",
        action="store_true",
        help="Also run structured secondary ablations (not for headline reporting)",
    )
    parser.add_argument(
        "--run-simple-variants",
        action="store_true",
        help="Deprecated alias for --run-secondary-ablations",
    )
    parser.add_argument(
        "--ablation-grid",
        type=Path,
        default=DEFAULT_ABLATION_GRID,
        help="Path to ablation grid JSON",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def check_required_artifacts(cfg: dict) -> None:
    base = REPO_ROOT / cfg["results_dir_extension"]
    required = required_artifacts_for_mode(cfg["mode"])
    missing = [str(base / p) for p in required if not (base / p).exists()]
    if missing:
        print("Missing required precomputed artifacts:")
        for m in missing:
            print(f"  - {m}")
        raise SystemExit(1)


def expected_primary_output(cfg: dict) -> Path:
    return (
        REPO_ROOT
        / cfg["results_dir_extension"]
        / cfg["model"]
        / "primary_results"
        / "primary_model_fold_metrics.csv"
    )


def run_primary(cfg: dict, config_path: Path, force: bool) -> Path:
    cmd = [
        sys.executable,
        str(PRIMARY_SCRIPT),
        "--config",
        str(config_path),
        "--results_dir_extension",
        cfg["results_dir_extension"],
    ]
    if force:
        cmd.append("--force")

    print("Running locked primary model:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)

    out_path = expected_primary_output(cfg)
    if not out_path.exists():
        raise RuntimeError(f"Expected primary output not found: {out_path}")
    return out_path


def run_secondary_ablations(cfg: dict, grid_path: Path, force: bool) -> tuple[Path, Path]:
    out_dir = REPO_ROOT / cfg["results_dir_extension"] / cfg["model"] / "ablation_results"
    out_summary = out_dir / "secondary_ablation_summary.csv"
    out_manifest = out_dir / "secondary_ablation_manifest.json"

    cmd = [
        sys.executable,
        str(SECONDARY_ABLATION_SCRIPT),
        "--grid",
        str(grid_path),
        "--results_dir_extension",
        cfg["results_dir_extension"],
    ]
    if force:
        cmd.append("--force")

    print("\nRunning structured secondary ablations:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)

    if not out_summary.exists() or not out_manifest.exists():
        raise RuntimeError("Secondary ablation outputs not found after run")
    return out_summary, out_manifest


def summarise_csv(path: Path) -> dict:
    df = pd.read_csv(path)
    if df.empty:
        return {"n_rows": 0}

    return {
        "n_rows": int(len(df)),
        "n_folds": int(df["Outer_Fold"].nunique()) if "Outer_Fold" in df.columns else None,
        "precision_mean": float(df["Precision_Outer"].mean()) if "Precision_Outer" in df.columns else None,
        "precision_std": float(df["Precision_Outer"].std()) if "Precision_Outer" in df.columns else None,
        "recall_mean": float(df["Recall_Outer"].mean()) if "Recall_Outer" in df.columns else None,
        "recall_std": float(df["Recall_Outer"].std()) if "Recall_Outer" in df.columns else None,
        "f05_mean": float(df["F05_Outer"].mean()) if "F05_Outer" in df.columns else None,
        "f05_std": float(df["F05_Outer"].std()) if "F05_Outer" in df.columns else None,
        "tp_total": int(df["TP"].sum()) if "TP" in df.columns else None,
        "fp_total": int(df["FP"].sum()) if "FP" in df.columns else None,
        "tn_total": int(df["TN"].sum()) if "TN" in df.columns else None,
        "fn_total": int(df["FN"].sum()) if "FN" in df.columns else None,
    }


def write_repro_bundle(
    cfg: dict,
    primary_csv: Path,
    run_name: str,
    secondary_summary: Path | None = None,
    secondary_manifest: Path | None = None,
) -> Path:
    out_dir = REPO_ROOT / "repro" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    copied_csv = out_dir / primary_csv.name
    shutil.copy2(primary_csv, copied_csv)

    copied_secondary_summary = None
    copied_secondary_manifest = None
    if secondary_summary is not None and secondary_manifest is not None:
        copied_secondary_summary = out_dir / secondary_summary.name
        copied_secondary_manifest = out_dir / secondary_manifest.name
        shutil.copy2(secondary_summary, copied_secondary_summary)
        shutil.copy2(secondary_manifest, copied_secondary_manifest)

    summary = summarise_csv(copied_csv)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
        "primary_csv_source": str(primary_csv),
        "primary_csv_copy": str(copied_csv),
        "ablation_csv_source": str(primary_csv),
        "ablation_csv_copy": str(copied_csv),
        "llm_calls": 0,
        "secondary_ablation_summary": (
            str(copied_secondary_summary) if copied_secondary_summary else None
        ),
        "secondary_ablation_manifest": (
            str(copied_secondary_manifest) if copied_secondary_manifest else None
        ),
        "notes": (
            "Offline reproduction from precomputed artifacts only. "
            "Primary model is locked; secondary ablations are optional."
        ),
    }
    with (out_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return out_dir


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    run_secondary = args.run_secondary_ablations or args.run_simple_variants
    run_name = args.run_name or cfg.get("name") or datetime.now().strftime("repro_%Y%m%d_%H%M%S")

    print("=" * 72)
    print("RRF Paper Reproduction (Offline / Precomputed)")
    print("=" * 72)
    print("Primary path is locked by config; secondary ablations are optional.")

    check_required_artifacts(cfg)
    primary_csv = run_primary(cfg, config_path=args.config, force=args.force)

    secondary_summary = None
    secondary_manifest = None
    if run_secondary:
        secondary_summary, secondary_manifest = run_secondary_ablations(
            cfg, grid_path=args.ablation_grid, force=args.force
        )

    out_dir = write_repro_bundle(
        cfg,
        primary_csv,
        run_name,
        secondary_summary=secondary_summary,
        secondary_manifest=secondary_manifest,
    )

    print("\nDone.")
    print(f"Primary CSV:  {primary_csv}")
    if secondary_summary:
        print(f"Ablation CSV: {secondary_summary}")
    print(f"Repro bundle: {out_dir}")


if __name__ == "__main__":
    main()
