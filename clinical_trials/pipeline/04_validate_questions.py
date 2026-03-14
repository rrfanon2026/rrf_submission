#!/usr/bin/env python3
"""Canonical stage 04 wrapper: filter/deduplicate/invert question matrices."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd

from _legacy import (
    default_results_dir_extension,
    legacy_script_path,
    normalise_phase,
    resolve_results_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 04: validate/filter questions.")
    parser.add_argument("--model", required=True, help="LLM model (e.g. gpt-4o-mini)")
    parser.add_argument("--phase", required=True, help="Phase tag: I/II/III.")
    parser.add_argument(
        "--splits",
        default="val_split,test",
        help="Comma-separated subset of: val_split,test.",
    )
    parser.add_argument(
        "--results-dir-extension",
        default=None,
        help=(
            "Legacy results extension. Default maps phase to "
            "../legacy_snapshot/results/phase_<PHASE>."
        ),
    )
    parser.add_argument(
        "--predictions-dir",
        default=None,
        help="Absolute/relative directory containing phase prediction CSVs.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions only.")
    return parser.parse_args()


def parse_splits(raw: str) -> list[str]:
    allowed = {"val_split", "test"}
    splits = [part.strip() for part in raw.split(",") if part.strip()]
    if not splits:
        raise ValueError("--splits cannot be empty.")
    unknown = [s for s in splits if s not in allowed]
    if unknown:
        raise ValueError(f"Unsupported split(s): {unknown}. Allowed: val_split,test")
    return splits


def ensure_mpl_cache_dir() -> None:
    """Avoid matplotlib writing into non-writable home directories."""
    cache_dir = Path("/tmp/matplotlib-cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))


def install_seaborn_stub() -> None:
    """Install a minimal seaborn stub for legacy plotting calls."""
    if "seaborn" in sys.modules:
        return

    stub = ModuleType("seaborn")

    def histplot(data, bins=20, kde=True, **_kwargs):  # noqa: ARG001
        import matplotlib.pyplot as plt

        plt.hist(data, bins=bins)

    stub.histplot = histplot
    sys.modules["seaborn"] = stub


def import_legacy_module(path: Path) -> ModuleType:
    """Import a module from an explicit path."""
    spec = importlib.util.spec_from_file_location("legacy_stage04", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load legacy module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_legacy_stage04_module() -> ModuleType:
    """Load the legacy stage-04 module without editing snapshot files."""
    ensure_mpl_cache_dir()
    path = legacy_script_path("06_analyse_final_predictions_V3.py")
    try:
        return import_legacy_module(path)
    except ModuleNotFoundError as exc:
        if exc.name != "seaborn":
            raise
        print("Warning: seaborn not installed; using lightweight histogram fallback.")
        install_seaborn_stub()
        return import_legacy_module(path)


def load_question_list_from_val(
    module: ModuleType, predictions_dir: Path, phase: str, model: str
) -> list[str]:
    """Load previously generated val-split filtered question list."""
    model_tag = model.replace("-", "_")
    val_file = (
        predictions_dir
        / f"phase_{phase}_filtered_combined_questions_with_success_{model_tag}_val_split.csv"
    )
    if not val_file.exists():
        raise FileNotFoundError(
            "Test split requested without val_split run in this execution, and existing "
            f"val file was not found: {val_file}"
        )
    val_df = pd.read_csv(val_file)
    mask = ~val_df["Question"].isin(module.SPECIAL_QUESTIONS)
    return val_df.loc[mask, "Question"].astype(str).tolist()


def main() -> int:
    args = parse_args()
    phase = normalise_phase(args.phase)
    splits = parse_splits(args.splits)

    ext = args.results_dir_extension or default_results_dir_extension(phase)
    if args.predictions_dir:
        predictions_dir = Path(args.predictions_dir).expanduser().resolve()
    else:
        predictions_dir = resolve_results_dir(ext)

    print(f"Predictions directory: {predictions_dir}")
    print(f"Requested splits: {', '.join(splits)}")
    if args.dry_run:
        print("Dry run enabled; no filtering executed.")
        return 0

    module = load_legacy_stage04_module()
    question_list: list[str] | None = None

    if "val_split" in splits:
        val_df = module.process_split("val_split", args.model, phase, predictions_dir)
        mask = ~val_df["Question"].isin(module.SPECIAL_QUESTIONS)
        question_list = val_df.loc[mask, "Question"].astype(str).tolist()

    if "test" in splits:
        if question_list is None:
            question_list = load_question_list_from_val(
                module, predictions_dir, phase, args.model
            )
        module.process_split(
            "test",
            args.model,
            phase,
            predictions_dir,
            question_order=question_list,
            questions_to_keep=question_list,
        )

    print("Stage 04 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
