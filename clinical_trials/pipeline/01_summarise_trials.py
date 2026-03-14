#!/usr/bin/env python3
"""Canonical stage 01 wrapper: clinical-trial summarisation."""

from __future__ import annotations

import argparse
import sys

from _legacy import default_results_dir_extension, run_legacy_script


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 01: summarise clinical trials.")
    parser.add_argument("--provider", required=True, help="LLM provider (e.g. openai)")
    parser.add_argument("--model", required=True, help="LLM model (e.g. gpt-4o-mini)")
    parser.add_argument(
        "--phase",
        default="I",
        help="Phase tag used for default results path (I/II/III).",
    )
    parser.add_argument(
        "--input-filename",
        required=True,
        help="Input filename under the selected results directory.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--runner",
        choices=["parallel", "sequential"],
        default="parallel",
        help="Which legacy summarisation backend to call.",
    )
    parser.add_argument(
        "--results-dir-extension",
        default=None,
        help=(
            "Legacy results extension. Default maps phase to "
            "../legacy_snapshot/results/phase_<PHASE>."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print command only.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ext = args.results_dir_extension or default_results_dir_extension(args.phase)
    script_name = (
        "01_build_clin_trial_summaries_parallel.py"
        if args.runner == "parallel"
        else "01_build_clin_trial_summaries.py"
    )
    cmd_args = [
        "--provider",
        args.provider,
        "--model",
        args.model,
        "--temperature",
        str(args.temperature),
        "--results_dir_extension",
        ext,
        "--input_filename",
        args.input_filename,
    ]
    rc = run_legacy_script(script_name, cmd_args, dry_run=args.dry_run)
    if rc != 0:
        print(f"Stage 01 failed with exit code {rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
