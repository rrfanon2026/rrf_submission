#!/usr/bin/env python3
"""Canonical stage 02 wrapper: clinical-trial question generation."""

from __future__ import annotations

import argparse
import sys

from _legacy import default_results_dir_extension, run_legacy_script


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 02: generate trial questions.")
    parser.add_argument("--provider", required=True, help="LLM provider (e.g. openai)")
    parser.add_argument("--model", required=True, help="LLM model (e.g. gpt-4o-mini)")
    parser.add_argument("--phase", required=True, help="Phase tag: I/II/III.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-questions", type=int, default=10)
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
    cmd_args = [
        "--provider",
        args.provider,
        "--model",
        args.model,
        "--temperature",
        str(args.temperature),
        "--results_dir_extension",
        ext,
        "--num_questions",
        str(args.num_questions),
        "--phase",
        args.phase,
    ]
    rc = run_legacy_script(
        "02_clin_trial_question_generation.py", cmd_args, dry_run=args.dry_run
    )
    if rc != 0:
        print(f"Stage 02 failed with exit code {rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
