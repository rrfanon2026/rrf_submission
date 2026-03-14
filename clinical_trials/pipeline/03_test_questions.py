#!/usr/bin/env python3
"""Canonical stage 03 wrapper: trial question testing/inference."""

from __future__ import annotations

import argparse
import sys

from _legacy import default_results_dir_extension, run_legacy_script


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 03: run trial inference.")
    parser.add_argument("--provider", required=True, help="LLM provider (e.g. openai)")
    parser.add_argument("--model", required=True, help="LLM model (e.g. gpt-4o-mini)")
    parser.add_argument("--phase", required=True, help="Phase tag: I/II/III.")
    parser.add_argument(
        "--mode",
        choices=["questions", "vanilla", "final"],
        default="questions",
        help="questions/vanilla use legacy parallel script; final uses final script.",
    )
    parser.add_argument("--test-set", default="val_split", help="e.g. val_split or test")
    parser.add_argument("--question-set", default=None, help="Question set id for questions mode.")
    parser.add_argument(
        "--questions-file",
        default=None,
        help="Path to combined/final question CSV (required in final mode).",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=5)
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Optional trial cap for quicker smoke tests.",
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

    if args.mode == "final":
        if not args.questions_file:
            raise ValueError("--questions-file is required when --mode final")
        cmd_args = [
            "--provider",
            args.provider,
            "--model",
            args.model,
            "--temperature",
            str(args.temperature),
            "--questions_file",
            args.questions_file,
            "--test_set",
            args.test_set,
            "--phase",
            args.phase,
            "--results_dir_extension",
            ext,
            "--batch_size",
            str(args.batch_size),
            "--num_workers",
            str(args.num_workers),
        ]
        if args.max_trials is not None:
            cmd_args.extend(["--max_trials", str(args.max_trials)])
        rc = run_legacy_script(
            "03_validate_questions_final.py", cmd_args, dry_run=args.dry_run
        )
    else:
        question_set = args.question_set if args.question_set is not None else "0"
        cmd_args = [
            "--provider",
            args.provider,
            "--model",
            args.model,
            "--temperature",
            str(args.temperature),
            "--question-set",
            question_set,
            "--test_set",
            args.test_set,
            "--phase",
            args.phase,
            "--results_dir_extension",
            ext,
            "--batch_size",
            str(args.batch_size),
            "--mode",
            args.mode,
            "--num_workers",
            str(args.num_workers),
        ]
        if args.max_trials is not None:
            cmd_args.extend(["--max_trials", str(args.max_trials)])
        rc = run_legacy_script(
            "03_validate_questions_parallel.py", cmd_args, dry_run=args.dry_run
        )

    if rc != 0:
        print(f"Stage 03 failed with exit code {rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
