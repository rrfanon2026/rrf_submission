#!/usr/bin/env python3
"""Shared helpers for clinical-trial pipeline wrappers."""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CLINICAL_TRIALS_ROOT = REPO_ROOT / "clinical_trials"
LEGACY_SCRIPTS_DIR = CLINICAL_TRIALS_ROOT / "legacy_snapshot" / "scripts"
LEGACY_RESULTS_DIR = CLINICAL_TRIALS_ROOT / "legacy_snapshot" / "results"


def normalise_phase(phase: str) -> str:
    """Normalize phase labels to I/II/III."""
    p = phase.strip().upper()
    p = p.replace("PHASE_", "").replace("PHASE ", "")
    if p not in {"I", "II", "III"}:
        raise ValueError(f"Unsupported phase: {phase}. Use I, II, or III.")
    return p


def default_results_dir_extension(phase: str) -> str:
    """Return extension compatible with legacy scripts.

    Legacy scripts resolve paths as:
    `project_root / "results" / results_dir_extension`
    where project_root is `<repo>/clinical_trials`.
    """
    phase_norm = normalise_phase(phase)
    return f"../legacy_snapshot/results/phase_{phase_norm}"


def resolve_results_dir(results_dir_extension: str) -> Path:
    """Resolve legacy extension to an absolute directory path."""
    return (CLINICAL_TRIALS_ROOT / "results" / results_dir_extension).resolve()


def legacy_script_path(script_name: str) -> Path:
    """Resolve a script inside legacy snapshot."""
    path = LEGACY_SCRIPTS_DIR / script_name
    if not path.exists():
        raise FileNotFoundError(f"Legacy script not found: {path}")
    return path


def run_legacy_script(script_name: str, args: list[str], dry_run: bool = False) -> int:
    """Run a legacy script with arguments."""
    script_path = legacy_script_path(script_name)
    cmd = [sys.executable, str(script_path), *args]
    print("Legacy command:")
    print(" " + shlex.join(cmd))
    if dry_run:
        return 0
    proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    return int(proc.returncode)
