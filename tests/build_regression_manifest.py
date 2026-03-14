#!/usr/bin/env python3
"""Build checksum manifest for key precomputed artifacts.

Usage:
    python tests/build_regression_manifest.py
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FILE_GLOBS = [
    "precomputed/01_question_training_data_anonymised.csv",
    "precomputed/02_question_validation_data_anonymised.csv",
    "precomputed/03_full_cross_validation_test_data_anonymised.csv",
    "precomputed/gpt_4o_mini/initial_questions/generated_questions_all_anonymised.csv",
    "precomputed/gpt_4o_mini/initial_questions/generated_questions_semantic_dedup_anonymised.csv",
    "precomputed/gpt_4o_mini/initial_questions/generated_questions_unique_anonymised.csv",
    "precomputed/gpt_4o_mini/similarity/questions_dedup_0_80_anonymised.csv",
    "precomputed/gpt_4o_mini/validation_predictions/predictions_val_set_all_question_anonymised.csv",
    "precomputed/gpt_4o_mini/validation_stats/high_precision_questions_anonymised_baseline_j1_0.csv",
    "precomputed/gpt_4o_mini/test_questions/selected_questions_all_anonymised.csv",
    "precomputed/gpt_4o_mini/test_questions/selected_questions_set_*_anonymised_EXPERT.csv",
    "precomputed/gpt_4o_mini/test_predictions/predictions_test_all_anonymised.csv",
    "precomputed/gpt_4o_mini/test_predictions/predictions_test_set_*_anonymised_EXPERT.csv",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_meta(path: Path) -> tuple[int | None, int | None]:
    if path.suffix.lower() != ".csv":
        return None, None
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return 0, 0
        rows = sum(1 for _ in reader)
    return rows, len(header)


def collect_files() -> list[Path]:
    files: list[Path] = []
    for pattern in FILE_GLOBS:
        files.extend(ROOT.glob(pattern))
    unique_files = sorted({p for p in files if p.is_file() and p.name != ".DS_Store"})
    return unique_files


def main() -> None:
    files = collect_files()
    manifest_files = []
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        rows, n_columns = csv_meta(path)
        manifest_files.append(
            {
                "path": rel,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "rows": rows,
                "n_columns": n_columns,
            }
        )

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(manifest_files),
        "notes": "Golden checksums for offline regression of precomputed pipeline artifacts.",
        "files": manifest_files,
    }

    out_path = ROOT / "tests" / "regression_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} with {len(manifest_files)} files")


if __name__ == "__main__":
    main()
