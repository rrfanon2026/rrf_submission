#!/usr/bin/env python3
"""Offline regression tests for precomputed RRF artifacts.

Run:
    python -m unittest tests.test_precomputed_regression -v
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "tests" / "regression_manifest.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            _ = next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


class TestPrecomputedRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not MANIFEST_PATH.exists():
            raise FileNotFoundError(
                f"Missing manifest: {MANIFEST_PATH}. "
                "Run `python tests/build_regression_manifest.py` first."
            )
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_manifest_has_expected_shape(self) -> None:
        self.assertEqual(self.manifest.get("schema_version"), 1)
        self.assertIn("files", self.manifest)
        self.assertGreater(len(self.manifest["files"]), 0)
        self.assertEqual(self.manifest.get("file_count"), len(self.manifest["files"]))

    def test_all_manifest_files_exist(self) -> None:
        missing: list[str] = []
        for item in self.manifest["files"]:
            path = ROOT / item["path"]
            if not path.exists():
                missing.append(item["path"])
        self.assertEqual(missing, [], msg=f"Missing files: {missing}")

    def test_all_manifest_hashes_match(self) -> None:
        mismatched: list[str] = []
        for item in self.manifest["files"]:
            path = ROOT / item["path"]
            expected = item["sha256"]
            actual = sha256_file(path)
            if expected != actual:
                mismatched.append(item["path"])
        self.assertEqual(
            mismatched,
            [],
            msg=(
                "Files changed relative to regression manifest. "
                "If changes are intentional, regenerate with "
                "`python tests/build_regression_manifest.py`. "
                f"Changed: {mismatched}"
            ),
        )

    def test_stage_output_counts(self) -> None:
        base = ROOT / "precomputed" / "gpt_4o_mini"
        self.assertTrue(
            (base / "initial_questions" / "generated_questions_all_anonymised.csv").exists()
        )
        self.assertTrue(
            (base / "initial_questions" / "generated_questions_unique_anonymised.csv").exists()
        )
        self.assertTrue(
            (base / "initial_questions" / "generated_questions_semantic_dedup_anonymised.csv").exists()
        )
        self.assertEqual(
            len(list((base / "initial_questions").glob("generated_questions_set_*_anonymised.csv"))),
            0,
        )
        self.assertEqual(
            len(list((base / "deduplicated").glob("selected_questions_all_anonymised.csv"))),
            0,
        )
        self.assertEqual(
            len(list((base / "validation_predictions").glob("predictions_val_set_all_question_anonymised.csv"))),
            1,
        )
        self.assertEqual(
            len(list((base / "test_questions").glob("selected_questions_all_anonymised.csv"))),
            1,
        )
        self.assertEqual(
            len(list((base / "test_predictions").glob("predictions_test_all_anonymised.csv"))),
            1,
        )
        self.assertEqual(
            len(list((base / "test_questions").glob("selected_questions_set_*_anonymised_EXPERT.csv"))),
            2,
        )
        self.assertEqual(
            len(list((base / "test_predictions").glob("predictions_test_set_*_anonymised_EXPERT.csv"))),
            2,
        )

    def test_reproduce_paper_smoke(self) -> None:
        if importlib.util.find_spec("pandas") is None:
            self.skipTest("pandas not installed in this interpreter")

        run_name = "unittest_offline_repro"
        run_dir = ROOT / "repro" / run_name
        if run_dir.exists():
            shutil.rmtree(run_dir)

        try:
            cmd = [sys.executable, "scripts/reproduce_paper.py", "--run-name", run_name]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)

            if proc.returncode != 0:
                print("STDOUT:\n", proc.stdout)
                print("STDERR:\n", proc.stderr)

            self.assertEqual(proc.returncode, 0, msg="reproduce_paper.py failed")
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "run_manifest.json").exists())

            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))

            n_rows = summary.get("rows", summary.get("n_rows"))
            precision_mean = summary.get("mean_precision_outer", summary.get("precision_mean"))
            recall_mean = summary.get("mean_recall_outer", summary.get("recall_mean"))
            f05_mean = summary.get("mean_f05_outer", summary.get("f05_mean"))

            self.assertIsNotNone(n_rows)
            self.assertIsNotNone(precision_mean)
            self.assertIsNotNone(recall_mean)
            self.assertIsNotNone(f05_mean)
            self.assertGreater(int(n_rows), 0)

            csv_copy = run_manifest.get("ablation_csv_copy")
            self.assertIsNotNone(csv_copy)
            csv_path = Path(csv_copy)
            self.assertTrue(csv_path.exists())
            self.assertEqual(int(n_rows), count_csv_rows(csv_path))
        finally:
            if run_dir.exists():
                shutil.rmtree(run_dir)


if __name__ == "__main__":
    unittest.main(verbosity=2)
