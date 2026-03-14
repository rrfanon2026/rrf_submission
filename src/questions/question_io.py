from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        vv = str(v).strip()
        if not vv or vv in seen:
            continue
        seen.add(vv)
        out.append(vv)
    return out


def load_questions(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Canonical questions file not found: {path}")

    df = pd.read_csv(path)
    if "Question" not in df.columns:
        raise ValueError(f"Missing 'Question' column in {path}")

    questions = (
        df["Question"]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    if not questions:
        raise ValueError(f"No usable questions found in {path}")
    return questions


def write_question_list(path: Path, questions: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Question": questions}).to_csv(path, index=False)
    return path


def write_canonical_selected_questions(
    high_precision_file: Path, test_question_dir: Path, suffix: str, logger: logging.Logger
) -> Path:
    if not high_precision_file.exists():
        raise FileNotFoundError(f"Missing high-precision question file: {high_precision_file}")

    df = pd.read_csv(high_precision_file)
    if "Question" not in df.columns:
        raise ValueError(f"Missing 'Question' column in {high_precision_file}")

    questions = dedupe_preserve_order(df["Question"].dropna().tolist())
    out_file = test_question_dir / f"selected_questions_all{suffix}.csv"
    write_question_list(out_file, questions)
    logger.info(f"Saved canonical selected questions: {out_file} (n={len(questions)})")
    return out_file
