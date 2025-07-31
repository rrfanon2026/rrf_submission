#!/usr/bin/env python3

import argparse
import logging
import sys
import pandas as pd
from pathlib import Path


# ── Setup ─────────────────────────────────────────────
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.dataset_utils import split_into_sets
from src.questions.generation import save_questions
from src.questions.anonymised_generation import generate_questions_anonymised, generate_updated_question_set_anonymised
from src.llms.utils import get_llm_client

# ── Logging ────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── CLI ────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description='Generate and test evaluation questions')
    parser.add_argument('--provider', required=True, help='LLM provider (e.g., openai)')
    parser.add_argument('--model', required=True, help='LLM model (e.g., gpt-4o-mini)')
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--results_dir_extension', type=str, default='precomputed')
    return parser.parse_args()

# ── Utility ─────────────────────────────────────────────
def generate_and_save_questions(set_idx, founder_data, llm_client, args, questions_dir):
    output_file = questions_dir / f"generated_questions_set_{set_idx}_anonymised.csv"

    questions = generate_questions_anonymised(llm_client, founder_data) if set_idx == 0 \
                    else generate_updated_question_set_anonymised(founder_data, llm_client, logger)

    save_questions(questions, output_file)
    logger.info(f"✓ Saved {len(questions)} questions → {output_file}")


# ── Main ───────────────────────────────────────────────
def main():
    args = parse_args()
    model_clean = args.model.replace('-', '_')

    questions_dir = project_root / args.results_dir_extension / model_clean / "initial_questions"
    questions_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data_path = project_root / args.results_dir_extension / "01_question_training_data_anonymised.csv"
    logger.info(f"Loading training data from: {data_path}")
    train_data = pd.read_csv(data_path)

    # Split and init
    sets = split_into_sets(train_data)
    llm_client = get_llm_client(args.provider, args.model, args.temperature)

    # Generate questions for each set
    for idx, founder_set in enumerate(sets):
        logger.info(f"Generating questions for set {idx + 1} / {len(sets)}")
        generate_and_save_questions(idx, founder_set, llm_client, args, questions_dir)

    logger.info("🎉 Done.")


if __name__ == "__main__":
    main()