### 3b to be merged with 3

import argparse
import pandas as pd
from pathlib import Path
import math
import logging

# === LOGGING SETUP ===
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# === ARGUMENTS ===
def parse_args():
    parser = argparse.ArgumentParser(description="Split deduplicated questions into multiple files")
    parser.add_argument('--model', type=str, required=True, help="Model name used to generate questions (e.g., gpt_4o_mini)")
    parser.add_argument('--results_dir_extension', type=str, default='precomputed')
    parser.add_argument('--n_chunks', type=int, default=10, help='Number of output splits')
    return parser.parse_args()

def main():
    args = parse_args()
    suffix = '_anonymised'

    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / args.results_dir_extension / args.model.replace('-','_')
    similarity_dir = results_dir / "similarity"
    dedup_dir = results_dir / "deduplicated"
    dedup_dir.mkdir(parents=True, exist_ok=True)
    input_file = similarity_dir / f"questions_dedup_0_80{suffix}.csv"

    if not input_file.exists():
        raise FileNotFoundError(f"❌ Input file not found: {input_file}")

    # === LOAD QUESTIONS ===
    df = pd.read_csv(input_file)
    questions = df["Question"].tolist()
    n_total = len(questions)
    chunk_size = math.ceil(n_total / args.n_chunks)

    logger.info(f"📦 Total questions: {n_total}, Chunk size: {chunk_size}, Suffix: {suffix}")

    # === SPLIT AND SAVE ===
    for i in range(args.n_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunk_questions = questions[start:end]

        out_df = pd.DataFrame({"Question": chunk_questions})
        out_path = dedup_dir / f"generated_questions_set_dedup_{i}{suffix}.csv"
        out_df.to_csv(out_path, index=False)
        logger.info(f"✅ Saved: {out_path.name} ({len(chunk_questions)} questions)")

if __name__ == "__main__":
    main()