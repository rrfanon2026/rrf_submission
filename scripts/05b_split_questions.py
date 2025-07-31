import os
import pandas as pd
import argparse
import sys
import numpy as np
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def parse_args():
    parser = argparse.ArgumentParser(description="Split questions into chunks for testing")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--results_dir_extension', type=str, default='precomputed')
    parser.add_argument('--prefix', type=str, default="gpt_4o_mini", help="Model name prefix for filenames")
    parser.add_argument('--n_chunks', type=int, default=7, help="Number of chunks to split into")
    return parser.parse_args()


def main():
    args = parse_args()
    suffix = "_anonymised"
    base_dir = Path(project_root) / args.results_dir_extension / args.model.replace('-','_')
    test_question_dir = base_dir / "test_questions"
    os.makedirs(test_question_dir, exist_ok=True)

    input_path = base_dir / f"validation_stats/high_precision_questions{suffix}_baseline_j1_0.csv"
    df = pd.read_csv(input_path)
    questions = df["Question"].dropna().tolist()

    # Split as evenly as possible
    chunks = np.array_split(questions, args.n_chunks)
    for i, chunk in enumerate(chunks):
        out_path = test_question_dir / f"selected_questions_set_{i}{suffix}.csv"
        pd.DataFrame(chunk, columns=["Question"]).to_csv(out_path, index=False)
        print(f"✅ Saved {len(chunk)} questions to: {out_path}")

if __name__ == "__main__":
    main()
