### Note: I struggle to run this locally

import argparse
import pandas as pd
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import logging

# Setup project root and logging
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from src.utils.plotting import plot_similarity_heatmap
from src.questions.similarity_utils import compute_similarity, extract_similar_pairs, deduplicate_questions

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# ------------------------
# ARGUMENT PARSING
# ------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Which model was used to generate the questions/')
    parser.add_argument('--threshold', type=float, required=True, help='Threshold of similarity to remove')
    parser.add_argument('--results_dir_extension', type=str, default='precomputed')
 
    return parser.parse_args()

def load_questions(load_dir: Path, search_suffix: str) -> list[str]:
    csv_files = list(load_dir.glob(f"generated_questions_set_*{search_suffix}.csv"))
    print("Loaded files:", csv_files)

    q_series = []
    for fp in csv_files:
        df = pd.read_csv(fp)
        if "Question" in df.columns:
            q_series.append(df["Question"])
        else:
            print(f"Warning: no 'Question' column in {fp}")

    all_questions = pd.concat(q_series, ignore_index=True)
    return all_questions.drop_duplicates().reset_index(drop=True).tolist()

# ------------------------
# MAIN
# ------------------------
def main():
    args = parse_args()
    suffix = "_anonymised"

    base_dir = project_root / 'results' / args.results_dir_extension / args.model.replace('-', '_')
    load_dir = base_dir / 'initial_questions'
    save_dir = base_dir / 'similarity'
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Load ---
    questions = load_questions(load_dir, suffix)
    logger.info(f"(0) Loaded {len(questions)} unique questions")

    # --- Similarity ---
    model = SentenceTransformer(str(project_root / "results" / "all-MiniLM-L6-v2"), local_files_only=True)
    sim_matrix = compute_similarity(model, questions)
    pairs_df = extract_similar_pairs(sim_matrix, questions, args.threshold)
    pd.DataFrame(sim_matrix).to_csv(save_dir / f"question_similarity_matrix{suffix}.csv", index=False)    
    pairs_df.to_csv(save_dir / f"high_similarity_pairs{suffix}.csv", index=False)
    plot_similarity_heatmap(sim_matrix, save_dir, suffix)

    # --- Deduplication ---
    questions_dedup = deduplicate_questions(questions, sim_matrix, args.threshold)
    embeddings_dedup = model.encode(questions_dedup, convert_to_tensor=True)
    sim_matrix_dedup = util.pytorch_cos_sim(embeddings_dedup, embeddings_dedup).cpu().numpy()
    pd.DataFrame(sim_matrix_dedup).to_csv(save_dir / f"question_similarity_matrix_dedup{suffix}.csv", index=False)
    pairs08_df = extract_similar_pairs(sim_matrix_dedup, questions_dedup, args.threshold)
    pairs08_df.to_csv(save_dir / f"high_similarity_pairs_dedup_0_8{suffix}.csv", index=False)
    pd.DataFrame({'Question': questions_dedup}).to_csv(save_dir / f"questions_dedup_0_80{suffix}.csv", index=False)

    logger.info("✅ Done.")

if __name__ == "__main__":
    main()