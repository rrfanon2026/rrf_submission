import numpy as np
import pandas as pd
import logging
from scipy.spatial.distance import hamming, pdist, squareform
from sklearn.metrics import jaccard_score
from scipy.cluster.hierarchy import linkage, fcluster
# from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)

# def compute_similarity(model: SentenceTransformer, questions: list[str]) -> np.ndarray:
#     embeddings = model.encode(questions, convert_to_tensor=True)
#     return util.pytorch_cos_sim(embeddings, embeddings).cpu().numpy()

def extract_similar_pairs(sim_matrix: np.ndarray, questions: list[str], threshold: float) -> pd.DataFrame:
    pairs = [(i, j, sim_matrix[i, j])
             for i in range(len(questions))
             for j in range(i+1, len(questions))
             if sim_matrix[i, j] > threshold]
    
    pairs_df = pd.DataFrame(pairs, columns=["idx1", "idx2", "similarity"])
    pairs_df["question1"] = pairs_df.idx1.map(lambda i: questions[i])
    pairs_df["question2"] = pairs_df.idx2.map(lambda i: questions[i])
    return pairs_df.sort_values("similarity", ascending=False)

def deduplicate_questions(questions: list[str], sim_matrix: np.ndarray, threshold: float) -> list[str]:
    logger.info(f"(0) Initial number of questions: {len(questions)}")

    similar_pairs = [(i, j, sim_matrix[i, j]) 
                     for i in range(len(questions)) 
                     for j in range(i+1, len(questions)) 
                     if sim_matrix[i, j] > threshold]
    
    logger.info(f"(1) Found {len(similar_pairs)} pairs with similarity > {threshold:.2f}")
    
    remove_indices = set()
    for i, j, sim in similar_pairs:
        if i in remove_indices or j in remove_indices:
            continue
        remove_indices.add(j)
        logger.info(f"Sim={sim:.2f} | Kept Q{i+1}: {questions[i]}")
        logger.info(f"              ❌ Removed Q{j+1}: {questions[j]}")
        logger.info("-" * 100)

    dedup_questions = [q for idx, q in enumerate(questions) if idx not in remove_indices]
    logger.info(f"(3) Remaining questions after deduplication: {len(dedup_questions)}")
    return dedup_questions

def deduplicate_exact_questions(df):
    while True:
        duplicate_counts = df['Question'].value_counts()
        duplicate_questions = duplicate_counts[duplicate_counts > 1].index.tolist()
        if not duplicate_questions:
            break
        df = df.drop_duplicates(subset=['Question'], keep='first').reset_index(drop=True)
    return df


def load_and_merge_predictions(predictions_dir, predictions_files, special_questions):
    """Load prediction files, tag each question with its source, and concatenate."""
    combined_df = pd.DataFrame()
    special_rows_added = False
    special_rows = pd.DataFrame()

    for file_name in predictions_files:
        df = pd.read_csv(predictions_dir / file_name)

        # Split special vs. regular rows
        is_special = df['Question'].isin(special_questions)
        regular_df = df[~is_special].copy()
        special_df = df[is_special]

        # Determine source only for regular questions
        if "EXPERT" in file_name.upper():
            regular_df["Source"] = "expert"
        else:
            regular_df["Source"] = "llm"

        if not special_rows_added:
            special_rows = special_df
            special_rows_added = True

        combined_df = pd.concat([combined_df, regular_df], ignore_index=True)

    return pd.concat([special_rows, combined_df], ignore_index=True)


def filter_low_precision_questions(df, special_questions, precision_cutoff):
    """Removes questions below the precision cutoff and logs them. Special rows are untouched."""
    filtered_df = df[~df['Question'].isin(special_questions)]
    low_precision = filtered_df[filtered_df['Prec'].fillna(0).astype(float) < precision_cutoff]

    if not low_precision.empty:
        logger.info("\nQuestions removed due to low precision:")
        for q, p in zip(low_precision['Question'], low_precision['Prec']):
            logger.info(f" - {q} (Prec={p:.4f})")

    return df[
        (~df['Question'].isin(special_questions)) &
        (df['Prec'].fillna(0).astype(float) >= precision_cutoff)
    ].reset_index(drop=True)


def clean_question_text(df):
    df['Question'] = df['Question'].astype(str).str.strip().str.lower()
    df['Question'] = df['Question'].apply(lambda x: ''.join(char for char in x if char.isprintable()))
    return df


def add_success_row(df, sample_file_path):
    sample_df = pd.read_csv(sample_file_path)
    success_row = sample_df[sample_df['Question'] == 'Success']
    
    if not success_row.empty:
        print("✅ 'Success' row found in the original prediction file.")
        return pd.concat([success_row, df], ignore_index=True)
    else:
        print("❌ 'Success' row not found in the sample file. Please check the file content.")
        return df
    

def remove_jaccard_similar_questions(df, threshold=0.9):
    df.replace(['NaN', 'nan', 'NAN'], np.nan, inplace=True)
    for col in df.columns:
        if col not in ['Question', 'Prec']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    question_results = df.iloc[:, 2:].fillna(0).astype(int).values
    question_names = df['Question'].values

    num_questions = question_results.shape[0]
    to_remove = set()

    for i in range(num_questions):
        for j in range(i + 1, num_questions):
            score = jaccard_score(question_results[i], question_results[j], average='micro')
            if score > threshold:
                q1, q2 = question_names[i], question_names[j]
                p1 = df[df['Question'] == q1]['Prec'].values[0]
                p2 = df[df['Question'] == q2]['Prec'].values[0]
                if p1 < p2:
                    logger.info(f"Removing similar question (lower precision): {q1} vs {q2}")
                    to_remove.add(q1)
                else:
                    logger.info(f"Removing similar question (lower precision): {q2} vs {q1}")
                    to_remove.add(q2)

    return df[~df['Question'].isin(to_remove)].reset_index(drop=True)
    
def remove_hamming_similar_questions(df, threshold=0.1):  # e.g., 10% mismatch allowed
    df.replace(['NaN', 'nan', 'NAN'], np.nan, inplace=True)
    for col in df.columns:
        if col not in ['Question', 'Prec']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    question_results = df.iloc[:, 2:].fillna(0).astype(int).values
    question_names = df['Question'].values

    num_questions = question_results.shape[0]
    to_remove = set()

    for i in range(num_questions):
        for j in range(i + 1, num_questions):
            score = hamming(question_results[i], question_results[j])
            if score < threshold:
                q1, q2 = question_names[i], question_names[j]
                p1 = df[df['Question'] == q1]['Prec'].values[0]
                p2 = df[df['Question'] == q2]['Prec'].values[0]
                if p1 < p2:
                    logger.info(f"Removing similar question (lower precision): {q1} vs {q2} [Hamming: {score:.3f}]")
                    to_remove.add(q1)
                else:
                    logger.info(f"Removing similar question (lower precision): {q2} vs {q1} [Hamming: {score:.3f}]")
                    to_remove.add(q2)

    return df[~df['Question'].isin(to_remove)].reset_index(drop=True)

def remove_colinear_questions_via_clustering(df, threshold=0.95, method="cosine"):
    """
    Removes colinear questions by clustering highly similar ones and keeping only the highest precision per cluster.
    Returns both the filtered dataframe and the distance matrix (for debugging/visualization).
    """
    df.replace(['NaN', 'nan', 'NAN'], np.nan, inplace=True)
    for col in df.columns:
        if col not in ['Question', 'Prec']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    question_results = df.iloc[:, 2:].fillna(0).astype(int).values
    question_names = df['Question'].values
    precisions = df['Prec'].values

    # Compute pairwise distance matrix
    if method == "cosine":
        distances = squareform(pdist(question_results, metric='cosine'))
    else:
        raise ValueError("Unsupported method: choose 'correlation' or 'cosine'")

    # Linkage and clustering
    Z = linkage(distances, method='average')
    cluster_labels = fcluster(Z, t=1-threshold, criterion='distance')

    unique_clusters = np.unique(cluster_labels)
    logger.info(f"🔍 Found {len(unique_clusters)} clusters (threshold={threshold})")

    to_remove = set()
    for cluster_id in unique_clusters:
        idxs = np.where(cluster_labels == cluster_id)[0]
        if len(idxs) <= 1:
            continue  # skip singletons

        cluster_questions = [question_names[i] for i in idxs]
        cluster_precisions = [precisions[i] for i in idxs]

        logger.info(f"🧩 Cluster {cluster_id} (size={len(idxs)}):")
        for q, p in zip(cluster_questions, cluster_precisions):
            logger.info(f"   - {q} (precision={p:.3f})")

        best_idx = idxs[np.argmax(cluster_precisions)]
        logger.info(f"✅ Keeping: {question_names[best_idx]}")
        for i in idxs:
            if i != best_idx:
                q_rm = question_names[i]
                logger.info(f"🗑️ Removing: {q_rm} (precision {precisions[i]:.3f})")
                to_remove.add(q_rm)

    filtered_df = df[~df['Question'].isin(to_remove)].reset_index(drop=True)

    return filtered_df, distances  # return distance matrix for inspection