import pandas as pd
import numpy as np
import glob
import logging
from itertools import combinations
from sklearn.metrics import precision_score, recall_score, jaccard_score
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def normalize_questions(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize question text and remove non-printable characters."""
    df = df.copy()
    df['Question'] = (
        df['Question']
        .astype(str)
        .str.strip()
        .str.lower()
        .apply(lambda x: ''.join(ch for ch in x if ch.isprintable()))
    )
    return df

def drop_duplicates(df: pd.DataFrame) -> (pd.DataFrame, int):
    """Drop duplicate questions, return deduped DF and count removed."""
    before = len(df)
    deduped = df.drop_duplicates(subset=['Question'], keep='first').reset_index(drop=True)
    removed = before - len(deduped)
    return deduped, removed

def find_similar_questions(
    results: np.ndarray,
    names: list[str],
    thresh: float
) -> list[tuple[str, str, float]]:
    """Return list of question pairs with Jaccard similarity above thresh."""
    
    pairs: list[tuple[str, str, float]] = []
    n = len(results)
    for i, j in combinations(range(n), 2):
        score = jaccard_score(results[i], results[j], average='micro')
        if score > thresh:
            pairs.append((names[i], names[j], float(score)))
    return pairs


def drop_similar(df: pd.DataFrame, similar_pairs: list[tuple[str,str,float]]) -> pd.DataFrame:
    """Remove questions based on similarity, preserving expert-only similarities."""
    df = df.copy().reset_index(drop=True)
    to_remove = set()
    # Build lookup maps for precision and set index
    prec_map = df.set_index('Question')['Prec'].fillna(0).astype(float).to_dict()
    set_map = df.set_index('Question')['__set'].fillna(0).astype(int).to_dict()

    for q1, q2, score in similar_pairs:
        s1, s2 = set_map.get(q1, 0), set_map.get(q2, 0)
        # both expert: skip
        if s1 > 5 and s2 > 5:
            continue
        # one expert: remove non-expert
        if s1 > 5 or s2 > 5:
            rem = q2 if s1 > 5 else q1
            to_remove.add(rem)
            continue
        # both non-expert: remove lower precision (tie => q2)
        p1, p2 = prec_map.get(q1, 0), prec_map.get(q2, 0)
        rem = q1 if p1 < p2 else q2
        to_remove.add(rem)

    filtered_df = df[~df['Question'].isin(to_remove)].reset_index(drop=True)
    return filtered_df

def compute_question_matrix(df: pd.DataFrame) -> np.ndarray:
    """Extract only prediction columns (binary values) as integer matrix."""
    non_pred_cols = {'Question', 'Prec', '__set', 'Source_File'}
    pred_cols = [c for c in df.columns if c not in non_pred_cols]
    mat = df[pred_cols].fillna(0)
    return mat.astype(int).values


def run_permutation_evaluation(
    predictions_dir: Path,
    output_path: Path,
    model: str,
    suffix: str,
    metric_cols: set,
    m: int,
    B_null: int,
    alpha: float,
    excluded: set,
    pattern=None,
) -> pd.DataFrame:
    if output_path.exists():
        logger.info(f"Cached permutation test results already exist: {output_path}")
        return pd.read_csv(output_path)

    # pattern = f"predictions_val_set_*_question_model{suffix}_{model}.csv"
    # pattern = f"predictions_all_founders_set_*_question_model_{model}_{suffix}.csv"
    logger.info(f"Using glob pattern: {pattern}")
    logger.info(f"Search path: {predictions_dir}")
    predictions_files = sorted(glob.glob(str(predictions_dir / pattern)))

    if not predictions_files:
        logger.warning("❌ No prediction files matched the pattern.")
    else:
        logger.info(f"✅ Found {len(predictions_files)} prediction files.")

    results = []
    for fn in predictions_files:
        logger.info(f"Processing file: {fn}")
        df = pd.read_csv(fn)
        founder_cols = [c for c in df.columns if c not in metric_cols]

        # ground truth row
        success_row = df.loc[df['Question'] == 'Success', founder_cols]
        if success_row.empty:
            logger.warning(f"No Success row in {fn}")
            continue

        y_true = pd.to_numeric(success_row.iloc[0], errors='coerce').fillna(0).astype(int).values[:m]
        total_pos = y_true.sum()

        for q in df['Question'].unique():
            if q in excluded:
                continue
            y_pred = pd.to_numeric(df.loc[df['Question'] == q, founder_cols].iloc[0],
                                   errors='coerce').fillna(0).astype(int).values[:m]
            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / total_pos if total_pos > 0 else 0.0
            f05 = (1.25 * prec * rec / (0.25 * prec + rec)) if (prec + rec) > 0 else 0.0
            n_pos = int(y_pred.sum())

            if n_pos == 0:
                p_val = 1.0
            else:
                null_prec = np.zeros(B_null)
                for b in range(B_null):
                    perm = np.zeros(m, dtype=int)
                    perm[np.random.choice(m, size=n_pos, replace=False)] = 1
                    null_prec[b] = precision_score(y_true, perm, zero_division=0)
                p_val = (1 + (null_prec >= prec).sum()) / (1 + B_null)

            results.append({
                'File': fn,
                'Question': q,
                'TP': tp,
                'FP': fp,
                'Precision': prec,
                'Recall': rec,
                'F0.5': f05,
                'n_preds': n_pos,
                'p_value': p_val,
                'significant': p_val < alpha
            })

    results_df = pd.DataFrame(results).sort_values(by="Precision", ascending=False)
    results_df.to_csv(output_path, index=False)
    logger.info(f"Saved permutation test results to: {output_path}")
    return results_df

def run_permutation_evaluation_f1(
    predictions_dir: Path,
    output_path: Path,
    model: str,
    suffix: str,
    metric_cols: set[str],
    m: int,
    B_null: int,
    alpha: float,
    excluded: set[str]
) -> pd.DataFrame:
    """
    Permutation test on F1 for each question CSV in `predictions_dir`.  
    Matches files like:
       predictions_test_trials_set_*_question_model_{model}*{suffix}.csv  
    Computes TP/FP/FN → precision, recall, F1, F0.5, then builds a null F1 distribution
    by drawing B_null random predictions of the same size.  Returns a DataFrame
    sorted by F1 descending (and ensures the column exists even if no files match).
    """
    # early load
    if output_path.exists():
        logger.info(f"Cached permutation test results already exist: {output_path}")
        return pd.read_csv(output_path)

    # new glob to match your clinical-trial outputs
    pattern = f"predictions_test_trials_set_*_question_model_{model}*{suffix}*.csv"
    files = sorted(glob.glob(str(predictions_dir / pattern)))
    if not files:
        logger.warning(f"No prediction files found with pattern {pattern} in {predictions_dir}")
    results = []

    for fn in files:
        logger.info(f"Processing file: {fn}")
        df = pd.read_csv(fn)
        founder_cols = [c for c in df.columns if c not in metric_cols]

        # ground truth
        success = df.loc[df['Question']=='Success', founder_cols]
        if success.empty:
            logger.warning(f"  → no ‘Success’ row in {fn}, skipping.")
            continue

        y_true = (
            pd.to_numeric(success.iloc[0], errors='coerce')
              .fillna(0).astype(int)
              .values[:m]
        )
        total_pos = int(y_true.sum())

        for q in df['Question'].unique():
            if q in excluded:
                continue

            row = df.loc[df['Question']==q, founder_cols]
            if row.shape[0] == 0:
                continue
            y_pred = (
                pd.to_numeric(row.iloc[0], errors='coerce')
                  .fillna(0).astype(int)
                  .values[:m]
            )

            tp = int(((y_pred==1)&(y_true==1)).sum())
            fp = int(((y_pred==1)&(y_true==0)).sum())
            fn = int(((y_pred==0)&(y_true==1)).sum())

            prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
            rec  = tp/total_pos if total_pos>0 else 0.0

            if prec+rec>0:
                f1  = 2*prec*rec/(prec+rec)
                f05 = (1+0.5**2)*prec*rec/((0.5**2)*prec+rec)
            else:
                f1 = f05 = 0.0

            # null distribution on F1
            n_pos = int(y_pred.sum())
            if n_pos == 0:
                p_val = 1.0
            else:
                null_f1 = np.zeros(B_null)
                for b in range(B_null):
                    perm = np.zeros(m, dtype=int)
                    perm[np.random.choice(m, size=n_pos, replace=False)] = 1
                    p0 = precision_score(y_true, perm, zero_division=0)
                    r0 = recall_score(   y_true, perm, zero_division=0)
                    null_f1[b] = 2*p0*r0/(p0+r0) if (p0+r0)>0 else 0.0
                p_val = (1 + (null_f1 >= f1).sum())/(1 + B_null)

            results.append({
                'File': fn,
                'Question': q,
                'TP': tp,
                'FP': fp,
                'FN': fn,
                'Precision': prec,
                'Recall': rec,
                'F1': f1,
                'F0.5': f05,
                'n_preds': n_pos,
                'p_value': p_val,
                'significant': p_val < alpha
            })

    # define all columns so even an empty df has them
    cols = ['File','Question','TP','FP','FN','Precision','Recall','F1','F0.5',
            'n_preds','p_value','significant']
    results_df = pd.DataFrame(results, columns=cols)

    # sort by F1 descending
    results_df = results_df.sort_values(by="F1", ascending=False)

    # save and return
    results_df.to_csv(output_path, index=False)
    logger.info(f"Saved permutation test results (F1) to: {output_path}")
    return results_df