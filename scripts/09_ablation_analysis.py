import argparse
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from src.utils.scoring_utils import (compute_weights,select_best_hyperparams,evaluate_test_fold)
from src.questions.question_filtering import filter_questions, construct_predictions_file_list

# -----------------------------------------------------------------------------
# Argument Parser
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Run nested CV with precision-based scoring.")
parser.add_argument('--results_dir_extension', type=str, default='precomputed')
parser.add_argument('--mode', type=str, choices=['llm', 'llm_expert', 'expert_only'], required=True, help="Which question mode to load")
parser.add_argument('--similarity-metric', type=str, choices=['jaccard', 'hamming', 'cosine-cluster'], required=True,help='Similarity metric used to remove redundant questions')
parser.add_argument('--similarity-threshold', type=float, required=True,help='Threshold for filtering similar questions (used with the specified similarity metric)')
parser.add_argument('--optimise-for', type=str, choices=['precision', 'f0.5', 'f1', 'f2'], default='f0.5', help="Metric to optimise: precision, f0.5, f1, or f2")
parser.add_argument('--sort-by', type=str, choices=['precision', 'f0.5'], default='precision', help="Sort final questions by 'precision' or 'f0.5'")

# Parse arguments
args = parser.parse_args()
n_splits     = 10
n_repeats    = 10
weighting    = 0
exponents    = [float(weighting)]

# -----------------------------------------------------------------------------
# Save Results
# -----------------------------------------------------------------------------
# build suffix for filenames
suffix = f"_anonymised"
similarity_str  = f"{args.similarity_threshold:.2f}".replace('.', '_')
predictions_dir = project_root / args.results_dir_extension / "gpt_4o_mini/test_predictions/"
opt_str = f"optimise{args.optimise_for.replace('.', '_').upper()}"
sort_str = f"sortby{args.sort_by.replace('.', '_')}"

# Set file path names
out_dir =  project_root / args.results_dir_extension / "gpt_4o_mini/ablation_results"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / (
    f"nested_cv_{args.mode}_{suffix}"
    f"similarity_{args.similarity_metric}_{similarity_str}_"
    f"{opt_str}_{sort_str}.csv"
)

# Check if output file already exists
print(f"Outpath is {out_path}")
if out_path.exists():
    print(f"🛑 Skipping analysis — results already exist at:\n{out_path}")
    sys.exit(0)

# -----------------------------------------------------------------------------
# Load and Prepare Data
# -----------------------------------------------------------------------------
predictions_files = construct_predictions_file_list(args, suffix)
df = filter_questions(predictions_dir, predictions_files, suffix, args, sort_by=args.sort_by)
special_rows = df[df['Question'].isin(['Founder Index','Dataset','Success'])]
filtered_df = df[~df['Question'].isin(['Founder Index','Dataset','Success'])].copy()

metric_cols  = ['Index','Question','Pass Rate','Prec','TP','FP','TN','FN','Rec','F1','F0.5','Prec_Train','Prec_Validation','Prec_Test','Prec_Mean']
founder_cols = [c for c in filtered_df.columns if c not in metric_cols]
filtered_df[founder_cols] = (filtered_df[founder_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int))

X = filtered_df[founder_cols].values
Q, F = X.shape

success_series = special_rows[special_rows['Question']=='Success'][founder_cols].iloc[0]
success_values = pd.to_numeric(success_series, errors='coerce').fillna(0).astype(int).values

# -----------------------------------------------------------------------------
# Nested CV With Repeats
# -----------------------------------------------------------------------------
results = []
founder_indices = np.arange(F)
founder_rows = []
score_thresholds = list(range(1, 71))

for repeat in range(1, n_repeats + 1):
    print(f"\n🔁 Repeat {repeat}/{n_repeats}")
    outer_skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42 + repeat)

    for outer_idx, (train_idx, test_idx) in enumerate(
        outer_skf.split(founder_indices, success_values), start=1):

        print(f"  🧪 Fold {outer_idx}/{n_splits}")
        train_ids = founder_indices[train_idx]
        test_ids  = founder_indices[test_idx]
        y_train   = success_values[train_ids]
        y_test    = success_values[test_ids]
        X_train = X[:, train_ids]

        best_combo, prec_array, ratio_array, best_mean_f05 = select_best_hyperparams(X_train, y_train, train_ids, weighting, exponents, score_thresholds, Q,success_values, n_splits, outer_idx + repeat, optimise_for=args.optimise_for)

        exp_opt, nq_opt, t_opt = best_combo
        print(f"    ✔ best(inner): exp={exp_opt}, n_q={nq_opt}, t={t_opt}, F0.5={best_mean_f05:.3f}")

        W_full, _ = compute_weights(weighting, exp_opt, prec_array, ratio_array)
        preds_test, yt, tp, fp, tn, fn, p_out, r_out, f05_out, f05_out_sk, f1_out, f2_out, mcc_out = evaluate_test_fold(X, W_full, nq_opt, t_opt, test_ids, success_values)

        # -- save per-founder predictions -----------------
        founder_rows.append(pd.DataFrame({
            'founder_id': test_ids,
            'repeat'    : repeat,
            'fold'      : outer_idx,
            'y_true'    : y_test,
            'y_rrf'     : preds_test
        }))

        print(f"    ✔ outer scores: prec={round(p_out, 3)}, rec={round(r_out, 3)}, F0.5={f05_out:.3f}")

        results.append({
            "Repeat": repeat,
            "Outer_Fold": outer_idx,
            "Best_Exp": exp_opt,
            "Best_n_q": nq_opt,
            "Best_Thr": t_opt,
            "Precision_Outer": round(p_out, 3),
            "Recall_Outer": round(r_out, 3),
            "F05_Outer": round(f05_out, 3),
            "TP": int(tp), "FP": int(fp),
            "TN": int(tn), "FN": int(fn),
            "Pred_Pos": int(preds_test.sum()),
            "Actual_Pos": int(yt.sum()),
            "f05_out_sk": round(f05_out_sk, 3),
            "f1_out": round(f1_out, 3),
            "f2_out": round(f2_out, 3),
            "mcc_out": round(mcc_out, 3),
        })


# ---- write fold-metrics ----
out_df = pd.DataFrame(results)
out_df.to_csv(out_path, index=False)
print(f"\n✅ Saved repeated results to {out_path}")
