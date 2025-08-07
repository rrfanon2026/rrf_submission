import pandas as pd
import logging
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


from src.questions.similarity_utils import (
    load_and_merge_predictions,
    filter_low_precision_questions,
    clean_question_text,
    deduplicate_exact_questions,
    add_success_row,
    remove_jaccard_similar_questions,
    remove_hamming_similar_questions,
    remove_colinear_questions_via_clustering,
)

# Define special questions (since it was a constant in your script)
SPECIAL_QUESTIONS = {'Founder Index', 'Dataset', 'SUCCESS_PROPORTION', 'Success'}

# Set up logging
logger = logging.getLogger(__name__)

def filter_and_save_questions(predictions_dir: Path, predictions_files: list[str], suffix: str, args):
    combined_df = load_and_merge_predictions(predictions_dir, predictions_files, SPECIAL_QUESTIONS)
    filtered_df = filter_low_precision_questions(combined_df, SPECIAL_QUESTIONS, args.precision_cutoff)
    filtered_df = clean_question_text(filtered_df)
    filtered_df = deduplicate_exact_questions(filtered_df)
    filtered_df = remove_jaccard_similar_questions(filtered_df, threshold=args.jaccard_threshold)

    filtered_df['Prec'] = pd.to_numeric(filtered_df['Prec'], errors='coerce')
    filtered_df = filtered_df.sort_values(by='Prec', ascending=False, na_position='last')

    sample_file = predictions_dir / f"predictions_test_set_1{suffix}.csv"
    final_df = add_success_row(filtered_df, sample_file)

    out_file = predictions_dir / f"filtered_combined_questions_with_success_{args.mode.replace('+', '_')}{suffix}_TEST.csv"
    final_df.to_csv(out_file, index=False)
    logger.info(f"✅ Saved filtered questions to: {out_file}")



def filter_questions(predictions_dir: Path, predictions_files: list[str], suffix: str, args, sort_by: str = "precision") -> pd.DataFrame:
    precision_cutoff = 0.02001
    
    combined_df = load_and_merge_predictions(predictions_dir, predictions_files, SPECIAL_QUESTIONS)
    
    # Step 1: Filter out low-precision questions early
    filtered_df = filter_low_precision_questions(combined_df, SPECIAL_QUESTIONS, precision_cutoff)

    # Step 2: Cap LLM questions *before* removing duplicates/similarity
    if args.mode == "llm_expert":
        original_total = filtered_df[filtered_df['Source'] == 'llm'].shape[0]
        filtered_df = cap_llm_questions_to_match_expert_count(filtered_df, original_total, sort_by=sort_by)

    # Step 3: Clean text and deduplicate
    filtered_df = clean_question_text(filtered_df)
    filtered_df = deduplicate_exact_questions(filtered_df)

    # Step 4: Filter similar questions
    if args.similarity_metric == 'jaccard':
        filtered_df = remove_jaccard_similar_questions(filtered_df, threshold=args.similarity_threshold)
    elif args.similarity_metric == 'hamming':
        filtered_df = remove_hamming_similar_questions(filtered_df, threshold=args.similarity_threshold)
    elif args.similarity_metric == 'cosine-cluster':
        filtered_df, _ = remove_colinear_questions_via_clustering(filtered_df, threshold=args.similarity_threshold, method="cosine")
    else:
        raise ValueError(f"Unknown similarity metric: {args.similarity_metric}")    

    # Final sort by selected metric
    sort_column = 'Prec' if sort_by == "precision" else 'F0.5'
    filtered_df[sort_column] = pd.to_numeric(filtered_df[sort_column], errors='coerce')
    filtered_df = filtered_df.sort_values(by=sort_column, ascending=False, na_position='last')

    # Final sort by precision
    # filtered_df['Prec'] = pd.to_numeric(filtered_df['Prec'], errors='coerce')
    # filtered_df = filtered_df.sort_values(by='Prec', ascending=False, na_position='last')

    sample_file = predictions_dir / f"predictions_test_set_1{suffix}.csv"
    final_df = add_success_row(filtered_df, sample_file)
    return final_df

def cap_llm_questions_to_match_expert_count(df: pd.DataFrame, original_total: int, sort_by: str = "precision") -> pd.DataFrame:
    """Trims low-precision LLM questions to keep total count fixed after adding expert questions."""
    # Use the 'Source' column instead of matching text
    expert_df = df[df['Source'] == 'expert']
    llm_df = df[df['Source'] == 'llm']

    # Cap LLM questions to retain the original total size
    keep_n_llm = max(original_total - len(expert_df), 0)
    llm_df_capped = llm_df.head(keep_n_llm)

    print(f"✂️ Trimmed LLM questions to {len(llm_df_capped)} to accommodate {len(expert_df)} expert questions.")

    combined_df = pd.concat([llm_df_capped, expert_df], ignore_index=True)
    # return combined_df.sort_values(by='Prec', ascending=False, na_position='last').reset_index(drop=True)

    sort_column = 'Prec' if sort_by == "precision" else 'F0.5'
    combined_df[sort_column] = pd.to_numeric(combined_df[sort_column], errors='coerce')
    return combined_df.sort_values(by=sort_column, ascending=False, na_position='last').reset_index(drop=True)

def construct_predictions_file_list(args, suffix):
    base = f"predictions_test_set_{{}}{suffix}.csv"
    args.mode = args.mode.lstrip('_')
    
    if args.mode == "llm":
        files = [base.format(i) for i in range(7)]
        # files = [base.format(i) for i in range(7) if i != 3]
    elif args.mode == "llm_expert":
        files = [base.format(i) for i in range(7)] + [f"predictions_test_set_{i}{suffix}_EXPERT.csv" for i in [7, 8]]
    elif args.mode == "expert_only":
        files = [f"predictions_test_set_{i}{suffix}_EXPERT.csv" for i in [7, 8]]
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    
    logger.info(f"📄 Found {len(files)} prediction files for mode: {args.mode}")
    print(f"📄 Found {len(files)} prediction files for mode: {args.mode}")
    return files