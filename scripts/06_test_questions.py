import argparse
import pandas as pd
import logging
from pathlib import Path
import sys
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Add your imports for LLM client and utility functions
from src.llms.utils import get_llm_client
from src.utils.file_utils import setup_logging
from src.utils.dataset_utils import load_dataset
from src.questions.testing import test_question, update_results_file, update_founder_index, update_success_proportion, load_selected_questions, test_question_anonymised
from src.questions.vanilla_testing import update_vanilla_results, test_vanilla_predictions_anonymised, test_vanilla_predictions,test_vanilla_predictions_few_shot, test_vanilla_predictions_few_shot_anonymised

def parse_args():
    parser = argparse.ArgumentParser(description='Generate and test evaluation questions')
    parser.add_argument('--provider', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--results_dir_extension', type=str, default='precomputed')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--question_set', type=int, default=7, help='Set number of questions to test')
    parser.add_argument('--mode', choices=['questions', 'vanilla', 'vanilla_few_shot'], default='questions', help="‘questions’ runs test_question over N selected prompts; ‘vanilla’ runs one pass of the vanilla baseline.")
    parser.add_argument('--rep', type=int, default=1)
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()

    # Setup logging
    logger = setup_logging("question_testing")

    suffix = '_anonymised'


    base_dir = f"{project_root}/{args.results_dir_extension}"
    train_data_path = f"{base_dir}/01_question_training_data{suffix}.csv"
    test_data_path  = f"{base_dir}/03_full_cross_validation_test_data{suffix}.csv"

    
    test_data = load_dataset(test_data_path, "test")
    train_data = load_dataset(train_data_path, "train")
    num_batches = len(test_data) // args.batch_size # Note number of bathces

    # Set up the LLM client
    llm_client = get_llm_client(args.provider, args.model, temperature=args.temperature)

    if args.mode == 'questions':

        # Load the selected questions
        questions_file = f"{project_root}/{args.results_dir_extension}/{args.model.replace('-', '_')}/test_questions/selected_questions_set_{args.question_set}{suffix}.csv"

        # If running expert questions, use this
        # questions_file = f"{project_root}/{args.results_dir_extension}/{args.model.replace('-', '_')}/test_questions/selected_questions_set_{args.question_set}{suffix}_EXPERT.csv"


        print(f"Question file: {questions_file}")
        questions = load_selected_questions(questions_file)
        
        output_dir = Path(f"{project_root}/{args.results_dir_extension}/{args.model.replace('-', '_')}/test_predictions")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare the prediction file path
        prediction_file = f"{output_dir}/predictions_test_set_{args.question_set}{suffix}.csv"

        # If running expert questions
        # prediction_file = f"{output_dir}/predictions_test_set_{args.question_set}{suffix}_EXPERT.csv"

        if Path(prediction_file).exists():
            df_existing_preds = pd.read_csv(prediction_file)
            answered_questions = set(df_existing_preds['Question'].unique())
            logger.info(f"Found {len(answered_questions)} questions already run.")
        else:
            df_existing_preds = pd.DataFrame()
            answered_questions = set()

        for idx, question in enumerate(questions, start=1):
            if question in answered_questions:
                logger.info(f"⏭️ Skipping question {idx}/{len(questions)} – already processed.")
                continue
            # start timer for this question
            start = time.perf_counter()

            combined_predictions = []

            # Loop through batches
            for i in range(num_batches):
                batch_start = i * args.batch_size
                batch_end = batch_start + args.batch_size
                batch = test_data.iloc[batch_start:batch_end]

                # Test the question on the current batch
                preds = test_question_anonymised(llm_client, question, batch)

                combined_predictions.extend(preds)

            # Update the common prediction file with results for this question
            update_results_file(
                results_file=prediction_file,
                question=question,
                responses=combined_predictions,
                test_founders=test_data,  # Using the full version with 'startup_success'
                logger=logger,
                dataset_assignments={founder['founder_uuid']: 'Validation' for _, founder in test_data.iterrows()}
            )

            # Add indices of founders
            update_founder_index(prediction_file)

            # Add how many questions a founder was predicted successful for
            update_success_proportion(prediction_file)

            # end timer and log
            elapsed = time.perf_counter() - start
            logger.info(f"⏱️ Question {idx}/{len(questions)} ‘{question[:50]}…’ took {elapsed:.2f}s")

        logger.info("Testing questions on validation dataset completed.")
    elif args.mode in ['vanilla', 'vanilla_few_shot']:

        logger.info(f"Beginning vanilla loop with {len(test_data)} founders in {num_batches} batches.")
        start = time.perf_counter()

        # Prepare paths
        results_dir = Path(project_root) / args.results_dir_extension / args.model.replace('-', '_') / "vanilla"
        if args.mode == "vanilla_few_shot":
            prefix = "vanilla_few_shot_"
        else:  # fallback to plain vanilla
            prefix = "vanilla_zero_shot_"
        
        summary_file = results_dir / f"{prefix}{args.model.replace('-', '_')}{suffix}_rep_{args.rep}.csv"

        # Precompute assignments dict once
        dataset_assignments = {
            row['founder_uuid']: 'Validation'
            for _, row in test_data.iterrows()
        }

        combined_predictions = []

        for i in range(num_batches):
        # for i in range(2): # testing
            logger.info(f"Running vanilla batch {i+1}/{num_batches}…")
            batch_start = i * args.batch_size
            batch_end   = batch_start + args.batch_size
            batch       = test_data.iloc[batch_start:batch_end]

            if args.mode == 'vanilla_few_shot':
                preds = test_vanilla_predictions_few_shot_anonymised(llm_client, batch, train_data)
            elif args.mode == 'vanilla':
                preds = test_vanilla_predictions_anonymised(llm_client, batch)
            else:
                raise ValueError(f"Unexpected mode: {args.mode}")
                

            # 3) incrementally update the summary using _just_ this batch
            update_vanilla_results(
                results_file=summary_file,
                predictions=preds,
                test_founders=test_data,
                dataset_assignments=dataset_assignments,
                logger=logger
            )
        logger.info(f"  ✓ Updated summary with batch {i+1}")

        elapsed = time.perf_counter() - start
        logger.info(f"⏱️ Vanilla baseline complete in {elapsed:.2f}s; summary at {summary_file}")

if __name__ == "__main__":
    main()