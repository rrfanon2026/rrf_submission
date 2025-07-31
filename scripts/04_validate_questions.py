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
from src.utils.dataset_utils import load_dataset
from src.questions.testing import test_question, update_results_file, update_founder_index, update_success_proportion, test_question_anonymised

def parse_args():
    parser = argparse.ArgumentParser(description='Generate and test evaluation questions')
    parser.add_argument('--provider', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--question-set', type=str, default='0')
    parser.add_argument('--results_dir_extension', type=str, default='precomputed')
    parser.add_argument('--batch_size', type=int, default=20)
    return parser.parse_args()

def load_questions(args, project_root):
    questions_file = Path(project_root) / args.results_dir_extension / args.model.replace("-", "_") / f"deduplicated/generated_questions_set_dedup_{args.question_set}{args.suffix}.csv"
    if not questions_file.exists():
        raise FileNotFoundError(f"❌ Questions file not found: {questions_file}")
    return pd.read_csv(questions_file)['Question'].tolist()

def main():
    # Parse arguments
    args = parse_args()

    args.suffix = "_anonymised"

    output_dir = Path(project_root) / args.results_dir_extension / args.model.replace("-", "_") / "validation_predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    val_file = f"02_question_validation_data{args.suffix}.csv"

    data_path = Path(project_root) / args.results_dir_extension / val_file
    validation_data = load_dataset(data_path, "validation")

    questions = load_questions(args, project_root)

    # Set up the LLM client (always use gpt-4o-mini due to cost)
    llm_client_test = get_llm_client('openai', 'gpt-4o-mini', args.temperature)

    # Prepare prediction file path (across all validation founders)
    prediction_file = output_dir / f"predictions_val_set_{args.question_set}_question{args.suffix}.csv"
    
    # Loop through the questions (testing the first two questions for now)
    for idx, question in enumerate(questions, start=1):
        # start timer for this question
        start = time.perf_counter()

        combined_predictions = []
        # Loop through validation data in batches of founders
        num_batches = len(validation_data) // args.batch_size

        # Loop through batches
        for i in range(num_batches):
        # for i in range(2):
            batch_start = i * args.batch_size
            batch_end = batch_start + args.batch_size
            batch = validation_data.iloc[batch_start:batch_end]

            # Test the question on the current batch
            preds = test_question_anonymised(llm_client_test, question, batch)
            combined_predictions.extend(preds)

        # Update the common prediction file with results for this question
        update_results_file(
            results_file=prediction_file,
            question=question,
            responses=combined_predictions,
            test_founders=validation_data,  # Using the full version with 'startup_success'
            logger=logger,
            dataset_assignments={founder['founder_uuid']: 'Validation' for _, founder in validation_data.iterrows()}
        )

        # Add indices of founders
        update_founder_index(prediction_file)

        # Add how many questions a founder was predicted successful for
        update_success_proportion(prediction_file)

        # end timer and log
        elapsed = time.perf_counter() - start
        logger.info(f"⏱️ Question {idx}/{len(questions)} ‘{question[:50]}…’ took {elapsed:.2f}s")

    logger.info("Testing questions on validation dataset completed.")


if __name__ == "__main__":
    main()
