import pandas as pd
import logging
from pathlib import Path
from src.llms.prompts import get_standard_question_prompt, get_standard_updated_prompt
from src.questions.generation_utils import batch_dataframe, try_llm_request
    
def generate_questions(llm_client, founders_df, num_questions=10, batch_size=20):
    """Generate questions using the LLM client by processing founders in batches."""
    questions = []

    # Only process one batch of 20 founders (which is 1 set in our case)
    founder_batches = batch_dataframe(founders_df, batch_size)[:1]

    for batch_idx, batch in enumerate(founder_batches):
        logging.info(f"Processing batch {batch_idx + 1} of {len(founder_batches)}")

        # Determine whether feature engineering was used
        feature_engineered = 'professional_background' in founders_df.columns

        def build_summary(row):
            summary = [
                f"Founder {row['founder_uuid']}:",
                f"Education: {row['university_degrees']}",
                f"Work History: {row['work_history']}",
                f"Previous Companies: {row['previous_companies_founded']}"
            ]

            if feature_engineered:
                if 'professional_background' in row:
                    summary.append(f"Professional Background: {row['professional_background']}")
                if 'company_location' in row:
                    summary.append(f"Company Location: {row['company_location']}")
            else:
                raw_cols = [
                    'org_state', 'org_city', 'press_media_coverage_count',
                    'personal_branding', 'vc_experience', 'angel_experience',
                    'quant_experience', 'board_advisor_roles',
                    'tier_1_vc_experience', 'worked_at_bank'
                ]
                for col in raw_cols:
                    if col in row:
                        summary.append(f"{col.replace('_', ' ').title()}: {row[col]}")

            return " ".join(summary)

        founder_summaries = "\n\n".join(batch.apply(build_summary, axis=1))

        system_message, user_prompt = get_standard_question_prompt(num_questions, founder_summaries)

        batch_questions = try_llm_request(llm_client, system_message, user_prompt, logging, num_questions)
        questions.extend(batch_questions)

    return questions

def save_questions(questions, output_file):
    """Save generated questions to a CSV file, appending if file already exists."""
    # Convert string path to Path object if it's not already
    output_file = Path(output_file)

    questions_df = pd.DataFrame({"Question": [q.strip() for q in questions if q.strip()]})
    
    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if output_file.exists():
        questions_df.to_csv(output_file, mode='a', header=False, index=False, quoting=1)
    else:
        questions_df.to_csv(output_file, index=False, quoting=1)
    
    logging.info(f"Questions saved to {output_file}")


def generate_updated_question_set(founder_data: pd.DataFrame, llm_client, logger: logging.Logger, num_questions=10) -> list:
    """
    Generates a new set of objective YES/NO questions based on provided *new* founder data.
    This function explicitly informs the LLM about the new founder data as context to
    inspire the generation of fresh, relevant questions for the current cycle.

    Args:
        founder_data: A DataFrame containing the *new* founder information for this cycle.
        llm_client: An initialized LLM client object (e.g., OpenAI client).
        logger: A logger instance for logging messages.

    Returns:
        A list of generated YES/NO questions.
    """
    founder_summaries_str = "\n\n".join(founder_data.apply(lambda row: 
        f"Founder {row['founder_uuid']}: "
        f"Education: {row['university_degrees']}, "
        f"Work History: {row.get('work_history', 'N/A')}, "
        f"Previous Companies: {row.get('previous_companies_founded', 'N/A')}, "
        f"Professional Background: {row.get('professional_background', 'N/A')}, "
        f"Company Location: {row.get('company_location', 'N/A')}", axis=1))

    system_msg, user_prompt = get_standard_updated_prompt(num_questions, founder_summaries_str)
    return try_llm_request(llm_client, system_msg, user_prompt, logger, num_questions)