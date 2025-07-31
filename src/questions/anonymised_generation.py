import logging
import pandas as pd

from src.llms.prompts import get_anonymised_question_prompt, get_anonymised_updated_question_prompt
from src.questions.generation_utils import batch_dataframe, try_llm_request

def generate_questions_anonymised(llm_client, founders_df, num_questions=10, batch_size=20):
    """Generate YES/NO questions using anonymised founder summaries."""
    questions = []

    founder_batches = batch_dataframe(founders_df, batch_size)[:1]

    for batch_idx, batch in enumerate(founder_batches):
        logging.info(f"Processing batch {batch_idx + 1} of {len(founder_batches)}")

        founder_summaries = "\n\n".join(
            [f"Founder {row['founder_uuid']}:\n{row['summary']}" for _, row in batch.iterrows()]
        )

        system_msg, user_prompt = get_anonymised_question_prompt(num_questions, founder_summaries)

        batch_questions = try_llm_request(llm_client, system_msg, user_prompt, logging, num_questions)
        questions.extend(batch_questions)

    return questions

def generate_updated_question_set_anonymised(
    founder_data: pd.DataFrame,
    llm_client,
    logger: logging.Logger,
    num_questions=10
) -> list:
    """
    Generates a new set of objective YES/NO questions based on anonymised founder summaries.

    Args:
        founder_data: DataFrame with anonymised founder summaries.
        llm_client: Initialized LLM client.
        logger: Logger instance.

    Returns:
        A list of generated YES/NO questions.
    """
    founder_summaries_str = "\n\n".join(
        [f"Founder {row['founder_uuid']}:\n{row['summary']}" for _, row in founder_data.iterrows()]
    )

    system_msg, user_prompt = get_anonymised_updated_question_prompt(num_questions, founder_summaries_str)
    return try_llm_request(llm_client, system_msg, user_prompt, logger, num_questions)