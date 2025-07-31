# src/llms/prompts.py

# ----------------------------
# FULL prompt templates per use-case
# ----------------------------

# src/prompts/question_generation.py

def get_standard_question_prompt(num_questions, founder_summaries):
    system_message = (
        "You are a VC analyst specializing in evaluating founders. "
        "Your task is to generate high-quality, objective YES/NO questions based on provided founder data. "
        "These questions should be designed to evaluate a founder's education, work history, and previous companies."
    )

    user_prompt = f"""Generate {num_questions} YES/NO questions that can be answered using founder data.
Each question should be objective and answerable using available information like education, work history, and previous companies.

IMPORTANT FORMATTING REQUIREMENTS:
1. Return ONLY the questions, one per line
2. Do NOT include any introductory text or explanations
3. Do NOT include any numbering or bullet points
4. Each line should contain exactly one question that can be answered with Yes or No

Example format (DO NOT USE THESE QUESTIONS):
Has the founder previously founded a successful company?

Founder Summaries:
{founder_summaries}"""

    return system_message, user_prompt

def get_standard_updated_prompt(num_questions: int, founder_summaries: str) -> tuple[str, str]:
    system_msg = (
        "You are a VC analyst specializing in evaluating founders. "
        "Your task is to generate high-quality, objective YES/NO questions based on provided founder data. "
        "These questions should be designed to evaluate a founder's education, work history, and previous companies. "
        "Each question must be answerable with 'Yes' or 'No' and directly verifiable from the provided data."
    )

    user_prompt = f"""We are now evaluating a new batch of founders. Please generate {num_questions} new objective YES/NO questions specifically inspired by the characteristics and information available in this *new set of founders*.
Each question should be based on relevant details found in founders' education, work history, previous companies, professional background, and company location.

IMPORTANT FORMATTING REQUIREMENTS:
1. Return ONLY the questions, one per line.
2. Do NOT include any introductory text, explanations, or conversational filler.
3. Do NOT include any numbering or bullet points.
4. Each line must contain exactly one question that can be answered with 'Yes' or 'No'.
5. Ensure the questions are distinct from common knowledge and require specific founder data for an accurate answer.

Example format (DO NOT USE THESE QUESTIONS):
Is the founder a graduate of an Ivy League university?
Has the founder held a leadership position in a tech company?

Your new questions based on this batch:

{founder_summaries}"""
    
    return system_msg, user_prompt

def get_anonymised_question_prompt(num_questions: int, founder_summaries: str) -> tuple[str, str]:
    system_message = (
        "You are a VC analyst specializing in evaluating founders. "
        "Your task is to generate high-quality, objective YES/NO questions based on provided founder summaries. "
        "These questions should evaluate factors such as education, work history, leadership, and relevant experience."
    )

    user_prompt = f"""Generate {num_questions} YES/NO questions that can be answered using the anonymised founder summaries below.
Each question should be objective and specific, relating to observable traits in the summaries (e.g., education, roles, founding experience).
IMPORTANT FORMATTING REQUIREMENTS:
1. Return ONLY the questions, one per line
2. Do NOT include any introductory text or explanations
3. Do NOT include any numbering or bullet points
4. Each line should contain exactly one question that can be answered with Yes or No

Example format (DO NOT USE THIS QUESTION):
Has the founder previously held a senior leadership role?

Founder Summaries:

{founder_summaries}"""
    
    return system_message, user_prompt

def get_anonymised_updated_question_prompt(num_questions: int, founder_summaries_str: str) -> tuple[str, str]:
    system_msg = (
        "You are a VC analyst specializing in evaluating startup founders. "
        "Your task is to generate objective YES/NO questions based on the anonymised founder summaries below. "
        "These questions should focus on traits like education, leadership, experience, and founding history. "
        "Each question must be answerable with 'Yes' or 'No' based solely on the summaries provided."
    )

    user_prompt = f"""We are now evaluating a new batch of anonymised founders. Please generate {num_questions} new objective YES/NO questions inspired by this batch.
    Each question must be specific, verifiable from the data, and helpful for distinguishing successful from unsuccessful founders.

    IMPORTANT FORMATTING REQUIREMENTS:
    1. Return ONLY the questions, one per line.
    2. Do NOT include any introductory text, explanations, or conversational filler.
    3. Do NOT include any numbering or bullet points.
    4. Each line must contain exactly one question that can be answered with 'Yes' or 'No'.

    Example format (DO NOT USE THESE QUESTIONS):
    Has the founder held a leadership role in a technology company?
    Did the founder complete postgraduate education?

    New Founder Summaries:\n\n{founder_summaries_str}"""

    return system_msg, user_prompt