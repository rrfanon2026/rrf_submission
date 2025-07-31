import logging

def batch_dataframe(df, batch_size):
    return [df.iloc[i:i + batch_size] for i in range(0, len(df), batch_size)]

def clean_llm_response(response):
    lines = response.strip().split('\n')
    return [
        line.strip() for line in lines
        if line.strip() and '?' in line and
        not line.strip().startswith(('```', '1.', '2.', '3.', '-', '*')) and
        not line.strip().lower().startswith(('here are', 'based on', 'questions:', 'new questions:'))
    ]

def try_llm_request(llm_client, system_msg, user_prompt, logger=None, num_questions=10):
    try:
        response = llm_client.send_prompt(system_msg, user_prompt)
        return clean_llm_response(response)[:num_questions]
    except Exception as e:
        if logger:
            logger.error(f"LLM error: {e}")
        else:
            logging.error(f"LLM error: {e}")
        return []