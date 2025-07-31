import logging
import json
import pandas as pd
from pathlib import Path
import numpy as np


def test_vanilla_predictions(llm_client, founders_df):
    """
    Generate vanilla GPT predictions for founder success based solely on their profile data.

    Returns a list of dicts:
    [
        {"Founder ID": "<uuid1>", "Prediction": "Successful"},
        {"Founder ID": "<uuid2>", "Prediction": "Unsuccessful"},
        ...
    ]
    """

    # 1. Drop PII and ground-truth
    columns_to_drop = ['startup_success', 'founder_name']
    founders_df = founders_df.drop(columns=columns_to_drop, errors='ignore')


    # 2. Build the system prompt from GPTree baseline
    system_message = (
        "You are an expert in venture capital tasked with distinguishing successful founders "
        "from unsuccessful ones. All founders under consideration are sourced from LinkedIn profiles "
        "of companies that have raised between $100K and $4M in funding. A successful founder is defined "
        "as one whose company has achieved either an exit or IPO valued at over $500M or raised "
        "more than $500M in funding. **Do NOT use the internet, external databases, or real-world " 
        "knowledge about the companies’ actual outcomes; rely exclusively on the information provided below.** "
    )

    # 3. Format each founder’s data as a short summary
    summaries = []
    for _, row in founders_df.iterrows():
        summary = (
            f"Founder {row['founder_uuid']}: "
            f"Education: {row.get('university_degrees', 'N/A')}; "
            f"Work History: {row.get('work_history', 'N/A')}; "
            f"Previous Companies Founded: {row.get('previous_companies_founded', 'N/A')}"
        )
        summaries.append(summary)
    founder_summaries = "\n\n".join(summaries)

    # 4. Create the user prompt
    user_prompt = f"""
    Based on the summaries below, classify each founder as "Successful" or "Unsuccessful".

    **Founder Summaries:**
    {founder_summaries}

    **Return your answer in JSON array form like:**
    ```json
    [
        {{"Founder ID": "uuid1", "Prediction": "Successful"}},
        {{"Founder ID": "uuid2", "Prediction": "Unsuccessful"}}
    ]
    ```
    """

    # 5. Send to LLM and parse
    try:
        raw = llm_client.send_prompt(system_message, user_prompt)
        logging.debug(f"LLM raw response: {raw}")
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        preds = json.loads(cleaned)
        if isinstance(preds, list):
            return preds
        else:
            logging.error("Unexpected JSON structure from LLM")
            return []
    except json.JSONDecodeError:
        logging.error(f"JSON decode failed on LLM response: {raw}")
        return []
    except Exception as e:
        logging.error(f"Error calling LLM: {e}")
        return []

def update_vanilla_results(
    results_file: Path,
    predictions: list,
    test_founders: pd.DataFrame,
    dataset_assignments: dict,
    logger: logging.Logger,
):
    """
    Incrementally append a batch of vanilla predictions onto
    a single 'Vanilla' row, re-computing metrics each time.
    """
    results_file = Path(results_file)
    metric_cols = [
        'Pass Rate','Prec','TP','FP','TN','FN','Rec','F1','F0.5',
        'Prec_Train','Prec_Validation','Prec_Test','Prec_Mean'
    ]

    # 1) Load or initialize the DataFrame
    if results_file.exists():
        df = pd.read_csv(results_file)
    else:
        df = pd.DataFrame(columns=['Index','Question'] + metric_cols)
        # Seed Dataset and Success rows
        base = [{'Question':'Dataset'}, {'Question':'Success'}, {'Question':'SUCCESS_PROPORTION'}]
        df = pd.concat([df, pd.DataFrame(base)], ignore_index=True)
        # Create blank Vanilla row
        vanilla = {'Question':'Vanilla', 'Index': 1}
        for m in metric_cols:
            vanilla[m] = 0
        df = pd.concat([df, pd.DataFrame([vanilla])], ignore_index=True)

    # 2) Ensure we have exactly one Vanilla row
    assert df['Question'].tolist().count('Vanilla') == 1, "Expected exactly one Vanilla row"

    # 3) Add any new founder-columns
    for p in predictions:
        short = p['Founder ID'][:5]
        if short not in df.columns:
            # append column at end
            df[short] = ''
            # fill Dataset and Success rows
            df.loc[df['Question']=='Dataset', short] = dataset_assignments.get(p['Founder ID'], 'Unknown')
            founder_match = test_founders.loc[test_founders['founder_uuid'] == p['Founder ID'], 'startup_success']
            if founder_match.empty:
                logger.warning(f"⚠️ No match found in test_founders for Founder ID: {p['Founder ID']}")
                continue  # optionally skip this prediction
            actual = int(founder_match.iloc[0])
            df.loc[df['Question']=='Success', short] = actual

    # 4) Write the batch’s predictions into that row
    vanilla_idx = df.index[df['Question']=='Vanilla'][0]
    for p in predictions:
        fid, pred_str = p['Founder ID'], p['Prediction']
        short = fid[:5]
        df.at[vanilla_idx, short] = 1 if pred_str=='Successful' else 0

    # 5) Recompute metrics across all filled founder-columns
    #    Gather all shorts
    shorts = [c for c in df.columns
              if c not in ['Index','Question'] + metric_cols]

    #    Collect true & pred arrays
    preds = df.loc[vanilla_idx, shorts].astype(int).values
    # trues = df.loc[df['Question']=='Success', shorts].astype(int).values.flatten()
    success_row = df.loc[df['Question'] == 'Success', shorts]
    success_row_clean = success_row.replace('', np.nan).astype(float).fillna(0).astype(int)
    trues = success_row_clean.values.flatten()
    splits = df.loc[df['Question']=='Dataset', shorts].values.flatten()

    tp = ((preds==1) & (trues==1)).sum()
    fp = ((preds==1) & (trues==0)).sum()
    tn = ((preds==0) & (trues==0)).sum()
    fn = ((preds==0) & (trues==1)).sum()

    precision = tp/(tp+fp) if tp+fp>0 else 0
    recall    = tp/(tp+fn) if tp+fn>0 else 0
    f1        = 2*precision*recall/(precision+recall) if precision+recall>0 else 0
    f05       = 1.25*precision*recall/(0.25*precision+recall) if precision+recall>0 else 0

    # per‐split precision
    split_metrics = {}
    for split in set(splits):
        mask = (splits==split)
        tp_s = ((preds==1)&(trues==1)&mask).sum()
        fp_s = ((preds==1)&(trues==0)&mask).sum()
        split_metrics[split] = tp_s/(tp_s+fp_s) if tp_s+fp_s>0 else 0

    prec_mean = sum(split_metrics.values())/len(split_metrics)

    # 6) Update the Vanilla row’s metric columns
    df.at[vanilla_idx, 'TP']   = tp
    df.at[vanilla_idx, 'FP']   = fp
    df.at[vanilla_idx, 'TN']   = tn
    df.at[vanilla_idx, 'FN']   = fn
    df.at[vanilla_idx, 'Pass Rate'] = tp + fp
    df.at[vanilla_idx, 'Prec'] = round(precision, 3)
    df.at[vanilla_idx, 'Rec']  = round(recall, 3)
    df.at[vanilla_idx, 'F1']   = round(f1, 3)
    df.at[vanilla_idx, 'F0.5'] = round(f05, 3)
    df.at[vanilla_idx, 'Prec_Train']      = round(split_metrics.get('Train', 0), 3)
    df.at[vanilla_idx, 'Prec_Validation'] = round(split_metrics.get('Validation', 0), 3)
    df.at[vanilla_idx, 'Prec_Test']       = round(split_metrics.get('Test', 0), 3)
    df.at[vanilla_idx, 'Prec_Mean']       = round(prec_mean, 3)

    # 7) Write back out
    df.to_csv(results_file, index=False)
    logger.info(f"Vanilla results incrementally updated in {results_file}")

    return df

def test_vanilla_predictions_anonymised(llm_client, founders_df):
    """
    Generate vanilla GPT predictions for founder success based on anonymised summaries.

    Returns a list of dicts:
    [
        {"Founder ID": "<uuid1>", "Prediction": "Successful"},
        {"Founder ID": "<uuid2>", "Prediction": "Unsuccessful"},
        ...
    ]
    """

    # 1. Build the system message
    system_message = (
        "You are an expert in venture capital asked to predict, as best you can, "
        "which founders will ultimately build companies that exit or IPO at a value "
        "greater than $500 M.  Current funding information is unavailable to you; "
        "assume each company is still early-stage with no meaningful difference in "
        "capital raised.  Base your judgement **only** on the information provided "
        "in each founder’s summary (background, prior exits, domain knowledge, etc.)."
    )

    system_message += (
        "\n\nRules:\n"
        "1. Do NOT use the internet, external databases, or real-world knowledge about these companies.\n"
        "2. Do NOT use – or attempt to infer – the amount of funding a company has raised.\n"
        "3. Use ONLY the anonymised information provided below.\n"
    )

    # 2. Format summaries
    summaries = []
    for _, row in founders_df.iterrows():
        summaries.append(f"Founder {row['founder_uuid']}: {row['summary']}")
    founder_summaries = "\n\n".join(summaries)

    # 3. Create user prompt
    user_prompt = f"""
    Based on the summaries below, classify each founder as "Successful" or "Unsuccessful".

    **Founder Summaries:**
    {founder_summaries}

    **Return ONLY your answer in JSON array form like:**
    ```json
    [
        {{"Founder ID": "uuid1", "Prediction": "Successful"}},
        {{"Founder ID": "uuid2", "Prediction": "Unsuccessful"}}
    ]
    ```

    **Reminder:** In this dataset, only ~2% of founders are actually Successful. Your predictions should reflect this rarity. If you’re unsure, lean toward classifying as "Unsuccessful".
    """

    # 4. Query LLM and parse response
    try:
        raw = llm_client.send_prompt(system_message, user_prompt)
        logging.debug(f"LLM raw response: {raw}")
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        preds = json.loads(cleaned)
        if isinstance(preds, list):
            return preds
        else:
            logging.error("Unexpected JSON structure from LLM")
            return []
    except json.JSONDecodeError:
        logging.error(f"JSON decode failed on LLM response: {raw}")
        return []
    except Exception as e:
        logging.error(f"Error calling LLM: {e}")
        return []

def test_vanilla_predictions_few_shot(llm_client, founders_df, examples_df, n_success=2, n_unsuccess=2, seed=None):
    """
    Generate GPT predictions using grouped few-shot examples.

    Parameters:
        llm_client: LLM client to send prompt.
        founders_df: Unlabeled test founders (with 'founder_uuid').
        examples_df: Labeled example founders (must include 'startup_success').
        n_success: Number of successful examples to sample.
        n_unsuccess: Number of unsuccessful examples to sample.
        seed: Optional random seed.

    Returns:
        A list of dicts: [{"Founder ID": "...", "Prediction": "Successful"}, ...]
    """
    # Sample few-shot examples
    rng = examples_df.sample(frac=1, random_state=seed) if seed else examples_df.sample(frac=1)
    success_examples = rng[rng["startup_success"] == 1].head(n_success)
    unsuccess_examples = rng[rng["startup_success"] == 0].head(n_unsuccess)

    def format_example(row):
        return (
            f"Education: {row.get('university_degrees', 'N/A')}; "
            f"Work History: {row.get('work_history', 'N/A')}; "
            f"Previous Companies Founded: {row.get('previous_companies_founded', 'N/A')}"
        )

    # Format the grouped few-shot examples
    success_block = "\n\n".join(format_example(row) for _, row in success_examples.iterrows())
    unsuccess_block = "\n\n".join(format_example(row) for _, row in unsuccess_examples.iterrows())

    # Drop PII and format test founders
    founders_df = founders_df.drop(columns=['startup_success', 'founder_name'], errors='ignore')
    test_summaries = []
    for _, row in founders_df.iterrows():
        summary = (
            f"Founder {row['founder_uuid']}: "
            f"Education: {row.get('university_degrees', 'N/A')}; "
            f"Work History: {row.get('work_history', 'N/A')}; "
            f"Previous Companies Founded: {row.get('previous_companies_founded', 'N/A')}"
        )
        test_summaries.append(summary)
    test_block = "\n\n".join(test_summaries)

    # Prompt construction
    system_message = (
        "You are a venture capital expert. Based on the labeled founder examples below, "
        "classify the new founders as either 'Successful' or 'Unsuccessful'. "
        "**Do NOT use the internet or any external data. Use only the examples and summaries.**"
    )

    user_prompt = f"""
    These are successful founders:

    {success_block}

    These are unsuccessful founders:

    {unsuccess_block}

    Now, classify the following new founders:

    {test_block}

    Return ONLY your answer in valid JSON array format, with NO explanations, NO comments, and NO text before or after.
    Your output should look exactly like this:
    [
    {{"Founder ID": "uuid1", "Prediction": "Successful"}},
    {{"Founder ID": "uuid2", "Prediction": "Unsuccessful"}}
    ]

    **Reminder:** In this dataset, only ~2% of founders are actually Successful. Your predictions should reflect this rarity. If you’re unsure, lean toward classifying as "Unsuccessful".
    """
    # Send prompt and parse response
    try:
        raw = llm_client.send_prompt(system_message, user_prompt)
        logging.debug(f"LLM raw response: {raw}")
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        preds = json.loads(cleaned)
        if isinstance(preds, list):
            return preds
        else:
            logging.error("Unexpected JSON structure from LLM")
            return []
    except json.JSONDecodeError:
        logging.error(f"JSON decode failed on LLM response: {raw}")
        return []
    except Exception as e:
        logging.error(f"Error calling LLM: {e}")
        return []
    
def test_vanilla_predictions_few_shot_anonymised(llm_client, founders_df, examples_df, n_success=2, n_unsuccess=2, seed=None):
    """
    Generate GPT predictions using anonymised summaries and few-shot examples.

    Parameters:
        llm_client: LLM client to send prompt.
        founders_df: Unlabeled test founders (with 'founder_uuid' and summary fields).
        examples_df: Labeled example founders (must include 'startup_success').
        n_success: Number of successful examples to sample.
        n_unsuccess: Number of unsuccessful examples to sample.
        seed: Optional random seed.

    Returns:
        A list of dicts: [{"Founder ID": "...", "Prediction": "Successful"}, ...]
    """
    # Sample few-shot examples
    rng = examples_df.sample(frac=1, random_state=seed) if seed else examples_df.sample(frac=1)
    success_examples = rng[rng["startup_success"] == 1].head(n_success)
    unsuccess_examples = rng[rng["startup_success"] == 0].head(n_unsuccess)

    # Format few-shot examples (anonymised)
    def format_example(row):
        return row.get("summary", "N/A")

    success_block = "\n\n".join(format_example(row) for _, row in success_examples.iterrows())
    unsuccess_block = "\n\n".join(format_example(row) for _, row in unsuccess_examples.iterrows())

    # Format test summaries
    test_summaries = []
    for _, row in founders_df.iterrows():
        test_summaries.append(f"Founder {row['founder_uuid']}: {row.get('summary', 'N/A')}")
    test_block = "\n\n".join(test_summaries)

    # Prompt construction
    system_message = (
        "You are an expert in venture capital asked to predict, as best you can, "
        "which founders will ultimately build companies that exit or IPO at a value "
        "greater than $500 M. You are provided with anonymised summaries only—no funding data or external knowledge."
        "\n\nRules:\n"
        "1. Do NOT use the internet, external databases, or real-world knowledge about these companies.\n"
        "2. Do NOT use – or attempt to infer – the amount of funding a company has raised.\n"
        "3. Use ONLY the anonymised information provided below.\n"
    )

    user_prompt = f"""
    These are successful founders:

    {success_block}

    These are unsuccessful founders:

    {unsuccess_block}

    Now, classify the following new founders:

    {test_block}

    Return ONLY your answer in valid JSON array format, with NO explanations, NO comments, and NO text before or after.
    Your output should look exactly like this:
    [
    {{"Founder ID": "uuid1", "Prediction": "Successful"}},
    {{"Founder ID": "uuid2", "Prediction": "Unsuccessful"}}
    ]

    **Reminder:** In this dataset, only ~2% of founders are actually Successful. Your predictions should reflect this rarity. If you’re unsure, lean toward classifying as "Unsuccessful".
    """

    # Send prompt and parse response
    try:
        raw = llm_client.send_prompt(system_message, user_prompt)
        logging.debug(f"LLM raw response: {raw}")
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        preds = json.loads(cleaned)
        if isinstance(preds, list):
            return preds
        else:
            logging.error("Unexpected JSON structure from LLM")
            return []
    except json.JSONDecodeError:
        logging.error(f"JSON decode failed on LLM response: {raw}")
        return []
    except Exception as e:
        logging.error(f"Error calling LLM: {e}")
        return []