import math
import time
import pandas as pd
import logging
import json
from pathlib import Path
import sys
import json, logging, textwrap
from typing import List, Dict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Testing functions
def test_question(llm_client, question, founders_df):
    """Test a single question against all 20 founders."""
    system_message = (
        "You are a VC analyst evaluating founders. "
        "Your task is to decide whether the provided question applies to each founder based on their data. "
        "Return your answer as 'Yes' or 'No' for each founder in JSON format."
    )

    # Remove the 'startup_success' column for testing
    columns_to_drop = ['startup_success', 'founder_name']
    founders_df = founders_df.drop(columns=columns_to_drop, errors='ignore')

    founder_summaries = "\n\n".join(founders_df.apply(lambda row: 
        f"Founder {row['founder_uuid']}: Education: {row['university_degrees']}, "
        f"Work History: {row['work_history']}, Previous Companies: {row['previous_companies_founded']}", axis=1))
    
    user_prompt = f"""
    Given the following founder backgrounds, answer the question concisely as 'Yes', 'No'.

    **Question:** {question}
    
    **Founder Summaries:**
    {founder_summaries}

    **Expected Output Format (JSON List):**
    ```json
    [
        {{"Founder ID": "uuid1", "Answer": "Yes"}},
        {{"Founder ID": "uuid2", "Answer": "No"}}
    ]
    ```
    """

    try:
        response = llm_client.send_prompt(system_message, user_prompt)
        
        # Print and log the raw response for debugging
        print(f"Raw response:\n{response}\n")
        logging.info(f"Raw response: {response}")
        
        cleaned_response = response.replace("```json", "").replace("```", "").strip()
        
        predictions = json.loads(cleaned_response)
        
        if isinstance(predictions, list):
            logging.info("Successfully parsed JSON response.")
            return predictions
        else:
            logging.error("Response was not a list.")
            return []

    except json.JSONDecodeError:
        logging.error(f"Failed to parse response as JSON. Raw response: {response}")
        print(f"Failed to parse response as JSON. Raw response:\n{response}")
        return []
    


def save_predictions(predictions, output_file):
    """Save predictions to a CSV file."""
    if isinstance(predictions, list) and len(predictions) > 0:
        results_df = pd.DataFrame(predictions)
        results_df.to_csv(output_file, index=False)
        logging.info(f"Predictions saved to {output_file}")
    else:
        logging.error("No valid predictions to save. Please check the input format.")


# Evaluation functions
def update_results_file(results_file: Path, question: str, responses: list, test_founders: pd.DataFrame, logger: logging.Logger, dataset_assignments: dict):
    """Update the results CSV file with new question responses."""
    # Convert string path to Path object if it's not already
    results_file = Path(results_file)

    # Create or load existing results
    if results_file.exists():
        results_df = pd.read_csv(results_file)
    else:
        # Initialize with Index, Question and metric columns first
        metric_columns = ['Pass Rate', 'Prec', 'TP', 'FP', 'TN', 'FN', 'Rec', 'F1', 'F0.5',
                          'Prec_Train', 'Prec_Validation', 'Prec_Test', 'Prec_Mean']
        results_df = pd.DataFrame(columns=['Index', 'Question'] + metric_columns)
        
        # Add dataset row
        dataset_row = {'Index': '', 'Question': 'Dataset'}
        results_df = pd.concat([results_df, pd.DataFrame([dataset_row])], ignore_index=True)
        
        # Add success row
        success_row = {'Index': '', 'Question': 'Success'}
        results_df = pd.concat([results_df, pd.DataFrame([success_row])], ignore_index=True)
        
        # Add success proportion row
        proportion_row = {'Index': '', 'Question': 'SUCCESS_PROPORTION'}
        results_df = pd.concat([results_df, pd.DataFrame([proportion_row])], ignore_index=True)
    
    # --- Insert "Founder Index" row at the very beginning if it does not exist ---
    if "Founder Index" not in results_df['Question'].values:
        # Identify founder columns as those not in ['Index', 'Question'] plus our metric columns.
        metric_columns = ['Pass Rate', 'Prec', 'TP', 'FP', 'TN', 'FN', 'Rec', 'F1', 'F0.5',
                          'Prec_Train', 'Prec_Validation', 'Prec_Test', 'Prec_Mean']
        founder_cols = [col for col in results_df.columns if col not in ['Index', 'Question'] + metric_columns]
        # Create a dictionary mapping each founder column to its sequential index (starting at 1)
        founder_index_dict = {col: i+1 for i, col in enumerate(founder_cols)}
        founder_index_row = {'Index': '', 'Question': 'Founder Index'}
        founder_index_row.update(founder_index_dict)
        # Insert this row at the very beginning
        results_df = pd.concat([pd.DataFrame([founder_index_row]), results_df], ignore_index=True)
    # --- End of "Founder Index" insertion ---
    
    # Add new row for this question
    question_row = {'Question': question}
    
    # Initialize metrics for each dataset
    dataset_metrics = {
        "Train": {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0},
        "Validation": {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0},
        "Test": {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
    }
    
    tp = fp = tn = fn = 0
    for response in responses:
        founder_id = response['Founder ID']
        answer = response['Answer']
        
        # Find the actual success value for this founder from the test_founders DataFrame
        founder_data = test_founders[test_founders['founder_uuid'] == founder_id]
        if len(founder_data) == 0:
            logger.warning(f"Skipping founder {founder_id} - not found in test_founders")
            continue
            
        actual = int(founder_data['startup_success'].iloc[0])
        pred = 1 if answer == 'Yes' else 0
        
        # Save the response for this founder using a shortened ID (first 5 characters)
        question_row[founder_id[:5]] = pred
        
        # If this founder column isn't present in results_df, add it and set dataset assignment and success value
        if founder_id[:5] not in results_df.columns:
            results_df[founder_id[:5]] = ''
            dataset_label = dataset_assignments.get(founder_id, "Unknown")
            results_df.loc[results_df['Question'] == 'Dataset', founder_id[:5]] = dataset_label
            results_df.loc[results_df['Question'] == 'Success', founder_id[:5]] = actual
        
        # Retrieve dataset label for the current founder
        dataset_label = dataset_assignments.get(founder_id, "Unknown")
        
        # Update overall metrics and dataset-specific metrics
        if pred == 1 and actual == 1:
            tp += 1
            dataset_metrics[dataset_label]['tp'] += 1
        elif pred == 1 and actual == 0:
            fp += 1
            dataset_metrics[dataset_label]['fp'] += 1
        elif pred == 0 and actual == 0:
            tn += 1
            dataset_metrics[dataset_label]['tn'] += 1
        elif pred == 0 and actual == 1:
            fn += 1
            dataset_metrics[dataset_label]['fn'] += 1
    
    precision = round(tp / (tp + fp), 3) if (tp + fp) > 0 else 0
    recall = round(tp / (tp + fn), 3) if (tp + fn) > 0 else 0
    f1 = round(2 * (precision * recall) / (precision + recall), 3) if precision + recall > 0 else 0
    f05 = round(1.25 * (precision * recall) / (0.25 * precision + recall), 3) if precision + recall > 0 else 0
    
    prec_by_dataset = {k: round(v['tp'] / (v['tp'] + v['fp']), 3) if (v['tp'] + v['fp']) > 0 else 0 
                       for k, v in dataset_metrics.items()}
    
    valid_precs = [v for v in prec_by_dataset.values() if v is not None]
    prec_mean = round(sum(valid_precs) / len(valid_precs), 3) if valid_precs else 0
    
    question_row.update({
        'Pass Rate': tp + fp,
        'Prec': precision,
        'TP': tp,
        'FP': fp,
        'TN': tn,
        'FN': fn,
        'Rec': recall,
        'F1': f1,
        'F0.5': f05,
        'Prec_Train': prec_by_dataset.get('Train', 0),
        'Prec_Validation': prec_by_dataset.get('Validation', 0),
        'Prec_Test': prec_by_dataset.get('Test', 0),
        'Prec_Mean': prec_mean
    })
    
    num_questions = len(results_df[~results_df['Question'].isin(['Dataset', 'Success', 'SUCCESS_PROPORTION', 'Founder Index'])])
    question_row['Index'] = num_questions + 1
    
    results_df = pd.concat([results_df, pd.DataFrame([question_row])], ignore_index=True)
    results_df.to_csv(results_file, index=False)
    return results_df


def update_founder_index(results_file: Path):
    """Update the 'Founder Index' row with the correct numbering of founders."""
    # Convert string path to Path object if it's not already
    results_file = Path(results_file)

    if not results_file.exists():
        logging.error(f"Results file does not exist: {results_file}")
        return
    
    results_df = pd.read_csv(results_file)
    
    # Define the metric columns to be placed at the beginning
    metric_columns = ['Index', 'Question', 'Pass Rate', 'Prec', 'TP', 'FP', 'TN', 'FN', 
                      'Rec', 'F1', 'F0.5', 'Prec_Train', 'Prec_Validation', 'Prec_Test', 'Prec_Mean']
    
    # Identify founder columns by excluding metric columns
    founder_columns = [col for col in results_df.columns if col not in metric_columns]

    # If there are no founder columns, warn and return
    if not founder_columns:
        logging.warning("No founder columns detected, cannot update founder index.")
        return

    # Create the Founder Index row dictionary
    founder_index_row = {col: (i + 1) if col in founder_columns else '' for i, col in enumerate(founder_columns)}
    founder_index_row.update({col: '' for col in metric_columns})
    founder_index_row['Index'] = ''
    founder_index_row['Question'] = 'Founder Index'
    
    # Remove the existing Founder Index row if it exists
    results_df = results_df[results_df['Question'] != 'Founder Index']
    
    # Insert the new Founder Index row at the top, after the header row
    results_df = pd.concat([pd.DataFrame([founder_index_row]), results_df], ignore_index=True)
    
    # Ensure metric columns are placed before founder columns
    ordered_columns = metric_columns + founder_columns
    results_df = results_df[ordered_columns]
    
    # Save the updated DataFrame back to the file
    results_df.to_csv(results_file, index=False)
    logging.info(f"Founder Index updated successfully in {results_file}")


def update_success_proportion(results_file: Path):
    """Update the 'SUCCESS_PROPORTION' row in the results file without deleting existing rows."""
    # Convert string path to Path object if it's not already
    results_file = Path(results_file)    

    if not results_file.exists():
        logging.error(f"Results file does not exist: {results_file}")
        return
    
    results_df = pd.read_csv(results_file)

    # Identify founder columns by excluding metric columns
    metric_columns = ['Index', 'Question', 'Pass Rate', 'Prec', 'TP', 'FP', 'TN', 'FN', 
                      'Rec', 'F1', 'F0.5', 'Prec_Train', 'Prec_Validation', 'Prec_Test', 'Prec_Mean']
    
    founder_cols = [col for col in results_df.columns if col not in metric_columns]

    if not founder_cols:
        logging.warning("No founder columns detected, cannot update SUCCESS_PROPORTION.")
        return

    # Filter the results_df to exclude rows related to Dataset, Success, and SUCCESS_PROPORTION
    question_rows = results_df[~results_df['Question'].isin(['Dataset', 'Success', 'SUCCESS_PROPORTION', 'Founder Index'])]

    # Calculate success proportion for each founder column
    success_proportions = {}
    for col in founder_cols:
        # Convert answers to numeric (1s and 0s)
        answers = pd.to_numeric(question_rows[col], errors='coerce')
        num_ones = (answers == 1).sum()
        total_answers = answers.notna().sum()
        
        # Calculate success rate as a percentage
        success_rate = num_ones / total_answers if total_answers > 0 else 0
        success_proportions[col] = f"{success_rate:.2%}"
    
    # Update the existing SUCCESS_PROPORTION row in-place
    success_row_index = results_df.index[results_df['Question'] == 'SUCCESS_PROPORTION'].tolist()
    
    if len(success_row_index) == 1:
        row_index = success_row_index[0]
        for col, value in success_proportions.items():
            results_df.at[row_index, col] = value
    else:
        logging.error("SUCCESS_PROPORTION row not found or duplicated. Make sure it's initialized properly.")
        return

    # Save the updated DataFrame back to the file
    results_df.to_csv(results_file, index=False)
    logging.info(f"SUCCESS_PROPORTION updated successfully in {results_file}")


def load_selected_questions(questions_file):
    """Load the selected questions from CSV file."""
    try:
        questions_df = pd.read_csv(questions_file)
        questions = questions_df['Question'].tolist()
        logger.info(f"Successfully loaded {len(questions)} selected questions.")
        return questions
    except Exception as e:
        logger.error(f"Error loading selected questions: {e}")
        sys.exit(1)


REDACTOR_TEMPLATE = textwrap.dedent("""\
    ### ROLE
    You are a redactor who converts detailed founder résumés into short, anonymised summaries.
    The goal is to keep the strongest venture-relevant signals while removing
    anything that would let another model look the person up online.

    ### WHAT TO REMOVE / GENERALISE
    • Replace all proper nouns (people, company names, universities, cities) with
      generic descriptors such as “Ivy-League university”, “top-tier public research
      university”, “major technology company (10001+ employees)”, “global investment
      bank”, or “well-known tech hub”.
    • Keep **degree subjects, level (BS, MS, MBA, PhD), company-size brackets,
      role type (e.g. CTO, algorithmic trader), and industry**.
    • If multiple roles are similar, combine them (“several years as a senior
      engineer at large software firms”).
    • De-identify locations to region level only (“in a major European capital”,
      “in the US Midwest”, “in Southeast Asia”).
    • Omit dates, exact years, GPAs, and minor details.
    • If data is missing, say nothing about it.

    ### OUTPUT FORMAT
    Write **one cohesive paragraph, 2–4 sentences**, third-person, present tense.
    ### INPUT
    {FOUNDERRAW}
    ### OUTPUT
    (Return a single JSON array, each element:
       {"Founder ID": "<uuid>", "Summary": "<paragraph>"} )
""")


def anonymise_founders(llm_client, founders_df) -> List[Dict[str, str]]:
    """
    Turn up to 20 founder records into de-identified paragraph summaries.

    Parameters
    ----------
    llm_client : object with .send_prompt(system_prompt, user_prompt) -> str
    founders_df : pd.DataFrame
        Must include columns:
        'founder_uuid', 'university_degrees', 'work_history',
        'previous_companies_founded'  (others are optional)

    Returns
    -------
    List[Dict[str, str]]
        [
          {"Founder ID": "uuid-1", "Summary": "This founder holds …"},
          ...
        ]
    """

    # 1. Drop PII/labels we never want the LLM to see.
    founders_df = founders_df.drop(
        columns=['startup_success', 'founder_name'], errors='ignore'
    )

    # 2. Build raw text block the redactor will receive.
    raw_blocks = []
    for _, row in founders_df.iterrows():
        block = (
            f"Founder UUID: {row['founder_uuid']}\n"
            f"Education: {row.get('university_degrees', 'N/A')}\n"
            f"Work History: {row.get('work_history', 'N/A')}\n"
            f"Previous Companies Founded: "
            f"{row.get('previous_companies_founded', 'N/A')}\n"
            f"Location Notes: {row.get('founder_location', '')}"
        )
        raw_blocks.append(block)

    founders_blob = "\n\n".join(raw_blocks)
    user_prompt = REDACTOR_TEMPLATE.replace("{FOUNDERRAW}", founders_blob)

    # 3. Ask the LLM for anonymised summaries.
    try:
        raw_response = llm_client.send_prompt("", user_prompt)
        logging.debug("Redactor LLM response: %s", raw_response)

        cleaned = raw_response.replace("```json", "").replace("```", "").strip()
        summaries = json.loads(cleaned)

        if isinstance(summaries, list):
            return summaries
        logging.error("Unexpected JSON structure from LLM.")
        return []
    except json.JSONDecodeError:
        logging.error("Failed to decode JSON from LLM response.")
        return []
    except Exception as e:
        logging.error(f"Error calling LLM: {e}")
        return []
    
def build_anonymised_csv(
    llm_client,
    founders_df,
    batch_size: int,
    out_path: Path,
    logger: logging.Logger,
):
    """
    Redact `founders_df` (≤20-row chunks) and write a CSV with columns:
        founder_uuid, summary
    """
    all_rows = []                       # ← collect dicts from every batch
    batches  = math.ceil(len(founders_df) / batch_size)
    start    = time.perf_counter()

    for i in range(batches):
        batch = founders_df.iloc[i*batch_size : (i+1)*batch_size]
        logger.info(f"Redacting batch {i+1}/{batches} ({len(batch)} founders)…")

        try:
            summaries = anonymise_founders(llm_client, batch)
            all_rows.extend(
                {"founder_uuid": item["Founder ID"], "summary": item["Summary"]}
                for item in summaries
            )
        except Exception as e:
            logger.error(f"Batch {i+1}: {e}")

    # --- write CSV ----------------------------------------------------------
    df_out = pd.DataFrame(all_rows, columns=["founder_uuid", "summary"])
    df_out.to_csv(out_path, index=False)
    logger.info(f"✓ Wrote {len(df_out)} summaries → {out_path} "
                f"in {time.perf_counter()-start:.1f}s")
    

def test_question_anonymised(llm_client, question, founders_df):
    """Test a single question against anonymised founder summaries."""
    system_message = (
        "You are a VC analyst evaluating founders. "
        "Your task is to decide whether the provided question applies to each founder based on their anonymised summary. "
        "Return your answer as 'Yes' or 'No' for each founder in JSON format."
    )

    # Remove the 'startup_success' column just in case
    founders_df = founders_df.drop(columns=['startup_success'], errors='ignore')

    founder_summaries = "\n\n".join(
        founders_df.apply(lambda row: f"Founder {row['founder_uuid']}:\n{row['summary']}", axis=1)
    )

    user_prompt = f"""
    Given the following anonymised founder summaries, answer the question concisely as 'Yes' or 'No'.

    **Question:** {question}

    **Founder Summaries:**
    {founder_summaries}

    **Expected Output Format (JSON List):**
    ```json
    [
        {{"Founder ID": "uuid1", "Answer": "Yes"}},
        {{"Founder ID": "uuid2", "Answer": "No"}}
    ]
    """

    try:
        response = llm_client.send_prompt(system_message, user_prompt)

        # Print and log the raw response for debugging
        print(f"Raw response:\n{response}\n")
        logging.info(f"Raw response: {response}")

        cleaned_response = response.replace("```json", "").replace("```", "").strip()
        predictions = json.loads(cleaned_response)

        if isinstance(predictions, list):
            logging.info("Successfully parsed JSON response.")
            return predictions
        else:
            logging.error("Response was not a list.")
            return []

    except json.JSONDecodeError:
        logging.error(f"Failed to parse response as JSON. Raw response: {response}")
        print(f"Failed to parse response as JSON. Raw response:\n{response}")
        return []

def test_trial_vanilla_prediction(llm_client, trials_df):
    """
    Generate vanilla GPT predictions for clinical trial success as Yes/No for each trial.

    Returns a list of dicts:
    [
        {"Trial ID": "<id1>", "Answer": "Yes"},
        {"Trial ID": "<id2>", "Answer": "No"},
        ...
    ]
    """

    # 1. Drop ground-truth or PII columns if present
    columns_to_drop = ['trial_success', 'trial_name']
    trials_df = trials_df.drop(columns=columns_to_drop, errors='ignore')

    # 2. System prompt
    system_message = (
        "You are a clinical trial analyst. "
        "For each clinical trial below, predict whether it is likely to be successful. "
        "If you believe the trial is likely to be successful based only on the structured summary, answer 'Yes'; "
        "otherwise, answer 'No'. "
        "Base your answer exclusively on the information provided. Do not use external data or real-world knowledge."
    )

    # 3. Format each trial’s data as a short summary
    summaries = []
    for _, row in trials_df.iterrows():
        summary = (
            f"Trial {row['Trial ID']}: "
            f"Phase: {row.get('Phase', 'N/A')}; "
            f"Diseases: {row.get('Diseases', 'N/A')}; "
            f"Drugs: {row.get('Drugs', 'N/A')}; "
            f"Inclusion: {row.get('Inclusion Summary', 'N/A')}; "
            f"Exclusion: {row.get('Exclusion Summary', 'N/A')}"
        )
        summaries.append(summary)
    trial_summaries = "\n\n".join(summaries)

    # 4. User prompt
    user_prompt = f"""
    Based on the summaries below, predict whether each clinical trial is likely to be successful.
    For each trial, answer 'Yes' if you believe it will be successful, otherwise answer 'No'.

    **Trial Summaries:**
    {trial_summaries}

    **Return your answer in JSON array form like:**
    ```json
    [
        {{"Trial ID": "NCT01234567", "Answer": "Yes"}},
        {{"Trial ID": "NCT07654321", "Answer": "No"}}
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


def test_trial_question(llm_client, question, trials_df):
    """Test a single YES/NO question against each clinical trial in the provided DataFrame."""
    system_message = (
        "You are a clinical trial analyst. "
        "Your task is to decide whether the provided YES/NO question applies to each clinical trial based on the structured summary. "
        "Return your answer as 'Yes' or 'No' for each trial in JSON format."
    )

    trial_summaries = "\n\n".join(trials_df.apply(lambda row: 
        f"Trial {row['Trial ID']}:\n"
        f"Phase: {row['Phase']}\n"
        f"Diseases: {row['Diseases']}\n"
        f"Drugs: {row['Drugs']}\n"
        f"Inclusion Summary: {row['Inclusion Summary']}\n"
        f"Exclusion Summary: {row['Exclusion Summary']}", axis=1))

    user_prompt = f"""
    Given the following clinical trial summaries, answer the question for each trial with 'Yes' or 'No'.

    **Question:** {question}

    **Trial Summaries:**
    {trial_summaries}

    **Expected Output Format (JSON List):**
    ```json
    [
        {{"Trial ID": "NCT01234567", "Answer": "Yes"}},
        {{"Trial ID": "NCT07654321", "Answer": "No"}}
    ]
    """
    try:
        response = llm_client.send_prompt(system_message, user_prompt)

        # Debugging info
        print(f"Raw response:\n{response}\n")
        logging.info(f"Raw response: {response}")

        cleaned_response = response.replace("```json", "").replace("```", "").strip()
        predictions = json.loads(cleaned_response)

        if isinstance(predictions, list):
            logging.info("Successfully parsed JSON response.")
            return predictions
        else:
            logging.error("Response was not a list.")
            return []

    except json.JSONDecodeError:
        logging.error(f"Failed to parse response as JSON. Raw response: {response}")
        print(f"Failed to parse response as JSON. Raw response:\n{response}")
        return []
    
def update_trial_results_file(results_file: Path, question: str, responses: list, test_trials: pd.DataFrame, logger: logging.Logger, dataset_assignments: dict):
    """Update the results CSV file with new question responses for clinical trials."""
    results_file = Path(results_file)

    # Load or initialize results
    if results_file.exists():
        results_df = pd.read_csv(results_file)
    else:
        metric_columns = ['Pass Rate', 'Prec', 'TP', 'FP', 'TN', 'FN', 'Rec', 'F1', 'F0.5',
                          'Prec_Train', 'Prec_Validation', 'Prec_Test', 'Prec_Mean', 'Accuracy',
                          'Specificity', 'Balanced_Acc', 'Youden_J']
        results_df = pd.DataFrame(columns=['Index', 'Question'] + metric_columns)

        for label in ['Dataset', 'Success', 'SUCCESS_PROPORTION']:
            row = {'Index': '', 'Question': label}
            results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)

    # Add Trial Index row if missing
    if "Trial Index" not in results_df['Question'].values:
        metric_columns = ['Pass Rate', 'Prec', 'TP', 'FP', 'TN', 'FN', 'Rec', 'F1', 'F0.5',
                          'Prec_Train', 'Prec_Validation', 'Prec_Test', 'Prec_Mean', 'Accuracy',
                          'Specificity', 'Balanced_Acc', 'Youden_J']
        trial_cols = [col for col in results_df.columns if col not in ['Index', 'Question'] + metric_columns]
        trial_index_row = {'Index': '', 'Question': 'Trial Index'}
        trial_index_row.update({col: i+1 for i, col in enumerate(trial_cols)})
        results_df = pd.concat([pd.DataFrame([trial_index_row]), results_df], ignore_index=True)

    question_row = {'Question': question}
    dataset_metrics = {"Train": {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0},
                       "Validation": {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0},
                       "Test": {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}}

    tp = fp = tn = fn = 0
    for r in responses:
        trial_id = r['Trial ID']
        answer = r['Answer']
        match = test_trials[test_trials['Trial ID'] == trial_id]

        if match.empty:
            logger.warning(f"Trial ID {trial_id} not found in test_trials.")
            continue

        actual = int(match['label'].iloc[0])
        pred = 1 if answer == "Yes" else 0
        col_id = trial_id
        question_row[col_id] = pred

        if col_id not in results_df.columns:
            results_df[col_id] = ''
            dataset_label = dataset_assignments.get(trial_id, "Unknown")
            results_df.loc[results_df['Question'] == 'Dataset', col_id] = dataset_label
            results_df.loc[results_df['Question'] == 'Success', col_id] = actual

        dataset_label = dataset_assignments.get(trial_id, "Unknown")
        if pred == 1 and actual == 1:
            tp += 1; dataset_metrics[dataset_label]['tp'] += 1
        elif pred == 1 and actual == 0:
            fp += 1; dataset_metrics[dataset_label]['fp'] += 1
        elif pred == 0 and actual == 0:
            tn += 1; dataset_metrics[dataset_label]['tn'] += 1
        elif pred == 0 and actual == 1:
            fn += 1; dataset_metrics[dataset_label]['fn'] += 1

    precision = round(tp / (tp + fp), 3) if (tp + fp) else 0
    recall = round(tp / (tp + fn), 3) if (tp + fn) else 0
    f1 = round(2 * precision * recall / (precision + recall), 3) if precision + recall else 0
    f05 = round(1.25 * precision * recall / (0.25 * precision + recall), 3) if precision + recall else 0

    prec_by_dataset = {k: round(v['tp'] / (v['tp'] + v['fp']), 3) if (v['tp'] + v['fp']) else 0 
                       for k, v in dataset_metrics.items()}
    prec_mean = round(sum(prec_by_dataset.values()) / len(prec_by_dataset), 3)
    accuracy = round((tp + tn) / (tp + tn + fp + fn), 3) if (tp + tn + fp + fn) else 0
    specificity = round(tn / (tn + fp), 3) if (tn + fp) else 0
    balanced_acc = round((recall + specificity) / 2, 3)
    youden_j = round(recall + specificity - 1, 3)

    question_row.update({
        'Pass Rate': tp + fp,
        'Prec': precision,
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
        'Rec': recall, 'F1': f1, 'F0.5': f05,
        'Prec_Train': prec_by_dataset.get('Train', 0),
        'Prec_Validation': prec_by_dataset.get('Validation', 0),
        'Prec_Test': prec_by_dataset.get('Test', 0),
        'Prec_Mean': prec_mean,
        'Accuracy': accuracy,
        'Specificity': specificity,
        'Balanced_Acc': balanced_acc,
        'Youden_J': youden_j,
    })

    question_row['Index'] = len(results_df[~results_df['Question'].isin(['Dataset', 'Success', 'SUCCESS_PROPORTION', 'Trial Index'])]) + 1
    results_df = pd.concat([results_df, pd.DataFrame([question_row])], ignore_index=True)
    results_df.to_csv(results_file, index=False)
    return results_df
