import json, math, time, re
import pandas as pd
from src.redaction.constants import REDACTOR_TEMPLATE, BACKGROUND_PROSE_PROMPT
from src.utils.dataset_utils import load_dataset


# ── helper: redact ≤20 founders in one LLM call ────────────────
def anonymise_founders(llm_client, founders_df, include_background=True):
    """Return list[{'Founder ID', 'Summary'}] for ≤20 founders."""
    # founders_df = founders_df.drop(columns=['startup_success', 'founder_name'], errors='ignore')
    founders_for_llm = founders_df.drop(columns=['startup_success', 'founder_name'], errors='ignore')

    raw_blocks = []
    for _, r in founders_for_llm.iterrows():
        block = (
            f"Founder UUID: {r['founder_uuid']}\n"
            f"Education: {r.get('university_degrees', 'N/A')}\n"
            f"Work History: {r.get('work_history', 'N/A')}\n"
            f"Previous Companies Founded: {r.get('previous_companies_founded', 'N/A')}\n"
        )
        if include_background:
            block += f"Professional Background: {r.get('professional_background', 'N/A')}\n"
        raw_blocks.append(block)

    prompt = REDACTOR_TEMPLATE.replace("{FOUNDERRAW}", "\n\n".join(raw_blocks))
    resp   = llm_client.send_prompt("", prompt)
    cleaned = resp.replace("```json", "").replace("```", "").strip()
    return json.loads(cleaned)    # list[dict]

def redact_and_save(input_path, output_path, llm_client, batch_size, logger, two_stage_redaction=False):
    if output_path.exists():
        logger.info(f"Anonymised file already exists: {output_path}. Skipping.")
        return

    founders = load_dataset(input_path, "test")

    print(founders.columns)
    print(founders[['founder_uuid', 'startup_success']].head())


    batches = math.ceil(len(founders) / batch_size)
    logger.info(f"Redacting {len(founders)} founders in {batches} batch(es)…")

    all_rows = []
    t0 = time.perf_counter()

    for i in range(batches):
        batch = founders.iloc[i*batch_size : (i+1)*batch_size]

        try:
            summaries = anonymise_founders(llm_client, batch, include_background=not two_stage_redaction)

            # If using two-stage, precompute background prose in one batch call
            if two_stage_redaction:
                background_prose_list = prose_from_professional_background_batch(llm_client, batch)
                background_prose_map = {
                    item["Founder ID"]: item["Background Prose"] for item in background_prose_list
                }

            for item in summaries:
                uuid = item["Founder ID"]
                summary = item["Summary"]

                row = founders[founders['founder_uuid'] == uuid]
                if not row.empty:
                    row_data = row.iloc[0]

                    if two_stage_redaction and uuid in background_prose_map:
                        summary += " " + background_prose_map[uuid]

                    state = extract_state_from_location(row_data.get("company_location", ""))
                    if state and not two_stage_redaction:
                        summary = summary.strip()
                        summary += f" This founder founded their company in {state}."

                # all_rows.append({
                #     "founder_uuid": uuid,
                #     "summary": summary
                # })
                all_rows.append({
                    "founder_uuid": uuid,
                    "summary": summary,
                    "startup_success": row_data.get("startup_success", None)
                })

            logger.info(f"  ✓ Batch {i+1}/{batches} ({len(batch)} founders) done.")
        except Exception as e:
            logger.error(f"Batch {i+1} failed: {e}")

    pd.DataFrame(all_rows, columns=["founder_uuid", "summary", "startup_success"]).to_csv(output_path, index=False)
    logger.info(f"Finished in {time.perf_counter()-t0:.1f}s. CSV written → {output_path}")

def prose_from_professional_background_batch(llm_client, founders_df):
    """Returns list[{'founder_uuid': ..., 'background_prose': ...}] in batch."""
    raw_blocks = []

    for _, r in founders_df.iterrows():
        uuid = r['founder_uuid']
        background = r.get('professional_background', '').strip()
        location_str = r.get('company_location', '')
        state = extract_state_from_location(location_str)

        if background:
            block = f"Founder UUID: {uuid}\n"
            block += f"Professional Background: {background}\n"
            if state:
                block += f"Company State: {state}\n"
            raw_blocks.append(block)

    if not raw_blocks:
        return []

    prompt = BACKGROUND_PROSE_PROMPT.format(RAW_BLOCKS="\n\n".join(raw_blocks))
    resp = llm_client.send_prompt("", prompt)
    
    cleaned = resp.replace("```json", "").replace("```", "").strip()
    return json.loads(cleaned)  # list of dicts


def extract_state_from_location(text):
    match = re.search(r',\s*([A-Za-z\s]+)$', str(text))
    return match.group(1) if match else None