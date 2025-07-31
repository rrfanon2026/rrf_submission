import textwrap

# ── redaction prompt template (verbatim) ───────────────────────
REDACTOR_TEMPLATE = textwrap.dedent("""\
    ### ROLE
    You are a redactor who converts detailed founder résumés into short, anonymised summaries.
    The goal is to keep the strongest venture-relevant signals while removing
    anything that would let another model look the person up online.

    ### WHAT TO REMOVE / GENERALISE
    • Replace all **proper nouns** (people, companies, universities, cities) with
      generic descriptors such as “Ivy-League university”, “top-tier public research
      university”, “major technology company (10001+ employees)”, or “global investment bank”.
    • When redacting universities, aim to preserve the prestige level (e.g., distinguish Ivy League from public universities).
    • Keep **degree subjects, level (BS, MS, MBA, PhD), company-size brackets,
      role type (e.g. CTO, algorithmic trader), and industry**.
    • If multiple roles are similar, combine them (“several years as a senior engineer at large software firms”).
    • Avoid all gendered language. Do not use "he", "she", "his", or "her" — use "they", "their", etc.
    • Omit dates, exact years, GPAs, and minor details.
    • If data is missing, say nothing about it.

    ### CRUCIAL SIGNALS TO INCLUDE
    • Include ALL information in the “professional background” notes.                        
    Where possible, ensure the summary retains:
    • Whether the founder is a graduate of an Ivy League university.
    • Any roles as angel investor or in venture capital.
    • Experience as a board member or advisor to a major company.
    • Employment at a FAANG company.
    • Involvement in AI / machine learning.

    ### OUTPUT FORMAT
    Write **one cohesive paragraph, 2–4 sentences**, third-person, present tense.

    ### INPUT
    {FOUNDERRAW}

    ### OUTPUT
    (Return a single JSON array, each element:
       {"Founder ID": "<uuid>", "Summary": "<paragraph>"} )
""")


BACKGROUND_PROSE_PROMPT = textwrap.dedent("""\
    ### ROLE
    You are a writer who transforms structured bullet-style facts into natural, free-flowing prose for multiple startup founders. The goal is to preserve all information while improving readability and flow.

    ### INSTRUCTIONS
    • For each founder, rewrite their background facts into a short, natural-sounding paragraph.
    • If a "Company State" is provided, include it as the location where they founded their company.
    • Keep the tone professional, neutral, and descriptive.
    • Do not omit or alter any information.
    • Do not include founder names or identifiable info.

    ### FORMAT
    Return a JSON array with objects of the form: 
    {"Founder ID": "<uuid>", "Background Prose": "<paragraph>"}

    ### INPUT
    {RAW_BLOCKS}

    ### OUTPUT
    (Return a single JSON array)
""")