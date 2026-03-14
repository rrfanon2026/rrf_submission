# Clinical Trials: RRF Applied to TOP Phase I

This folder contains a second dataset application: RRF applied to clinical trial outcome prediction (TOP Phase I).

## For Reviewers

Only `results/phase_I/` is used by the paper reproduction notebook. Specifically:

- `results/phase_I/primary_results/primary_model_repeat_metrics.csv` — RRF primary model metrics
- `results/phase_I/logreg_results/logreg_repeat_metrics.csv` — elastic net baseline metrics
- `results/phase_I/ablation_results/beta_sweep_summary.csv` — beta sweep for Pareto figure

To re-run the offline evaluation scripts (no API calls):

```bash
python clinical_trials/pipeline/09_primary_model.py
python clinical_trials/pipeline/11_logreg_baseline.py
python clinical_trials/pipeline/10_ablation_analysis.py
```

## Folder Structure

- `data/`: Input CSVs (filtered question-response matrices for val and test splits)
- `results/phase_I/`: Phase I primary, logistic regression, and ablation results (used by notebook)
- `pipeline/`: Pipeline scripts (stages 01–04 require API calls; 09/10/11 are offline)
- `configs/`: Locked model and ablation grid configs
- `docs/`: Inventory and curation manifests
- `legacy_snapshot/`: Read-only provenance archive (gitignored; not needed for reproduction)

## Pipeline Stages

### Stages 01–04: Data Preparation (require API calls)

These stages generate and filter questions using LLM calls. They require `legacy_snapshot/` for raw training data and an OpenAI API key.

```bash
# 01 – Summarise trials
python clinical_trials/pipeline/01_summarise_trials.py \
  --provider openai --model gpt-4o-mini --phase I \
  --input-filename phase_I_train_split.csv

# 02 – Generate questions
python clinical_trials/pipeline/02_question_generation.py \
  --provider openai --model gpt-4o-mini --phase I --num-questions 10

# 03 – Run trial inference
python clinical_trials/pipeline/03_test_questions.py \
  --provider openai --model gpt-4o-mini --phase I \
  --mode questions --test-set val_split --question-set 0

# 04 – Filter/deduplicate/invert
python clinical_trials/pipeline/04_validate_questions.py \
  --model gpt-4o-mini --phase I --splits val_split,test
```

### Stages 09–11: Offline Evaluation (no API calls)

These stages read precomputed data from `data/` and write results to `results/phase_I/`.

**Primary model (locked):**
```bash
python clinical_trials/pipeline/09_primary_model.py
```

**Logistic regression baseline (locked):**
```bash
python clinical_trials/pipeline/11_logreg_baseline.py
```

**Secondary ablations:**
```bash
python clinical_trials/pipeline/10_ablation_analysis.py
```

Smoke tests (fast, 2 repeats):
```bash
python clinical_trials/pipeline/09_primary_model.py --n-repeats 2 --force
python clinical_trials/pipeline/11_logreg_baseline.py --n-repeats 2 --force
python clinical_trials/pipeline/10_ablation_analysis.py --n-repeats 2 --force
```

## Analysis Guardrails

- Primary headline model is fixed in: `configs/primary_model_locked.json`
- Logistic baseline is fixed in: `configs/logreg_baseline_locked.json`
- Ablations are secondary analysis only: `configs/secondary_ablation_grid.json`
