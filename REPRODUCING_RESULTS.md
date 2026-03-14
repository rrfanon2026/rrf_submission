# Reproducing Paper Results

This guide explains how to reproduce the figures and tables from the paper.

## Prerequisites

```bash
pip install -r requirements.txt
```

## Option A: Run the Notebook (Recommended)

Open and run all cells in `notebooks/paper_reproduction.ipynb`:

```bash
jupyter notebook notebooks/paper_reproduction.ipynb
```

This is the fastest way to reproduce the paper results. The notebook reads precomputed CSV artifacts and generates all figures and tables. It does not make any API calls or require external access — it uses only standard Python libraries (pandas, numpy, matplotlib).

## Option B: Offline Primary Model Verification

To verify the primary model results from precomputed predictions:

```bash
python scripts/reproduce_paper.py
```

This reads precomputed LLM predictions, runs the locked ensemble/cross-validation logic deterministically, and writes a verification bundle to `repro/`. No API key is required.

To also run secondary ablations:

```bash
python scripts/reproduce_paper.py --run-secondary-ablations
```

## Option C: End-to-End with LLM API Calls

To re-run the full pipeline from scratch (requires an OpenAI API key and incurs API costs):

```bash
export OPENAI_API_KEY="sk-..."

# Step 1: Generate questions
python scripts/01_question_generation.py --provider openai --model gpt-4o-mini

# Step 2: Generate predictions
python scripts/02_test_questions.py --provider openai --model gpt-4o-mini --mode questions

# Step 3: Run primary model evaluation
python scripts/04_primary_model.py
```

**Note:** LLM outputs are stochastic, so exact numbers will differ slightly from the paper. The precomputed artifacts in Options A and B ensure exact reproduction.

## Why Precomputed Artifacts?

RRF uses LLMs for question generation and scoring. LLM outputs are **non-deterministic** and **API-dependent**: the same prompt can produce different responses across runs, and results depend on model versions that may change over time. The repository ships CSV snapshots of all LLM-generated outputs so that the paper results can be reproduced exactly without requiring API access.

## Artifact Map

The table below maps each section of the paper reproduction notebook to the precomputed artifacts it reads. Section names correspond to markdown headers in the notebook.

| Notebook Section | Artifact Path(s) | Paper Output |
|---|---|---|
| **Setup** | `precomputed/gpt_4o_mini/ablation_results/` (sets base path) | — |
| **Optimisation Target Comparison** | `precomputed/gpt_4o_mini/ablation_results/nested_cv_llm__anonymised*.csv` (filtered by hamming_0_15 + sortbyf0_5) | Table: optimisation target comparison |
| **Main Results: RRF vs Baselines** | `precomputed/gpt_4o_mini/ablation_results/nested_cv_llm__anonymised*hamming_0_15*.csv`, `precomputed/gpt_4o_mini/benchmark_results/benchmark_gpt_4o_mini.csv`, `precomputed/gpt_4o_mini/vanilla/vanilla_*.csv` (6 files) | Table: main results |
| **Question-Generating Model Comparison** | `precomputed/gpt_4o_mini/ablation_results/*.csv`, `precomputed/claude_sonnet_4_5/ablation_results/*.csv`, `precomputed/gemini_2_5_pro/ablation_results/*.csv`, `precomputed/gpt_5_2/ablation_results/*.csv`, `precomputed/*/benchmark_results/*.csv` | Table: cross-model comparison |
| **Combined Ablation Figure** | `precomputed/gpt_4o_mini/ablation_results/nested_cv_llm__anonymised*.csv` (multiple variants: similarity metric, threshold, chronological), `nested_cv_llm_expert__anonymised*.csv`, `nested_cv_expert_only__anonymised*.csv` | Figure: 5-panel ablation figure |
| **Optimisation Target Ablation (Detailed)** | `precomputed/gpt_4o_mini/ablation_results/nested_cv_llm__anonymised*.csv` (filtered) | Table: detailed ablation metrics |
| **Hyperparameter Sensitivity** | `precomputed/gpt_4o_mini/grid_search_results_anonymised.csv`, `precomputed/gpt_4o_mini/ablation_results/nested_cv_llm__anonymised*hamming_0_15*.csv` | Figure: 2-panel hyperparameter sensitivity |
| **Clinical Trials: TOP Phase I Results** | `clinical_trials/results/phase_I/primary_results/primary_model_repeat_metrics.csv`, `clinical_trials/results/phase_I/logreg_results/logreg_repeat_metrics.csv` | Table: clinical trial results |
| **Clinical Trials: Operating-Point Diagnostics** | Same as above | Table: operating-point diagnostics (appendix) |
| **Clinical Trials: Recall–Specificity Pareto** | `clinical_trials/results/phase_I/ablation_results/beta_sweep_summary.csv` | Figure: recall–specificity Pareto (appendix) |

## Regression Tests

To verify that precomputed artifacts have not been inadvertently modified:

```bash
python -m unittest tests.test_precomputed_regression -v
```

This checks file checksums against a golden manifest (`tests/regression_manifest.json`).
