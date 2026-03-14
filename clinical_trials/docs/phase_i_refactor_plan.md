# Phase I Refactor Plan (Curated)

This curation pass only updates documentation tables in this repository. The imported
snapshot under `clinical_trials/legacy_snapshot/` is read-only, and the original
legacy source folders under ``
were not modified.

## Scope

- Prioritize Phase I artifacts for paper migration.
- Keep Phase II/III and stability outputs as future-reference provenance.
- Mark duplicates/legacy variants for merge-and-archive.

## Immediate next execution order

1. Canonicalize Phase I scripts 01-04 into a minimal deterministic pipeline.
2. Build `09_primary_model.py` (locked config) for Phase I RRF headline results.
3. Build `10_ablation_analysis.py` for secondary sensitivity studies.
4. Keep logistic-regression baseline in a separate explicit script.

## Progress

- Step 1 complete: canonical wrappers created for stages 01-04 in
  `clinical_trials/pipeline/`.
- Legacy snapshot remains immutable; wrappers call or reuse legacy logic without
  editing snapshot files.
- Step 2 complete: locked primary script added at
  `clinical_trials/pipeline/09_primary_model.py` with fixed config at
  `clinical_trials/configs/primary_model_locked.json`.
- Step 3 complete: secondary ablation script added at
  `clinical_trials/pipeline/10_ablation_analysis.py` with fixed grid at
  `clinical_trials/configs/secondary_ablation_grid.json`.
- Step 4 complete: logistic baseline script added at
  `clinical_trials/pipeline/11_logreg_baseline.py` with fixed config at
  `clinical_trials/configs/logreg_baseline_locked.json`.

## Guardrails to retain

- Fixed primary config defined before running.
- Secondary analyses cannot overwrite headline model definition.
- Table-ready metrics printed directly by scripts.
