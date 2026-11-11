# Task 2 - Reusable Feature Pipeline (`features-1.0.0`)

**Documentation:** `Task2_Documentation.pdf` | **Notebook:** `task2_feature_pipeline.ipynb` | **Code:** `src/features/` | **Tests:** `tests/test_features.py`

## What it does
One saved object turns raw exported data into model-ready features: repairs impossible values, fixes salary typos, adds engineered features and "was missing" flags, caps outliers, fills gaps, scales numbers and one-hot encodes categories. Leakage columns and IDs are dropped. It learns **only from training rows**.

## Files in `src/features/`
`config.py` (column lists) | `data.py` (load, de-duplicate, split) | `transformers.py` (RowCleaner, SalaryTypoFixer, FeatureEngineer, Winsorizer) | `pipeline.py` (build, fit, save, load) | `cli.py` (one-command run) | `attendance.py` (used by Task 4)

## Run
```bash
python -m src.features.cli run --input data/raw/02_CortexHR_AI_ML_Dataset.xlsx --output-dir artifacts
```
Saved pipeline and metadata: `outputs/artifacts/features-1.0.0.joblib` and `.meta.json`.

## Key results
74 output columns (29 numeric, 4 flags, 41 one-hot), 0 missing values, loaded pipeline gives identical output, 15 tests for this task.

## Likely review questions
1. **Why a Pipeline?** Same steps in training and production, no leakage inside cross-validation, one object to save.
2. **What does "fit on train only" mean?** Medians, caps, category lists and department pay levels are computed from training rows, then only applied to test rows.
3. **Why is `pay_vs_dept_median` risky?** The department median must come from train; the notebook shows how it differs from the all-rows median.
4. **How is the version stored?** `features-1.0.0` is in the saved file's metadata, the `.meta.json`, and the package.
5. **How do you prove same input gives same output?** Tests transform twice (and after save/load) and compare the frames.
