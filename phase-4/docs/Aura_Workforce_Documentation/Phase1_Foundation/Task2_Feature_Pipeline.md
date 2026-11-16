# Task 2: Reusable Feature Pipeline (`features-1.0.0`)

**Phase 1 (Foundation) | Status: Done | Code:** `src/features/` | **Notebook:** `task2_feature_pipeline.ipynb` | **Tests:** `tests/test_features.py` | **Report:** `Task2_Documentation.pdf`

## 1. Objective
Turn the Task 1 decisions into code: one saved object that converts raw exported data into model-ready numbers, the same way every time.

## 2. Why it matters
If training and production prepare data differently, the model silently breaks. One pipeline object guarantees identical steps, prevents leakage during cross-validation, and can be saved and reloaded.

## 3. Input data
The cleaned `attrition_dataset` (2,383 rows after de-duplication), split 80/20 into a training part and a locked test part (stratified, seed 42).

## 4. What was done, step by step
1. **RowCleaner:** repairs impossible values found in Task 1.
2. **SalaryTypoFixer:** fixes pay that is more than 5 times what the role's comp ratio implies.
3. **FeatureEngineer:** adds new columns: log pay, pay versus department median, tenure in years, attendance gap, overtime per year of tenure, plus four "was missing" flags (never promoted, no raise yet, not reviewed yet, engagement missing).
4. **Winsorizer:** caps extreme values at the 1st and 99th percentile.
5. **Fill, scale, encode:** median fill, scaling for numbers, one-hot encoding for categories (department, location, work mode, employment type, education).
6. **Drop** leakage columns and ID columns.
7. **Fit on training rows only.** Medians, caps, category lists and department pay levels come from training rows and are only applied to test rows.
8. **Save** as `features-1.0.0.joblib` with a `.meta.json` (version, columns, training size).

## 5. Deliverables
- `src/features/`: `config.py` (column lists), `data.py` (load, de-duplicate, split), `transformers.py`, `pipeline.py` (build, fit, save, load), `cli.py` (one-command run), `attendance.py` (used by Task 4)
- Saved pipeline and metadata in `outputs/artifacts/`
- Notebook and `Task2_Documentation.pdf`

## 6. Key results
74 output columns (29 numeric, 4 flags, 41 one-hot), 0 missing values, a reloaded pipeline gives identical output, and 15 tests cover this task.

## 7. How to run
```bash
python -m src.features.cli run --input data/raw/02_CortexHR_AI_ML_Dataset.xlsx --output-dir artifacts
pytest -q
```

## 8. Interview questions
1. **Why a Pipeline?** Same steps in training and production, no leakage inside cross-validation, and one object to save.
2. **What does "fit on train only" mean?** Medians, caps, category lists and department pay levels are learned from training rows, then only applied to test rows.
3. **Why is `pay_vs_dept_median` risky?** The department median must come from training rows only, or information from the test set leaks in.
4. **How is the version stored?** `features-1.0.0` is in the saved file's metadata, in the `.meta.json` and in the package.
5. **How do you prove same input gives same output?** Tests transform twice, and again after save and load, and compare the results.

## 9. Limits
The pipeline encodes decisions made on this dataset; a different export with new columns needs a new version number.
