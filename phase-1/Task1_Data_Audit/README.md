# Task 1 - Data Audit & Exploration

**Documentation:** `Task1_Documentation.pdf` (3 pages, plain language) | **Notebook:** `task1_data_audit.ipynb` | **Code:** `src/audit/checks.py` | **Tests:** `tests/test_audit_checks.py`

## What it does
Profiles the `attrition_dataset` sheet before any modelling: structure and data dictionary, duplicates, missing values, outliers (IQR vs z-score), impossible values, class imbalance and leakage. Every issue ends in a written decision (26 in total: D1-D20, M1-M6) saved in `outputs/tables/decision_log.csv`.

## Key results
- 2,389 rows, 42 columns; 6 exact duplicates removed -> 2,383 employees.
- Missing values are mostly "not applicable" (never promoted, no raise yet, not reviewed yet): flag and fill, do not drop.
- IQR flags 93 salaries and z-score 26, but only 3 are real typos (10x too high).
- 14.6% left; "nobody leaves" is 85.4% accurate and useless.
- 6 columns leak the answer (exit date, resignation date, exit reason, exit interview, settlement, voluntary flag): a model using them scores ROC-AUC 1.00 instead of about 0.71.

## Likely review questions
1. **A column has 40% missing. Drop, fill or flag?** Find out why it is missing first. If the gap means something, fill and add a "was missing" flag. Drop only if mostly empty and unrelated to the target, or leaking.
2. **Why did IQR and z-score disagree?** Pay and overtime are skewed; IQR over-flags. A business rule, not a statistic, says what is an error.
3. **What is data leakage?** Using information that would not exist at prediction time, like studying with the answer key.
4. **Why is accuracy misleading here?** Only 14.6% leave, so always predicting "stays" scores 85.4%.
5. **Why remove duplicates before splitting?** Otherwise a copy lands in test and the score is inflated.
