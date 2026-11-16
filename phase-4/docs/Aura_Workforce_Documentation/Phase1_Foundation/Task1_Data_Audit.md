# Task 1: Data Audit and Exploration

**Phase 1 (Foundation) | Status: Done | Code:** `src/audit/checks.py` | **Notebook:** `task1_data_audit.ipynb` | **Tests:** `tests/test_audit_checks.py` | **Report:** `Task1_Documentation.pdf`

## 1. Objective
Check whether the employee data is fit to model, before building anything. Task 1 reads the `attrition_dataset` sheet, finds every problem in it, and writes down a decision for each problem.

## 2. Why it matters
A model is only as good as its data. Wrong or leaking data gives a model that looks excellent in a notebook and fails in real use. Every issue found here protects the next three tasks.

## 3. Input data
Sheet `attrition_dataset` of the CortexHR workbook: 2,389 rows and 42 columns, one row per employee observation. The column to predict is `left_within_12m` (did the person leave within 12 months).

## 4. What was done, step by step
1. **Structure and data dictionary:** shape, column types, and a plain-language meaning for every column.
2. **Duplicates:** found 6 exact duplicate rows and removed them, leaving 2,383 employees. This must happen before the train/test split, otherwise a copy lands in the test set and inflates the score.
3. **Missing values:** most gaps mean "not applicable" (never promoted, no raise yet, not reviewed yet). Decision: fill and add a "was missing" flag instead of dropping, because the gap itself carries meaning.
4. **Outliers:** compared two rules. IQR flagged 93 salaries and z-score flagged 26, but only 3 were real typos (10 times too high). Pay is skewed, so a business rule, not a statistic, decides what is an error.
5. **Impossible values:** listed values that cannot be true (saved in `impossible_values.csv`).
6. **Class balance:** 14.6% of employees left. Always predicting "stays" is 85.4% accurate and useless.
7. **Leakage:** found 6 columns that reveal the answer (exit date, resignation date, exit reason, exit interview, settlement amount, voluntary-leaving flag). A model using them scores ROC-AUC 1.00 instead of about 0.71.
8. **Decision log:** every issue became a written decision, 26 in total (D1 to D20 and M1 to M6), saved in `decision_log.csv`.

## 5. Deliverables
- Notebook `task1_data_audit.ipynb` (reruns top to bottom)
- Report `Task1_Documentation.pdf` (3 pages, plain language)
- Tables in `outputs/tables/`: `decision_log.csv`, `missing_values.csv`, `outlier_comparison.csv`, `class_balance.csv`, `leakage_columns.csv`, `impossible_values.csv`
- Charts in `outputs/figures/`: class balance, missing values, salary outliers, leakage demo

## 6. Key results
| Finding | Number |
|---|---|
| Rows after removing duplicates | 2,383 |
| Written decisions | 26 |
| Employees who left | 14.6% |
| Leakage columns | 6 |
| Real salary typos among 93 IQR flags | 3 |

## 7. How to run
```bash
pip install -r requirements.txt
# open task1_data_audit.ipynb in VS Code, select the .venv kernel, Run All
pytest -q
```
The dataset must be at `data/raw/02_CortexHR_AI_ML_Dataset.xlsx`.

## 8. Interview questions
1. **A column has 40% missing. Drop, fill or flag?** First find out why it is missing. If the gap means something (for example "never promoted"), fill it and add a "was missing" flag. Drop only if it is mostly empty and unrelated to the target, or if it leaks.
2. **Why did IQR and z-score disagree?** Pay and overtime are skewed, so IQR over-flags. A business rule decides what is an error.
3. **What is data leakage?** Using information that would not exist at prediction time, like studying with the answer key.
4. **Why is accuracy misleading here?** Only 14.6% leave, so "everyone stays" scores 85.4%.
5. **Why remove duplicates before splitting?** Otherwise a copy lands in the test set and the score is inflated.

## 9. Limits
The audit uses the data as exported. The workbook also holds answer-key sheets; findings here were produced from the data first and compared with those sheets only afterwards.
