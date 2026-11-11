# Task 3 - Attrition Baseline Models & Evaluation

**Documentation:** `Task3_Documentation.pdf` | **Notebook:** `task3_baseline_models.ipynb` | **Code:** `src/models/baselines.py`

## What it does
Trains a dummy ("everyone stays"), logistic regression and a depth-4 decision tree (plain and class-weighted), each behind the Task 2 feature pipeline. 5-fold stratified cross-validation, mean +/- spread for precision, recall, F1, ROC-AUC. Confusion matrix, ROC curve, an HR "review the top X%" table, and one final check on the locked test set.

## Key results
| Model | Recall | Precision | ROC-AUC |
|---|---|---|---|
| Dummy | 0.00 | 0.00 | 0.50 (accuracy 85%) |
| Logistic Regression (balanced) | 0.61 | 0.25 | 0.70 |
| Decision Tree depth 4 (balanced) | 0.56 | 0.22 | 0.64 |

Chosen baseline: **balanced logistic regression**. Locked test set: ROC-AUC 0.69, recall 0.59, consistent with cross-validation. Outputs: `outputs/tables/baseline_table.csv`, `outputs/figures/`.

## Likely review questions
1. **Precision or recall for HR?** Recall: a missed leaver is a lost chance; a false alarm costs one conversation. Watch precision so HR still trusts the list.
2. **What do class weights do?** Make a missed leaver cost about 6x more, which raises recall and lowers precision; ranking (ROC-AUC) stays the same.
3. **Why refit the pipeline inside each fold?** Otherwise validation rows influence medians and caps (leakage).
4. **Why lock the test set?** To get one honest, unbiased final score.
5. **Why is the dummy 85% accurate?** Because 85% of employees stay. That is why accuracy is not the metric.
