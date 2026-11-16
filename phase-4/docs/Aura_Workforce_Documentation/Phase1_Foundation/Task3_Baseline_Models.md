# Task 3: Attrition Baseline Models and Evaluation

**Phase 1 (Foundation) | Status: Done | Code:** `src/models/baselines.py` | **Notebook:** `task3_baseline_models.ipynb` | **Report:** `Task3_Documentation.pdf`

## 1. Objective
Set the score that every later model has to beat, and learn which metric to trust when only 14.6% of people leave.

## 2. Why it matters
Without a baseline, nobody can say whether a fancy model helps. Without the right metric, "85% accurate" looks good while catching nobody.

## 3. Input data
The Task 2 pipeline output on the training rows. The 20% test set stays locked until the final check.

## 4. What was done, step by step
1. Trained five baselines, each behind the Task 2 pipeline: dummy ("everyone stays"), logistic regression, logistic regression with balanced class weights, decision tree depth 4, and decision tree depth 4 with balanced class weights.
2. **5-fold stratified cross-validation** (shuffled, seed 42). The pipeline is refitted inside every fold so validation rows never influence medians or caps.
3. Reported mean and spread of precision, recall, F1 and ROC-AUC.
4. Drew the confusion matrix and the ROC curve.
5. Built an HR "review the top X%" table (how many leavers sit in the top 5%, 10%, 20%, 30% of scores).
6. One final check on the locked test set.

## 5. Deliverables
Notebook, `Task3_Documentation.pdf`, `outputs/tables/baseline_table.csv`, charts in `outputs/figures/`, `outputs/results.json`.

## 6. Key results
| Model | Recall | Precision | ROC-AUC |
|---|---|---|---|
| Dummy | 0.00 | 0.00 | 0.50 (accuracy 85%) |
| Logistic Regression (balanced) | 0.61 | 0.25 | 0.70 |
| Decision Tree depth 4 (balanced) | 0.56 | 0.22 | 0.64 |

**Chosen baseline: balanced logistic regression.** On the locked test set: ROC-AUC 0.69 and recall 0.59, consistent with cross-validation.

## 7. How to run
Open `task3_baseline_models.ipynb`, select the `.venv` kernel, Run All.

## 8. Think for yourself: precision or recall for HR?
Recall. A missed leaver is a lost chance to help. A false alarm costs one conversation. Precision still matters, because if the list is full of false alarms HR stops trusting it.

## 9. Interview questions
1. **What do class weights do?** A missed leaver counts about 6 times more, which raises recall and lowers precision. Ranking (ROC-AUC) stays the same.
2. **Why refit the pipeline inside each fold?** Otherwise validation rows influence medians and caps (leakage).
3. **Why lock the test set?** To get one honest, unbiased final score.
4. **Why is the dummy 85% accurate?** Because 85% of employees stay. That is why accuracy is not the metric.

## 10. Limits
ROC-AUC of about 0.70 means the model ranks people only moderately well. It is a support signal for HR, not a prediction of what a person will do.
