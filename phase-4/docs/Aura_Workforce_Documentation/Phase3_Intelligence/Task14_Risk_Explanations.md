# Task 14: Explain Every Person's Risk

**Phase 3 (Intelligence) | Status: Done (read-aloud test pending) | Code:** `src/attrition/explain.py` | **Output:** `outputs/attrition_risk_2026.csv` | **Folder:** `phase-3/Task14_Risk_Explanations`

## 1. Objective
Score all 430 active employees and give the top 3 reasons for each, in plain English, in the format of `sample_attrition_risk`.

## 2. Why it matters
A score alone cannot help an employee. Reasons help HR start a real conversation. Good teams also say on the output itself that drivers are not certainty.

## 3. What was done, step by step
1. Scored the 430 people in `attrition_scoring_2026` with attrition-v2 (calibrated 12-month probability, range 0.023 to 0.448).
2. **Risk bands** by `risk_bands()`: High = top 10% (43 people), Medium = next 20% (86), Low = the rest (301).
3. **SHAP** (TreeExplainer) gives each person's drivers. One-hot columns are added up per feature.
4. A lookup turns each driver into a plain sentence using the person's real numbers, for example "Low engagement score (2.9/5)".
5. Every row's `model_note` says it is a support signal that needs human review.
6. Matched the sample's columns and exported a CSV.

Biggest drivers overall (`figures/shap_summary.png`): engagement score, months since last raise, tenure, work-from-home share, manager changes.

## 4. Output
430 rows with `prediction_id, employee_id, department_id, as_of_date, risk_score_12m, risk_band, top_reason_1..3, model_version, model_note`. Plus `sample_10_rows_to_read_aloud.csv` and the SHAP summary chart.

## 5. Sensitive-data care
- Gender, age and other protected attributes are not features.
- Education is a feature but is **never shown as a reason**.
- Department, location and work mode appear only as context from past data.
- A reason says what the model weighed, not what the person will do.

## 6. Think for yourself: the word "tenure"
Wording used: "Shorter time at the company (13 months): people with less time here left more often in past data". It describes a pattern in past data and does not judge new joiners.

## 7. Still to do (by you): read-aloud check
Read the 10 rows in `sample_10_rows_to_read_aloud.csv` aloud to a friend. Mark any confusing or unfair-sounding reason, change its wording in `TEXT` in `src/attrition/explain.py`, and re-run `python scripts/run_phase3.py`.

## 8. Interview questions
1. **What does SHAP tell you?** How much each feature pushed this person's score up or down compared with an average person.
2. **Why is an explanation not a cause?** It shows what the model used, not why the person might leave.
3. **Why hide some features from reasons?** Not everything the model sees is something HR should discuss or act on.
