# Task 14: Explain every person's risk
Code: `src/attrition/explain.py`, run via `scripts/run_phase3.py`. **Output: `outputs/attrition_risk_2026.csv`, 430 rows, same columns as `sample_attrition_risk`.**

## Steps
1. Score the 430 people in `attrition_scoring_2026` with attrition-v2 (calibrated 12-month probability, range 0.023 to 0.448).
2. `risk_bands()` (in `src/fairness.py`): **High = top 10% (43), Medium = next 20% (86), Low = the rest (301)**, the rule stated in the sample.
3. SHAP (TreeExplainer) gives each person's drivers. One-hot columns are added up per feature (one reason for "work location", not one per city).
4. A lookup turns each driver into a plain sentence using the person's real numbers, e.g. "Low engagement score (2.9/5)". Risk-raising drivers come first; low-risk people get their top drivers too (drivers, not certainty).
5. Every row's `model_note` says it is a support signal that needs human review.
6. Biggest drivers overall (`figures/shap_summary.png`): engagement score, months since last raise, tenure, work-from-home share, manager changes.

## Sensitive-data care
- Gender, age and other protected attributes are not features.
- Education is a feature but **never shown as a reason**.
- Department, location and work mode appear only as "the team/location context has seen more leavers in past data".
- Explanations show drivers, not certainty: a reason says what the model weighed, not what the person will do.

## Think for yourself: "tenure"
Wording used: "Shorter time at the company (13 months): people with less time here left more often in past data". It describes a pattern in past data and does not judge new joiners.

## Read-aloud check (TODO for you)
Read the 10 rows in `outputs/sample_10_rows_to_read_aloud.csv` aloud to a friend. Mark any confusing or unfair-sounding reason and change its wording in `TEXT` in `src/attrition/explain.py`, then re-run `python scripts/run_phase3.py`.

## Interview questions
1. *What does SHAP tell you?* How much each feature pushed this person's score up or down compared with an average person.
2. *Why is an explanation not a cause?* It shows what the model used, not why the person might leave.
3. *Why hide some features from reasons?* Not everything the model sees is something HR should discuss or act on.
