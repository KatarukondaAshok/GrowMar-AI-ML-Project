# Aura Workforce (CortexHR): Project Overview

**Student:** Katarukonda Ashok (GM00154) | **Track:** AI/ML Engineer | **Programme:** GrowMar, Aura Workforce | **Repository:** https://github.com/KatarukondaAshok/GrowMar-AI-ML-Project

## 1. What this project is
Aura Workforce is an HR analytics project for a fictional company, Zenith Nova Technologies (all data is made up for training). The AI/ML track builds, step by step, a system that:

- **understands the data** and prepares it safely (Phase 1),
- **detects mistakes** in payroll and attendance, explains each one, serves them through an API and answers HR questions in a chatbot (Phase 2),
- **ranks resumes, checks itself for fairness, predicts who may leave, explains why, and recommends fair pay bands** (Phase 3).

Everything uses one dataset, the workbook `02_CortexHR_AI_ML_Dataset.xlsx` (employee attrition, attendance, payroll, chatbot messages, resumes and salary records).

## 2. The three phases and 14 tasks at a glance
| Phase | Theme | Task | What it does | Main result |
|---|---|---|---|---|
| 1 | Foundation | 1 Data Audit | Finds data problems and writes 26 decisions | 2,383 clean employees; 6 leakage columns found |
| 1 | Foundation | 2 Feature Pipeline | One saved object that prepares data the same way every time | 74 model-ready columns, 0 missing |
| 1 | Foundation | 3 Baseline Models | Sets the score later models must beat | Balanced logistic regression: ROC-AUC 0.70, recall 0.61 |
| 1 | Foundation | 4 Anomaly Detection | Finds unusual attendance months | 85% of injected anomalies caught, 1.9% false alarms |
| 2 | Core development | 6 Payroll Detector | Catches wrong payslips | 98 of 100 injected errors caught |
| 2 | Core development | 7 Reason Codes | Reason, sentence and rupees at risk per flag | 223 flags in the official format |
| 2 | Core development | 8 Anomaly API | Serves detectors through FastAPI | Validated endpoints, 422 on bad input |
| 2 | Core development | 9 Intent Classifier | Understands HR questions, safe fallback | 81.8% test accuracy; 93.1% when it answers |
| 2 | Core development | 10 Chatbot Slots and Permissions | Memory, mock HR calls, role rules | 12 dialogues pass; others' salary always refused |
| 3 | Intelligence | 11 Resume Ranking | Ranks resumes without using identity | Top-5 precision 0.96 |
| 3 | Intelligence | 12 Fairness Audit | Twin-resume test and fix | Twin gap 0.056 to 0.000 |
| 3 | Intelligence | 13 Attrition v2 | Gradient boosting versus the baseline | Gain small and not proven; calibrated |
| 3 | Intelligence | 14 Risk Explanations | Top 3 plain-English reasons for 430 people | 43 High, 86 Medium, 301 Low |
| 3 | Intelligence | 15 Pay Bands | Fair pay range recommendation | MAE 54k versus 116k baseline; 80.7% inside range |

The programme's Phase 2 task sheet refers to a "Task 5". It is not part of the Phase 1 material delivered here (Phase 1 has four tasks). If your portal lists a Task 5, add its document to Phase 1.

## 3. How the pieces connect
```
 Attrition chain
   Task 1 audit -> Task 2 pipeline -> Task 3 baseline -> Task 13 v2 -> Task 14 reasons

 Anomaly and chatbot chain
   Task 4 attendance -> Task 6 payroll -> Task 7 reasons -> Task 8 API -> Tasks 9-10 chatbot

 Hiring and pay chain
   Task 11 ranking -> Task 12 fairness audit -> Task 15 fair pay bands
```
- Task 1 decisions become Task 2 code; Task 2 feeds Tasks 3 and 13.
- Task 4's idea (Isolation Forest plus rules) carries into Task 6; Task 7 makes the flags understandable; Task 8 serves them; Tasks 9 and 10 let people ask about them.
- Task 12 tests Task 11; Task 15 reuses the fairness method.

## 4. Repository layout
```
GrowMar-AI-ML-Project/
  README.md      project overview and phase table
  phase-1/       Task1_Data_Audit ... Task4_Anomaly_Detection, src, tests, Dockerfile
  phase-2/       Task6 ... Task10, app, src, mock_services, models, tests, reports/phase2
  phase-3/       Task11 ... Task15, src, models, tests, reports/phase3
```
Git tags: `phase-1`, `phase-2` (pushed) and `phase-3` (push after the Phase 3 commit). The programme dataset is not committed; place it at `data/raw/02_CortexHR_AI_ML_Dataset.xlsx` inside each phase folder to rerun everything.

## 5. Tools and methods used
Python, pandas, scikit-learn (pipelines, cross-validation, Isolation Forest, logistic regression, TF-IDF), LightGBM, SHAP, FastAPI and Pydantic, joblib, pytest, Docker files, Git and GitHub.

## 6. Honest limitations (read before presenting)
- All errors and anomalies are **synthetic**, so real-world recall would be lower.
- Phase 2's rules do most of the payroll work; the model adds little (5 of 130 model-only flags were real).
- Attrition v2's gain over the baseline is **within noise**; its value is calibration and explainability.
- Resume labels are closely tied to skill coverage, so ranking quality is probably optimistic; the 25-pair twin test is small.
- Every attrition, ranking and pay output is a **support signal for a human**, never a decision.
- The programme's starter files (`api_skeleton`, `mock_services`, `cortexhr_helpers.py`, the vault unlock sheets) were not supplied, so equivalents were built from the workbook.

## 7. Where the project stands today
| Item | Status |
|---|---|
| Phase 1 pushed to GitHub (`phase-1/`, tag `phase-1`) | Done |
| Phase 2 pushed to GitHub (`phase-2/`, tag `phase-2`) | Done |
| Phase 3 folder and report built | Done, push and tag `phase-3` |
| Task 7 friend test (`FRIEND_FEEDBACK.md`) | To do |
| Task 14 read-aloud check of 10 rows | To do |
| Rewrite the "What I learned" sections in your own words | To do |
| Add FOBES review account (`fobesitech`) as repository collaborator | To do |
| Upload report PDFs and submit the Phase 3 review form | To do |

## 8. Daily routine and next steps
- **Keep marking attendance every weekday: login 9:00 AM, logout 6:00 PM.** Task and project work is flexible; only these timings are fixed.
- Open the Growmar site daily and check Tasks and Attendance.
- If you have already submitted, use the time to study the Aura Workforce overview document and practise the interview questions in each task document, out loud, in your own words.
- Phase 4 (forecasting and going live) comes next; it will be a new `phase-4/` folder in the same repository.

## 9. Glossary
| Term | Plain meaning |
|---|---|
| Data leakage | Using information that would not exist at prediction time, like studying with the answer key |
| Pipeline | One saved object that applies the same preparation steps in training and production |
| Cross-validation | Training and testing several times on different slices to get a steadier score |
| Precision | Of the people flagged, how many were right |
| Recall | Of the people who really left or erred, how many were caught |
| ROC-AUC | How well the model ranks risky cases above safe ones (0.5 = random, 1.0 = perfect) |
| Isolation Forest | An anomaly method: odd points are isolated in fewer random splits |
| TF-IDF | A text score that rewards distinctive words and discounts common ones |
| SHAP | A way to show how much each feature pushed one prediction up or down |
| Calibration | Making "0.20" really mean about 20 in 100 |
| Counterfactual (twin) test | Change one irrelevant detail and check the output does not change |
| Slot | A detail the chatbot must find in a message, such as the month or a person's name |
