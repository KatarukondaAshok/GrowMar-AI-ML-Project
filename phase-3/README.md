# Phase 3: Intelligence (Tasks 11 to 15)

Student: Katarukonda Ashok | Track: AI/ML Engineer (Aura Workforce, CortexHR) | Git tag: `phase-3`

By Day 15 the project **ranks, explains and checks itself for fairness**: it ranks resumes without looking at who the person is, audits that screener with twin resumes, builds a stronger attrition model and proves (or doubts) the gain, explains every person's risk in plain English, and recommends pay bands that do not copy past bias.

## Results at a glance
| Task | Folder | Result |
|---|---|---|
| 11 Resume ranking | [Task11_Resume_Ranking](Task11_Resume_Ranking) | Top-5 precision **0.96**, NDCG@5 0.968 vs human labels (random: 0.18). No name, gender, age, city or college tier in the score |
| 12 Fairness audit | [Task12_Fairness_Audit](Task12_Fairness_Audit) | Twin gap **0.056 -> 0.000** points on all 25 pairs; ranking quality unchanged |
| 13 Attrition v2 | [Task13_Attrition_v2](Task13_Attrition_v2) | ROC-AUC 0.725 vs 0.703 on the same folds: small gain, **not proven** (3 of 5 folds). Calibrated probabilities |
| 14 Risk explanations | [Task14_Risk_Explanations](Task14_Risk_Explanations) | 430 rows with score, band (43 High / 86 Medium / 301 Low) and 3 plain-English reasons, official format |
| 15 Pay bands | [Task15_Pay_Bands](Task15_Pay_Bands) | MAE 54k vs 116k baseline; 80.7% of true offers inside the range; model does not copy the career-gap / gender pay gap |

Report (one PDF): [reports/phase3/AIML_AshokKatarukonda_Phase3_Report.pdf](reports/phase3/AIML_AshokKatarukonda_Phase3_Report.pdf)

## How to run (from inside this folder)
```bash
cd phase-3
pip install -r requirements-dev.txt
pytest -q                          # 15 tests, no dataset needed
# to re-create every table, figure and model, put the workbook in data/raw/ (see data/raw/README.md):
python scripts/run_phase3.py       # about 1 minute
python scripts/build_phase3_report.py
```
Outputs land in each `TaskNN/outputs/`; models are saved in `models/` (`attrition-v2.joblib`, `paybands-v1.joblib`).

## What is where
| Path | Content |
|---|---|
| `src/resume/` | masking, skill and experience extraction, scoring, ranking measures (Tasks 11, 12) |
| `src/attrition/` | attrition v2 training, calibration, SHAP explanations (Tasks 13, 14) |
| `src/pay/` | pay-band model and range (Task 15) |
| `src/fairness.py` | `group_gap()` and `risk_bands()` (the starter `cortexhr_helpers.py` was not supplied, so I wrote them) |
| `src/features/`, `src/models/` | Phase 1 feature pipeline and baselines, copied so this folder runs alone |
| `tests/` | 15 tests: masking in every layout, twin test, bands, reasons, saved models |
| `reports/phase3/` | Phase 3 report PDF and `results.json` |

## Honest notes
- Task 13's gain is within noise; the case for v2 is calibration and explainability, not accuracy.
- Resume labels look tied to skill coverage, so ranking quality is probably optimistic.
- The 25-pair twin test is small; a zero gap there does not prove fairness elsewhere.
- Task 14's "read 10 rows aloud" test and the wording fixes are a TODO for the student (`Task14_Risk_Explanations/README.md`).
- Every attrition and pay output is a support signal for a human, never a decision.
