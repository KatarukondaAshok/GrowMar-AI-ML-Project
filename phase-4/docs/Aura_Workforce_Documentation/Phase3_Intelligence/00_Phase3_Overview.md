# Phase 3: Intelligence (Tasks 11 to 15)

**Folder:** `phase-3/` | **Tag:** `phase-3` | **Status:** Built; push and tag next | **Report:** `phase-3/reports/phase3/AIML_AshokKatarukonda_Phase3_Report.pdf`

## What Phase 3 achieves
By Day 15 the project **ranks, explains and checks itself for fairness**: it ranks resumes without looking at who the person is, audits that screener with twin resumes, builds a stronger attrition model and proves (or doubts) the gain, explains every person's risk in plain English, and recommends pay bands that do not copy past bias.

| Task | What it delivers | Document |
|---|---|---|
| 11 Resume Ranking | Masked skill, similarity and experience score; top-5 precision 0.96 | Task 11 document |
| 12 Fairness Audit | Twin test: gap 0.056 to 0.000; honest fairness statement | Task 12 document |
| 13 Attrition v2 | LightGBM on the same folds; small, not proven gain; calibrated | Task 13 document |
| 14 Risk Explanations | 430 rows with score, band and 3 plain-English reasons | Task 14 document |
| 15 Pay Bands | Ridge model, range output, gender and career-gap checks | Task 15 document |

## How the tasks connect
Task 12 audits Task 11's screener. Task 13 returns to attrition with the Phase 1 pipeline and the same folds. Task 14 explains Task 13's model with SHAP. Task 15 applies Task 12's fairness method to pay.

## Headline numbers
- Resume ranking: precision@5 0.96, NDCG@5 0.968 (random order: 0.18 and 0.30).
- Twin gap: 0.056 points before masking, exactly 0 after, with ranking quality unchanged.
- Attrition v2: ROC-AUC 0.725 versus 0.703 on the same folds; wins 3 of 5 folds; gain not proven.
- Risk bands: 43 High, 86 Medium, 301 Low.
- Pay model: MAE 54k versus 116k for the role average; 80.7% of true offers inside the recommended range.
- 15 tests pass, without needing the dataset.

## How to run Phase 3
```bash
cd phase-3
pip install -r requirements-dev.txt
pytest -q
# to recreate every table, figure and model (about 1 minute), put the workbook in data/raw/:
python scripts/run_phase3.py
python scripts/build_phase3_report.py
```

## Still to do
Task 14 read-aloud check of 10 rows, rewriting the report's learning sections in your own words, pushing `phase-3/` and the `phase-3` tag, adding `fobesitech` as a collaborator, and submitting the Phase 3 review form.

## Honest notes
Task 13's gain is within noise. Resume labels look tied to skill coverage. The 25-pair twin test is small. All attrition and pay outputs are support signals for a human, never decisions.
