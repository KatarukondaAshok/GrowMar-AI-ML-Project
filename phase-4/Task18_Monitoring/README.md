# Task 18: Watch the models: drift and fairness monitoring
Code: `src/monitor/monitor.py`. Run: `python -m src.monitor.monitor` (exit code 2 = ALERT). Schedule: `.github/workflows/monitor.yml` (copy to the repo root). Live: `GET /v1/monitoring`. Latest output: `outputs/monitoring_report.txt` and `data/monitoring_latest.json`.

## What it checks
1. **Drift (PSI)** on the 8 biggest SHAP drivers of attrition-v2, between `attrition_dataset` (what the model learned) and `attrition_scoring_2026` (what it scores now). PSI below 0.10 = OK, 0.10 to 0.25 = WATCH, above 0.25 = ALERT.
2. **Twin-resume gap:** re-scores the 25 twin pairs with the masked screener. Any gap is a WATCH; more than 0.10 points an ALERT.
3. **Pay gap:** re-measures the model's predicted pay (as a share of the band mid-point) by gender and by career gap. Under 1 point OK, 1 to 2 WATCH, over 2 ALERT.
It prints one report and returns one overall status (the worst of all checks). The checks run from small ID-free extracts in `data/monitor/` (rebuild with `scripts/export_monitor_extracts.py`), so the scheduled job needs no secrets and no workbook.

## Result of the latest run: OVERALL ALERT (a real finding)
| Feature | PSI | Verdict |
|---|---|---|
| months_since_last_raise | 2.783 | **ALERT** |
| attendance_rate_3m | 1.467 | **ALERT** |
| tenure_months | 0.209 | WATCH |
| comp_ratio | 0.104 | WATCH |
| wfh_share_3m | 0.106 | WATCH |
| last_engagement_score, manager_changes_12m, avg_overtime_hours_per_month_3m | 0.063, 0.001, 0.036 | OK |

Twin resumes: 25 pairs, average gap 0.0, **OK**. Pay gap: gender 0.58 points (history 1.10), career gap 1.68 points (history 2.45), **WATCH**.

**What the numbers show (looked at, not yet explained):** in the 2026 scoring file "months since last raise" is 5 for most people (median 5, 75th percentile 5, max 17), while in the training data it is spread from 0 to 50 (median 9). Attendance rate in the 2026 file is capped at 1.0 and contains zeros, while the training data goes above 1 and has a minimum of 0.20. **My hypotheses, not confirmed:** (a) raises were given together in April 2026 (the data lists an April 2026 bonus cycle), so most people are about 5 months from their raise at the 30 Sep snapshot; (b) the two files define attendance rate differently. Check both with whoever produced the files. Until then, treat attrition-v2 scores as less reliable on these two features.

## Think for yourself: PSI shows an ALERT. 4-step plan
1. **Confirm it is not a data bug:** compare the column definition, units and filters between the two files; look at the raw values of the drifting feature.
2. **Find the cause:** split by department, month and employee type; check events (a raise cycle, a policy change, a new attendance device).
3. **Measure the damage:** re-score with the model and compare the risk bands and the top reasons with the last run; if the feature is a top driver, the scores may be wrong.
4. **Decide and write it down:** if it is a data bug, fix the data; if the world changed, recalibrate or retrain and re-run the fairness checks; if unsure, tell HR to treat scores as indicative only. Record who decided, when, and why, and re-run the monitor.

## Schedule
Mondays 03:30 UTC and on demand, via GitHub Actions (`monitor.yml`; it uploads the report as an artifact). **Still to do:** copy the workflow to `.github/workflows/` at the repo root, push, open the Actions tab, press "Run workflow" once and screenshot the run.
