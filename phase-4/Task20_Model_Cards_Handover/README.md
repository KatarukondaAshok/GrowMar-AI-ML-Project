# Task 20: Model cards, final audit and handover
| Deliverable | File |
|---|---|
| Model card for every model (7) | `model_cards/*.md` (purpose, use and not-use, data, method, metrics, fairness, limits, monitoring) |
| Architecture picture | `ARCHITECTURE.md`, `architecture.png` |
| Release notes (included, not included, known limits) | `../RELEASE_NOTES.md` |
| Final audit: every evaluation re-run | tests: `test_results.txt` (37 passed); forecast, monitor and smoke outputs in Tasks 16 to 18 |
| Demo plan and consolidated report outline | `DEMO_SCRIPT.md`, `CONSOLIDATED_REPORT_OUTLINE.md` |
| Final Challenge (yours) | `FINAL_CHALLENGE.md` |
| Phase 4 report PDF (one file) | `../reports/phase4/AIML_AshokKatarukonda_Phase4_Report.pdf` |

## Final audit: numbers match
The numbers in the model cards come from the saved results of Tasks 6 to 16 and the live smoke test (`/v1/health` lists the same versions as the cards). Re-run order: `pytest -q`, `python scripts/run_task16.py`, `python -m src.monitor.monitor`, `python smoke_test.py URL`.

## Think for yourself: which model would I not release?
The **monthly attrition-rate forecast**: with only a few exits per month its backtest error is about 68%, so it looks like information but is mostly noise. The next one I would hold back is attrition-v2 *scores for 2026* until the monitor's ALERT on "months since last raise" and "attendance rate" is explained.

## Still to do (only you can do these)
Tag `v1.0.0`, record the 5-minute demo (`DEMO_SCRIPT.md`), do the Final Challenge on its own branch, fill the consolidated report and submit it with the live links and video in the Google Form (not on the site).
