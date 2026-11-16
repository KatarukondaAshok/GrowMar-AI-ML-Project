# Dataset (not committed)
Phase 4 runs from the saved models and the small tables in `data/`, so the API, console, monitor and tests need **no workbook**.
To re-create the forecast, the resume screener and the monitor extracts, place the programme workbook here as `02_CortexHR_AI_ML_Dataset.xlsx` (it is not committed) and run:
`python scripts/run_task16.py`, `python scripts/train_phase4_models.py`, `python scripts/export_monitor_extracts.py`.
