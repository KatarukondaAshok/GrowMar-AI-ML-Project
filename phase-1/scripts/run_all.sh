#!/usr/bin/env bash
# Rebuild every notebook and PDF from scratch (run from the project root). Needs requirements-dev.txt.
set -e
python scripts/build_task1_notebook.py
(cd Task1_Data_Audit && jupyter nbconvert --to notebook --execute --inplace task1_data_audit.ipynb)
python scripts/build_task1_report.py
python scripts/build_task2_notebook.py && python scripts/build_task2_report.py
python scripts/build_task3_notebook.py && python scripts/build_task3_report.py
python scripts/build_task4_notebook.py && python scripts/build_task4_report.py
pytest -q
