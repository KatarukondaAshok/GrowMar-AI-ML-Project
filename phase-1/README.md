# Aura Workforce - Phase 1: Data Foundations & Baseline Models

**Student:** Katarukonda Ashok (GM00154) | **Domain:** AI/ML Engineer | **First review:** 17 Oct 2026

Everything for Phase 1 is in this one folder. There are four tasks, each in its own folder with a **documentation PDF**, a **notebook**, and an **outputs/** folder.

```
Aura_Phase1_Foundation/
|-- README.md                      <- you are here
|-- Task1_Data_Audit/              Task 1
|   |-- Task1_Documentation.pdf        the audit report (read this first)
|   |-- task1_data_audit.ipynb         the rerunnable notebook
|   |-- README.md                      what it is, key results, review questions
|   `-- outputs/ (figures, tables)
|-- Task2_Feature_Pipeline/        Task 2
|   |-- Task2_Documentation.pdf
|   |-- task2_feature_pipeline.ipynb
|   |-- README.md
|   `-- outputs/ (artifacts/features-1.0.0.joblib + metadata, results.json)
|-- Task3_Baseline_Models/         Task 3
|   |-- Task3_Documentation.pdf
|   |-- task3_baseline_models.ipynb
|   |-- README.md
|   `-- outputs/ (figures, tables, results.json)
|-- Task4_Anomaly_Detection/       Task 4
|   |-- Task4_Documentation.pdf
|   |-- task4_anomaly_detection.ipynb
|   |-- README.md
|   `-- outputs/ (artifacts/attendance-anomaly-1.0.0.joblib + metadata, figures, tables)
|-- src/                           shared, reusable code used by all notebooks
|   |-- audit/checks.py                Task 1 audit helpers
|   |-- features/                      Task 2 pipeline (config, data, transformers, pipeline, cli) + attendance.py (Task 4)
|   |-- models/baselines.py            Task 3 helpers
|   `-- anomaly/                       Task 4 (inject.py, detect.py)
|-- tests/                         automated tests (pytest) for tasks 1, 2 and 4
|-- data/raw/                      02_CortexHR_AI_ML_Dataset.xlsx (the dataset used)
|-- scripts/                       scripts that rebuild the notebooks and PDFs (optional)
|-- Dockerfile, requirements.txt, requirements-dev.txt, pyproject.toml, pytest.ini
```

## The four tasks in one table

| Task | Question it answers | Main result | Documentation |
|---|---|---|---|
| 1 Data Audit | Is the data fit to model? | 2,383 employees after removing 6 duplicates; 26 written decisions (D1-D20, M1-M6); 14.6% left; 6 leakage columns found | `Task1_Data_Audit/Task1_Documentation.pdf` |
| 2 Feature Pipeline | Can every model prepare data the same way? | `features-1.0.0`: 74 model-ready columns, 0 missing values, fitted on train only, saved and reloadable | `Task2_Feature_Pipeline/Task2_Documentation.pdf` |
| 3 Baseline Models | What score must later models beat? | Balanced logistic regression: ROC-AUC about 0.70, recall about 0.61 (locked test confirms) | `Task3_Baseline_Models/Task3_Documentation.pdf` |
| 4 Anomaly Detection | Which attendance months look unusual? | Isolation Forest flags 3.5% of months, catching 85% of injected anomalies with 1.9% false alarms | `Task4_Anomaly_Detection/Task4_Documentation.pdf` |

The tasks build on each other: Task 1 decisions become Task 2 code, Task 2 feeds Task 3, and Task 4 reuses the same fit/transform style.

## How to run it

### Option A: VS Code (easiest)
1. Open this folder in VS Code (File > Open Folder). Install the *Python* and *Jupyter* extensions.
2. Create an environment and install the packages (about 500 MB):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate            # Windows   (Mac/Linux: source .venv/bin/activate)
   pip install --no-cache-dir -r requirements.txt
   ```
3. Open any `taskN_*.ipynb`, choose **Select Kernel > .venv**, click **Run All**. Notebooks find the project root themselves, so open them from any folder.
4. Run the tests: `pytest -q` (expect all passed).

### Option B: Docker
```bash
docker build -t aura-attrition .
docker run --rm -p 8888:8888 -v "$PWD:/app" aura-attrition                    # Jupyter Lab at http://localhost:8888
docker run --rm -v "$PWD:/app" aura-attrition pytest -q                       # tests
docker run --rm -v "$PWD:/app" aura-attrition python -m src.features.cli run --output-dir artifacts   # Task 2 in one command
```

### Task 2 in one command (no notebook)
```bash
python -m src.features.cli run --input data/raw/02_CortexHR_AI_ML_Dataset.xlsx --output-dir artifacts
```

## Suggested order for a student
1. Read `Task1_Documentation.pdf`, then run `task1_data_audit.ipynb`.
2. Read Task 2, run its notebook, open `src/features/` to see the same logic as code.
3. Read Task 3, run it, look at `outputs/figures/`.
4. Read Task 4, run it, open `outputs/tables/injected_rows_log.csv` to see exactly what was injected.
5. Read each task's README for likely review questions and practise answering in your own words.

## Submitting on the portal
Attach the whole folder as a zip, or a GitHub link to it, plus the four documentation PDFs if the form allows several files.

## Notes and honesty
- The workbook contains extra sheets (`answer_key_planted_dirt`, `answer_key_notes`, `baseline_benchmarks`, `anomaly_baselines`, provided anomaly labels). Findings here were produced from the data first and only compared with those sheets afterwards. The provided anomaly labels are used in Task 4 only as a second, independent exam.
- Docker files are provided but were not executed in the environment where this folder was prepared. Run `docker build` once before your review.
- All data is fictional (Zenith Nova Technologies), supplied for training.

## Troubleshooting
| Problem | Fix |
|---|---|
| `No space left on device` during `pip install` | Free disk space, `pip cache purge`, delete `.venv` and recreate; use `--no-cache-dir` |
| `ModuleNotFoundError: src` | Open the folder that contains `src/` (this one). `pytest.ini` already sets the path |
| `FileNotFoundError ... xlsx` | Dataset must be at `data/raw/02_CortexHR_AI_ML_Dataset.xlsx` |
| Kernel not listed in VS Code | `pip install ipykernel`, then reload the window |

Dataset: place 02_CortexHR_AI_ML_Dataset.xlsx in data/ (provided by the programme).
