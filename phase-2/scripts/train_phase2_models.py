"""Train and save every Phase 2 model into models/ (payroll detector, intent classifier, chat resources).

Run from the project root:  python scripts/train_phase2_models.py
The attendance detector (Phase 1, Task 4) is copied from Task4_Anomaly_Detection/outputs/artifacts.
No label is used for training anywhere: anomaly labels only measure, intent labels train the classifier on the
training sheet only (the test sheet is never fitted on).
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.chat.nlu import IntentClassifier  # noqa: E402
from src.chat.resources import build_resources, save_resources  # noqa: E402
from src.payroll.detector import PayrollDetector  # noqa: E402

CORTEX = ROOT / "data/raw/02_CortexHR_AI_ML_Dataset.xlsx"
LEDGER = ROOT / "data/raw/01_LedgerForge_Python_Master_Dataset.xlsx"
MODELS = ROOT / "models"
INTENT_THRESHOLD = 0.5     # chosen in Task 9 from grouped cross-validation on training data only


def train_attendance():
    """Re-run the Task 4 recipe (same seed, same rule) so the saved model matches the installed scikit-learn."""
    from src.anomaly.detect import anomaly_scores, choose_threshold, make_detector, save_detector, sweep_thresholds
    from src.anomaly.inject import SEED, inject_anomalies
    att = pd.read_excel(CORTEX, sheet_name="attendance_monthly_features")
    labels = pd.read_excel(CORTEX, sheet_name="attendance_anomaly_labels")
    base = att[~att.attendance_feature_id.isin(labels.attendance_feature_id)].reset_index(drop=True)
    inj, _ = inject_anomalies(base, seed=SEED)
    det = make_detector(n_estimators=300).fit(inj)
    best = choose_threshold(sweep_thresholds(anomaly_scores(det, inj), inj.is_injected.values), max_flag_rate=0.05)
    save_detector(det, best.threshold, str(MODELS), extra={"trained_on_rows": len(inj), "chosen_flag_rate": float(best.flag_rate)})


def main():
    MODELS.mkdir(exist_ok=True)
    pay = pd.read_excel(CORTEX, sheet_name="payroll_monthly_features")
    PayrollDetector().fit(pay).save(str(MODELS))
    train_attendance()
    tr = pd.read_excel(CORTEX, sheet_name="chatbot_nlu_training")
    clf = IntentClassifier(threshold=INTENT_THRESHOLD).fit(tr["text"], tr["intent"])
    clf.save(str(MODELS / "intent-tfidf-lr-1.0.0.joblib"), {"threshold": INTENT_THRESHOLD, "trained_on": len(tr)})
    depts = pd.read_excel(LEDGER, sheet_name="departments")["department_name"].tolist() if LEDGER.exists() else []
    roles = pd.read_excel(LEDGER, sheet_name="roles")["role_title"].tolist() if LEDGER.exists() else []
    save_resources(build_resources(str(CORTEX), departments=depts, extra_roles=roles), str(MODELS / "chat_resources.json"))
    print("models saved:", sorted(p.name for p in MODELS.iterdir()))


if __name__ == "__main__":
    main()
