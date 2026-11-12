"""Re-create every table, figure and transcript used in the Phase 2 report.  python scripts/run_phase2_evidence.py"""
import json
import sys
import warnings
from datetime import date, datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.chat.nlu import IntentClassifier, TypoFixer, clean_text  # noqa: E402
from src.payroll.detector import PayrollDetector  # noqa: E402
from src.payroll.evaluate import TYPE_ORDER, overall, per_type  # noqa: E402
from src.payroll.reasons import REASONS, build_results  # noqa: E402

DATA = ROOT / "data/raw/02_CortexHR_AI_ML_Dataset.xlsx"
SHEETS = ["payroll_monthly_features", "payroll_anomaly_labels", "anomaly_baselines", "sample_anomaly_results",
          "chatbot_nlu_training", "chatbot_test_set"]
X = pd.read_excel(DATA, sheet_name=SHEETS)
RESULTS = {}


def task6_and_7():
    pay, lab, base = X["payroll_monthly_features"], X["payroll_anomaly_labels"], X["anomaly_baselines"]
    t6, t7 = ROOT / "Task6_Payroll_Detector/outputs", ROOT / "Task7_Reason_Codes/outputs"
    det = PayrollDetector().fit(pay)
    s = det.score(pay); s.index = pay["payroll_item_id"].to_numpy()
    truth = s.index.to_series().isin(lab["payroll_item_id"])
    b = base[base["task"].str.startswith("Payroll")].iloc[0]
    comp = pd.DataFrame([
        {"detector": "Reference baseline (Isolation Forest, 2.5%)", "flagged": int(b.flagged), "caught": int(round(b.recall * 100)),
         "precision": b.precision, "recall": b.recall},
        {"detector": "Model alone (my ratios, Isolation Forest, 2.5%)", **overall(s["model_flag"], truth)},
        {"detector": "Rules alone", **overall(s["rule_flag"], truth)},
        {"detector": "Rules + model (final)", **overall(s["flagged"], truth)}])
    comp.to_csv(t6 / "tables/detector_comparison.csv", index=False)
    pt = per_type(s, lab); pt.to_csv(t6 / "tables/per_error_type_results.csv", index=False)
    model_only = s["model_flag"] & ~s["rule_flag"]
    extra = {"model_only_flags": int(model_only.sum()), "model_only_true": int((model_only & truth).sum()),
             "rule_flags": int(s["rule_flag"].sum()), "rule_false_alarms": int((s["rule_flag"] & ~truth).sum())}
    # examples of flags the rules caught (one per rule) and what the model alone missed on those rows
    rows = []
    for code in [c for c in s.columns if c.startswith("rule_") and c not in ("rule_flag", "rule_code")]:
        ids = s.index[s[code] & truth]
        if len(ids):
            i = ids[0]; r = pay.set_index("payroll_item_id").loc[i]
            rows.append({"rule": code[5:], "payroll_item_id": i, "employee_id": r.employee_id, "gross_pay": r.gross_pay,
                         "net_pay": r.net_pay, "model_flagged_it": bool(s.loc[i, "model_flag"])})
    pd.DataFrame(rows).to_csv(t6 / "tables/rule_examples.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 4)); w = 0.27; xs = np.arange(len(pt))
    for k, (col, lab_, c) in enumerate([("recall_model_alone", "model alone", "#7aa6c2"), ("recall_rules_alone", "rules alone", "#e0a458"),
                                       ("recall_combined", "rules + model", "#4c8c4a")]):
        ax.bar(xs + (k - 1) * w, pt[col], w, label=lab_, color=c)
    ax.set_xticks(xs); ax.set_xticklabels(pt["error_type"], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("recall (share of injected errors caught)"); ax.set_ylim(0, 1.08); ax.legend(ncol=3, fontsize=8)
    ax.set_title("Payroll errors caught, by error type"); plt.tight_layout(); plt.savefig(t6 / "figures/recall_by_error_type.png", dpi=140); plt.close()
    # Task 7 outputs
    when = datetime(2026, 10, 1, 2, 0)
    res = build_results(pay, det, scored=det.score(pay), scored_at=when)
    res.to_csv(t7 / "tables/payroll_flags_official_format.csv", index=False)
    res.head(20).to_csv(t7 / "tables/flags_20_rows.csv", index=False)
    pd.DataFrame([{"reason_code": k, "sentence_template": v} for k, v in REASONS.items()]).to_csv(t7 / "tables/reason_codes.csv", index=False)
    sample = X["sample_anomaly_results"]
    RESULTS["task6"] = {"comparison": comp.to_dict("records"), "extra": extra, "flags_total": int(s["flagged"].sum())}
    RESULTS["task7"] = {"rows": len(res), "columns_match_sample": list(res.columns) == list(sample.columns),
                        "ranked_desc": bool(res["money_at_risk"].is_monotonic_decreasing),
                        "money_total_rule_flags_rs": float(res[res.reason_code != "STATISTICAL_OUTLIER"].money_at_risk.sum())}
    # mini challenge: follow one real flagged payslip all the way to an API answer
    return det, pay, res


def task9():
    tr, te = X["chatbot_nlu_training"], X["chatbot_test_set"]
    out = ROOT / "Task9_HR_Intent_Classifier/outputs"
    thr = 0.5
    m = IntentClassifier(threshold=thr).fit(tr["text"], tr["intent"])
    P = m.predict_proba(te["text"]); pred = m.classes_[P.argmax(1)]; conf = P.max(1)
    acc = accuracy_score(te["expected_intent"], pred)
    answered = conf >= thr
    RESULTS["task9"] = {"test_messages": len(te), "accuracy_always_answer": round(acc, 3),
                        "threshold": thr, "answered_share": round(float(answered.mean()), 3),
                        "accuracy_when_answered": round(float((pred[answered] == te["expected_intent"].values[answered]).mean()), 3),
                        "fallback_share": round(float((~answered).mean()), 3)}
    fx = TypoFixer().fit([clean_text(t) for t in tr["text"]])
    groups = pd.factorize(pd.Series([fx.fix(clean_text(t)) for t in tr["text"]]))[0]
    allc = np.array(sorted(tr["intent"].unique())); oof = np.zeros((len(tr), len(allc)))
    for a, b in GroupKFold(5).split(tr["text"], tr["intent"], groups):
        mm = IntentClassifier().fit(tr["text"].iloc[a], tr["intent"].iloc[a]); Pp = mm.predict_proba(tr["text"].iloc[b])
        for j, c in enumerate(mm.classes_):
            oof[b, list(allc).index(c)] = Pp[:, j]
    cv_pred, cv_conf = allc[oof.argmax(1)], oof.max(1)
    rows = []
    for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        a = cv_conf >= t
        rows.append({"threshold": t, "answered_share": round(a.mean(), 3), "accuracy_when_answered": round((cv_pred[a] == tr["intent"].values[a]).mean(), 3),
                     "wrong_confident_answers": int((cv_pred[a] != tr["intent"].values[a]).sum()),
                     "test_answered_share": round(float((conf >= t).mean()), 3),
                     "test_accuracy_when_answered": round(float((pred[conf >= t] == te["expected_intent"].values[conf >= t]).mean()), 3)})
    pd.DataFrame(rows).to_csv(out / "tables/threshold_choice.csv", index=False)
    rep = pd.DataFrame(classification_report(te["expected_intent"], pred, output_dict=True, zero_division=0)).T
    rep.round(3).to_csv(out / "tables/classification_report_test.csv")
    errs = te.assign(predicted=pred, confidence=conf.round(2))[pred != te["expected_intent"].values]
    errs[["text", "expected_intent", "predicted", "confidence"]].to_csv(out / "tables/test_errors.csv", index=False)
    pairs = errs.groupby(["expected_intent", "predicted"]).size().sort_values(ascending=False).rename("count").reset_index()
    pairs.to_csv(out / "tables/most_confused_pairs.csv", index=False)
    labels = sorted(set(te["expected_intent"]) | set(pred))
    cm = confusion_matrix(te["expected_intent"], pred, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8.5)); ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6); ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("predicted"); ax.set_ylabel("expected"); ax.set_title(f"Intent confusion matrix on the 132 test messages (accuracy {acc:.1%})")
    plt.tight_layout(); plt.savefig(out / "figures/confusion_matrix.png", dpi=140); plt.close()
    RESULTS["task9"]["top_confused"] = pairs.head(5).to_dict("records")
    RESULTS["task9"]["grouped_cv_accuracy"] = round(float((cv_pred == tr["intent"].values).mean()), 3)


if __name__ == "__main__":
    task6_and_7(); task9()
    (ROOT / "reports/phase2").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports/phase2/results.json").write_text(json.dumps(RESULTS, indent=2, default=str))
    print(json.dumps(RESULTS, indent=2, default=str))
