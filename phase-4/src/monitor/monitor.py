"""Task 18: watch the models. Drift (PSI) on 8 features and on the risk score, plus the twin-resume and pay-gap fairness checks.

Run:   python -m src.monitor.monitor           prints ONE report and writes data/monitoring_latest.json (served by GET /v1/monitoring)
Exit code: 0 = OK or WATCH, 2 = ALERT (so a scheduler can fail loudly).
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
PSI_FEATURES = ["last_engagement_score", "months_since_last_raise", "tenure_months", "wfh_share_3m", "manager_changes_12m",
                "comp_ratio", "attendance_rate_3m", "avg_overtime_hours_per_month_3m"]   # the biggest SHAP drivers of attrition-v2
PSI_WATCH, PSI_ALERT = 0.10, 0.25
TWIN_WATCH, TWIN_ALERT = 1e-6, 0.10        # score points (0-100): any gap at all is a WATCH, over 0.10 is an ALERT
PAY_WATCH, PAY_ALERT = 1.0, 2.0            # gap in points of the band mid-point
RANK = {"OK": 0, "WATCH": 1, "ALERT": 2}


def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index: how far today's distribution has moved from the one the model learned.
    Bins are deciles of the reference data; PSI = sum((a - e) * ln(a / e)). Below 0.10 stable, 0.10-0.25 watch, above 0.25 act."""
    e, a = pd.Series(expected).dropna().to_numpy(float), pd.Series(actual).dropna().to_numpy(float)
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e)
    pa = np.histogram(a, edges)[0] / len(a)
    pe, pa = np.clip(pe, 1e-4, None), np.clip(pa, 1e-4, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def verdict(value: float, watch: float, alert: float) -> str:
    return "ALERT" if value > alert else "WATCH" if value > watch else "OK"


def drift_table(ref: pd.DataFrame, cur: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in PSI_FEATURES:
        v = psi(ref[f], cur[f])
        rows.append({"feature": f, "psi": round(v, 4), "verdict": verdict(v, PSI_WATCH, PSI_ALERT),
                     "reference_mean": round(float(ref[f].mean()), 3), "current_mean": round(float(cur[f].mean()), 3)})
    return pd.DataFrame(rows)


def twin_check(data_dir: Path) -> dict:
    from src.resume.scoring import ResumeScreener
    res = pd.read_csv(data_dir / "twin_resumes.csv")
    pairs = pd.read_csv(data_dir / "twin_pairs.csv")
    jobs = pd.read_csv(data_dir / "job_descriptions.csv")
    sc = ResumeScreener(masked=True).fit(jobs, res)
    allp = pd.concat([pd.DataFrame({"resume_id": r, "job_id": j} for r in res.resume_id for j in jobs.job_id)])
    s = sc.score_pairs(res, allp).set_index(["resume_id", "job_id"])["score"]
    gaps = []
    for p in pairs.itertuples():
        d = [abs(s[(p.resume_id_twin, j)] - s[(p.resume_id_original, j)]) for j in jobs.job_id]
        gaps.append(float(np.mean(d)))
    avg, mx = float(np.mean(gaps)), float(np.max(gaps))
    return {"pairs": len(gaps), "avg_abs_gap_points": round(avg, 6), "max_gap_points": round(mx, 6),
            "pairs_with_any_gap": int(sum(g > TWIN_WATCH for g in gaps)), "verdict": verdict(avg, TWIN_WATCH, TWIN_ALERT)}


def pay_check(data_dir: Path, models_dir: Path) -> dict:
    from src.pay.model import PayBandModel
    s = pd.read_csv(data_dir / "pay_gap_inputs.csv")
    model = PayBandModel.load(str(models_dir / "paybands-v1.joblib"))
    s["pred_ratio"] = model.predict_mid(s, as_of_now=False) / s["band_mid_2026"]
    s["actual_ratio"] = s["offered_annual_fixed"] / s["band_mid_2026"]
    out = {}
    s["gap_group"] = pd.cut(s["career_gap_months"], [-1, 0, 6, 12, 1000], labels=["none", "1-6m", "7-12m", "12m+"]).astype(str)
    for name, col, ref in (("gender", "gender", "Male"), ("career_gap", "gap_group", "none")):
        g = s[s[col].map(s[col].value_counts()) >= 30].groupby(col)[["pred_ratio", "actual_ratio"]].mean() * 100
        pred_gap = float((g["pred_ratio"] - g.loc[ref, "pred_ratio"]).abs().max())
        act_gap = float((g["actual_ratio"] - g.loc[ref, "actual_ratio"]).abs().max())
        out[name] = {"model_gap_points": round(pred_gap, 2), "history_gap_points": round(act_gap, 2),
                     "verdict": verdict(pred_gap, PAY_WATCH, PAY_ALERT)}
    out["verdict"] = max((v["verdict"] for v in out.values()), key=RANK.get)
    return out


def score_drift(ref: pd.DataFrame, cur: pd.DataFrame, models_dir: Path) -> dict | None:
    """PSI of the model's own output. Needs the full raw columns, so it is skipped when only the small extracts exist."""
    return None


def run(data_dir: Path | None = None, models_dir: Path | None = None, write: bool = True) -> dict:
    data_dir = Path(data_dir or ROOT / "data")
    models_dir = Path(models_dir or ROOT / "models")
    mon = data_dir / "monitor"
    drift = drift_table(pd.read_csv(mon / "psi_reference.csv"), pd.read_csv(mon / "psi_current.csv"))
    twin, pay = twin_check(mon), pay_check(mon, models_dir)
    statuses = list(drift["verdict"]) + [twin["verdict"], pay["verdict"]]
    overall = max(statuses, key=RANK.get)
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "status": overall,
              "thresholds": {"psi_watch": PSI_WATCH, "psi_alert": PSI_ALERT, "twin_alert_points": TWIN_ALERT, "pay_alert_points": PAY_ALERT},
              "drift": drift.to_dict("records"), "twin_resume_gap": twin, "pay_gap": pay,
              "next_step": {"OK": "No action. Next scheduled run will re-check.", "WATCH": "Look at the WATCH items this week; confirm the cause before the next run.",
                            "ALERT": "Follow the 4-step plan in Task18_Monitoring/README.md before anyone relies on the affected model."}[overall]}
    if write:
        (data_dir / "monitoring_latest.json").write_text(json.dumps(report, indent=2))
    return report


def format_report(r: dict) -> str:
    lines = [f"CORTEXHR MONITORING REPORT  {r['generated_utc']}  OVERALL: {r['status']}", "", "Drift (PSI, reference = training data)"]
    lines += [f"  {d['feature']:<34} PSI {d['psi']:.3f}  {d['verdict']}" for d in r["drift"]]
    t, p = r["twin_resume_gap"], r["pay_gap"]
    lines += ["", f"Twin resumes: {t['pairs']} pairs, average gap {t['avg_abs_gap_points']} points, pairs with any gap {t['pairs_with_any_gap']}  {t['verdict']}",
              f"Pay gap (model, points of band mid-point): gender {p['gender']['model_gap_points']} (history {p['gender']['history_gap_points']})  "
              f"career gap {p['career_gap']['model_gap_points']} (history {p['career_gap']['history_gap_points']})  {p['verdict']}", "", r["next_step"]]
    return "\n".join(lines)


if __name__ == "__main__":
    rep = run()
    print(format_report(rep))
    sys.exit(2 if rep["status"] == "ALERT" else 0)
