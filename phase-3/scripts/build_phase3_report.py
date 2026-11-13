"""Build the one-file Phase 3 report.  python scripts/build_phase3_report.py"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from reportkit import P, B, tbl, img, h1, h2, sub, build, W, Spacer
from reportlab.lib.units import mm
ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "reports/phase3/results.json").read_text())
REPO = "https://github.com/KatarukondaAshok/GrowMar-AI-ML-Project/tree/main/phase-3"
OUT = ROOT / "reports/phase3/AIML_AshokKatarukonda_Phase3_Report.pdf"
q = R["task11"]["quality"]["Final screener (masked): skills + similarity + experience"]
t13 = R["task13"]; t12 = R["task12"]; t14 = R["task14"]
s = [P("Phase 3 Report: Intelligence", h1),
     P(f"Name: Ashok Katarukonda | Track: AI/ML Engineer (Aura Workforce, CortexHR) | Phase: 3 | Git tag: <b>phase-3</b><br/>Repository: <a href='{REPO}' color='blue'>{REPO}</a>", sub),
     P("1. Task status and evidence", h2),
     tbl([["Task", "Status", "Evidence (path in the repository)"],
          ["11 Resume ranking", "Done", "phase-3/Task11_Resume_Ranking/outputs/tables/ranking_quality.csv, misranks_explained.csv"],
          ["12 Fairness audit", "Done", "phase-3/Task12_Fairness_Audit/outputs/tables/fairness_summary.csv"],
          ["13 Attrition v2", "Done", "phase-3/Task13_Attrition_v2/outputs/tables/comparison_table.csv, tuning_log.csv"],
          ["14 Risk explanations", "Done (read-aloud test pending)", "phase-3/Task14_Risk_Explanations/outputs/attrition_risk_2026.csv"],
          ["15 Pay bands", "Done", "phase-3/Task15_Pay_Bands/outputs/tables/group_gap_table.csv"]],
         [40 * mm, 38 * mm, W - 78 * mm]),
     P("2. Proof it works", h2),
     B("<b>Tests:</b> <font name='Courier'>pytest -q</font> gives <b>15 passed</b> (masking in every resume layout, twin resumes, risk bands 43/86/301, plain-English reasons, saved models, pay band inside role band)."),
     B(f"<b>Task 11:</b> top-5 precision <b>{q['precision_at_5']}</b>, NDCG@5 {q['ndcg_at_5']}, Spearman {q['spearman']} against the human labels (random order: 0.18 / 0.30). No name, gender, age, city or college tier is used."),
     B(f"<b>Task 12:</b> twin gap before masking {t12[0]['mean_abs_gap_points']} points (all 25 pairs differ, max {t12[0]['max_abs_gap_points']}); after masking <b>0.000</b> (0 of 25). Ranking quality unchanged ({t12[1]['precision_at_5']} / {t12[1]['ndcg_at_5']})."),
     tbl([["Attrition model (5-fold CV, same folds)", "ROC-AUC", "PR-AUC", "Precision top 30%", "Recall top 30%"],
          ["Task 3 baseline: LogReg balanced", "0.703", "0.301", "0.273", "0.559"], ["Attrition v2: LightGBM", "0.725", "0.339", "0.284", "0.581"]],
         [62 * mm, 24 * mm, 24 * mm, 32 * mm, W - 142 * mm]),
     B(f"<b>Task 13:</b> per-fold gain +0.022 on average but spread 0.036 and 3 of 5 folds won, so the gain is <b>not proven</b>. Calibration: Brier 0.114 vs 0.125 base rate; calibration error 0.028 to 0.020. No leakage (strongest single feature AUC 0.61)."),
     B(f"<b>Task 14:</b> 430 rows in the official format; bands {t14['bands']['High']} High / {t14['bands']['Medium']} Medium / {t14['bands']['Low']} Low; three plain-English reasons each; every row says it needs human review."),
     B("<b>Task 15:</b> pay error (MAE) Rs 54,293 vs Rs 115,964 for the role-average baseline; 80.7% of true offers fall inside the recommended range. A model that sees career gap and gender copies the past gap (-2.5 points); the final job-only model does not."),
     Spacer(1, 3), img(str(ROOT / "Task14_Risk_Explanations/outputs/figures/shap_summary.png"), W * 0.55),
     P("3. What I learned", h2),
     B("Fairness is measured, not assumed: even without gender, names and tier words moved scores until I masked them."),
     B("A fix must be tested in every layout: cutting the first lines of a resume missed names in two-column and table layouts."),
     B("Compare models fold by fold: a +0.02 gain with 0.04 spread is not a proven gain."),
     B("Explanations show drivers, not certainty, and must be worded so they do not judge the person (tenure, education)."),
     P("4. Problems I hit and how I solved them", h2),
     tbl([["Problem", "Fix"],
          ["Masking by position left twin gaps in 4 pairs", "Removed name, city and e-mail by value; added a two-column test"],
          ["First risk reasons printed model-scaled numbers (e.g. -1.5/5)", "Used the person's real values; missing values leave the number out"],
          ["Pay range covered only 43% of offers", "Clipping to the 2026 band hid older, lower offers; evaluate without clipping (80.7%), clip only new offers"],
          ["Starter helpers (cortexhr_helpers.py) not supplied", "Wrote group_gap() and risk_bands() with tests"],
          ["v2 is not clearly better than the baseline", "Reported it honestly; ship for calibration and explainability with the baseline as fallback"]],
         [75 * mm, W - 75 * mm]),
     P("5. Plan for the next phase", h2),
     B("Phase 4: forecasting and putting everything live in the same repository (folder phase-4)."),
     B("To add before submitting: read 10 risk rows aloud to a friend and fix confusing wording (Task 14).")]
build(str(OUT), s, "Phase 3 Report", "Ashok Katarukonda | Aura Workforce Phase 3 | tag phase-3")
