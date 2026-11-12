"""Build the one-file Phase 2 report.  python scripts/build_phase2_report.py"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from reportkit import P, B, tbl, img, h1, h2, sub, build, W, Spacer
from reportlab.lib.units import mm
ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "reports/phase2/results.json").read_text())
REPO = "https://github.com/KatarukondaAshok/GrowMar-AI-ML-Project/tree/main/phase-2"
OUT = ROOT / "reports/phase2/AIML_AshokKatarukonda_Phase2_Report.pdf"
s = [P("Phase 2 Report: Core Development", h1),
     P(f"Name: Ashok Katarukonda | Track: AI/ML Engineer (Aura Workforce, CortexHR) | Phase: 2 | Git tag: <b>phase-2</b><br/>Repository: <a href='{REPO}' color='blue'>{REPO}</a>", sub),
     P("1. Task status and evidence", h2),
     tbl([["Task", "Status", "Evidence (path in the repository)"],
          ["6 Payroll detector", "Done", "phase-2/Task6_Payroll_Detector/outputs/tables/per_error_type_results.csv, figures/recall_by_error_type.png"],
          ["7 Reason codes + money", "Done (friend test pending)", "phase-2/Task7_Reason_Codes/outputs/tables/flags_20_rows.csv"],
          ["8 Anomaly API", "Done", "phase-2/Task8_Anomaly_API/outputs/curl_session.md, tests/test_api.py"],
          ["9 Intent classifier", "Done", "phase-2/Task9_HR_Intent_Classifier/outputs/figures/confusion_matrix.png"],
          ["10 Slots, memory, permissions", "Done", "phase-2/Task10_Chatbot_Slots_Permissions/outputs/transcripts.md"]],
         [38 * mm, 32 * mm, W - 70 * mm]),
     P("2. Proof it works", h2),
     B("<b>Tests:</b> <font name='Courier'>pytest -q</font> gives <b>110 passed</b> (34 from Phase 1, 76 new): rules, output format, API (11), intent model, slots, 12 dialogues, refusals, mock server."),
     B("<b>Task 6:</b> rules + Isolation Forest catch <b>98 of 100</b> injected payroll errors (recall 0.98) against the baseline's 0.40; rules alone reach precision 1.00. Of 130 model-only flags just 5 were real, so the rules do the work."),
     tbl([["Detector", "Flagged", "Caught", "Precision", "Recall"],
          ["Reference baseline", "184", "40", "0.217", "0.40"], ["Model alone (my ratios)", "184", "59", "0.321", "0.59"],
          ["Rules alone", "93", "93", "1.000", "0.93"], ["Rules + model", "223", "98", "0.439", "0.98"]],
         [60 * mm, 25 * mm, 25 * mm, 28 * mm, W - 138 * mm]),
     Spacer(1, 4), img(str(ROOT / "Task6_Payroll_Detector/outputs/figures/recall_by_error_type.png"), W * 0.78),
     B("<b>Task 7:</b> 223 flags in the official 12 columns (match the sample sheet), ranked by money at risk; top flag PI-003304, Rs 21,02,520."),
     B("<b>Task 8:</b> <font name='Courier'>POST /v1/anomaly/payroll</font> and <font name='Courier'>/attendance</font> return the official format; bad input gives 422; models load once."),
     B(f"<b>Task 9:</b> test accuracy <b>{R['task9']['accuracy_always_answer']*100:.1f}%</b>; with the 0.5 confidence threshold the bot answers {R['task9']['answered_share']*100:.0f}% at {R['task9']['accuracy_when_answered']*100:.0f}% accuracy and politely falls back on the rest."),
     B("<b>Task 10:</b> 12 sample dialogues pass; someone else's salary is refused every time (employee and HR, 10 test cases) with no data call; missing month triggers a question."),
     P("3. Mini challenge: a flagged payslip, raw data to API answer", h2),
     B("Row PI-003304 (EMP-0304, Dec 2025): gross 28,23,640 vs fixed pay 1,97,044. Ratio <i>gross_vs_expected</i> = 10.0."),
     B("Rule GROSS_ABOVE_NORM fires (limit 1.5); reason: \"Gross is 10.0x what the fixed pay and days worked allow\"; money at risk Rs 21,02,520; ranked 1 of 223."),
     B("Returned by the API in the official format (see curl_session.md)."),
     P("4. What I learned", h2),
     B("Readable rules beat a clever model on known payroll mistakes; the model is a safety net."),
     B("Every flag needs a reason and a rupee amount, or nobody acts on it."),
     B("A confidence threshold turns wrong confident answers into honest \"I am not sure\"."),
     B("Permissions belong in server code that uses the real role, never in the message text."),
     P("5. Problems I hit and how I solved them", h2),
     tbl([["Problem", "Fix"],
          ["Double payments and stale salaries look normal row by row (model recall 0%)", "Added context rules (same employee-month, pay vs formula)"],
          ["Model adds 125 false alarms (precision 0.44)", "Rules ranked first; model-only flags marked STATISTICAL_OUTLIER, low priority"],
          ["Test accuracy 81.8% vs 94.2% in cross-validation", "Reported the test number; chose threshold from grouped CV; plan more paraphrases"],
          ["Bot answered \"How is LOP calculated?\" with a PF answer", "Near-exact FAQ match overrides an unsure model; personal-data intents never overridden"],
          ["Starter api_skeleton and mock_services not supplied", "Built my own FastAPI app and mock HR system from the workbooks"]],
         [75 * mm, W - 75 * mm]),
     P("6. Plan for the next phase", h2),
     B("Phase 3 (Tasks 11-15): resume ranking, twin-resume fairness audit, attrition v2, SHAP reasons, fair pay bands; tag phase-3."),
     B("To add before submitting: friend feedback on the 20 flags (Task 7).")]
build(str(OUT), s, "Phase 2 Report", "Ashok Katarukonda | Aura Workforce Phase 2 | tag phase-2")
