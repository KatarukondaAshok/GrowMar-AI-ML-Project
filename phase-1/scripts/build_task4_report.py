import json
import pandas as pd
from reportkit import *

T = "Task4_Anomaly_Detection/"
R = json.load(open(T + "outputs/results.json"))
SW = pd.read_csv(T + "outputs/tables/threshold_sweep.csv")
FIG = T + "outputs/figures/"
FOOT = "Task 4 - Attendance Anomaly Detection | Aura Workforce | Katarukonda Ashok (GM00154)"
c = R["chosen"]; ex = R["external"]; ref = R["reference"]; hr = R["hours_rule"]

s = [P("Task 4: Attendance Anomaly Detection", h1),
     P("Documentation | Isolation Forest, model version attendance-anomaly-1.0.0 | Prepared by Katarukonda Ashok (GM00154) | October 2026", sub)]

s += [P("1. What this task is, in plain words", h2),
      P("HR needs to see unusual attendance quickly: someone absent for two weeks without a note, an impossible 19-hour average day, or a sudden break from a person's normal pattern. "
        "There are no labels saying which months are \"bad\", so this is <b>unsupervised</b> learning: the model learns what normal looks like and scores how far each month is from it."),
      P("The problem with no labels is that we cannot tell whether the model is any good. So we <b>plant known anomalies</b> in a copy of the data, remember exactly which rows we changed, "
        "and measure how many the model finds (<b>catch rate</b>) and how many normal rows it wrongly flags (<b>false-alarm rate</b>).")]

s += [P("2. Data and features", h2),
      P(f"The attendance table has <b>{R['rows']:,} employee-months</b> for {R['employees']} employees (Apr 2025 to Sep 2026), with no missing values. "
        "The feature pipeline (<font name='Courier'>src/features/attendance.py</font>, same fit/transform style as Task 2) keeps eight raw measures and adds six engineered ones:"),
      tbl([["Feature", "Meaning"],
           ["attendance_rate, unpaid_absence_days, longest_unpaid_streak_days, late_arrivals, avg_hours_worked, overtime_hours, half_days_unpaid, paid_leave_days", "Taken as they are"],
           ["present_ratio, wfh_share", "Present days / scheduled days; work-from-home days / present days"],
           ["rate_change_vs_prev, hours_change_vs_prev", "Change from the person's previous month"],
           ["rate_vs_own_mean, hours_vs_own_mean", "Distance from <b>this person's own</b> usual level (learned in fit). This is what detects a sudden pattern change"]],
          [88 * mm, W - 88 * mm])]

s += [P("3. Creating ground truth: the injection script", h2),
      P(f"<font name='Courier'>src/anomaly/inject.py</font> is repeatable with a fixed seed (42): running it twice gives identical rows (checked: {R['repeatable']}). "
        f"The 150 rows labelled in the workbook are set aside first, leaving {R['base_rows']:,} rows treated as normal; then <b>{R['injected']}</b> anomalies are injected into healthy months and logged "
        f"(<font name='Courier'>outputs/tables/injected_rows_log.csv</font> records the id, type, changed columns, old and new values)."),
      tbl([["Injected type", "Rows", "What changes"],
           ["LONG_ABSENCE", R["injected_by_type"]["LONG_ABSENCE"], "7-15 unpaid days in a row; present days and attendance rate fall to match"],
           ["IMPOSSIBLE_HOURS", R["injected_by_type"]["IMPOSSIBLE_HOURS"], "Average hours per day set to 13.5-21"],
           ["PATTERN_CHANGE", R["injected_by_type"]["PATTERN_CHANGE"], "Attendance falls to 50-75%, late arrivals jump, hours dip"]],
          [40 * mm, 18 * mm, W - 58 * mm])]

s += [P("4. How Isolation Forest works", h2),
      P("It builds many random trees. Each tree splits the data on a random feature at a random value. A strange month is separated from the others after only a few splits (a <b>short path</b>); "
        "a normal month is buried in the crowd and needs many splits. The average path length becomes the <b>anomaly score</b>: shorter path, higher score. "
        "The model (300 trees, seed 42) never sees which rows were injected."),
      img(FIG + "score_distribution.png", 105 * mm)]

rows = [["Rows flagged", "Threshold (score)", "Caught", "Missed", "False alarms", "Catch rate", "False-alarm rate", "Precision"]]
for r in SW[SW.flag_rate.isin([0.01, 0.02, 0.03, 0.035, 0.04, 0.05, 0.10, 0.20])].itertuples():
    mark = "<b>" if abs(r.flag_rate - c["flag_rate"]) < 1e-9 else ""
    end = "</b>" if mark else ""
    rows.append([f"{mark}{r.flag_rate:.1%}{end}", f"{r.threshold:.3f}", r.caught, r.missed, r.false_alarms, f"{r.catch_rate:.0%}", f"{r.false_alarm_rate:.1%}", f"{r.precision:.0%}"])
s += [keep(P("5. Threshold tuning", h2),
      P("For each share of rows flagged (the \"contamination\" idea) we recorded how many injected anomalies were caught and how many normal rows were wrongly flagged:"),
      tbl(rows, [24 * mm, 26 * mm, 16 * mm, 16 * mm, 22 * mm, 20 * mm, 28 * mm, W - 152 * mm])),
      Spacer(1, 4), img(FIG + "precision_recall_tradeoff.png", 150 * mm)]

s += [keep(P("6. Chosen threshold and the reason", h2),
           P(f"<b>Flag the top {c['flag_rate']:.1%} most unusual months (anomaly score at least {c['threshold']:.3f}).</b> "
             f"This catches <b>{c['catch_rate']:.0%}</b> of injected anomalies ({c['caught']} of {R['injected']}), wrongly flags only <b>{c['false_alarm_rate']:.1%}</b> of normal rows, "
             f"and has precision {c['precision']:.0%}. That is about <b>{R['reviews_per_month']:.0f} employee-months per calendar month</b> for HR to review."),
           P("<b>Why:</b> rule = best F1 among thresholds that flag at most 5% of rows. Below about 3% the catch rate falls steeply; above about 4% each extra anomaly costs many extra false alarms "
             "and precision falls toward 35% and below. The hint warns that flagging 20% of rows makes HR stop trusting the list, and the right-hand chart shows why: "
             "the false-alarm line stays low up to about 5%, while going to 10% would catch every injected anomaly but flood HR with wrong flags."))]

bt = R["catch_by_type"]
s += [P("7. What it catches and what it misses", h2),
      tbl([["Type", "Share flagged at the chosen threshold"],
           ["LONG_ABSENCE", f"{bt['LONG_ABSENCE']:.0%}"], ["PATTERN_CHANGE", f"{bt['PATTERN_CHANGE']:.0%}"],
           ["IMPOSSIBLE_HOURS", f"{bt['IMPOSSIBLE_HOURS']:.0%}"], ["normal rows (false alarms)", f"{bt['normal']:.1%}"]], [70 * mm, W - 70 * mm]),
      Spacer(1, 3),
      P("Anomalies that move several features at once (absences, pattern changes) are found every time. A single extreme column (impossible hours) is diluted among 14 features, so only about half are caught. "
        f"The fix is a plain rule: <b>average hours above 12</b> catches {hr['caught']} of {hr['of']} impossible-hours rows with {hr['false_alarms']} false alarms (the highest normal month is 11.1 hours). "
        "A good production design is <b>hard rules for impossible values, Isolation Forest for subtle patterns</b>.")]

et = R["external_by_type"]
s += [P("8. Second exam: the labels shipped with the dataset", h2),
      P(f"The workbook has 150 labelled anomalies (absence streaks, impossible hours, late spikes, sudden drops, zero attendance). They were <b>set aside before training</b>, so the model has never seen them. "
        f"Scoring the full original table with the saved model and threshold flags {ex['flagged']} months ({ex['flagged_share']:.1%}): "
        f"<b>{ex['caught']} of the 150 caught ({ex['catch_rate']:.0%})</b>, precision {ex['precision']:.0%}."),
      tbl([["Provided type", "Share flagged"]] + [[k, f"{v:.0%}"] for k, v in et.items()], [70 * mm, W - 70 * mm]),
      Spacer(1, 3),
      P(f"The dataset authors' reference Isolation Forest (3% contamination) reports precision {ref['precision']:.2f} and recall {ref['recall']:.2f}; this model is comparable or slightly better. "
        "The exam also exposed a blind spot the first test could not: <b>late-arrival spikes</b> (13+ late arrivals in a month) are mostly missed, because the injection design had no such type. "
        "That is a reason to add a late-count rule and a LATE_SPIKE injection in the next iteration.")]

s += [P("9. Saved model and how to run it", h2),
      P("Saved with <font name='Courier'>joblib</font>: the full pipeline, the threshold and a metadata file (version, seed, features, threshold rule, catch and false-alarm rates). "
        f"Reloading gives identical scores ({R['reload_identical']})."),
      C("jupyter nbconvert --to notebook --execute --inplace Task4_Anomaly_Detection/task4_anomaly_detection.ipynb"),
      C("from src.anomaly.detect import load_detector, anomaly_scores\n"
        "pipe, threshold, meta = load_detector('Task4_Anomaly_Detection/outputs/artifacts/attendance-anomaly-1.0.0.joblib')\n"
        "flags = anomaly_scores(pipe, new_month_rows) >= threshold")]

s += [keep(P("10. Limitations (say these in the review)", h2),
           B("Injected anomalies are synthetic and fairly obvious, so the first catch rate is optimistic; the provided-label exam is the more realistic number."),
           B("Single-feature anomalies (hours, late spikes) are weak spots and need hard rules."),
           B("Some \"false alarms\" may be real, unlabelled oddities in the data."),
           B("The threshold is a flag rate. Revisit it when HR gives feedback on the flagged list."))]

s += [keep(P("11. Completion criteria", h2),
           checklist([("Injection script repeatable with a fixed seed", "Done", "src/anomaly/inject.py, seed 42; test and notebook confirm identical reruns"),
                      ("Catch rate and false-alarm rate reported", "Done", f"Section 5 table; chosen point {c['catch_rate']:.0%} caught, {c['false_alarm_rate']:.1%} false alarms"),
                      ("Threshold chosen with a written reason", "Done", "Section 6"),
                      ("Model saved with metadata", "Done", "outputs/artifacts/attendance-anomaly-1.0.0.joblib and .meta.json")])),
      P("<b>Next:</b> Phase 2 applies the same thinking to payroll, then wraps all models in a secure, tested service.")]
build(T + "Task4_Documentation.pdf", s, "Task 4 - Attendance Anomaly Detection", FOOT)
