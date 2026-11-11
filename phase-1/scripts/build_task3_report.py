import json
import pandas as pd
from reportkit import *

T = "Task3_Baseline_Models/"
R = json.load(open(T + "outputs/results.json"))
S = pd.read_csv(T + "outputs/tables/cv_summary_full.csv").set_index("model")
LIFT = pd.DataFrame(R["lift"]); TH = pd.read_csv(T + "outputs/tables/threshold_tradeoff.csv")
FIG = T + "outputs/figures/"
FOOT = "Task 3 - Attrition Baseline Models | Aura Workforce | Katarukonda Ashok (GM00154)"
ch = R["chosen_model"]; tm = R["test_metrics"]; cm = R["oof_confusion"]; tc = R["test_confusion"]; ts = R["time_split"]


def pm(model, m): return f"{S.loc[model, m + '_mean']:.3f} +/- {S.loc[model, m + '_std']:.3f}"


s = [P("Task 3: Attrition Baseline Models and Evaluation", h1),
     P("Documentation | Prepared by Katarukonda Ashok (GM00154) | October 2026", sub)]

s += [P("1. What this task is, in plain words", h2),
      P("A <b>baseline</b> is the first honest score. Every smarter model built later has to beat it, otherwise we cannot prove an improvement. "
        "We predict who will leave within 12 months using simple, explainable models: a dummy that always says \"stays\", a logistic regression and a shallow decision tree, "
        "each with and without class weights. All of them use the Task 2 feature pipeline."),
      P(f"<b>The challenge:</b> only {R['leaver_rate']:.1%} of employees leave. A model that says \"nobody leaves\" is <b>{R['dummy_accuracy']:.1%} accurate</b> and finds no one, "
        "so accuracy is the wrong yardstick. We use precision, recall, F1 and ROC-AUC.")]

s += [P("2. How it was done (and the rules that keep it honest)", h2),
      B(f"Stratified 80/20 split ({R['train_rows']:,} train, {R['test_rows']:,} test). The <b>test set is locked</b> and used exactly once, at the end."),
      B("Each model is <i>feature pipeline + model</i> in one object, so cross-validation refits the cleaning, caps, medians and encodings <b>inside every fold</b>."),
      B("5-fold stratified cross-validation; every metric is reported as mean +/- spread (standard deviation over the 5 folds)."),
      B("Class weights (balanced) tried for both model types and compared with the plain versions.")]

models = list(S.index)
rows = [["Model", "Precision", "Recall", "F1", "ROC-AUC", "Accuracy"]]
for m in models:
    rows.append([("<b>" + m + "</b>") if m == ch else m, pm(m, "precision"), pm(m, "recall"), pm(m, "f1"), pm(m, "roc_auc"), pm(m, "accuracy")])
s += [P("3. Results: cross-validated baseline table", h2),
      tbl(rows, [50 * mm, 25 * mm, 25 * mm, 25 * mm, 25 * mm, W - 150 * mm]),
      Spacer(1, 3),
      P("<b>How to read it.</b> <i>Precision</i>: of the people flagged as leavers, how many truly left. <i>Recall</i>: of the people who truly left, how many we flagged. "
        "<i>ROC-AUC</i>: how well the model ranks leavers above stayers (0.5 is a coin flip). The plain models look accurate only because they rarely predict \"leaves\" (recall near 0.1)."),
      img(FIG + "model_comparison.png", 150 * mm)]

s += [P("4. Class weights: what changed", h2),
      P(f"Setting <font name='Courier'>class_weight=\"balanced\"</font> tells the model that missing a leaver costs about 6 times more than a false alarm. "
        f"For logistic regression, recall rises from {S.loc['Logistic Regression','recall_mean']:.2f} to {S.loc['Logistic Regression (balanced)','recall_mean']:.2f}, "
        f"precision falls from {S.loc['Logistic Regression','precision_mean']:.2f} to {S.loc['Logistic Regression (balanced)','precision_mean']:.2f}, and ROC-AUC stays near "
        f"{S.loc['Logistic Regression (balanced)','roc_auc_mean']:.2f}. Weights move the cut-off, not the ranking: the same people are ranked highest, but more of them are flagged."),
      P(f"<b>Chosen baseline: {ch}.</b> Rule written before touching the test set: among balanced models within 0.01 ROC-AUC of the best, take the one with the highest recall. "
        f"It also beats the decision tree on ROC-AUC ({S.loc[ch,'roc_auc_mean']:.2f} vs {S.loc['Decision Tree depth 4 (balanced)','roc_auc_mean']:.2f}) and is easier to explain.")]

s += [keep(P("5. Confusion matrix and ROC curve", h2), img(FIG + "confusion_and_roc.png", 150 * mm)),
      P(f"Out-of-fold on the training rows: the model caught <b>{cm['tp']} of {cm['tp'] + cm['fn']}</b> leavers ({cm['tp'] / (cm['tp'] + cm['fn']):.0%}) and wrongly flagged "
        f"{cm['fp']} of {cm['tn'] + cm['fp']} stayers ({cm['fp'] / (cm['tn'] + cm['fp']):.0%}). The bottom-left cell (missed leavers) is the costly mistake; the top-right cell is a wasted conversation.")]

lt = [["Review the top...", "People reviewed", "Leavers found", "Share of all leavers found", "Hit rate in that group", "Better than random by"]]
for r in R["lift"]:
    lt.append([f"{r['review_top_pct']}% highest risk", r["people_reviewed"], r["leavers_found"], f"{r['share_of_all_leavers_found']:.0%}", f"{r['precision_in_group']:.0%}", f"{r['lift_vs_random']:.1f}x"])
s += [keep(P("6. What it means for HR", h2),
           P("HR cannot talk to everyone. If they only review the highest-risk people, the model finds far more leavers than random picking would:"),
           tbl(lt, [30 * mm, 24 * mm, 22 * mm, 34 * mm, 30 * mm, W - 140 * mm])),
      Spacer(1, 3),
      P(f"The model is a <b>screening aid</b>, not a verdict: the typical hit rate in the highest-risk group is about 30-40%, so most flagged people will not leave. "
        "Use it to prioritise conversations, never to judge an individual."),
      img(FIG + "top_drivers.png", 100 * mm),
      P("<b>Readable drivers (logistic regression):</b> people who have <b>never had a raise</b> or have waited long since the last one, who have changed managers more often, who have longer total careers, or who commute farther are more likely to leave. "
        "Higher job level, higher engagement score, a better compensation ratio (pay versus role midpoint) and longer tenure point toward staying. Pay-related columns overlap, so do not over-read any single coefficient. "
        f"Department matters too (highest: {', '.join(k.replace('department_id_', '') for k in R['dept_effects']['highest'])}; lowest: {', '.join(k.replace('department_id_', '') for k in R['dept_effects']['lowest'])}). "
        "These are associations in this dataset, not proven causes.")]

s += [keep(P("7. Think for yourself: precision or recall?", h2),
           P("<b>Recall matters most.</b> A leaver we miss is a resignation with no chance to act. A stayer we flag by mistake costs one friendly conversation. "
             "But precision still has a job: if most flags are wrong, HR stops trusting the list, so we watch both and show the trade-off:"),
           tbl([["Flag if risk score at least", "Share of staff flagged", "Recall", "Precision"]] +
               [[f"{r.threshold:.1f}", f"{r.flagged_pct:.0%}", f"{r.recall:.0%}", f"{r.precision:.0%}"] for r in TH.itertuples()],
               [45 * mm, 45 * mm, 35 * mm, W - 125 * mm]))]

t_cv = S.loc[ch]
s += [P("8. Final check on the locked test set (used once)", h2),
      tbl([["Metric", "Cross-validation (mean +/- std)", "Locked test set"],
           ["Precision", pm(ch, "precision"), f"{tm['precision']:.3f}"], ["Recall", pm(ch, "recall"), f"{tm['recall']:.3f}"],
           ["F1", pm(ch, "f1"), f"{tm['f1']:.3f}"], ["ROC-AUC", pm(ch, "roc_auc"), f"{tm['roc_auc']:.3f}"]],
          [40 * mm, 70 * mm, W - 110 * mm]),
      Spacer(1, 3),
      P(f"Test confusion matrix [[TN, FP], [FN, TP]] = {tc}. The test scores sit within the cross-validation spread ({'yes' if R['test_within_cv_range'] else 'no'}), "
        "so the baseline numbers are trustworthy and nothing leaked."),
      P(f"<b>Time-based robustness:</b> training on 2021-2024 observations and testing on 2025 gives ROC-AUC {ts['roc_auc']:.2f}, recall {ts['recall']:.2f} and precision {ts['precision']:.2f} "
        f"({int(ts['train_rows']):,} train rows, {int(ts['test_rows']):,} test rows). Ranking quality holds on newer data; recall at the 0.5 cut-off is lower, "
        "so the cut-off should be reviewed periodically.")]

s += [P("9. How to run it", h2),
      C("jupyter nbconvert --to notebook --execute --inplace Task3_Baseline_Models/task3_baseline_models.ipynb"),
      P("Code: <font name='Courier'>src/models/baselines.py</font> (model list, cross-validation, lift table). Outputs: <font name='Courier'>Task3_Baseline_Models/outputs/</font> "
        "(tables and figures). Reference scores in the workbook's baseline_benchmarks sheet (logistic regression ROC-AUC about 0.71, recall about 0.63) are consistent with these results.")]

s += [keep(P("10. Completion criteria", h2),
           checklist([("Dummy, logistic and tree baselines compared", "Done", "Table in section 3 (plain and balanced versions)"),
                      ("Cross-validated metrics saved", "Done", "outputs/tables/cv_summary_full.csv, cv_per_fold.csv, baseline_table.csv"),
                      ("Test set untouched until final check", "Done", "Used once in Step 9 of the notebook"),
                      ("Report written for non-technical readers", "Done", "This document: sections 1, 6 and 7")]))]
s += [P("<b>Next:</b> Task 4 changes the kind of problem: finding unusual rows without labels.")]
build(T + "Task3_Documentation.pdf", s, "Task 3 - Baseline Models", FOOT)
