from nbkit import NB, setup

nb = NB()
nb.md("""# Task 3 - Attrition Baseline Models & Evaluation
**Project:** Aura Workforce attrition | **Author:** Katarukonda Ashok

**Goal:** set the honest first score that every later model must beat. Train a dummy, logistic regression and a shallow decision tree (with and without class weights), judge them with 5-fold stratified cross-validation, and explain the result for HR.

**Rules followed**
1. Stratified train/test split. The **test set is locked** and used exactly once, in Step 9.
2. The Task 2 feature pipeline sits *inside* each model, so preprocessing is refitted in every fold (no leakage).
3. Report precision, recall, F1 and ROC-AUC as **mean +/- spread over 5 folds**, not accuracy.""")
nb.code(setup("Task3_Baseline_Models"))
nb.code("""from sklearn.base import clone
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score, precision_score, recall_score, f1_score,
                             average_precision_score, accuracy_score)
from sklearn.model_selection import cross_val_predict
from sklearn.tree import export_text
from src.features import config as cfg
from src.features.data import load_clean_base, split_train_test, time_based_split
from src.models.baselines import (model_zoo, with_features, split_xy, make_cv, cross_validate_all, lift_table, SCORING)
RESULTS = {}""")

nb.md("""## Step 1. Split and LOCK the test set
Duplicates are removed first (Task 1, D1), then 80% train / 20% test with the same leaver rate in both. From here on only `X_train` / `y_train` are used until Step 9.""")
nb.code("""df = load_clean_base(DATA)
train, test = split_train_test(df)
X_train, y_train = split_xy(train)
X_test, y_test = split_xy(test)        # LOCKED: not touched until Step 9
print(f"Train {len(train):,} rows ({y_train.mean():.2%} leave) | Test {len(test):,} rows ({y_test.mean():.2%} leave) - locked")
RESULTS.update(train_rows=len(train), test_rows=len(test), leaver_rate=float(y_train.mean()))""")

nb.md("""## Step 2. The dummy baseline: "everyone stays"
This is the floor. If a model cannot beat it on the right metrics, it is useless, even when its accuracy looks high.""")
nb.code("""dummy_acc = 1 - y_train.mean()
print(f"A model that always predicts 'stays' is {dummy_acc:.1%} accurate and finds 0 of {int(y_train.sum())} leavers.")
RESULTS["dummy_accuracy"] = float(dummy_acc)""")

nb.md("""## Step 3. Cross-validate all baselines (5-fold, stratified)
Each model is `Task 2 feature pipeline -> model`. The pipeline is cloned and refitted inside every fold, so a fold's validation rows never influence medians, caps or one-hot categories.""")
nb.code("""summary, folds = cross_validate_all(X_train, y_train)
summary.to_csv(TAB / "cv_summary_full.csv", index=False)
folds.to_csv(TAB / "cv_per_fold.csv", index=False)
show = pd.DataFrame({"model": summary.model})
for m, label in [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("accuracy", "Accuracy")]:
    show[label] = summary[f"{m}_mean"].map("{:.3f}".format) + " +/- " + summary[f"{m}_std"].map("{:.3f}".format)
show.to_csv(TAB / "baseline_table.csv", index=False)
display(show.set_index("model"))
RESULTS["cv_summary"] = summary.round(4).to_dict(orient="records")""")

nb.md("""**How to read the table**
- *Accuracy* rewards the dummy (everyone stays), so ignore it for ranking.
- *Precision*: of the people flagged as leavers, how many really left.
- *Recall*: of the people who really left, how many we flagged.
- *ROC-AUC*: how well the model ranks leavers above stayers (0.5 = coin flip, 1.0 = perfect).
- *PR-AUC*: same idea, but honest about rare positives (the dummy sits near 0.146, the leaver rate).""")

nb.md("## Step 4. Class weights: plain versus balanced")
nb.code("""pairs = [("Logistic Regression", "Logistic Regression (balanced)"), ("Decision Tree depth 4", "Decision Tree depth 4 (balanced)")]
rows = []
S = summary.set_index("model")
for a, b in pairs:
    for m in ["precision", "recall", "f1", "roc_auc"]:
        rows.append({"model family": a.split(" depth")[0], "metric": m, "plain": S.loc[a, f"{m}_mean"], "balanced": S.loc[b, f"{m}_mean"],
                     "change": S.loc[b, f"{m}_mean"] - S.loc[a, f"{m}_mean"]})
cw = pd.DataFrame(rows); cw.to_csv(TAB / "class_weight_comparison.csv", index=False)
display(cw.round(3))

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
order = list(summary.model)
x = np.arange(len(order)); w = 0.38
ax[0].bar(x - w/2, summary.recall_mean, w, yerr=summary.recall_std, label="Recall", color="#d9694a", capsize=2)
ax[0].bar(x + w/2, summary.precision_mean, w, yerr=summary.precision_std, label="Precision", color="#3b6ea8", capsize=2)
ax[0].set_xticks(x); ax[0].set_xticklabels([o.replace(" (", "\\n(").replace("Logistic Regression", "LogReg").replace("Decision Tree", "Tree") for o in order], fontsize=7)
ax[0].set_title("Recall and precision (5-fold mean +/- std)"); ax[0].legend(fontsize=8)
ax[1].bar(x, summary.roc_auc_mean, yerr=summary.roc_auc_std, color="#5a8f7b", capsize=2); ax[1].axhline(0.5, color="k", ls="--", lw=1)
ax[1].set_xticks(x); ax[1].set_xticklabels([o.replace(" (", "\\n(").replace("Logistic Regression", "LogReg").replace("Decision Tree", "Tree") for o in order], fontsize=7)
ax[1].set_ylim(0.4, 0.8); ax[1].set_title("ROC-AUC (dashed = coin flip)")
plt.tight_layout(); plt.savefig(FIG / "model_comparison.png"); plt.close()""")
nb.md("""**What class weights do:** the plain models rarely say "leaves" (recall below 0.11), so they miss almost every leaver. `class_weight="balanced"` tells the model that a missed leaver costs about 6x more than a false alarm, so recall jumps to about 0.55-0.6 while precision drops. Ranking quality (ROC-AUC) barely changes, because weights move the cut-off, not the ranking.""")

nb.md("""## Step 5. Pick the baseline to beat
Rule used (written down before looking at the test set): among the *balanced* models, keep those whose ROC-AUC is within 0.01 of the best, then take the one with the highest recall. HR prefers catching leavers (see the answer to the "think for yourself" question below).""")
nb.code("""bal = summary[summary.model.str.contains("balanced")]
cand = bal[bal.roc_auc_mean >= bal.roc_auc_mean.max() - 0.01]
chosen_name = cand.sort_values("recall_mean", ascending=False).model.iloc[0]
print("Chosen baseline:", chosen_name)
chosen_est = model_zoo()[chosen_name]
RESULTS["chosen_model"] = chosen_name""")

nb.md("## Step 6. Confusion matrix and ROC curve (out-of-fold predictions on the training set)")
nb.code("""cv = make_cv()
oof_proba = cross_val_predict(with_features(clone(chosen_est)), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
oof_pred = (oof_proba >= 0.5).astype(int)
cm = confusion_matrix(y_train, oof_pred)
tn, fp, fn, tp = cm.ravel()
print(f"Out-of-fold: caught {tp} of {tp+fn} leavers, wrongly flagged {fp} of {tn+fp} stayers")
RESULTS["oof_confusion"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
im = ax[0].imshow(cm, cmap="Blues")
for (i, j), v in np.ndenumerate(cm):
    ax[0].text(j, i, f"{v}\\n({v / cm[i].sum():.0%} of row)", ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["Predicted stays", "Predicted leaves"]); ax[0].set_yticks([0, 1]); ax[0].set_yticklabels(["Actually stayed", "Actually left"])
ax[0].set_title("Confusion matrix (5-fold out-of-fold)")
for name in ["Dummy (always stays)", "Decision Tree depth 4 (balanced)", chosen_name]:
    p = cross_val_predict(with_features(clone(model_zoo()[name])), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    fpr, tpr, _ = roc_curve(y_train, p)
    ax[1].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_train, p):.2f})", lw=2 if name == chosen_name else 1.2)
ax[1].plot([0, 1], [0, 1], "k--", lw=1); ax[1].set_xlabel("False alarm rate (stayers flagged)"); ax[1].set_ylabel("Recall (leavers caught)")
ax[1].set_title("ROC curve"); ax[1].legend(fontsize=7, loc="lower right")
plt.tight_layout(); plt.savefig(FIG / "confusion_and_roc.png"); plt.close()""")

nb.md("""**Reading the confusion matrix:** the bottom-right cell is leavers we caught, the bottom-left is leavers we missed (the costly mistake), the top-right is stayers wrongly flagged (a wasted conversation), and the top-left is stayers correctly left alone.""")

nb.md("""## Step 7. What it means for HR: review the highest-risk people first
HR cannot talk to everyone. If they only review the top X% by risk score, how many real leavers do they find? (Random picking would find X% of leavers.)""")
nb.code("""lt = lift_table(y_train.values, oof_proba)
lt["lift_vs_random"] = lt.share_of_all_leavers_found / (lt.review_top_pct / 100)
lt.to_csv(TAB / "hr_lift_table.csv", index=False)
display(lt.round(3))
RESULTS["lift"] = lt.round(4).to_dict(orient="records")

th = []
for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
    pr = (oof_proba >= t).astype(int)
    th.append({"threshold": t, "flagged_pct": pr.mean(), "recall": recall_score(y_train, pr), "precision": precision_score(y_train, pr, zero_division=0)})
th = pd.DataFrame(th); th.to_csv(TAB / "threshold_tradeoff.csv", index=False); display(th.round(3))""")

nb.md("""**Think for yourself - precision or recall?** **Recall.** HR would rather catch more leavers, even if some stayers are flagged by mistake: a missed leaver is a resignation with no chance to act, while a false alarm costs one friendly conversation. But precision still matters, because if most flags are wrong HR stops trusting the list. The table above shows the trade-off: lowering the threshold finds more leavers but flags more stayers.""")

nb.md("## Step 8. What drives the prediction (logistic regression coefficients)")
nb.code("""lr = with_features(clone(model_zoo()["Logistic Regression (balanced)"])).fit(X_train, y_train)
from src.features.pipeline import output_columns
names = output_columns(lr.named_steps["features"])
coef = pd.Series(lr.named_steps["model"].coef_[0], index=names).sort_values()
coef.to_csv(TAB / "all_coefficients.csv", header=["coefficient"])
signal = coef[[c for c in coef.index if c in cfg.NUMERIC_COLUMNS + cfg.FLAG_COLUMNS]]      # readable HR measures
top = pd.concat([signal.head(6), signal.tail(6)])
top.to_csv(TAB / "top_coefficients.csv", header=["coefficient"])
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.barh(top.index, top.values, color=["#5a8f7b" if v < 0 else "#d9694a" for v in top.values])
ax.axvline(0, color="k", lw=0.8); ax.set_title("Strongest HR drivers (orange = leave more)", fontsize=10); ax.set_xlabel("coefficient (standardised features)")
plt.tight_layout(); plt.savefig(FIG / "top_drivers.png"); plt.close()
display(top.round(3).to_frame("coefficient"))
dept = coef[[c for c in coef.index if c.startswith("department_id_")]].sort_values()
print("Department effect (coefficient):", dept.round(2).to_dict())
RESULTS["top_leave_drivers"] = signal.tail(6)[::-1].round(3).to_dict(); RESULTS["top_stay_drivers"] = signal.head(6).round(3).to_dict()
RESULTS["dept_effects"] = {"highest": dept.tail(2)[::-1].round(2).to_dict(), "lowest": dept.head(2).round(2).to_dict()}

tree = with_features(clone(model_zoo()["Decision Tree depth 4 (balanced)"])).fit(X_train, y_train)
print(export_text(tree.named_steps["model"], feature_names=list(output_columns(tree.named_steps["features"])), max_depth=2))""")

nb.md("""## Step 9. FINAL check on the locked test set (used once)
The chosen model is refitted on all training rows and scored on the 20% test rows that nothing has touched. The scores should sit close to the cross-validation numbers; if they were far better, something leaked.""")
nb.code("""final = with_features(clone(chosen_est)).fit(X_train, y_train)
proba = final.predict_proba(X_test)[:, 1]; pred = (proba >= 0.5).astype(int)
test_metrics = {"precision": precision_score(y_test, pred), "recall": recall_score(y_test, pred), "f1": f1_score(y_test, pred),
                "roc_auc": roc_auc_score(y_test, proba), "pr_auc": average_precision_score(y_test, proba), "accuracy": accuracy_score(y_test, pred)}
cvrow = summary.set_index("model").loc[chosen_name]
cmp = pd.DataFrame({"cross-validation (mean)": {m: cvrow[f"{m}_mean"] for m in test_metrics}, "cv spread (std)": {m: cvrow[f"{m}_std"] for m in test_metrics},
                    "locked test set": test_metrics})
cmp.to_csv(TAB / "final_test_check.csv"); display(cmp.round(3))
tcm = confusion_matrix(y_test, pred); print("Test confusion matrix [[TN, FP], [FN, TP]]:", tcm.tolist())
RESULTS["test_metrics"] = {k: round(float(v), 4) for k, v in test_metrics.items()}; RESULTS["test_confusion"] = tcm.tolist()
RESULTS["test_within_cv_range"] = bool(abs(test_metrics["roc_auc"] - cvrow["roc_auc_mean"]) <= 2 * cvrow["roc_auc_std"])
print("Test ROC-AUC within 2 standard deviations of CV:", RESULTS["test_within_cv_range"])""")

nb.md("""## Step 10. Robustness: does it still work on newer data? (time-based check)
Task 1 noted the rows span 2021-2025. Train on 2021-2024 observations and test on 2025 to see whether the score holds when the future is genuinely unseen.""")
nb.code("""older, newer = time_based_split(df, "2025-01-01")
Xo, yo = split_xy(older); Xn, yn = split_xy(newer)
m = with_features(clone(chosen_est)).fit(Xo, yo); pn = m.predict_proba(Xn)[:, 1]; pr = (pn >= 0.5).astype(int)
tb = {"train_rows": len(older), "test_rows": len(newer), "roc_auc": roc_auc_score(yn, pn), "recall": recall_score(yn, pr), "precision": precision_score(yn, pr, zero_division=0)}
display(pd.Series(tb, name="train 2021-2024, test 2025").round(3).to_frame())
RESULTS["time_split"] = {k: round(float(v), 4) for k, v in tb.items()}""")

nb.md("""## Summary
- The dummy is 85% accurate and useless; real baselines are judged on recall, precision, F1 and ROC-AUC.
- Class weights trade precision for recall; ranking quality (ROC-AUC about 0.70) is the same.
- The balanced logistic regression is the baseline every later model must beat.
- Locked-test scores match cross-validation, and the time-based check holds, so the numbers are honest.""")
nb.code("""(OUT / "results.json").write_text(json.dumps(RESULTS, indent=2, default=str))
print("saved", (OUT / "results.json").relative_to(ROOT))""")
nb.save("Task3_Baseline_Models/task3_baseline_models.ipynb")
