from nbkit import NB, setup

nb = NB()
nb.md("""# Task 4 - Attendance Anomaly Detection
**Project:** Aura Workforce attrition | **Author:** Katarukonda Ashok

**Goal:** flag unusual monthly attendance with an **Isolation Forest** (an unsupervised model: no labels needed), tune its threshold, and *measure* how well it works by injecting anomalies we know about.

**Why injection?** With no labels you cannot tell whether the model is any good. So we plant known problems (impossible hours, long unexplained absences, sudden pattern changes) in a copy of the data, save exactly which rows we changed, and check how many the model finds and how many normal rows it wrongly flags.

**Plan:** features (Task 2 style pipeline) -> inject with a fixed seed -> train -> sweep thresholds -> choose one and say why -> save model + threshold + metadata -> second, independent exam on the labels shipped with the dataset.""")
nb.code(setup("Task4_Anomaly_Detection"))
nb.code("""from sklearn.metrics import precision_recall_curve, average_precision_score
from src.features.attendance import build_attendance_pipeline, FEATURE_COLUMNS, RAW_COLUMNS, ENGINEERED
from src.anomaly.inject import inject_anomalies, SEED
from src.anomaly.detect import (make_detector, anomaly_scores, sweep_thresholds, choose_threshold, save_detector, load_detector, VERSION)
RESULTS = {}""")

nb.md("## Step 1. Load the monthly attendance table")
nb.code("""att = pd.read_excel(DATA, sheet_name="attendance_monthly_features")
labels = pd.read_excel(DATA, sheet_name="attendance_anomaly_labels")
print(f"{len(att):,} employee-months | {att.employee_id.nunique()} employees | {att.month.min():%b %Y} to {att.month.max():%b %Y}")
print("Missing values:", int(att.isna().sum().sum()), "| duplicate employee-months:", int(att.duplicated(["employee_id", "month"]).sum()))
display(att[RAW_COLUMNS + ["scheduled_days", "present_days"]].describe().T[["mean", "std", "min", "50%", "max"]].round(2))
RESULTS.update(rows=len(att), employees=int(att.employee_id.nunique()))""")
nb.md("""The workbook also ships a sheet with **150 pre-planted anomaly labels** (`attendance_anomaly_labels`). To keep the experiment honest we do **not** train with those labels (Isolation Forest never uses labels). We only (a) set those 150 rows aside so our own ground truth is not contaminated, and (b) use them in Step 9 as a second, independent exam.""")

nb.md("""## Step 2. Per-employee monthly features (the pipeline)
`AttendanceFeatureBuilder` keeps the eight raw measures and adds six engineered ones. The key idea for *sudden pattern changes*: compare each month with **that person's own usual behaviour**, not with the whole company.""")
nb.code("""rows = [("attendance_rate, unpaid_absence_days, longest_unpaid_streak_days, late_arrivals, avg_hours_worked, overtime_hours, half_days_unpaid, paid_leave_days", "taken as they are"),
        ("present_ratio", "present days / scheduled days"), ("wfh_share", "work-from-home days / present days"),
        ("rate_change_vs_prev", "attendance rate minus last month's"), ("hours_change_vs_prev", "hours worked minus last month's"),
        ("rate_vs_own_mean", "attendance rate minus this person's usual rate (learned in fit)"),
        ("hours_vs_own_mean", "hours worked minus this person's usual hours (learned in fit)")]
display(pd.DataFrame(rows, columns=["feature(s)", "meaning"]).set_index("feature(s)"))
pipe_feat = build_attendance_pipeline().fit(att)
F = pd.DataFrame(pipe_feat.transform(att), columns=FEATURE_COLUMNS)
print("Feature matrix:", F.shape, "| missing after the pipeline:", int(F.isna().sum().sum()))""")

nb.md("""## Step 3. Create ground truth: inject known anomalies (fixed seed)
- First remove the 150 rows labelled in the workbook, so every remaining row is treated as *normal*.
- Then inject three kinds of problem into randomly chosen healthy months (seed 42):

| Type | What we change |
|---|---|
| `LONG_ABSENCE` | 7-15 unpaid days in a row; present days and attendance rate drop to match |
| `IMPOSSIBLE_HOURS` | average hours per day set to 13.5-21 |
| `PATTERN_CHANGE` | attendance falls to 50-75%, late arrivals jump, hours dip: a sudden break from the person's own pattern |
- Save which rows were injected and what changed.""")
nb.code("""is_provided = att.attendance_feature_id.isin(labels.attendance_feature_id)
base = att[~is_provided].reset_index(drop=True)
inj, log = inject_anomalies(base, seed=SEED)
log.to_csv(TAB / "injected_rows_log.csv", index=False)
inj[["attendance_feature_id", "employee_id", "month", "is_injected"]].to_csv(TAB / "injected_row_flags.csv", index=False)
print(f"Base table {len(base):,} rows (150 provided-label rows set aside). Injected {int(inj.is_injected.sum())} anomalies = {inj.is_injected.mean():.2%} of rows")
display(log.injection_type.value_counts().rename("rows").to_frame())
ex = log.groupby("injection_type").head(1)[["attendance_feature_id", "injection_type", "changed_columns", "old_values", "new_values"]]
display(ex.set_index("attendance_feature_id"))
inj2, log2 = inject_anomalies(base, seed=SEED)
print("Repeatable (run twice with seed 42, identical):", inj.equals(inj2) and log.equals(log2))
RESULTS.update(base_rows=len(base), injected=int(inj.is_injected.sum()), injected_by_type=log.injection_type.value_counts().to_dict(), repeatable=bool(inj.equals(inj2)))""")

nb.md("""## Step 4. Train the Isolation Forest and score every row
Isolation Forest builds many random trees that split on random features and random values. A strange row is cut off from the rest in only a few splits (a **short path**), a normal row needs many. Short average path = high **anomaly score**. The model never sees `is_injected`.""")
nb.code("""detector = make_detector(n_estimators=300).fit(inj)
scores = anomaly_scores(detector, inj)
y = inj.is_injected.values
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.hist(scores[y == 0], bins=60, alpha=0.7, color="#7aa6c2", label="normal rows", density=True)
ax.hist(scores[y == 1], bins=30, alpha=0.8, color="#d9694a", label="injected anomalies", density=True)
ax.set_xlabel("anomaly score (higher = more unusual)"); ax.set_ylabel("density"); ax.legend(); ax.set_title("Anomaly scores: normal vs injected")
plt.tight_layout(); plt.savefig(FIG / "score_distribution.png"); plt.close()
print("Average precision (area under the precision-recall curve):", round(average_precision_score(y, scores), 3), "| random guessing would give", round(y.mean(), 3))
RESULTS["average_precision"] = float(average_precision_score(y, scores))""")

nb.md("""## Step 5. Tune the threshold
"Contamination" is the share of rows we decide to flag. For each share we record:
- **catch rate** = injected anomalies found / all injected (recall)
- **false-alarm rate** = normal rows wrongly flagged / all normal rows
- **precision** = flagged rows that really are injected""")
nb.code("""sweep = sweep_thresholds(scores, y)
sweep.to_csv(TAB / "threshold_sweep.csv", index=False)
view = sweep.assign(flag_rate=(sweep.flag_rate * 100).round(1).astype(str) + "%", catch_rate=(sweep.catch_rate * 100).round(1),
                    false_alarm_rate=(sweep.false_alarm_rate * 100).round(2), precision=(sweep.precision * 100).round(1), f1=sweep.f1.round(3), threshold=sweep.threshold.round(3))
display(view[["flag_rate", "threshold", "flagged", "caught", "missed", "false_alarms", "catch_rate", "false_alarm_rate", "precision", "f1"]]
        .rename(columns={"catch_rate": "catch %", "false_alarm_rate": "false-alarm %", "precision": "precision %"}))""")

nb.md("""## Step 6. Precision-recall trade-off and the choice
**Rule:** take the threshold with the best F1 (balance of catching and being right) **among thresholds that flag at most 5% of rows**. The hint warns that flagging 20% makes HR stop trusting the list, so we cap the flag rate.""")
nb.code("""best = choose_threshold(sweep, max_flag_rate=0.05)
prec, rec, thr = precision_recall_curve(y, scores)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(rec, prec, color="#3b6ea8", lw=2, label="precision-recall curve")
ax[0].scatter(sweep.catch_rate, sweep.precision, color="k", s=18, zorder=3)
for _, r in sweep.iloc[[1, 3, 5, 8, 10, 13]].iterrows():
    ax[0].annotate(f"{r.flag_rate:.1%}", (r.catch_rate, r.precision), textcoords="offset points", xytext=(5, 5), fontsize=7)
ax[0].scatter([best.catch_rate], [best.precision], color="#d9694a", s=90, zorder=4, label=f"chosen: flag {best.flag_rate:.1%} of rows")
ax[0].set_xlabel("catch rate (recall)"); ax[0].set_ylabel("precision"); ax[0].set_title("Precision-recall trade-off (labels = % of rows flagged)"); ax[0].legend(fontsize=8)
ax[1].plot(sweep.flag_rate * 100, sweep.catch_rate * 100, "o-", color="#5a8f7b", label="catch rate %")
ax[1].plot(sweep.flag_rate * 100, sweep.false_alarm_rate * 100, "o-", color="#d9694a", label="false-alarm rate %")
ax[1].axvline(best.flag_rate * 100, color="k", ls="--", lw=1); ax[1].axvspan(5, 20, color="grey", alpha=0.12)
ax[1].set_xlabel("% of rows flagged"); ax[1].set_ylabel("%"); ax[1].set_title("Catch rate vs false alarms (grey = HR stops trusting it)"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "precision_recall_tradeoff.png"); plt.close()
months = att.month.nunique()
per_month = best.flagged / len(inj) * (len(att) / months)
print(f"CHOSEN: flag the top {best.flag_rate:.1%} most unusual months (score >= {best.threshold:.3f})")
print(f"  catch rate {best.catch_rate:.1%} ({int(best.caught)} of {int(inj.is_injected.sum())})  |  false-alarm rate {best.false_alarm_rate:.2%} ({int(best.false_alarms)} normal rows)  |  precision {best.precision:.1%}")
print(f"  about {per_month:.0f} flagged employee-months per calendar month for HR to review")
RESULTS["chosen"] = {k: (float(v) if k not in ("flagged", "caught", "missed", "false_alarms") else int(v)) for k, v in best.items()}
RESULTS["reviews_per_month"] = float(per_month)""")
nb.md("""**Why this threshold (written reason):** below about 3% we miss too many real problems (catch rate falls steeply); above about 4% each extra anomaly found costs many extra false alarms and precision falls toward 35% and below. The chosen point keeps the false-alarm rate under 2% of normal rows, flags a number of months HR can actually review, and sits at the top of the F1 curve. Going to 10% would catch everything injected but flood HR with wrong flags.""")

nb.md("## Step 7. Which kinds of anomaly are caught?")
nb.code("""typ = inj.attendance_feature_id.map(log.set_index("attendance_feature_id").injection_type).fillna("normal")
flag = scores >= best.threshold
bytype = pd.DataFrame({"type": typ, "flagged": flag}).groupby("type").flagged.agg(["sum", "count", "mean"])
bytype.columns = ["flagged", "rows", "share_flagged"]
bytype.to_csv(TAB / "catch_rate_by_type.csv"); display(bytype.round(3))
RESULTS["catch_by_type"] = bytype.share_flagged.round(3).to_dict()""")
nb.md("""**Finding:** long absences and sudden pattern changes are caught every time, because they move several features at once. Impossible hours are caught only about half the time: the forest splits across 14 features, and a single extreme column is diluted. A simple **rule** fixes this (check below), so a sensible production design is *hard rules for impossible values + Isolation Forest for subtle patterns*.""")
nb.code("""rule = (inj.avg_hours_worked > 12).values            # nobody averages more than 12 hours a day
ih = (typ == "IMPOSSIBLE_HOURS").values
print(f"Rule 'average hours > 12': catches {int((rule & ih).sum())} of {int(ih.sum())} impossible-hours rows, false alarms on normal rows: {int((rule & ~ih & (y == 0)).sum())}")
print("Highest average hours among normal rows:", inj.loc[y == 0, "avg_hours_worked"].max())
RESULTS["hours_rule"] = {"caught": int((rule & ih).sum()), "of": int(ih.sum()), "false_alarms": int((rule & ~ih & (y == 0)).sum())}""")

nb.md("""## Step 8. Save the model and the threshold with metadata, then reload it""")
nb.code("""art = OUT / "artifacts"
path = save_detector(detector, best.threshold, art, extra={
    "trained_on_rows": len(inj), "injected_rows": int(inj.is_injected.sum()), "chosen_flag_rate": float(best.flag_rate),
    "catch_rate_on_injected": float(best.catch_rate), "false_alarm_rate": float(best.false_alarm_rate), "precision_on_injected": float(best.precision),
    "threshold_rule": "best F1 among thresholds flagging <= 5% of rows", "injection_seed": SEED})
print("Saved:", path.relative_to(ROOT))
pipe2, thr2, meta = load_detector(path)
same = np.allclose(anomaly_scores(pipe2, inj), scores)
print("Reloaded model gives identical scores:", same, "| threshold:", round(thr2, 4), "| version:", meta["version"])
newrows = att.sample(5, random_state=1)
print("Scoring 5 sample rows with the saved model:")
display(newrows[["employee_id", "month", "attendance_rate", "avg_hours_worked"]].assign(anomaly_score=anomaly_scores(pipe2, att).take(newrows.index).round(3)).assign(flag=lambda d: d.anomaly_score >= thr2))
RESULTS["reload_identical"] = bool(same)""")

nb.md("""## Step 9. Second exam: the labels that came with the dataset
This is the realistic test. The 150 rows labelled in the workbook were **set aside before training**, so the model has never seen them. We score the full original table with the saved model and threshold and check how many of the 150 are flagged.""")
nb.code("""s_all = anomaly_scores(detector, att)
flag_all = s_all >= best.threshold
truth = is_provided.values
tp = int((flag_all & truth).sum()); fp = int((flag_all & ~truth).sum())
ext = {"flagged": int(flag_all.sum()), "caught": tp, "of_provided": int(truth.sum()), "catch_rate": tp / truth.sum(),
       "false_alarms": fp, "precision": tp / max(tp + fp, 1), "flagged_share": float(flag_all.mean())}
print(ext)
t = labels.merge(att[["attendance_feature_id"]].assign(flagged=flag_all), on="attendance_feature_id").groupby("anomaly_type").flagged.agg(["sum", "count", "mean"])
t.columns = ["flagged", "rows", "share_flagged"]; t.to_csv(TAB / "provided_labels_by_type.csv"); display(t.round(3))
ref = pd.read_excel(DATA, sheet_name="anomaly_baselines").iloc[0]
print(f"Reference from the dataset authors (Isolation Forest, 3% contamination): precision {ref.precision}, recall {ref.recall}, flagged {ref.flagged}")
RESULTS["external"] = {k: round(float(v), 4) if isinstance(v, float) else int(v) for k, v in ext.items()}
RESULTS["external_by_type"] = t.share_flagged.round(3).to_dict()
RESULTS["reference"] = {"precision": float(ref.precision), "recall": float(ref.recall), "flagged": int(ref.flagged)}""")
nb.md("""**Reading it:** the model finds most of the provided anomalies (absence streaks, sudden drops and zero attendance are all caught), is comparable to or a little better than the dataset's own reference, and its **blind spots are exactly the single-column ones**: impossible hours (partly) and late-arrival spikes (mostly). Our injection design did not include late spikes, which is why this second exam is valuable: it exposes a gap the first exam could not.""")

nb.md("""## Limitations (say these out loud in the review)
1. Injected anomalies are synthetic and fairly obvious, so the first catch rate is optimistic; the provided-label exam is the more realistic number.
2. Single-feature anomalies (hours, late spikes) are weak spots; add hard rules for them.
3. "Normal" rows may hide real, unlabelled oddities, so some "false alarms" may be genuine.
4. The threshold is a *rate of flagging*; revisit it when HR gives feedback on the flagged list.

## Summary
Isolation Forest + per-employee features flags a manageable few percent of months, finds almost all multi-feature anomalies, is repeatable (fixed seed), and is saved with its threshold and metadata. Phase 2 reuses this thinking for payroll.""")
nb.code("""(OUT / "results.json").write_text(json.dumps(RESULTS, indent=2, default=str))
print("saved", (OUT / "results.json").relative_to(ROOT))""")
nb.save("Task4_Anomaly_Detection/task4_anomaly_detection.ipynb")
