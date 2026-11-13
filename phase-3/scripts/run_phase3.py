"""Re-create every Phase 3 table, figure, model and the results file.   python scripts/run_phase3.py
Needs data/raw/02_CortexHR_AI_ML_Dataset.xlsx (not committed). Models are saved to models/."""
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold, train_test_split

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.attrition import explain as ex  # noqa: E402
from src.attrition import v2  # noqa: E402
from src.fairness import group_gap, risk_bands  # noqa: E402
from src.pay import model as pay  # noqa: E402
from src.resume.scoring import ResumeScreener, ranking_quality  # noqa: E402

DATA = ROOT / "data/raw/02_CortexHR_AI_ML_Dataset.xlsx"
SHEETS = ["job_descriptions", "resumes", "resume_job_labels", "resume_bias_pairs", "resume_protected_attrs", "baseline_benchmarks",
          "attrition_scoring_2026", "sample_attrition_risk", "salary_training", "protected_attributes"]
X = pd.read_excel(DATA, sheet_name=SHEETS)
RES = {}
T = lambda t: ROOT / t  # noqa: E731


# ------------------------------------------------------------------ Task 11 and 12
def task11_12():
    J, R, L, B, PA = (X[k] for k in ["job_descriptions", "resumes", "resume_job_labels", "resume_bias_pairs", "resume_protected_attrs"])
    out11, out12 = T("Task11_Resume_Ranking/outputs/tables"), T("Task12_Fairness_Audit/outputs/tables")
    fair = ResumeScreener(masked=True).fit(J, R)
    naive = ResumeScreener(masked=False).fit(J, R)
    pairs = L[["resume_id", "job_id"]]
    sc = fair.score_pairs(R, pairs)
    variants = {"Final screener (masked): skills + similarity + experience": ranking_quality(sc, L),
                "Required-skill coverage only": ranking_quality(ResumeScreener(True, {"required": 1, "nice": 0, "similarity": 0, "experience": 0}).fit(J, R).score_pairs(R, pairs), L),
                "Text similarity only (TF-IDF)": ranking_quality(ResumeScreener(True, {"required": 0, "nice": 0, "similarity": 1, "experience": 0}).fit(J, R).score_pairs(R, pairs), L)}
    rng = np.random.default_rng(42); rnd = sc.assign(score=rng.random(len(sc)))
    variants["Random order (reference)"] = ranking_quality(rnd, L)
    pd.DataFrame(variants).T.reset_index().rename(columns={"index": "ranking"}).to_csv(out11 / "ranking_quality.csv", index=False)
    # what we extract, checked against the labels' own coverage columns and the stated experience
    m = sc.merge(L, on=["resume_id", "job_id"]); prof = fair.profile(R)
    chk = {"required_coverage_MAE_vs_label": round(float((m.required_coverage - m.required_skill_coverage).abs().mean()), 3),
           "nice_coverage_MAE_vs_label": round(float((m.nice_coverage - m.nice_skill_coverage).abs().mean()), 3),
           "years_MAE_vs_total_experience_years": round(float((prof.years_found - R.set_index("resume_id").total_experience_years).abs().mean()), 2)}
    pd.Series(chk).rename("value").to_csv(out11 / "extraction_check.csv")
    # top-5 per job among all 185 resumes (ids only: no names, no gender, no tier)
    rows = []
    for jid in J.job_id:
        top = fair.rank(R, jid, 5); top.insert(1, "rank", range(1, 6)); rows.append(top)
    pd.concat(rows)[["job_id", "rank", "resume_id", "score", "required_coverage", "nice_coverage", "similarity", "experience_fit", "years_found", "missing_required"]].to_csv(out11 / "top5_per_job.csv", index=False)
    # five misranks, explained
    m["rank_pct_score"] = m.groupby("job_id")["score"].rank(pct=True); m["rank_pct_human"] = m.groupby("job_id")["human_suitability_score"].rank(pct=True)
    m["disagreement"] = (m.rank_pct_score - m.rank_pct_human)
    lay = R.set_index("resume_id")["layout_difficulty"]; m["layout"] = m.resume_id.map(lay)
    fp = m[(m.human_suitability_score <= 2)].nlargest(3, "disagreement"); fn = m[(m.human_suitability_score >= 4)].nsmallest(2, "disagreement")
    mis = pd.concat([fp.assign(kind="ranked too high"), fn.assign(kind="ranked too low")])

    def why(r):
        need = float(J.set_index("job_id").loc[r.job_id, "min_experience_years"]); bits = []
        if r.kind == "ranked too high":
            bits.append(f"covers {r.required_coverage:.0%} of required skills but reviewer says '{r.reviewer_note}' (human score {r.human_suitability_score})")
            if r.years_found is not None and r.years_found < need: bits.append(f"only {r.years_found} years against {need:g} needed (experience carries 15% of the score)")
            else: bits.append("skills match on paper; the score cannot see depth or fit that a reviewer reads between the lines")
        else:
            bits.append(f"human score {r.human_suitability_score} ('{r.reviewer_note}') but my required coverage is {r.required_coverage:.0%} (label says {r.required_skill_coverage:.0%})")
            bits.append("missing: " + ", ".join(r.missing_required) if r.missing_required else "skills found; text similarity or experience pulled the score down")
        if r.layout != "Clean": bits.append(f"resume layout is '{r.layout}', which can hide text from a parser")
        return "; ".join(bits)
    mis["why_it_went_wrong"] = mis.apply(why, axis=1)
    mis[["resume_id", "job_id", "kind", "score", "human_suitability_score", "shortlist", "required_coverage", "required_skill_coverage", "years_found", "layout", "why_it_went_wrong"]].to_csv(out11 / "misranks_explained.csv", index=False)
    RES["task11"] = {"quality": variants, "extraction": chk, "baseline_shortlist_rate": round(float((L.shortlist == "Y").mean()), 3)}

    # ---- Task 12: twin resumes
    Rr = R.set_index("resume_id"); pa = PA.set_index("resume_id")
    def twin_table(screener):
        rows = []
        for _, p in B.iterrows():
            job = Rr.loc[p.resume_id_original, "target_job_family"]
            ids = [p.resume_id_original, p.resume_id_twin]
            s = screener.score_pairs(R, pd.DataFrame({"resume_id": ids, "job_id": [job] * 2}))
            allr = screener.score_pairs(R, pd.DataFrame({"resume_id": R.resume_id, "job_id": job})).sort_values(["score", "resume_id"], ascending=[False, True]).reset_index(drop=True)
            sc_ = dict(zip(allr.resume_id, allr.score)); rk = {r: 1 + int((allr.score > sc_[r]).sum()) for r in ids}   # ties share a rank
            rows.append({"pair_id": p.pair_id, "job_id": job, "attribute_changed": p.attribute_changed, "score_original": s.score.iloc[0], "score_twin": s.score.iloc[1],
                         "gap_original_minus_twin": round(s.score.iloc[0] - s.score.iloc[1], 4), "rank_original": rk[ids[0]], "rank_twin": rk[ids[1]],
                         "orig_gender": pa.loc[ids[0], "inferred_gender"], "twin_gender": pa.loc[ids[1], "inferred_gender"],
                         "orig_tier": pa.loc[ids[0], "college_tier"], "twin_tier": pa.loc[ids[1], "college_tier"]})
        return pd.DataFrame(rows)
    tb, ta = twin_table(naive), twin_table(fair)
    tb.to_csv(out12 / "twin_gaps_before.csv", index=False); ta.to_csv(out12 / "twin_gaps_after.csv", index=False)

    def summ(t, name, screener):
        g = t.gap_original_minus_twin.abs()
        male_minus_female = np.where(t.orig_gender.eq("Male") & t.twin_gender.eq("Female"), t.gap_original_minus_twin, np.where(t.orig_gender.eq("Female") & t.twin_gender.eq("Male"), -t.gap_original_minus_twin, np.nan))
        q = ranking_quality(screener.score_pairs(R, pairs), L)
        return {"screener": name, "mean_abs_gap_points": round(g.mean(), 4), "max_abs_gap_points": round(g.max(), 4), "pairs_with_any_gap": int((g > 1e-9).sum()),
                "mean_male_minus_female": round(float(np.nanmean(male_minus_female)), 4), "pairs_where_rank_differs": int((t.rank_original != t.rank_twin).sum()),
                "precision_at_5": q["precision_at_5"], "ndcg_at_5": q["ndcg_at_5"]}
    summary = pd.DataFrame([summ(tb, "Before: naive screener reads the raw resume text", naive), summ(ta, "After: masked screener (final)", fair)])
    summary.to_csv(out12 / "fairness_summary.csv", index=False)
    # group gaps over all 185 resumes (score for their own target job), raw vs masked
    gg = []
    for name, scr in [("before", naive), ("after", fair)]:
        s = scr.score_pairs(R, pd.DataFrame({"resume_id": R.resume_id, "job_id": R.target_job_family})).merge(PA, on="resume_id")
        for col in ["inferred_gender", "college_tier"]:
            t = group_gap(s, "score", col); t.insert(0, "group_by", col); t.insert(0, "screener", name); gg.append(t.rename(columns={c: "gap_vs_reference" for c in t.columns if c.startswith("gap_vs_")}))
    pd.concat(gg).to_csv(out12 / "group_gaps_all_resumes.csv", index=False)
    RES["task12"] = summary.to_dict("records")


# ------------------------------------------------------------------ Task 13 and 14
def task13_14():
    out13, fig13, fig14, out14 = T("Task13_Attrition_v2/outputs/tables"), T("Task13_Attrition_v2/outputs/figures"), T("Task14_Risk_Explanations/outputs/figures"), T("Task14_Risk_Explanations/outputs")
    Xtr, ytr, Xte, yte = v2.load_train_test(str(DATA))
    oof_lr, f_lr = v2.cv_oof(v2.baseline_lr(), Xtr, ytr)
    # tuning log: 12 random settings, same folds, chosen by mean ROC-AUC (PR-AUC kept for reference)
    rng = np.random.default_rng(42); grid = []
    for i in range(12):
        grid.append({"n_estimators": int(rng.choice([100, 200, 400])), "learning_rate": float(rng.choice([0.02, 0.03, 0.05])), "num_leaves": int(rng.choice([4, 8, 16])),
                     "min_child_samples": int(rng.choice([20, 40, 80])), "reg_lambda": float(rng.choice([1.0, 5.0, 20.0])), "colsample_bytree": float(rng.choice([0.6, 0.8, 1.0]))})
    log, store = [], {}
    default = {}
    for name, p in [("default", default)] + [(f"try-{i + 1}", p) for i, p in enumerate(grid)]:
        oof, f = v2.cv_oof(v2.lgbm(**p), Xtr, ytr)
        log.append({"setting": name, **p, **{f"cv_{k}": round(f[k].mean(), 4) for k in ["roc_auc", "pr_auc", "recall_top30", "precision_top30", "brier"]}, "cv_roc_auc_sd": round(f["roc_auc"].std(), 4)})
        store[name] = (p, oof, f)
    tl = pd.DataFrame(log); tl.to_csv(out13 / "tuning_log.csv", index=False)
    best = tl.sort_values("cv_roc_auc", ascending=False).iloc[0]["setting"]; bp, oof_g, f_g = store[best]
    # same-fold comparison
    comp = pd.DataFrame({"fold": f_lr.fold, "LR_roc_auc": f_lr.roc_auc, "v2_roc_auc": f_g.roc_auc, "diff": f_g.roc_auc - f_lr.roc_auc, "LR_pr_auc": f_lr.pr_auc, "v2_pr_auc": f_g.pr_auc, "diff_pr": f_g.pr_auc - f_lr.pr_auc}).round(4)
    comp.to_csv(out13 / "same_fold_comparison.csv", index=False)
    bench = X["baseline_benchmarks"]
    def row(name, f): return {"model": name, "roc_auc": round(f.roc_auc.mean(), 3), "roc_auc_sd": round(f.roc_auc.std(), 3), "pr_auc": round(f.pr_auc.mean(), 3), "precision_top30": round(f.precision_top30.mean(), 3), "recall_top30": round(f.recall_top30.mean(), 3)}
    table = [{"model": "Reference: LogReg balanced (baseline_benchmarks)", "roc_auc": 0.713, "roc_auc_sd": 0.01, "pr_auc": 0.332, "precision_top30": np.nan, "recall_top30": np.nan},
             row("My Task 3 baseline: LogReg balanced (same folds, recomputed)", f_lr), row(f"Attrition v2: LightGBM ({best})", f_g)]
    pd.DataFrame(table).to_csv(out13 / "comparison_table.csv", index=False)
    # calibration: cross-fitted Platt scaling on the out-of-fold scores
    p = np.clip(oof_g, 1e-6, 1 - 1e-6); lg = np.log(p / (1 - p)); cal = np.zeros_like(p)
    from sklearn.linear_model import LogisticRegression
    for a, b in v2.folds().split(lg.reshape(-1, 1), ytr):
        cal[b] = LogisticRegression(C=1e6).fit(lg[a].reshape(-1, 1), ytr.iloc[a]).predict_proba(lg[b].reshape(-1, 1))[:, 1]
    calt = pd.DataFrame([{"scores": "LightGBM raw", "brier": round(v2.brier_score_loss(ytr, oof_g), 4), "ECE": round(v2.expected_calibration_error(ytr, oof_g), 4)},
                         {"scores": "LightGBM + Platt calibration", "brier": round(v2.brier_score_loss(ytr, cal), 4), "ECE": round(v2.expected_calibration_error(ytr, cal), 4)},
                         {"scores": "LogReg balanced (baseline)", "brier": round(v2.brier_score_loss(ytr, oof_lr), 4), "ECE": round(v2.expected_calibration_error(ytr, oof_lr), 4)},
                         {"scores": "Base rate only", "brier": round(v2.brier_score_loss(ytr, np.full(len(ytr), ytr.mean())), 4), "ECE": 0.0}])
    calt.to_csv(out13 / "calibration.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.2, 4.6)); ax.plot([0, 0.6], [0, 0.6], "k--", lw=1, label="perfect")
    for name, s in [("raw", oof_g), ("calibrated", cal)]:
        q = pd.qcut(s, 8, duplicates="drop"); g = pd.DataFrame({"p": s, "y": ytr.to_numpy()}).groupby(q).mean(); ax.plot(g.p, g.y, "o-", label=name)
    ax.set_xlabel("predicted probability"); ax.set_ylabel("share who really left"); ax.set_title("Reliability of attrition v2 (out-of-fold)"); ax.legend(); plt.tight_layout(); plt.savefig(fig13 / "reliability.png", dpi=140); plt.close()
    # leakage re-check: no single clean feature should predict the target almost perfectly
    model = v2.fit_final(Xtr, ytr, bp, oof_g)
    Xt = model.transform(Xtr); from sklearn.metrics import roc_auc_score
    single = sorted(((roc_auc_score(ytr, Xt[c].fillna(Xt[c].median())), c) for c in Xt.columns if Xt[c].nunique() > 1), key=lambda t: -abs(t[0] - 0.5))[:8]
    pd.DataFrame([{"feature": c, "single_feature_roc_auc": round(a, 3)} for a, c in single]).to_csv(out13 / "leakage_check.csv", index=False)
    # locked test set: used exactly once
    pt = model.predict_proba(Xte)
    from sklearn.linear_model import LogisticRegression as LR
    lr_pipe = v2.Pipeline([("features", v2.build_feature_pipeline()), ("model", v2.baseline_lr())]).fit(Xtr, ytr); pl = lr_pipe.predict_proba(Xte)[:, 1]
    test = pd.DataFrame([{"model": "LogReg balanced (baseline)", **{k: round(v, 3) for k, v in v2.score_all(yte, pl).items() if k != "brier"}},
                         {"model": "Attrition v2 (calibrated)", **{k: round(v, 3) for k, v in v2.score_all(yte, pt).items()}}])
    test.to_csv(out13 / "locked_test_once.csv", index=False)
    (ROOT / "models").mkdir(exist_ok=True)
    model.save(str(ROOT / "models/attrition-v2.joblib"), {"params": bp, "cv_roc_auc": float(f_g.roc_auc.mean()), "trained_rows": int(len(Xtr)), "leakage_columns_in_features": 0,
                                                          "note": "Support signal for HR, never a decision"})
    RES["task13"] = {"best": best, "params": bp, "table": table, "fold_diff_roc": comp["diff"].tolist(), "diff_mean": round(float(comp["diff"].mean()), 4), "diff_sd": round(float(comp["diff"].std()), 4),
                     "wins": int((comp["diff"] > 0).sum()), "calibration": calt.to_dict("records"), "test": test.to_dict("records"), "max_single_feature_auc": round(max(abs(a - 0.5) + 0.5 for a, _ in single), 3)}
    # ---- Task 14: 430 people
    S, SAMPLE = X["attrition_scoring_2026"], X["sample_attrition_risk"]
    Xs = S.drop(columns=[c for c in S.columns if c == v2.TARGET], errors="ignore"); Xst = model.transform(Xs)
    Xshow = Xs.reset_index(drop=True).copy(); Xshow["tenure_years"] = Xshow["tenure_months"] / 12      # real values for the sentences
    prob = model.predict_proba(Xs); sv = ex.shap_values(model.booster, Xst); g = ex.grouped_shap(sv, list(Xst.columns))
    bands = risk_bands(prob); NOTE = "Support signal for HR, never a decision; needs human review. Bands: High = top 10%, Medium = next 20%."
    rows = []
    for i in range(len(S)):
        r = ex.reasons_for_row(g.iloc[i], Xshow.iloc[i])
        rows.append({"prediction_id": f"AR-{i + 1:04d}", "employee_id": S.employee_id.iloc[i], "department_id": S.department_id.iloc[i], "as_of_date": pd.Timestamp(S.observation_date.iloc[i]).date(),
                     "risk_score_12m": round(float(prob[i]), 3), "risk_band": bands.iloc[i], "top_reason_1": r[0], "top_reason_2": r[1], "top_reason_3": r[2], "model_version": v2.VERSION, "model_note": NOTE})
    outdf = pd.DataFrame(rows); assert list(outdf.columns) == list(SAMPLE.columns) and len(outdf) == 430
    outdf.to_csv(out14 / "attrition_risk_2026.csv", index=False)
    outdf.sample(10, random_state=7).sort_values("risk_score_12m", ascending=False).to_csv(out14 / "sample_10_rows_to_read_aloud.csv", index=False)
    imp = pd.DataFrame({"feature_group": g.columns, "mean_abs_shap": g.abs().mean().to_numpy()}).sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(out14 / "shap_importance.csv", index=False)
    top = imp.head(12).iloc[::-1]; fig, ax = plt.subplots(figsize=(7, 4.6)); ax.barh(top.feature_group, top.mean_abs_shap, color="#4c78a8"); ax.set_xlabel("mean |SHAP| (effect on log-odds of leaving)")
    ax.set_title("What drives attrition v2 most (430 employees)"); plt.tight_layout(); plt.savefig(fig14 / "shap_summary.png", dpi=140); plt.close()
    RES["task14"] = {"rows": len(outdf), "bands": outdf.risk_band.value_counts().to_dict(), "columns_match_sample": True, "score_min": float(outdf.risk_score_12m.min()), "score_max": float(outdf.risk_score_12m.max()),
                     "top_drivers": imp.head(5).feature_group.tolist(), "blank_reasons": int((outdf[["top_reason_1", "top_reason_2", "top_reason_3"]] == "").sum().sum())}


# ------------------------------------------------------------------ Task 15
def task15():
    out, fig = T("Task15_Pay_Bands/outputs/tables"), T("Task15_Pay_Bands/outputs/figures")
    S, PA = X["salary_training"], X["protected_attributes"]
    D = pay.prepare(S).merge(PA[PA.record_type == "salary_hire"][["person_ref", "gender"]], left_on="hire_id", right_on="person_ref", how="left")
    D["career_gap_group"] = pd.cut(D.career_gap_months, [-1, 0, 6, 12, 100], labels=["no gap", "1-6 months", "7-12 months", "over 12 months"])
    D["gender_female"] = (D.gender == "Female").astype(int); D["has_gap"] = (D.career_gap_months > 0).astype(int)
    tr, te = train_test_split(D, test_size=0.2, random_state=42)
    kf = KFold(5, shuffle=True, random_state=42)

    def oof_pred(kind, feats_num, feats_cat, frame):
        o = np.zeros(len(frame))
        for a, b in kf.split(frame):
            m = pay.make_model(kind, feats_num, feats_cat).fit(frame.iloc[a][feats_num + feats_cat], np.log(frame.iloc[a].ratio))
            o[b] = np.exp(m.predict(frame.iloc[b][feats_num + feats_cat])) * frame.iloc[b].band_mid_2026
        return o
    y = tr.offered_annual_fixed.to_numpy()
    base_oof = np.zeros(len(tr))
    for a, b in kf.split(tr): base_oof[b] = pay.RoleAverage().fit(tr.iloc[a]).predict(tr.iloc[b])
    cands = {"Baseline: average pay of the role": base_oof, "Linear (Ridge), job-related features": oof_pred("linear", pay.NUM, pay.CAT, tr),
             "Gradient boosting (LightGBM), job-related features": oof_pred("gbm", pay.NUM, pay.CAT, tr)}
    met = pd.DataFrame([{"model": k, **{f"cv_{a}": b for a, b in pay.pay_metrics(y, v).items()}} for k, v in cands.items()])
    best_kind = "linear" if met.iloc[1].cv_MAE_rupees <= met.iloc[2].cv_MAE_rupees else "gbm"
    # locked 20% test, once
    ba = pay.RoleAverage().fit(tr).predict(te); fin = pay.make_model(best_kind).fit(tr[pay.FEATURES], np.log(tr.ratio))
    pt = np.exp(fin.predict(te[pay.FEATURES])) * te.band_mid_2026.to_numpy(); yt = te.offered_annual_fixed.to_numpy()
    met["test_MAE_rupees"] = [pay.pay_metrics(yt, ba)["MAE_rupees"]] + [np.nan, np.nan]; met["test_R2"] = [pay.pay_metrics(yt, ba)["R2"], np.nan, np.nan]
    met.loc[met.model.str.startswith("Linear" if best_kind == "linear" else "Gradient"), ["test_MAE_rupees", "test_R2"]] = [pay.pay_metrics(yt, pt)["MAE_rupees"], pay.pay_metrics(yt, pt)["R2"]]
    met.to_csv(out / "metrics_table.csv", index=False)
    # honest predictions for every row (out-of-fold for train, test predictions for test) for fairness checks
    D["pred_final"] = np.nan; D.loc[tr.index, "pred_final"] = cands[("Linear (Ridge), job-related features" if best_kind == "linear" else "Gradient boosting (LightGBM), job-related features")]; D.loc[te.index, "pred_final"] = pt
    # the variant that is allowed to see gender and career gap = a model that copies history
    nb = pay.NUM + ["gender_female", "has_gap", "career_gap_months"]
    tmp = pd.concat([tr, te]); hist = np.zeros(len(D))
    for a, b in KFold(5, shuffle=True, random_state=42).split(D):
        m = pay.make_model(best_kind, nb, pay.CAT).fit(D.iloc[a][nb + pay.CAT], np.log(D.iloc[a].ratio)); hist[b] = np.exp(m.predict(D.iloc[b][nb + pay.CAT])) * D.iloc[b].band_mid_2026
    D["pred_history"] = hist
    for c in ["pred_final", "pred_history"]: D["pr_" + c] = D[c] / D.band_mid_2026
    D["actual_ratio"] = D.ratio
    for c in ["pred_final", "pred_history"]: D["res_" + c] = D.actual_ratio - D["pr_" + c]
    gaps = []
    for gcol, ref in [("gender", "Male"), ("career_gap_group", "no gap")]:
        for label, pc in [("Actual pay in the data (pay / band mid)", "actual_ratio"), ("Model WITH gender and career gap (copies history)", "pr_pred_history"), ("Final model (job-related only)", "pr_pred_final"), ("Final model: actual minus predicted", "res_pred_final")]:
            t = group_gap(D.dropna(subset=[gcol]), pc, gcol, ref); t.insert(0, "measure", label); t.insert(0, "group_by", gcol)
            t = t.rename(columns={c: "gap_vs_reference" for c in t.columns if c.startswith("gap_vs_")}); gaps.append(t)
    gp = pd.concat(gaps); gp["gap_percent_points_of_band_mid"] = (gp.gap_vs_reference * 100).round(2); gp.to_csv(out / "group_gap_table.csv", index=False)
    # significance of the history gap for career gap (Welch t-test)
    a, b = D[D.career_gap_months > 0].actual_ratio, D[D.career_gap_months == 0].actual_ratio; tt = stats.ttest_ind(a, b, equal_var=False)
    RES["task15_gap_test"] = {"career_gap_mean_ratio": round(float(a.mean()), 4), "no_gap_mean_ratio": round(float(b.mean()), 4), "p_value": float(tt.pvalue), "n_gap": int(len(a)), "n_nogap": int(len(b))}
    # final model and its band: low / mid / high from the 10th and 90th percentile of out-of-fold log errors
    err = np.log(tr.offered_annual_fixed.to_numpy() / cands[("Linear (Ridge), job-related features" if best_kind == "linear" else "Gradient boosting (LightGBM), job-related features")])
    q_lo, q_hi = float(np.quantile(err, 0.10)), float(np.quantile(err, 0.90))
    now = dict(now_market_index=float(D.market_index.max()), now_year=2026)
    eval_model = pay.PayBandModel(pay.make_model(best_kind).fit(tr[pay.FEATURES], np.log(tr.ratio)), q_lo, q_hi, pay.FEATURES, **now)   # trained on train only
    cov = eval_model.recommend(te, as_of_now=False, clip_to_band=False)
    final = pay.PayBandModel(pay.make_model(best_kind).fit(D[pay.FEATURES], np.log(D.ratio)), q_lo, q_hi, pay.FEATURES, **now)
    final.save(str(ROOT / "models/paybands-v1.joblib"), {"kind": best_kind, "q_low": q_lo, "q_high": q_hi, "excluded_features": ["gender", "age", "career_gap_months"]})
    ex_rows = te.head(12); rec = final.recommend(ex_rows)
    pd.concat([ex_rows[["hire_id", "role_id", "job_level", "prior_experience_years", "band_min_2026", "band_max_2026", "offered_annual_fixed"]].reset_index(drop=True), rec.reset_index(drop=True)], axis=1).to_csv(out / "band_examples.csv", index=False)
    inside = float(((te.offered_annual_fixed >= cov.low) & (te.offered_annual_fixed <= cov.high)).mean())
    RES["task15"] = {"best_kind": best_kind, "metrics": met.to_dict("records"), "band_hit_rate_on_test": round(inside, 3), "q_low": round(q_lo, 4), "q_high": round(q_hi, 4)}
    fig_df = gp[(gp.group_by == "career_gap_group") & gp.measure.isin(["Actual pay in the data (pay / band mid)", "Model WITH gender and career gap (copies history)", "Final model (job-related only)"])]
    fig_df = fig_df[fig_df["career_gap_group"] != "no gap"] if "career_gap_group" in fig_df.columns else fig_df
    fig_, ax = plt.subplots(figsize=(7.5, 3.8)); xs = np.arange(3); wdt = 0.27
    for k, (m_, c) in enumerate([("Actual pay in the data (pay / band mid)", "#c0504d"), ("Model WITH gender and career gap (copies history)", "#e0a458"), ("Final model (job-related only)", "#4c8c4a")]):
        d_ = fig_df[fig_df.measure == m_]; ax.bar(xs + (k - 1) * wdt, d_.gap_percent_points_of_band_mid.to_numpy(), wdt, label=m_.split(" (")[0], color=c)
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(xs); ax.set_xticklabels(["1-6 months", "7-12 months", "over 12 months"]); ax.set_ylabel("gap vs no career gap (points of band mid)"); ax.set_title("Pay gap by career gap"); ax.legend(fontsize=7); plt.tight_layout(); plt.savefig(fig / "career_gap_gap.png", dpi=140); plt.close()


if __name__ == "__main__":
    task11_12(); task13_14(); task15()
    (ROOT / "reports/phase3").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports/phase3/results.json").write_text(json.dumps(RES, indent=2, default=str))
    print(json.dumps({k: v for k, v in RES.items()}, indent=1, default=str)[:9000])
