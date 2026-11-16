"""Task 11: score resumes against jobs (TF-IDF similarity + skill overlap + experience fit) and rank them."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.resume.text import build_vocabulary, extract_skills, extract_years, mask_resume, split_vocab

# Fixed in advance from how a recruiter reads a resume (must-have skills first); never tuned on the labels.
WEIGHTS = {"required": 0.50, "nice": 0.15, "similarity": 0.20, "experience": 0.15}


class ResumeScreener:
    """masked=True is the fair screener (no name, city, e-mail, college tier). masked=False reads the raw text
    and exists only so Task 12 can show what the audit finds in a naive build."""

    def __init__(self, masked: bool = True, weights: dict | None = None):
        self.masked, self.weights = masked, weights or WEIGHTS

    def fit(self, jobs: pd.DataFrame, resumes: pd.DataFrame):
        self.jobs_ = jobs.set_index("job_id")
        self.vocab_ = build_vocabulary(jobs, resumes)
        docs = [self._doc(r.resume_text, r.candidate_name, r.city) for r in resumes.itertuples()]
        self.job_text_ = {j: self._job_doc(r) for j, r in self.jobs_.iterrows()}
        self.tfidf_ = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english").fit(docs + list(self.job_text_.values()))
        return self

    def _doc(self, text: str, name=None, city=None) -> str:
        return mask_resume(text, name, city) if self.masked else str(text)

    @staticmethod
    def _job_doc(row) -> str:
        return f"{row['job_title']}. {row['required_skills']}. {row['nice_to_have_skills']}. {row['description_text']}"

    def profile(self, resumes: pd.DataFrame) -> pd.DataFrame:
        """What we read from each resume: skills and years."""
        rows = []
        for r in resumes.itertuples():
            rid, m = r.resume_id, mask_resume(r.resume_text, r.candidate_name, r.city)
            rows.append({"resume_id": rid, "skills_found": sorted(extract_skills(m, self.vocab_)), "years_found": extract_years(m)})
        return pd.DataFrame(rows).set_index("resume_id")

    def score_pairs(self, resumes: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
        """Score each (resume_id, job_id) pair. Returns the components and the 0-100 score."""
        res = resumes.set_index("resume_id")
        prof = self.profile(resumes)
        docs = {rid: self._doc(res.loc[rid, "resume_text"], res.loc[rid, "candidate_name"], res.loc[rid, "city"]) for rid in res.index}
        Rv = self.tfidf_.transform([docs[r] for r in res.index]); ridx = {r: i for i, r in enumerate(res.index)}
        jids = list(self.job_text_); Jv = self.tfidf_.transform([self.job_text_[j] for j in jids]); jidx = {j: i for i, j in enumerate(jids)}
        sim = cosine_similarity(Rv, Jv)
        sim_max = sim.max(axis=0).clip(min=1e-9)
        out = []
        for rid, jid in zip(pairs["resume_id"], pairs["job_id"]):
            job = self.jobs_.loc[jid]
            req, nice = set(split_vocab(job["required_skills"])), set(split_vocab(job["nice_to_have_skills"]))
            have = set(prof.loc[rid, "skills_found"])
            req_cov = len(req & have) / max(len(req), 1); nice_cov = len(nice & have) / max(len(nice), 1)
            yrs = prof.loc[rid, "years_found"]; need = float(job["min_experience_years"])
            exp_fit = 1.0 if need <= 0 else min(1.0, (yrs or 0.0) / need)
            s = sim[ridx[rid], jidx[jid]] / sim_max[jidx[jid]]
            w = self.weights
            score = 100 * (w["required"] * req_cov + w["nice"] * nice_cov + w["similarity"] * s + w["experience"] * exp_fit)
            out.append({"resume_id": rid, "job_id": jid, "required_coverage": round(req_cov, 3), "nice_coverage": round(nice_cov, 3),
                        "similarity": round(float(s), 3), "experience_fit": round(exp_fit, 3), "years_found": yrs,
                        "score": round(score, 3), "missing_required": sorted(req - have)})
        return pd.DataFrame(out)

    def rank(self, resumes: pd.DataFrame, job_id: str, top: int = 5) -> pd.DataFrame:
        pairs = pd.DataFrame({"resume_id": resumes["resume_id"], "job_id": job_id})
        return self.score_pairs(resumes, pairs).sort_values(["score", "resume_id"], ascending=[False, True]).head(top)


def ranking_quality(scored: pd.DataFrame, labels: pd.DataFrame, k: int = 5) -> dict:
    """Top-k quality per job against the human labels: precision@k (shortlist=Y), mean human score of the top k, NDCG@k."""
    m = scored.merge(labels, on=["resume_id", "job_id"])
    prec, mean_h, ndcg = [], [], []
    for _, g in m.groupby("job_id"):
        g = g.sort_values(["score", "resume_id"], ascending=[False, True])
        top = g.head(k)
        prec.append((top["shortlist"] == "Y").mean()); mean_h.append(top["human_suitability_score"].mean())
        gains = (2 ** top["human_suitability_score"] - 1).to_numpy(); disc = 1 / np.log2(np.arange(2, len(top) + 2))
        ideal = np.sort((2 ** g["human_suitability_score"] - 1).to_numpy())[::-1][:k]
        ndcg.append((gains * disc).sum() / max((ideal * disc[: len(ideal)]).sum(), 1e-9))
    return {"precision_at_5": round(float(np.mean(prec)), 3), "mean_human_score_top5": round(float(np.mean(mean_h)), 3),
            "ndcg_at_5": round(float(np.mean(ndcg)), 3), "spearman": round(float(m["score"].corr(m["human_suitability_score"], method="spearman")), 3)}
