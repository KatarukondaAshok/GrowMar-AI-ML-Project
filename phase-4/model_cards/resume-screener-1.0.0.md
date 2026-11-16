# Model card: Resume screener (`resume-screener-1.0.0`)

| Field | Value |
|---|---|
| Version | resume-screener-1.0.0 |
| Owner | Katarukonda Ashok (GM00154), AI/ML Engineer, Aura Workforce (CortexHR) |
| Status | Released as v1.0.0 for review; a support signal for people, never a decision |

## Purpose
Ranks resumes against a job description to save recruiter time.

## Intended use and out-of-scope use
- **Use:** A recruiter reads the ranked list, the matched and missing skills, and decides.
- **Do not use for:** Never to reject anyone automatically.

## Data
10 jobs, 185 resumes, 555 human-labelled pairs; 25 twin pairs for the fairness test. All data is fictional (Zenith Nova Technologies).

## Method
Masks name, e-mail, city and college tier first; score = 50% required skills + 15% nice-to-have + 20% TF-IDF similarity + 15% experience fit (weights fixed in advance).

## Metrics
Top-5 precision 0.96 and NDCG@5 0.968 against human labels (random order: 0.18 and 0.30).

## Fairness
Twin test: average gap 0.056 points before masking, exactly 0 after; 25 of 25 pairs differ before, 0 after. Group averages (gender, college tier) still differ because of skills in the pool, not names.

## Limits (honest)
Labels look tied to skill coverage, so quality is probably optimistic. Resumes without a skills section are under-scored (layout bias). 25 pairs is a small test. Other proxies (career gaps, non-English names) are untested.

## Monitoring
Twin gap re-run every week (Task 18).
