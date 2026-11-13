# Task 11: Rank resumes against job descriptions
Code: `src/resume/text.py`, `src/resume/scoring.py`. Evidence: `outputs/tables/`. **Result: top-5 precision 0.96 and NDCG@5 0.968 against the human labels (random order: 0.18 and 0.30).**

## Steps
1. **Read 5 resumes and 3 jobs.** A good match = the must-have skills, enough years of experience, relevant titles.
2. **Mask first.** Name, e-mail, city and college tier are removed before anything is scored (`mask_resume`). They are removed *by value*, not by position: two-column and table layouts put the name on the same line as the first section (see Task 12).
3. **Extract** skills (word-boundary match against every skill named in a job or on a resume) and years of experience (sum of `(2.2 yrs)` lines). Checked against the labels' own coverage columns: required-coverage error 0.026, nice-to-have 0.007; years error 0.39.
4. **Score each resume-job pair (0-100)** with weights fixed in advance, never tuned on the labels: 50% required-skill coverage, 15% nice-to-have coverage, 20% TF-IDF cosine similarity, 15% experience fit (years / minimum, capped at 1).
5. **Rank per job** and compare with `resume_job_labels` using top-5 measures.

## Ranking quality (555 labelled pairs, 10 jobs)
| Ranking | Precision@5 | Mean human score of top 5 | NDCG@5 | Spearman |
|---|---|---|---|---|
| **Final: skills + similarity + experience** | **0.96** | 4.36 | **0.968** | 0.841 |
| Required-skill coverage only | 0.92 | 4.26 | 0.883 | 0.792 |
| Text similarity only (TF-IDF) | 0.90 | 4.18 | 0.860 | 0.743 |
| Random order | 0.18 | 2.30 | 0.298 | 0.034 |

Caveat: the human labels look closely tied to skill coverage, so the numbers are probably higher than on real recruiter decisions.

## Five misranks explained (`outputs/tables/misranks_explained.csv`)
Three resumes ranked **too high** (human score 1-2): they cover only 20% of required skills but still outrank others, because the job has few strong candidates and experience/similarity add points. Two ranked **too low** (human score 4-5): resumes with the layout "Missing sections" have no skills section, so the parser finds only 40% of required skills where the label says 80-100%. Lesson: the parser depends on the resume layout, which is a fairness risk by itself.

## Think for yourself: years of experience alone?
No. Experience alone would disadvantage career changers, people returning from a break (often women), and recent graduates with strong skills. Here it carries only 15% of the score and is capped, so extra years never outweigh missing must-have skills.

## Never used in the score
Name, gender, age, e-mail, city, college tier. Tests prove masking works in every layout and that changing the tier changes nothing (`tests/test_resume.py`).

## Ranking endpoint idea (API notes, not built)
`POST /v1/resumes/rank`. Input: `{"job_id": "J04", "resumes": [{"resume_id": "...", "candidate_name": "...", "city": "...", "resume_text": "..."}], "top": 5}` (max 200 resumes). Output: ranked list with `resume_id`, `score`, `required_coverage`, `missing_required` and `years_found`; **no names, no protected attributes**. Every response carries `model_version` and the sentence "support for a recruiter, not a decision".

## Interview questions
1. *Why can a resume screener be unfair without anyone intending it?* Proxies: name, college, city, gaps in experience, layouts that parsers cannot read.
2. *Why TF-IDF plus skill overlap rather than a neural model?* Explainable to a recruiter ("matches 4 of 5 required skills"), no training labels needed, easy to audit.
3. *What does NDCG@5 measure?* Whether the best candidates are at the top, rewarding high human scores placed earlier.
