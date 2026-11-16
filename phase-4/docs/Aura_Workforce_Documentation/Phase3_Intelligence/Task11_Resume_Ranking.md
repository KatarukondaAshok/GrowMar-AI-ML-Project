# Task 11: Rank Resumes Against Job Descriptions

**Phase 3 (Intelligence) | Status: Done | Code:** `src/resume/text.py`, `src/resume/scoring.py` | **Tests:** `tests/test_resume.py` | **Folder:** `phase-3/Task11_Resume_Ranking`

## 1. Objective
Score resumes against job descriptions, rank the candidates for each job, and compare the ranking with human judgement, without ever using name, gender, age or college tier.

## 2. Why it matters
Recruiters read hundreds of resumes. A fair, explainable ranking saves time, but only if it never uses who the person is. Hiring tools are among the most legally sensitive uses of AI.

## 3. Input data
`job_descriptions` (10 jobs), `resumes` (185), `resume_job_labels` (555 human-labelled resume-job pairs).

## 4. What was done, step by step
1. **Read 5 resumes and 3 jobs.** A good match has the must-have skills, enough years of experience, and relevant titles.
2. **Mask first.** Name, e-mail, city and college tier are removed before scoring, by value and not by position, because two-column layouts put the name on the same line as the first section.
3. **Extract** skills (word-boundary match against every skill named in a job or on a resume) and years of experience (sum of lines such as `(2.2 yrs)`).
4. **Score each resume-job pair (0 to 100)** with weights fixed in advance and never tuned on the labels: 50% required-skill coverage, 15% nice-to-have coverage, 20% TF-IDF text similarity, 15% experience fit (years divided by the minimum, capped at 1).
5. **Rank per job** and compare with the human labels using top-5 measures.
6. **Collect 5 misranked resumes** and explain each.

## 5. Results (555 labelled pairs, 10 jobs)
| Ranking | Precision@5 | NDCG@5 | Spearman |
|---|---|---|---|
| **Final: skills + similarity + experience** | **0.96** | **0.968** | 0.841 |
| Required-skill coverage only | 0.92 | 0.883 | 0.792 |
| Text similarity only | 0.90 | 0.860 | 0.743 |
| Random order | 0.18 | 0.298 | 0.034 |

## 6. Five misranks explained
Three resumes were ranked too high (human score 1 to 2): they cover only 20% of required skills but still outrank others because the job has few strong candidates. Two were ranked too low (human score 4 to 5): resumes with the layout "Missing sections" have no skills section, so the parser finds only 40% of required skills where the label says 80 to 100%. Lesson: the parser depends on the resume layout, which is itself a fairness risk.

## 7. Think for yourself: rank on years of experience alone?
No. It would disadvantage career changers, people returning from a break (often women) and strong recent graduates. Here experience carries only 15% of the score and is capped, so extra years never outweigh missing must-have skills.

## 8. Ranking endpoint idea (API notes)
`POST /v1/resumes/rank`. Input: a `job_id` and a list of resumes (maximum 200). Output: a ranked list with `resume_id`, `score`, `required_coverage`, `missing_required` and `years_found`; no names and no protected attributes. Every response carries `model_version` and the sentence "support for a recruiter, not a decision".

## 9. Interview questions
1. **Why can a resume screener be unfair without anyone intending it?** Proxies: name, college, city, career gaps, and layouts that parsers cannot read.
2. **Why TF-IDF plus skill overlap rather than a neural model?** It is explainable to a recruiter ("matches 4 of 5 required skills"), needs no training labels and is easy to audit.
3. **What does NDCG@5 measure?** Whether the best candidates are at the top, rewarding high human scores placed earlier.

## 10. Limits
The human labels look closely tied to skill coverage, so the numbers are probably higher than on real recruiter decisions.
