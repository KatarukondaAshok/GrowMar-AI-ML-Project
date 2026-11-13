# Task 12: Audit the screener for fairness
Evidence: `outputs/tables/`. **Result: the average twin gap fell from 0.056 points (all 25 pairs differ) to exactly 0 (no pair differs), and ranking quality did not change (0.96 / 0.968).**

## Method (counterfactual twin test)
1. Take the 25 pairs in `resume_bias_pairs`: identical skills and experience, only the gender-coded name (and in 13 pairs the college tier) differ. A fair screener must give both the same score.
2. Score both twins, compute the gap per pair and the average. Gaps are on the 0-100 score.
3. Use `resume_protected_attrs` (the unlock) to see which twin is male-/female-coded and which has the higher tier.
4. Reduce the gap by masking, re-measure the gap **and** the ranking quality.

## Before and after
| Screener | Mean abs gap | Max gap | Pairs with any gap | Mean male minus female | Precision@5 | NDCG@5 |
|---|---|---|---|---|---|---|
| Before: naive, reads the whole resume text | 0.0562 | 0.378 | 25 of 25 | +0.027 | 0.96 | 0.968 |
| **After: masked (final)** | **0.0000** | **0.000** | **0 of 25** | **0.000** | 0.96 | 0.968 |

**What I found:** the naive screener never looks at gender or tier on purpose, yet names, e-mails and "Tier 1/2 institution" words change the TF-IDF vectors, so all 25 twins scored differently (largest gap 0.38 points). The gap is small here, but in a 10-point shortlist cut-off, small gaps decide who is in. **My first fix was not enough:** cutting the first lines of the resume left gaps in 4 pairs, because in two-column and table layouts the name sits on the same line as the first section. Removing name, city and e-mail *by value* closed them (this is also a unit test).

## What remains (honest fairness statement)
- **Tested:** direct effect of name (gender-coded) and college tier on the score, 25 pairs, one job each. **Found:** small non-zero gaps before masking, none after.
- **Not fixed by masking:** group averages over all 185 resumes differ (female-coded resumes score 2.2 points higher on average; Tier 1 resumes 6.7 points above Tier 2) and are *identical before and after masking*. They come from the skills and experience in the resumes (and the job mix), not from names. This is a signal to monitor, not proof of bias, and it needs a human look at the job requirements.
- **Not tested:** other proxies (career breaks, employment gaps, resume layout/parsing quality, non-English names), intersections, and real recruiter outcomes.
- 25 pairs is a small sample, so a zero gap here does not prove fairness elsewhere.

## Think for yourself: who decides the trade-off?
Not the data scientist alone. If a fix lowered accuracy a little, the decision belongs to HR leadership with legal/compliance input, using the documented numbers (here the fix cost nothing). Equal accuracy is not equal treatment: a screener can be 96% accurate overall and still score one group lower; accuracy averages over everyone, fairness is checked group by group.

## Interview questions
1. *What is a counterfactual test?* Change one irrelevant attribute, keep everything else equal, and check that the output does not change.
2. *Why is "we do not use gender" not a fairness proof?* Text, names and proxies carry it indirectly; you must measure.
3. *What did masking by position miss?* Layouts where the name is on a section line; fixed by removing the name by value.
