# Task 12: Audit the Screener for Fairness

**Phase 3 (Intelligence) | Status: Done | Code:** `src/fairness.py` (`group_gap`), `src/resume/` | **Tests:** `tests/test_fairness.py` | **Folder:** `phase-3/Task12_Fairness_Audit`

## 1. Objective
Test the Task 11 screener with twin resumes that differ in only one detail, measure any unfair score gap, reduce it, and write an honest fairness statement.

## 2. Why it matters
Fairness is checked, not assumed. A gap of a few points can mean thousands of rejected people over time.

## 3. Input data
`resume_bias_pairs` (25 twin pairs: identical skills and experience, only the gender-coded name differs, and in 13 pairs the college tier too), and `resume_protected_attrs` (which twin is male- or female-coded, and which has the higher tier).

## 4. What was done, step by step
1. Took the 25 twin pairs and scored both twins with the screener.
2. Computed the gap for each pair and the average gap (on the 0 to 100 score).
3. Opened the protected attributes behind the pairs.
4. **Reduced the gap** by masking name, city and e-mail by value, then re-measured the gap and the ranking quality.
5. Wrote the fairness statement.

## 5. Before and after
| Screener | Mean abs gap | Max gap | Pairs with any gap | Precision@5 | NDCG@5 |
|---|---|---|---|---|---|
| Before: naive, reads the whole text | 0.0562 | 0.378 | 25 of 25 | 0.96 | 0.968 |
| **After: masked (final)** | **0.0000** | **0.000** | **0 of 25** | 0.96 | 0.968 |

The naive screener never looks at gender or tier on purpose, yet names, e-mails and "Tier 1/2 institution" words change the TF-IDF vectors, so all 25 twins scored differently. **My first fix was not enough:** cutting the first lines left gaps in 4 pairs, because in two-column and table layouts the name sits on the same line as the first section. Removing name, city and e-mail by value closed them (this is also a unit test).

## 6. Honest fairness statement
- **Tested:** the direct effect of name and college tier, 25 pairs. **Found:** small non-zero gaps before masking, none after.
- **Not fixed by masking:** group averages across all 185 resumes still differ (female-coded resumes score 2.2 points higher on average; Tier 1 resumes 6.7 points above Tier 2), identical before and after masking. They come from the skills and experience in the resumes, not from names. This is a signal to monitor, not proof of bias.
- **Not tested:** other proxies (career breaks, resume layout and parsing quality, non-English names), intersections, and real recruiter outcomes.
- 25 pairs is a small sample, so a zero gap here does not prove fairness elsewhere.

## 7. Think for yourself: who decides the trade-off?
Not the data scientist alone. If a fix lowered accuracy a little, HR leadership decides, with legal and compliance input, using documented numbers (here the fix cost nothing). **Equal accuracy is not equal treatment:** a screener can be 96% accurate overall and still score one group lower, because accuracy averages over everyone while fairness is checked group by group.

## 8. Interview questions
1. **What is a counterfactual test?** Change one irrelevant attribute, keep everything else equal, and check that the output does not change.
2. **Why is "we do not use gender" not a fairness proof?** Text, names and proxies carry it indirectly, so you must measure.
3. **What did masking by position miss?** Layouts where the name is on a section line; fixed by removing the name by value.
