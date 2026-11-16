# Model card: HR chatbot intent classifier (`intent-tfidf-lr-1.0.0`)

| Field | Value |
|---|---|
| Version | intent-tfidf-lr-1.0.0 |
| Owner | Katarukonda Ashok (GM00154), AI/ML Engineer, Aura Workforce (CortexHR) |
| Status | Released as v1.0.0 for review; a support signal for people, never a decision |

## Purpose
Understands what an employee asks (33 intents) and answers FAQs, with a safe fallback.

## Intended use and out-of-scope use
- **Use:** Employees ask about leave, payslips, attendance and policy.
- **Do not use for:** Never to give someone else's salary; never to answer HR-only questions for employees.

## Data
`chatbot_nlu_training` 858 messages; `chatbot_test_set` 132 unseen paraphrases; `hr_faq`. All data is fictional (Zenith Nova Technologies).

## Method
Text cleaning, TF-IDF (words and characters), logistic regression, confidence threshold 0.5 (chosen on training data only), role and permission checks in code, session memory.

## Metrics
Test accuracy 81.8%. At threshold 0.5 it answers 65.9% of messages with 93.1% accuracy and falls back politely on the rest. 12 sample dialogues pass; salary of others is always refused.

## Fairness
Permissions come from the HR system's role, never from the message. Refusals read the same whether a name exists or not.

## Limits (honest)
About 1 in 5 unseen messages is misunderstood without the threshold. Analytics intents answer 'not available yet' in chat.

## Monitoring
Share of fallbacks and refusals per week.
