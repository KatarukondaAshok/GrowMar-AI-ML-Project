# Task 9: HR intent classifier with a safe fallback
Code: `src/chat/nlu.py`, `src/chat/faq.py`. Evidence: `outputs/tables/*`, `outputs/figures/confusion_matrix.png`.

## Steps
1. Read the 858 training messages (33 intents), including messy ones (typos, filler like "quick question:").
2. Clean: lowercase, strip filler, drop punctuation, collapse spaces; fix typos by snapping unknown words to close training words (`laeve` -> `leave`).
3. TF-IDF (words 1-2 grams + character 3-5 grams) + logistic regression.
4. Evaluate on `chatbot_test_set` (132 messages, never trained on).
5. Confidence threshold **0.5**, chosen from grouped cross-validation on training data only: answered accuracy 98.1% there. Below it the bot says it is not sure and gives the HR contact; exact FAQ questions are answered from `hr_faq`.
6. `/v1/chat` endpoint in the same API.

## Results
- **Test accuracy 81.8%** (always answering). With the 0.5 threshold the bot answers 65.9% of messages with **93.1%** accuracy and falls back on 34.1%.
- Grouped CV accuracy 94.2%: the test set uses new paraphrases, so it is harder than CV. Trust the test number.
- Threshold table: `outputs/tables/threshold_choice.csv`. Errors: `test_errors.csv`.

## Think for yourself: most confused intents
No pair is confused more than once (errors are scattered over unseen paraphrases). Two families confuse: small talk (`greet`, `help`, `chitchat`) and HR analytics (`attrition_risk_query`, `headcount_forecast`, `salary_band_query`, `resume_score_request`). Fix: add 15 to 20 new paraphrases per intent from real user logs, especially short messages ("yo", "cheers"), and add the near-duplicate analytics intents with distinguishing keywords.

## TF-IDF in two lines
It scores a word high if it is frequent in this message but rare across all messages, so common words count little and distinctive words count a lot. Each message becomes a vector of those scores that a classifier can learn from.

## Interview questions
1. *Why a threshold?* A wrong confident answer is worse than "I am not sure".
2. *Why character n-grams?* Typos still share most letter sequences with the right word.
3. *Why trust test over CV here?* Training messages are templated; CV leaks near-duplicates, test paraphrases do not.
