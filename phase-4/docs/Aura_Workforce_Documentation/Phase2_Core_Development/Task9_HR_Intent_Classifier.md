# Task 9: HR Intent Classifier with a Safe Fallback

**Phase 2 (Core Development) | Status: Done | Code:** `src/chat/nlu.py`, `src/chat/faq.py` | **Folder:** `phase-2/Task9_HR_Intent_Classifier`

## 1. Objective
Understand what an employee is asking (the "intent"), answer FAQ questions, and say "I am not sure" safely when the model is unsure.

## 2. Why it matters
Employees ask the same HR questions every day. A good assistant saves hours and gives consistent answers. A wrong confident answer is worse than "I am not sure".

## 3. Input data
`chatbot_nlu_training` (858 messages, 33 intents, including typos and filler like "quick question:"), `chatbot_test_set` (132 messages, never trained on) and `hr_faq`.

## 4. What was done, step by step
1. Read the training messages, including the messy ones.
2. **Cleaned the text:** lowercase, strip filler, drop punctuation, collapse spaces, snap unknown words to close training words (`laeve` to `leave`).
3. **TF-IDF** (word 1-2 grams plus character 3-5 grams) with **logistic regression**.
4. Evaluated on the test set with accuracy and a confusion matrix.
5. **Confidence threshold 0.5**, chosen by grouped cross-validation on training data only. Below it the bot says it is not sure and offers the HR contact.
6. Exact FAQ questions are answered from `hr_faq`.
7. Added the `/v1/chat` endpoint to the service.

## 5. Results
- **Test accuracy 81.8%** when always answering.
- With the 0.5 threshold the bot answers 65.9% of messages at **93.1%** accuracy and falls back on 34.1%.
- Grouped CV accuracy was 94.2%. The test set uses new paraphrases, so it is harder than CV. Trust the test number.

## 6. TF-IDF in two lines
It scores a word high if it is frequent in this message but rare across all messages, so common words count little and distinctive words count a lot. Each message becomes a vector of those scores that a classifier can learn from.

## 7. Think for yourself: which two intents are confused most?
No pair is confused more than once; errors are scattered over unseen paraphrases. Two families confuse: small talk (`greet`, `help`, `chitchat`) and HR analytics (`attrition_risk_query`, `headcount_forecast`, `salary_band_query`, `resume_score_request`). Fix: add 15 to 20 new paraphrases per intent from real user logs, especially short messages ("yo", "cheers"), and add distinguishing keywords.

## 8. Interview questions
1. **Why a threshold?** A wrong confident answer is worse than "I am not sure".
2. **Why character n-grams?** Typos still share most letter sequences with the right word.
3. **Why trust test over CV here?** Training messages are templated, so CV leaks near-duplicates; the test paraphrases do not.

## 9. Limits
81.8% accuracy means about 1 in 5 unseen messages is misunderstood without the threshold. The fallback is what keeps the bot safe.
