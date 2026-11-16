# Task 10: Smart Chatbot: Slots, Memory and Permissions

**Phase 2 (Core Development) | Status: Done | Code:** `src/chat/bot.py`, `slots.py`, `session.py`, `hrclient.py`; `mock_services/mock_server.py` | **Tests:** `tests/test_dialogues.py`, `tests/test_mock_server.py` | **Folder:** `phase-2/Task10_Chatbot_Slots_Permissions`

## 1. Objective
Make the chatbot handle real conversations: pick out months and names, remember them across turns, call a mock HR system, and enforce who may ask what.

## 2. Why it matters
Real conversations are not one-shot. And a bot that leaks someone's salary is a serious incident.

## 3. What was built
- **Slots:** month ("March", "last month"), department, role, leave type, person name. "Last month" counts from the demo date 2026-09-30.
- **Memory:** one session per conversation, bound to the person who started it. Follow-ups like "and last month?" reuse the earlier intent. Person names are never remembered.
- **Missing slot:** the bot asks ("Which month do you mean?") and uses the answer.
- **Mock HR calls** with the header `X-Demo-User`. The role comes from the HR system, never from the message, so "I am HR" changes nothing.
- **Permissions:** HR-only intents are refused for employees, and the mock server enforces the same rules (defence in depth).
- **Always refuses `salary_of_other`,** even for HR, even with tricks, even if the classifier is wrong (a safety net: a person name plus a pay word). A refusal makes no data call.
- **Chit-chat and policy never trigger a data call.**

The starter kit's mock service was not supplied, so one was built from the workbook data.

## 4. Tests
All 12 sample dialogues, 10 refusal cases, role-from-system, the clarifying question, no data call for chit-chat, session isolation, and the low-confidence fallback, plus 4 mock-server tests.

## 5. Think for yourself: two people with the same name
In the sample data 26 names repeat. For any lookup by name the HR system returns a list, and the bot asks: "I found 2 people named Priya Nair: EMP-0004 (Customer Support) or EMP-0005 (Engineering). Which one?" It uses ID and department, never salary. Only HR or Admin may search. Refusals read the same whether the name exists or not, so nobody can use the bot to find out who works here.

## 6. Deliverables
`outputs/transcripts.md` (two multi-turn chats), the refusal test, the mock HR service.

## 7. Interview questions
1. **How do you stop a bot leaking salary?** Permission checks in code, on the server, using the caller's real role; never in a prompt.
2. **How does multi-turn memory work?** A session record with the last intent and slots, keyed by session id and user id.
3. **Why ask instead of guessing a missing month?** A wrong guess gives a confident wrong answer about someone's pay.

## 8. Limits
HR-analytics intents (attrition, salary band, headcount, resume score) are recognised and role-checked but answer "not available yet"; they belong to Phase 3. Payroll flags cannot yet be filtered by department.
