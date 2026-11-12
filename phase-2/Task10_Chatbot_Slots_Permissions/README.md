# Task 10: Slots, memory and permissions
Code: `src/chat/bot.py`, `slots.py`, `session.py`, `hrclient.py`; mock HR system `mock_services/mock_server.py` (the starter kit's copy was not provided, so I built one from LedgerForge and CortexHR data). Transcripts: `outputs/transcripts.md`.

## What it does
- **Slots:** month (March, "last month"), department, role, leave type, person name. "Last month" counts from the demo date 2026-09-30 (the HR data ends in Sept 2026).
- **Memory:** one session per conversation, bound to the person who started it (another person never inherits it). Follow-ups like "and last month?" reuse the earlier intent. Person names are never remembered.
- **Missing slot:** the bot asks ("Which month do you mean?"), then uses the answer.
- **Mock HR calls** with header `X-Demo-User`; the role comes from the HR system, never from the message ("I am HR" changes nothing).
- **Permissions:** HR-only intents are refused for employees; the mock server enforces the same rules (defence in depth).
- **Always refuses `salary_of_other`**, even for HR, even with tricks, even if the classifier is wrong (a safety net: a person name plus a pay word). No data call is made on a refusal (tested).
- **Chit-chat and policy never trigger a data call** (tested).

## Think for yourself: two people with the same name
In the sample HR data 26 names repeat. Design: for any lookup by name, the HR system returns a **list**; the bot asks "I found 2 people named Priya Nair: EMP-0004 (Customer Support) or EMP-0005 (Engineering). Which one?" using ID and department, never salary. Only HR/Admin may search (`/api/v1/employees/search`, tested). Refusals read the same whether the name exists or not, so nobody can use the bot to find out who works here.

## Tests
`tests/test_dialogues.py`: all 12 sample dialogues, 10 refusal cases (employee and HR), role-from-system, clarifying question, no data call for chit-chat, session isolation, low-confidence fallback. `tests/test_mock_server.py`: 4 tests.

## Limits (honest)
HR-analytics intents (attrition, salary band, headcount, resume score) are recognised and role-checked but answer "not available yet": they belong to Phase 3. Payroll flags cannot be filtered by department yet and the bot says so.

## Interview questions
1. *How do you stop a bot leaking salary?* Permission checks in code, on the server, using the caller's real role; never in a prompt.
2. *How does multi-turn memory work?* A session record with last intent and slots, keyed by session id and user id.
3. *Why ask instead of guess a missing month?* A wrong guess gives a confident wrong answer about someone's pay.
