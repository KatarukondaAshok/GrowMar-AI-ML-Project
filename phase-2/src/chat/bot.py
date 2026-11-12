"""Tasks 9 and 10: the HR assistant. Intent -> slots -> memory -> permission check -> (mock) HR call -> reply.

Rules the bot never breaks:
  * Someone else's pay is always refused (even for HR, even if asked nicely, even if the classifier got it wrong).
  * Data calls only ever ask for the caller's own data (employee=me); the caller's role comes from the HR system,
    never from the words of the message.
  * Chit-chat, help and policy questions make no data call.
  * A missing detail (which month?) triggers a question, not a guess.
"""
from __future__ import annotations

import re
from datetime import date

from src.chat.faq import FaqRetriever
from src.chat.hrclient import HRClient, HRServiceError
from src.chat.session import SessionStore
from src.chat.slots import extract_slots, mentions_other_persons_pay, month_label

FAQ_EXACT = 0.8             # near-exact match with a stored FAQ question
FAQ_OVERRIDES_BELOW = 0.9   # ...used only when the intent model is below this confidence
FALLBACK = ("I am not sure I understood that, so I would rather not guess. Could you rephrase it? "
            "You can also write to hr@znova-demo.example.")
SIGN_IN = "Please sign in first so I can look up your own HR data."
NO_ACCESS = "Sorry, that information is only available to {who}."
NOT_BUILT = "That analysis is not available in this version yet. Please contact HR for now."
REFUSAL = "Sorry, I cannot share another person's pay. You can ask about your own payslip."

REQUIRED = {"payslip_status": ["month"], "attendance_summary": ["month"], "unusual_attendance": ["month"],
            "payroll_flags": ["month"], "attrition_risk_query": ["department"], "salary_band_query": ["role"],
            "resume_score_request": ["role"]}
ASK = {"month": "Which month do you mean? For example: March, or last month.",
       "department": "Which department do you mean?", "role": "Which role do you mean?"}
NOT_BUILT_INTENTS = {"attrition_risk_query", "salary_band_query", "headcount_forecast", "resume_score_request"}
KEEP_IN_MEMORY = ("month", "department", "role", "leave_type")      # person names are never remembered
FOLLOW_UP = re.compile(r"^(and|what about|how about|also|then)\b|\bonly\??$", re.I)
SMALL_TALK = {"hi": "greet", "hii": "greet", "hello": "greet", "hey": "greet", "yo": "greet", "hey there": "greet",
              "good morning": "greet", "good evening": "greet", "bye": "goodbye", "goodbye": "goodbye", "see you": "goodbye",
              "thanks": "thanks", "thank you": "thanks", "thx": "thanks", "ty": "thanks", "cheers": "thanks",
              "appreciate it": "thanks"}      # a tiny lexicon: one-word messages give a text model almost nothing to use
DETAIL = {"show details", "details", "why is it high", "why", "tell me more", "show more"}


class ChatEngine:
    def __init__(self, classifier, resources: dict, hr: HRClient, anomaly=None, today: date | None = None,
                 sessions: SessionStore | None = None):
        self.clf, self.res, self.hr, self.anomaly = classifier, resources, hr, anomaly
        self.today = today or date.today()
        self.sessions = sessions or SessionStore()
        self.faq = FaqRetriever(resources["faq"])
        self.threshold = classifier.threshold

    # ------------------------------------------------------------------ public
    def reply(self, session_id: str, message: str, user_id: str | None = None) -> dict:
        s = self.sessions.get(session_id, user_id)
        slots = extract_slots(message, self.res, self.today)
        intent, conf, source = self._decide(message, slots, s)
        out = self._respond(intent, conf, source, slots, s, message, user_id)
        for k in KEEP_IN_MEMORY:
            if k in slots:
                s.slots[k] = slots[k]
        if out["intent"] and not out.get("clarify") and out["intent"] not in ("greet", "thanks", "goodbye"):
            s.last_intent = out["intent"]
        self.sessions.add_turn(s, message, out["reply"])
        return out

    # ---------------------------------------------------------------- decide
    def _decide(self, message, slots, s):
        low = message.lower().strip(" ?!.")
        if mentions_other_persons_pay(message, slots):
            return "salary_of_other", 1.0, "safety_net"
        if s.pending and any(m in slots for m in s.pending["missing"]):
            return s.pending["intent"], 1.0, "answer_to_question"
        if s.last_intent and (low in DETAIL or (FOLLOW_UP.search(low) and slots)):
            return s.last_intent, 1.0, "follow_up"
        if low in SMALL_TALK:
            return SMALL_TALK[low], 1.0, "small_talk"
        intents, confs, best = self.clf.predict([message])
        row, sim = self.faq.best(message)
        personal = intents[0] is not None and self.res["responses"][intents[0]]["requires_login"] == "Y"
        if row is not None and sim >= FAQ_EXACT and not personal and confs[0] < FAQ_OVERRIDES_BELOW:
            return "faq", sim, "faq_exact"           # the message IS a stored FAQ question and the model is not sure
        if intents[0] is None:
            if s.last_intent and slots:
                return s.last_intent, float(confs[0]), "follow_up"
            return None, float(confs[0]), "low_confidence"
        return intents[0], float(confs[0]), "classifier"

    # --------------------------------------------------------------- respond
    def _respond(self, intent, conf, source, slots, s, message, user):
        base = {"intent": intent, "confidence": round(conf, 3), "slots": {}, "clarify": False, "refused": False,
                "source": source}
        if intent == "faq":
            row, _ = self.faq.best(message)
            return {**base, "reply": f"{row['answer']} (Source: {row['policy_source']})"}
        if intent is None:                                   # not sure: try the FAQ, else fall back politely
            row, sim = self.faq.best(message)
            if row:
                return {**base, "intent": "faq", "reply": f"{row['answer']} (Source: {row['policy_source']})", "source": "faq"}
            return {**base, "reply": FALLBACK, "source": "fallback"}
        spec = self.res["responses"][intent]
        merged = {**{k: v for k, v in s.slots.items()}, **slots}
        base["slots"] = {k: v for k, v in merged.items() if k in KEEP_IN_MEMORY}
        call = spec["backend_or_service_call"]
        if call == "REFUSE":                                  # salary_of_other: never answered, for anyone
            return {**base, "reply": spec["response_template"], "refused": True}
        if spec["requires_login"] == "Y" and not user:
            return {**base, "reply": SIGN_IN, "refused": True}
        allowed = spec["allowed_roles"]
        if allowed != "ALL" and user:
            role = self._role(user)
            if role is None:
                return {**base, "reply": spec["on_failure"] or FALLBACK}
            if role not in [r.strip() for r in allowed.split(",")]:
                who = "HR or Admin" if "EMPLOYEE" not in allowed else "signed-in staff"
                s.pending = None
                return {**base, "reply": NO_ACCESS.format(who=who), "refused": True}
        missing = [m for m in REQUIRED.get(intent, []) if m not in merged]
        if missing:
            s.pending = {"intent": intent, "missing": missing}
            return {**base, "reply": ASK[missing[0]], "clarify": True}
        s.pending = None
        if call is None or str(call).startswith("faq:"):      # policy / chit-chat: no data call at all
            return {**base, "reply": self._faq_or_template(spec)}
        if intent in NOT_BUILT_INTENTS:
            return {**base, "reply": NOT_BUILT, "source": "not_built"}
        try:
            text = self._with_data(intent, spec, merged, user, source, slots)
        except HRServiceError:
            text = spec["on_failure"] or FALLBACK
        return {**base, "reply": text}

    # ------------------------------------------------------------- helpers
    def _faq_or_template(self, spec):
        return spec["response_template"].replace("{next_holiday}", "the next holiday on the calendar")

    def _role(self, user):
        try:
            return self.hr.get(user, "/api/v1/employees/me")["role"].upper()
        except HRServiceError:
            return None

    def _with_data(self, intent, spec, m, user, source, cur=None):
        t = spec["response_template"]
        if intent == "leave_balance":
            b = self.hr.get(user, "/api/v1/leave/balance", {"employee": "me"})
            if m.get("leave_type") and source in ("follow_up", "classifier") and m.get("leave_type"):
                k = m["leave_type"]
                return f"You have {b[k.lower()]:g} {k} days left."
            return t.format(cl=f"{b['cl']:g}", sl=f"{b['sl']:g}", el=f"{b['el']:g}")
        if intent == "leave_request_status":
            r = self.hr.get(user, "/api/v1/leave/requests", {"employee": "me", "limit": 1})["requests"]
            return t.format(dates=r[0]["dates"], status=r[0]["status"]) if r else "You have no leave requests on record."
        if intent == "payslip_status":
            p = self.hr.get(user, f"/api/v1/payslips/me/{m['month']}")
            return t.format(month=month_label(m["month"]), status=p["status"])
        if intent in ("my_manager", "my_department"):
            e = self.hr.get(user, "/api/v1/employees/me")
            if intent == "my_manager":
                return t.format(manager_name=e["manager_name"]) if e.get("manager_name") else "I could not find a manager on record for you."
            return t.format(department_name=e["department_name"])
        if intent == "team_members":
            tm = self.hr.get(user, "/api/v1/employees/me/team")["direct_reports"]
            return t.format(team_list=", ".join(tm)) if tm else "You have no direct reports on record."
        if intent == "attendance_summary":
            a = self.hr.get(user, "/api/v1/attendance/summary", {"employee": "me", "month": m["month"]})
            return t.format(month=month_label(m["month"]), present=a["present"], scheduled=a["scheduled"], late=a["late"])
        if intent == "unusual_attendance":
            rec = self.hr.get(user, "/api/v1/attendance/records", {"employee": "me", "month": m["month"]})["record"]
            res = self.anomaly.attendance([rec]) if self.anomaly else []
            if res:
                return (f"Your attendance for {month_label(m['month'])} looks unusual: {res[0]['reason_text']}. "
                        "Flags are leads to check, not conclusions.")
            return f"Nothing looks unusual in your attendance for {month_label(m['month'])}."
        if intent == "holiday_list":
            h = self.hr.get(user, "/api/v1/holidays", {"after": self.today.isoformat()})["holidays"]
            return t.format(next_holiday=f"{h[0]['holiday_name']} on {h[0]['holiday_date']}") if h else \
                self._faq_or_template(spec)
        if intent == "payroll_flags":
            items = self.hr.get(user, "/api/v1/payroll/items", {"month": m["month"]})["items"]
            if not items:
                return f"There is no payroll data for {month_label(m['month'])} yet."
            res = self.anomaly.payroll(items) if self.anomaly else []
            if not res:
                return f"No payslips were flagged for {month_label(m['month'])}."
            text = t.format(count=len(res), month=month_label(m["month"]), reason=res[0]["reason_text"])
            if cur and cur.get("department"):
                text = "I cannot filter payroll flags by department yet, so this covers everyone. " + text
            if source == "follow_up":
                lines = [f"{r['record_id']} ({r['employee_id']}): {r['reason_text']}, about Rs {r['money_at_risk']:,.0f} at risk"
                         for r in res[:3]]
                return text + " Top flags: " + "; ".join(lines) + ". Flags are leads to check, not conclusions."
            return text
        return t
