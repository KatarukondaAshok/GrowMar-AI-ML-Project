"""Task 10: multi-turn memory. One small record per conversation; bound to the person who started it."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

MAX_SESSIONS = 2000     # oldest conversation is dropped first, so memory cannot grow without limit
MAX_TURNS = 20


@dataclass
class Session:
    user_id: str | None
    last_intent: str | None = None
    slots: dict = field(default_factory=dict)
    pending: dict | None = None          # {"intent": ..., "missing": [...]} while we wait for an answer
    turns: list = field(default_factory=list)


class SessionStore:
    def __init__(self, max_sessions: int = MAX_SESSIONS):
        self._data: OrderedDict[str, Session] = OrderedDict()
        self.max_sessions = max_sessions

    def get(self, session_id: str, user_id: str | None) -> Session:
        s = self._data.get(session_id)
        if s is None or s.user_id != user_id:      # a different person never inherits someone else's memory
            s = Session(user_id=user_id)
        self._data[session_id] = s
        self._data.move_to_end(session_id)
        while len(self._data) > self.max_sessions:
            self._data.popitem(last=False)
        return s

    def add_turn(self, s: Session, user: str, bot: str):
        s.turns.append({"user": user, "bot": bot})
        del s.turns[:-MAX_TURNS]

    def __len__(self):
        return len(self._data)
