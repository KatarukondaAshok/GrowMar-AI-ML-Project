"""Task 10: calls to the (mock) HR system. Only ever asks about the caller: employee=me."""
from __future__ import annotations


class HRServiceError(Exception):
    def __init__(self, status: int, detail: str = ""):
        super().__init__(f"HR service answered {status}: {detail}")
        self.status = status


class HRClient:
    """`http` is anything with .get(url, params=, headers=) -> response (httpx.Client or a FastAPI TestClient)."""

    def __init__(self, http):
        self.http = http
        self.calls: list[tuple[str, str]] = []      # (user, path): lets tests prove chit-chat makes no data call

    def get(self, user: str | None, path: str, params: dict | None = None) -> dict:
        self.calls.append((user, path))
        try:
            r = self.http.get(path, params=params, headers={"X-Demo-User": user} if user else {})
        except Exception as e:                       # network down, timeout, ...
            raise HRServiceError(503, str(e)) from e
        if r.status_code >= 400:
            raise HRServiceError(r.status_code, r.text[:200])
        return r.json()
