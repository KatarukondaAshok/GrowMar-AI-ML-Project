"""Task 17 'think for yourself': 100 people call at once. Usage: python scripts/load_test.py http://127.0.0.1:8000 [calls]"""
import os, sys, time
from concurrent.futures import ThreadPoolExecutor

import httpx

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 100
H = {"X-API-Key": os.environ["AURA_API_KEY"]} if os.environ.get("AURA_API_KEY") else {}
BODY = {"job_id": "J04", "resumes": [{"resume_id": "A", "resume_text": "SUMMARY\nML engineer with 5 years of experience.\nSKILLS\nPython, SQL, Machine Learning\n"}]}


def one(i):
    t = time.time()
    try:
        r = httpx.post(BASE + "/v1/resumes/rank", json=BODY, headers=H, timeout=120)
        return r.status_code, time.time() - t
    except Exception as e:
        return type(e).__name__, time.time() - t


t0 = time.time()
with ThreadPoolExecutor(max_workers=N) as ex:
    res = list(ex.map(one, range(N)))
secs = sorted(s for _, s in res)
ok = sum(1 for c, _ in res if c == 200)
print(f"{N} parallel calls in {time.time() - t0:.1f}s: {ok} ok, {N - ok} failed; median {secs[len(secs) // 2]:.2f}s, slowest {secs[-1]:.2f}s")
print("failures:", {c for c, _ in res if c != 200} or "none")
