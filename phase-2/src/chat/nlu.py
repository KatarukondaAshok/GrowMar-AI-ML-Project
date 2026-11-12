"""Task 9: clean messy text, TF-IDF + logistic regression intent classifier, confidence threshold with fallback."""
from __future__ import annotations

import difflib
import json
import re
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

SEED = 42
VERSION = "intent-tfidf-lr-1.0.0"
FILLERS = [r"quick question:?", r"i want to know", r"\bplease\b", r"\bpls\b", r"\burgently\b", r"\bkindly\b"]


def clean_text(text: str) -> str:
    """Lowercase, strip filler phrases, drop punctuation noise, collapse spaces."""
    t = str(text).lower().strip()
    for f in FILLERS:
        t = re.sub(f, " ", t)
    t = re.sub(r"[^a-z0-9&' ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


class TypoFixer:
    """Replace a word that is not in the training vocabulary by its closest vocabulary word (if very close)."""

    def __init__(self, min_count: int = 3, cutoff: float = 0.8):
        self.min_count, self.cutoff = min_count, cutoff

    def fit(self, texts):
        c = Counter(w for t in texts for w in t.split())
        self.known_ = {w for w, n in c.items() if n >= self.min_count}
        self.all_ = set(c)
        return self

    def fix(self, text: str) -> str:
        out = []
        for w in text.split():
            if w in self.all_ or len(w) < 4 or w.isdigit():
                out.append(w); continue
            m = difflib.get_close_matches(w, self.known_, n=1, cutoff=self.cutoff)
            out.append(m[0] if m else w)
        return " ".join(out)


class IntentClassifier:
    """Words (1-2 grams) + character n-grams (so typos still overlap) -> logistic regression."""

    def __init__(self, C: float = 20.0, threshold: float = 0.5):
        self.C, self.threshold = C, threshold

    def _prep(self, texts):
        return [self.fixer_.fix(clean_text(t)) for t in texts]

    def fit(self, texts, labels):
        cleaned = [clean_text(t) for t in texts]
        self.fixer_ = TypoFixer().fit(cleaned)
        prepared = self._prep(texts)
        self.word_ = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1).fit(prepared)
        self.char_ = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2).fit(prepared)
        X = hstack([self.word_.transform(prepared), self.char_.transform(prepared)]).tocsr()
        self.clf_ = LogisticRegression(C=self.C, max_iter=3000, random_state=SEED).fit(X, labels)
        self.classes_ = self.clf_.classes_
        return self

    def _X(self, texts):
        p = self._prep(texts)
        return hstack([self.word_.transform(p), self.char_.transform(p)]).tocsr()

    def predict_proba(self, texts):
        return self.clf_.predict_proba(self._X(texts))

    def predict(self, texts, threshold: float | None = None):
        """Return (intent, confidence). intent is None when confidence is below the threshold (use the fallback)."""
        thr = self.threshold if threshold is None else threshold
        P = self.predict_proba(texts)
        idx = P.argmax(axis=1)
        conf = P[np.arange(len(P)), idx]
        return [self.classes_[i] if c >= thr else None for i, c in zip(idx, conf)], conf, [self.classes_[i] for i in idx]

    def save(self, path: str, metadata: dict | None = None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self, "metadata": {"version": VERSION, **(metadata or {})}}, path)

    @staticmethod
    def load(path: str) -> "IntentClassifier":
        return joblib.load(path)["model"]
