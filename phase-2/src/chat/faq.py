"""Task 9: answer questions from hr_faq when a message closely matches a stored FAQ question."""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.chat.nlu import clean_text

FAQ_MIN_SIMILARITY = 0.45


class FaqRetriever:
    def __init__(self, faq: list[dict], min_similarity: float = FAQ_MIN_SIMILARITY):
        self.faq, self.min_similarity = faq, min_similarity
        self.vec_ = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), sublinear_tf=True)
        self.M_ = self.vec_.fit_transform([clean_text(f["question"]) for f in faq])

    def best(self, text: str):
        """(faq row, similarity) or (None, similarity) when nothing is close enough."""
        s = cosine_similarity(self.vec_.transform([clean_text(text)]), self.M_)[0]
        i = int(s.argmax())
        return (self.faq[i] if s[i] >= self.min_similarity else None), float(s[i])
