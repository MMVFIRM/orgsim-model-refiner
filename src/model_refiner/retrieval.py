from __future__ import annotations

import math
import re
from collections import Counter

from .types import Example

# Python's Unicode-aware \w keeps non-Latin training corpora retrievable. Apostrophes are retained inside words.
_TOKEN_RE = re.compile(r"\w+(?:['’]\w+)?", flags=re.UNICODE)


def _tokens(text: str) -> list[str]:
    out: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        token = raw.casefold()
        out.append(token)
        # Scripts that commonly omit spaces (CJK, etc.) otherwise become one giant token.
        # Add namespaced character bigrams so substring concepts remain retrievable without
        # changing Latin token behavior.
        if any(ord(ch) > 127 for ch in token) and len(token) >= 2:
            out.extend("§" + token[i:i + 2] for i in range(len(token) - 1))
    return out


class TfidfMemory:
    """Small dependency-free Unicode-aware TF-IDF memory for retrieving training demonstrations."""

    def __init__(self, examples: list[Example]):
        if not examples:
            raise ValueError("Memory requires at least one training example")
        self.examples = list(examples)
        docs = [_tokens(e.input) for e in examples]
        self._doc_tf = [Counter(d) for d in docs]
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        self._idf = {term: math.log((n + 1) / (freq + 1)) + 1.0 for term, freq in df.items()}
        self._doc_vecs = [self._vector(tf) for tf in self._doc_tf]

    def _vector(self, tf: Counter[str]) -> dict[str, float]:
        if not tf:
            return {}
        total = sum(tf.values())
        vec = {term: (count / total) * self._idf.get(term, 0.0) for term, count in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {k: v / norm for k, v in vec.items()}

    def query(self, text: str, k: int) -> list[Example]:
        if k <= 0:
            return []
        q = self._vector(Counter(_tokens(text)))
        scored: list[tuple[float, int]] = []
        for i, d in enumerate(self._doc_vecs):
            score = sum(q.get(term, 0.0) * value for term, value in d.items())
            scored.append((score, i))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [self.examples[i] for _, i in scored[: min(k, len(scored))]]
