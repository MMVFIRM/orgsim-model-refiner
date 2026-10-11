from __future__ import annotations

import math
import re
from dataclasses import dataclass


def normalize(text: str) -> str:
    text = text.strip().casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip("`'\" .,:;!?\n\t")


def exact_match(prediction: str, target: str) -> float:
    return float(normalize(prediction) == normalize(target))


@dataclass
class Metrics:
    count: int
    accuracy: float


def score_pairs(pairs: list[tuple[str, str]]) -> Metrics:
    if not pairs:
        return Metrics(count=0, accuracy=0.0)
    scores = [exact_match(pred, target) for pred, target in pairs]
    return Metrics(count=len(scores), accuracy=sum(scores) / len(scores))


def exact_binomial_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Probability-ordered two-sided binomial test, evaluated in log space."""
    if type(n) is not int or type(k) is not int or not 0 <= k <= n or n > 1_000_000:
        raise ValueError("Require integer 0 <= k <= n <= 1000000")
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("p must be a probability")
    if n == 0:
        return 1.0
    if p in (0, 1):
        return float(k == (n if p else 0))
    logp, logq, base = math.log(p), math.log1p(-p), math.lgamma(n + 1)
    def logpmf(i):
        return base - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * logp + (n - i) * logq
    observed = logpmf(k)
    if observed >= logpmf(min(n, math.floor((n + 1) * p))) - 1e-12:
        return 1.0
    return min(1.0, math.fsum(math.exp(logpmf(i)) for i in range(n + 1) if logpmf(i) <= observed + 1e-10))
