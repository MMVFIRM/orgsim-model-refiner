"""CPU-only external Artificial Weights adapter for black-box APIs.

Fits a regularized multi-recipe reward predictor to a SEPARATE router split.
Only local hashed feature weights are updated. No provider gradients, logits,
activations or native weights are accessed. This is a new integration, distinct
from the preserved `aw` research controllers, and requires its own ablation.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import math
import random
import re

from .common import integer, keys, number


def features(text: str, dimension: int) -> dict[int, float]:
    words = re.findall(r"\w+", text.casefold())
    tokens = words + [a + "|" + b for a, b in zip(words, words[1:])]
    for word in words:
        if any(ord(c) > 127 for c in word):
            tokens.extend("§" + word[i:i+2] for i in range(len(word)-1))
    vector = Counter()
    for token in tokens:
        h = hashlib.sha256(token.encode()).digest()
        index = 1 + int.from_bytes(h[:4], "little") % (dimension - 1)
        vector[index] += 1 if h[4] & 1 else -1
    norm = math.sqrt(sum(v*v for v in vector.values())) or 1
    result = {k: v / norm for k, v in vector.items() if v}
    result[0] = 1.0
    return result


def similarity(a: dict[int, float], b: dict[int, float]) -> float:
    return sum(value * b.get(k, 0) for k, value in a.items() if k != 0)


@dataclass(frozen=True)
class Route:
    recipe_id: str
    support: int
    similarity: float
    margin: float
    fallback: bool
    scores: dict[str, float]


class ArtificialWeights:
    VERSION = "external-linear-recipe-policy-v1"

    def __init__(self, recipe_ids: list[str], dimension: int = 512, *, min_similarity: float = 0.1, min_support: int = 2):
        integer(dimension, "dimension", 32, 2048)
        if not isinstance(recipe_ids, list) or not 1 <= len(recipe_ids) <= 64 or len(set(recipe_ids)) != len(recipe_ids):
            raise ValueError("Invalid recipe identifiers")
        if any(not isinstance(x,str) or not x or len(x)>128 for x in recipe_ids):
            raise ValueError("Invalid recipe identifier")
        self.recipe_ids = list(recipe_ids)
        self.dimension = dimension
        self.min_similarity = number(min_similarity, "min_similarity", 0, 1)
        self.min_support = integer(min_support, "min_support", 1, 100)
        self.weights = [[0.0] * dimension for _ in recipe_ids]
        self.support_vectors: list[dict[int, float]] = []

    def fit(self, inputs: list[str], rewards: list[list[float]], *, epochs: int = 40, seed: int = 42):
        integer(epochs, "epochs", 1, 200)
        if not inputs or len(inputs) != len(rewards):
            raise ValueError("Missing/misaligned router rewards")
        for row in rewards:
            if len(row) != len(self.recipe_ids):
                raise ValueError("Reward shape mismatch")
            for value in row:
                number(value, "reward", 0, 1)
        self.support_vectors = [features(x, self.dimension) for x in inputs]
        self.weights = [[0.0] * self.dimension for _ in self.recipe_ids]
        for j in range(len(self.recipe_ids)):
            self.weights[j][0] = sum(row[j] for row in rewards) / len(rewards)
        rng = random.Random(seed)
        order = list(range(len(inputs)))
        for epoch in range(epochs):
            rng.shuffle(order)
            lr = 0.08 / math.sqrt(1 + epoch / 5)
            for i in order:
                x = self.support_vectors[i]
                for j, w in enumerate(self.weights):
                    estimate = sum(w[k] * value for k, value in x.items())
                    err = estimate - rewards[i][j]
                    for k, value in x.items():
                        w[k] -= lr * (err * value + (0.02 * w[k] if k else 0))
        return self

    def route(self, text: str, default: str) -> Route:
        if default not in self.recipe_ids:
            raise ValueError("Default recipe is absent from weights")
        x = features(text, self.dimension)
        similarities = [max(0.0, similarity(x, v)) for v in self.support_vectors]
        support = sum(s >= self.min_similarity for s in similarities)
        scores = {name: max(0.0,min(1.0,sum(w[k]*value for k,value in x.items())))
                  for name,w in zip(self.recipe_ids,self.weights)}
        rank = sorted(self.recipe_ids, key=lambda name:(-scores[name], name != default, name))
        margin = scores[rank[0]] - scores[rank[1]] if len(rank)>1 else 1.0
        fallback = support < self.min_support
        return Route(default if fallback else rank[0], support, max(similarities,default=0.0), margin, fallback, scores)

    def to_dict(self):
        return {"version":self.VERSION,"recipe_ids":self.recipe_ids,"dimension":self.dimension,
                "weights":self.weights,"support_vectors":[{str(k):v for k,v in x.items()} for x in self.support_vectors],
                "min_similarity":self.min_similarity,"min_support":self.min_support}

    @classmethod
    def from_dict(cls,obj):
        keys(obj,{"version","recipe_ids","dimension","weights","support_vectors","min_similarity","min_support"},"weights")
        if obj.get("version")!=cls.VERSION:
            raise ValueError("Unsupported external weights version")
        model=cls(obj["recipe_ids"],obj["dimension"],min_similarity=obj["min_similarity"],min_support=obj["min_support"])
        weights=obj["weights"]
        if not isinstance(weights,list) or len(weights)!=len(model.recipe_ids):
            raise ValueError("Weight shape mismatch")
        for row in weights:
            if not isinstance(row,list) or len(row)!=model.dimension:
                raise ValueError("Weight dimension mismatch")
            for value in row:
                number(value,"weight",-1e6,1e6)
        vectors=obj["support_vectors"]
        if not isinstance(vectors,list) or len(vectors)>10_000:
            raise ValueError("Invalid support vectors")
        parsed=[]
        for row in vectors:
            if not isinstance(row,dict) or len(row)>model.dimension:
                raise ValueError("Invalid support vector")
            vector={}
            for key,value in row.items():
                try: index=int(key)
                except (ValueError,TypeError) as exc: raise ValueError("Invalid feature index") from exc
                integer(index,"feature index",0,model.dimension-1)
                vector[index]=number(value,"feature value",-1,1)
            if vector.get(0)!=1:
                raise ValueError("Missing bias feature")
            parsed.append(vector)
        model.weights=[list(row) for row in weights]
        model.support_vectors=parsed
        return model
