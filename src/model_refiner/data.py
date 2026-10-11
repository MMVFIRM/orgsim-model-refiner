from __future__ import annotations

import json
import random
from pathlib import Path

from .types import Example


def load_jsonl(path: str | Path) -> list[Example]:
    items: list[Example] = []
    seen_ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {line_no}: invalid JSON: {exc.msg}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no}: each record must be a JSON object")
            if "input" not in obj or "target" not in obj:
                raise ValueError(f"Line {line_no}: each record needs input and target")
            if not isinstance(obj["input"], str) or not obj["input"].strip():
                raise ValueError(f"Line {line_no}: input must be a non-empty string")
            if not isinstance(obj["target"], str) or not obj["target"].strip():
                raise ValueError(f"Line {line_no}: target must be a non-empty string")
            identifier = str(obj.get("id", line_no))
            if identifier in seen_ids:
                raise ValueError(f"Line {line_no}: duplicate id {identifier!r}")
            seen_ids.add(identifier)
            metadata = obj.get("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError(f"Line {line_no}: metadata must be an object")
            split = obj.get("split")
            if split is not None and not isinstance(split, str):
                raise ValueError(f"Line {line_no}: split must be a string")
            items.append(
                Example(
                    id=identifier,
                    input=obj["input"],
                    target=obj["target"],
                    split=split,
                    metadata=dict(metadata),
                )
            )
    if not items:
        raise ValueError("Dataset is empty")
    return items


def split_examples(
    examples: list[Example], seed: int = 42, train_fraction: float = 0.6, val_fraction: float = 0.2
) -> tuple[list[Example], list[Example], list[Example]]:
    explicit = [e for e in examples if e.split]
    if explicit:
        train = [e for e in examples if (e.split or "").lower() == "train"]
        val = [e for e in examples if (e.split or "").lower() in {"val", "validation"}]
        test = [e for e in examples if (e.split or "").lower() == "test"]
        unknown = [e for e in examples if (e.split or "").lower() not in {"train", "val", "validation", "test"}]
        if unknown:
            raise ValueError("If any split is explicit, all examples must use train/val/test")
        if not train or not val or not test:
            raise ValueError("Explicit split requires non-empty train, val, and test groups")
        return train, val, test

    if len(examples) < 3:
        raise ValueError("At least 3 examples are required for an automatic train/val/test split")
    if not 0 < train_fraction < 1 or not 0 < val_fraction < 1 or train_fraction + val_fraction >= 1:
        raise ValueError("Invalid train/validation fractions")

    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = max(1, int(n * train_fraction))
    n_val = max(1, int(n * val_fraction))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    return shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]
