"""Leakage-aware datasets, group-preserving splits, and preference records."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import random
from typing import Any

from .common import canonical, digest, keys, nonempty, normalized_input, strict_json
from .tasks import TaskSpec

SPLITS = {"train", "router", "validation", "test"}


@dataclass(frozen=True)
class Record:
    id: str
    input: str
    target: str
    split: str | None = None
    group: str | None = None
    slice: str = "default"
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, row: dict, index: int = 0) -> "Record":
        keys(row, set(cls.__dataclass_fields__) | {"messages"}, "record")
        row = dict(row)
        if "messages" in row:
            if "input" in row:
                raise ValueError("Use input OR messages, not both")
            messages = row.pop("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError("messages must be a nonempty list")
            for msg in messages:
                keys(msg, {"role", "content"}, "message")
                if not isinstance(msg.get("role"),str) or msg["role"] not in {"user", "assistant", "system", "developer"}:
                    raise ValueError("Unsupported message role")
                nonempty(msg.get("content"), "message content")
            # Historical role labels are data, not elevated live system instructions.
            row["input"] = canonical({"conversation": messages})
        nonempty(row.get("input"), "input")
        nonempty(row.get("target"), "target")
        row.setdefault("id", str(index + 1))
        nonempty(row["id"], "record id", 256)
        split = row.get("split")
        if split is not None and not isinstance(split,str):
            raise ValueError("split must be a string")
        if split in {"val", "calibration"}:
            split = {"val": "validation", "calibration": "router"}[split]
        if split is not None and split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}")
        row["split"] = split
        if row.get("group") is not None:
            nonempty(row["group"], "group", 256)
        nonempty(row.get("slice", "default"), "slice", 256)
        if not isinstance(row.get("metadata", {}), dict):
            raise ValueError("metadata must be an object")
        if len(canonical(row.get("metadata", {}))) > 8192:
            raise ValueError("metadata exceeds 8192 characters")
        return cls(**row)


@dataclass(frozen=True)
class Preference:
    id: str
    input: str
    chosen: str
    rejected: str

    @classmethod
    def from_dict(cls, row: dict) -> "Preference":
        keys(row, set(cls.__dataclass_fields__), "preference")
        for key in cls.__dataclass_fields__:
            nonempty(row.get(key), key, 256 if key == "id" else 100_000)
        if row["chosen"] == row["rejected"]:
            raise ValueError("Preference chosen and rejected are identical")
        return cls(**row)


def load_records(path: str | Path) -> list[Record]:
    return [Record.from_dict(x, i) for i, x in enumerate(load_jsonl(path))]


def load_jsonl(path: str | Path, max_bytes: int = 20_000_000) -> list[dict]:
    p = Path(path)
    if p.stat().st_size > max_bytes:
        raise ValueError("Dataset file exceeds size limit")
    rows = []
    with p.open(encoding="utf-8") as stream:
        for i, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(strict_json(line))
                except (ValueError, RecursionError) as exc:
                    raise ValueError(f"Invalid JSONL line {i}") from exc
    if not rows:
        raise ValueError("Empty dataset")
    return rows


@dataclass
class Dataset:
    train: list[Record]
    router: list[Record]
    validation: list[Record]
    test: list[Record]
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def prepare(cls, records: list[Record], task: TaskSpec, seed: int = 42) -> "Dataset":
        if not 8 <= len(records) <= 10_000:
            raise ValueError("Suite requires 8-10000 records and at least four independent groups")
        ids, inputs = set(), {}
        for ex in records:
            if ex.id in ids:
                raise ValueError(f"Duplicate record id: {ex.id}")
            ids.add(ex.id)
            errors = task.validate_output(ex.target)
            if errors:
                raise ValueError(f"Invalid target for {ex.id}: {errors}")
            norm = normalized_input(ex.input)
            if norm in inputs:
                raise ValueError(f"Duplicate or conflicting normalized input: {inputs[norm]} / {ex.id}")
            inputs[norm] = ex.id
        if any(ex.split for ex in records):
            if not all(ex.split for ex in records):
                raise ValueError("Mixed explicit and missing splits")
            splits = {key: [e for e in records if e.split == key] for key in SPLITS}
            if not all(splits[key] for key in ("train", "validation", "test")):
                raise ValueError("Explicit data needs train, validation, and test")
            seen_groups = {}
            for key, group_rows in splits.items():
                for ex in group_rows:
                    group = ex.group or ex.id
                    if group in seen_groups and seen_groups[group] != key:
                        raise ValueError(f"Group {group!r} leaks across splits")
                    seen_groups[group] = key
            if not splits["router"]:
                splits["train"], splits["router"] = cls._partition(splits["train"], (0.75, 0.25), seed)
        else:
            groups = cls._partition(records, (0.5, 0.15, 0.15, 0.20), seed)
            splits = dict(zip(("train", "router", "validation", "test"), groups))
        warnings = []
        if len(splits["test"]) < 100:
            warnings.append("Fewer than 100 test examples: insufficient for the default release gate")
        if len(splits["router"]) < 30:
            warnings.append("Small router calibration set; contextual weights may be unreliable")
        return cls(**splits, warnings=warnings)

    @staticmethod
    def _partition(rows: list[Record], fractions: tuple[float, ...], seed: int) -> list[list[Record]]:
        grouped = {}
        for ex in rows:
            grouped.setdefault(ex.group or ex.id, []).append(ex)
        if len(grouped) < len(fractions):
            raise ValueError("Not enough independent groups for leakage-safe splits")
        # Sorted before shuffling: row input order does not change the split.
        groups = [grouped[k] for k in sorted(grouped)]
        random.Random(seed).shuffle(groups)
        counts, remain = [], len(groups)
        for i, fraction in enumerate(fractions[:-1]):
            n = max(1, min(round(len(groups) * fraction), remain - (len(fractions) - i - 1)))
            counts.append(n)
            remain -= n
        counts.append(remain)
        out, offset = [], 0
        for n in counts:
            out.append([ex for g in groups[offset:offset + n] for ex in sorted(g, key=lambda x: x.id)])
            offset += n
        return out

    def audit_preferences(self, preferences: list[Preference]) -> None:
        training_inputs = {normalized_input(ex.input) for ex in self.train}
        seen = set()
        for pref in preferences:
            if pref.id in seen:
                raise ValueError("Duplicate preference id")
            seen.add(pref.id)
            if normalized_input(pref.input) not in training_inputs:
                raise ValueError("Preference inputs must match train-only records (never router/validation/test)")

    def manifest(self) -> dict:
        result = {}
        for key in ("train", "router", "validation", "test"):
            rows = getattr(self, key)
            result[key] = {"count": len(rows), "groups": len({x.group or x.id for x in rows}),
                           "sha256": digest([asdict(x) for x in sorted(rows, key=lambda x: x.id)]),
                           "input_hashes": sorted(digest(normalized_input(x.input)) for x in rows),
                           "group_hashes": sorted({digest(x.group or x.id) for x in rows})}
        return {"splits": result, "warnings": self.warnings,
                "policy": "group-disjoint; normalized duplicate rejection; train/router/validation/test"}
