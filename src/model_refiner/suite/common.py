"""Strict serialization and crash-safe local persistence for the suite."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import unicodedata
from typing import Any


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json(text: str) -> Any:
    def bad_constant(value):
        raise ValueError(f"Non-finite JSON number: {value}")
    value = json.loads(text, object_pairs_hook=_object, parse_constant=bad_constant)
    # Also catches overflowing ordinary numbers, e.g. 1e999.
    canonical(value)
    return value


def atomic_json(path: str | Path, value: Any, *, overwrite: bool = True) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(tmp, target)
        else:
            os.link(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def normalized_input(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def integer(value: Any, name: str, lo: int, hi: int) -> int:
    if type(value) is not int or not lo <= value <= hi:
        raise ValueError(f"{name} must be an integer in [{lo}, {hi}]")
    return value


def number(value: Any, name: str, lo: float, hi: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lo <= value <= hi:
        raise ValueError(f"{name} must be finite in [{lo}, {hi}]")
    return float(value)


def nonempty(value: Any, name: str, max_chars: int = 100_000) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > max_chars:
        raise ValueError(f"{name} must be a nonempty string of at most {max_chars} characters")
    return value


def keys(value: dict, allowed: set[str], name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    extra = set(value) - allowed
    if extra:
        raise ValueError(f"Unknown {name} fields: {sorted(extra)}")
