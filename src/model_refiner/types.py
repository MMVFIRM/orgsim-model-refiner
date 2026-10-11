from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Example:
    id: str
    input: str
    target: str
    split: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResult:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
