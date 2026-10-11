"""Portable, immutable deployment artifacts for Model Refiner."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from .types import Example

ARTIFACT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ArtifactExample:
    id: str
    input: str
    target: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_example(cls, ex: Example) -> "ArtifactExample":
        return cls(id=ex.id, input=ex.input, target=ex.target, metadata=dict(ex.metadata))

    def to_example(self) -> Example:
        return Example(id=self.id, input=self.input, target=self.target, metadata=dict(self.metadata))


@dataclass(frozen=True)
class RefinerArtifact:
    """Everything needed to reproduce the evaluated refined inference arm.

    The artifact deliberately stores no API keys or provider credentials.  Its
    ``artifact_id`` is a SHA-256 digest of the logical payload, so corruption or
    accidental edits are detected when loading.
    """

    artifact_id: str
    schema_version: int
    created_at: str
    provider: str
    model: str
    optimizer_model: str | None
    base_instruction: str
    compiled_rules: str
    retrieval_k: int
    examples: tuple[ArtifactExample, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def _logical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider": self.provider,
            "model": self.model,
            "optimizer_model": self.optimizer_model,
            "base_instruction": self.base_instruction,
            "compiled_rules": self.compiled_rules,
            "retrieval_k": self.retrieval_k,
            "examples": [asdict(x) for x in self.examples],
            "metadata": self.metadata,
        }

    @staticmethod
    def _digest(payload: dict[str, Any]) -> str:
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    @classmethod
    def create(
        cls,
        *,
        provider: str,
        model: str,
        optimizer_model: str | None,
        base_instruction: str,
        compiled_rules: str,
        retrieval_k: int,
        examples: list[Example] | tuple[Example, ...],
        metadata: dict[str, Any] | None = None,
        created_at: str | None = None,
    ) -> "RefinerArtifact":
        if not provider.strip():
            raise ValueError("Artifact provider must be non-empty")
        if not model.strip():
            raise ValueError("Artifact model must be non-empty")
        if not base_instruction.strip():
            raise ValueError("Artifact instruction must be non-empty")
        if retrieval_k < 0:
            raise ValueError("retrieval_k must be >= 0")
        packed = tuple(ArtifactExample.from_example(x) for x in examples)
        if retrieval_k and not packed:
            raise ValueError("A retrieval artifact needs at least one memory example")
        obj = cls(
            artifact_id="",
            schema_version=ARTIFACT_SCHEMA_VERSION,
            created_at=created_at or datetime.now(timezone.utc).isoformat(),
            provider=provider,
            model=model,
            optimizer_model=optimizer_model,
            base_instruction=base_instruction,
            compiled_rules=compiled_rules,
            retrieval_k=int(retrieval_k),
            examples=packed,
            metadata=dict(metadata or {}),
        )
        return cls(**{**asdict(obj), "examples": packed, "artifact_id": cls._digest(obj._logical_payload())})

    def verify(self) -> None:
        if self.schema_version != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported artifact schema {self.schema_version}; expected {ARTIFACT_SCHEMA_VERSION}"
            )
        expected = self._digest(self._logical_payload())
        if not self.artifact_id or self.artifact_id != expected:
            raise ValueError("Artifact integrity check failed")
        if self.retrieval_k < 0:
            raise ValueError("Invalid retrieval_k")
        if self.retrieval_k and not self.examples:
            raise ValueError("Artifact retrieval memory is empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "provider": self.provider,
            "model": self.model,
            "optimizer_model": self.optimizer_model,
            "base_instruction": self.base_instruction,
            "compiled_rules": self.compiled_rules,
            "retrieval_k": self.retrieval_k,
            "examples": [asdict(x) for x in self.examples],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "RefinerArtifact":
        try:
            artifact = cls(
                artifact_id=str(obj["artifact_id"]),
                schema_version=int(obj["schema_version"]),
                created_at=str(obj["created_at"]),
                provider=str(obj["provider"]),
                model=str(obj["model"]),
                optimizer_model=(None if obj.get("optimizer_model") is None else str(obj.get("optimizer_model"))),
                base_instruction=str(obj["base_instruction"]),
                compiled_rules=str(obj.get("compiled_rules", "")),
                retrieval_k=int(obj["retrieval_k"]),
                examples=tuple(
                    ArtifactExample(
                        id=str(x["id"]),
                        input=str(x["input"]),
                        target=str(x["target"]),
                        metadata=dict(x.get("metadata", {})),
                    )
                    for x in obj.get("examples", [])
                ),
                metadata=dict(obj.get("metadata", {})),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed Model Refiner artifact: {exc}") from exc
        artifact.verify()
        return artifact

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)

    @classmethod
    def load(cls, path: str | Path) -> "RefinerArtifact":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
