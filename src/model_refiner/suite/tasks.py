"""Task-aware scoring and bounded, offline structural output contracts.

This is an explicit JSON Schema subset, not a permissive partial validator.
Unsupported keywords fail at configuration time; remote references and regex
execution are deliberately unavailable.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import math
import re
from typing import Any

from .common import canonical, integer, keys, nonempty, number, strict_json

SCHEMA_KEYS = {"type", "properties", "required", "additionalProperties", "items", "enum", "const", "anyOf",
               "minItems", "maxItems", "minimum", "maximum", "minLength", "maxLength", "description", "title"}
JSON_TYPES = {"object", "array", "string", "number", "integer", "boolean", "null"}


def check_schema(schema: dict, depth: int = 0) -> None:
    if depth > 20:
        raise ValueError("Output schema exceeds depth limit 20")
    keys(schema, SCHEMA_KEYS, "output schema")
    if len(canonical(schema)) > 32_000:
        raise ValueError("Output schema exceeds 32000 characters")
    types = schema.get("type")
    if types is not None:
        types = types if isinstance(types, list) else [types]
        if not types or any(not isinstance(t, str) or t not in JSON_TYPES for t in types):
            raise ValueError("Unsupported schema type")
    if "required" in schema and (not isinstance(schema["required"], list) or
                                 any(not isinstance(x, str) for x in schema["required"])):
        raise ValueError("required must be a list of property names")
    if "properties" in schema:
        if not isinstance(schema["properties"], dict):
            raise ValueError("properties must be an object")
        for child in schema["properties"].values():
            check_schema(child, depth + 1)
    if "items" in schema:
        check_schema(schema["items"], depth + 1)
    if "additionalProperties" in schema:
        value = schema["additionalProperties"]
        if isinstance(value, dict):
            check_schema(value, depth + 1)
        elif type(value) is not bool:
            raise ValueError("additionalProperties must be boolean or a schema")
    if "anyOf" in schema:
        if not isinstance(schema["anyOf"], list) or not 1 <= len(schema["anyOf"]) <= 16:
            raise ValueError("anyOf requires 1-16 schemas")
        for child in schema["anyOf"]:
            check_schema(child, depth + 1)
    if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]):
        raise ValueError("enum must be a nonempty list")
    for k in ("minItems", "maxItems", "minLength", "maxLength"):
        if k in schema:
            integer(schema[k], k, 0, 1_000_000)
    for k in ("minimum", "maximum"):
        if k in schema:
            number(schema[k], k, -1e100, 1e100)
    for lo, hi in (("minimum", "maximum"), ("minItems", "maxItems"), ("minLength", "maxLength")):
        if lo in schema and hi in schema and schema[lo] > schema[hi]:
            raise ValueError(f"{lo} exceeds {hi}")


def _is_type(value: Any, kind: str) -> bool:
    return {"object": isinstance(value, dict), "array": isinstance(value, list),
            "string": isinstance(value, str), "number": type(value) in (int, float),
            "integer": type(value) is int or (type(value) is float and value.is_integer()),
            "boolean": type(value) is bool, "null": value is None}[kind]


def schema_errors(value: Any, schema: dict, path: str = "$", depth: int = 0) -> list[str]:
    if depth > 32:
        return [f"{path}: output nesting exceeds limit"]
    errors = []
    kind = schema.get("type")
    if kind is not None and not any(_is_type(value, t) for t in (kind if isinstance(kind, list) else [kind])):
        return [f"{path}: wrong type"]
    if "enum" in schema and canonical(value) not in [canonical(x) for x in schema["enum"]]:
        errors.append(f"{path}: not an allowed value")
    if "const" in schema and canonical(value) != canonical(schema["const"]):
        errors.append(f"{path}: const mismatch")
    if "anyOf" in schema and all(schema_errors(value, s, path, depth + 1) for s in schema["anyOf"]):
        errors.append(f"{path}: no anyOf branch matches")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}.{key}: required")
        for key, child in value.items():
            sub = props.get(key, schema.get("additionalProperties", True))
            if sub is False:
                errors.append(f"{path}.{key}: additional property")
            elif isinstance(sub, dict):
                errors.extend(schema_errors(child, sub, f"{path}.{key}", depth + 1))
            if len(errors) >= 20:
                break
    if isinstance(value, list):
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", math.inf):
            errors.append(f"{path}: array length out of range")
        if "items" in schema:
            for i, child in enumerate(value):
                errors.extend(schema_errors(child, schema["items"], f"{path}[{i}]", depth + 1))
                if len(errors) >= 20:
                    break
    if isinstance(value, str) and not schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", math.inf):
        errors.append(f"{path}: string length out of range")
    if type(value) in (int, float) and not schema.get("minimum", -math.inf) <= value <= schema.get("maximum", math.inf):
        errors.append(f"{path}: number out of range")
    return errors[:20]


@dataclass(frozen=True)
class TaskSpec:
    instruction: str
    kind: str = "text"
    metric: str = "normalized_exact"
    labels: tuple[str, ...] = ()
    output_schema: dict = field(default_factory=dict)
    numeric_tolerance: float = 0.01
    max_output_chars: int = 32_000

    def __post_init__(self):
        nonempty(self.instruction, "instruction")
        valid = {"classification": {"accuracy"}, "json": {"json_exact", "field_f1"},
                 "text": {"exact", "normalized_exact", "token_f1", "rubric"}, "numeric": {"numeric_tolerance"}}
        if not isinstance(self.kind,str) or not isinstance(self.metric,str) or self.kind not in valid or self.metric not in valid[self.kind]:
            raise ValueError("Incompatible task kind/metric")
        integer(self.max_output_chars, "max_output_chars", 1, 1_000_000)
        number(self.numeric_tolerance, "numeric_tolerance", 0, 1e50)
        if self.kind == "classification":
            if not isinstance(self.labels, (tuple, list)) or not 2 <= len(self.labels) <= 10_000:
                raise ValueError("Classification requires 2-10000 labels")
            for label in self.labels:
                nonempty(label, "label", 256)
                if label != label.strip():
                    raise ValueError("Labels cannot have surrounding whitespace")
            if len(set(self.labels)) != len(self.labels):
                raise ValueError("Duplicate labels")
        if self.kind == "json":
            check_schema(self.output_schema)
        elif self.output_schema:
            raise ValueError("output_schema is only supported for json tasks")

    @classmethod
    def from_dict(cls, obj: dict) -> "TaskSpec":
        keys(obj, set(cls.__dataclass_fields__), "task")
        data = dict(obj)
        nonempty(data.get("instruction"), "instruction")
        if not isinstance(data.get("labels",()), (list,tuple)):
            raise ValueError("labels must be a list")
        data["labels"] = tuple(data.get("labels", ()))
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)

    def contract(self) -> str:
        if self.kind == "classification":
            return "Return exactly one of these case-sensitive labels, with no explanation: " + canonical(self.labels)
        if self.kind == "json":
            return "Return JSON only (no Markdown), matching this output contract: " + canonical(self.output_schema)
        if self.kind == "numeric":
            return "Return exactly one finite JSON number, with no units or explanation."
        return "Return the requested answer with no unsolicited preamble."

    def validate_output(self, text: str) -> list[str]:
        if not isinstance(text, str) or not text.strip():
            return ["empty output"]
        if len(text) > self.max_output_chars:
            return ["output exceeds character limit"]
        if self.kind == "classification":
            return [] if text.strip() in self.labels else ["unknown label or invalid label format"]
        if self.kind in {"json", "numeric"}:
            try:
                obj = strict_json(text)
            except (ValueError, RecursionError):
                return ["invalid JSON"]
            if self.kind == "numeric":
                return [] if type(obj) in (int, float) else ["not a finite number"]
            try:
                return schema_errors(obj, self.output_schema)
            except RecursionError:
                return ["output nesting exceeds limit"]
        return []

    def score(self, prediction: str, target: str) -> float:
        if self.metric == "rubric":
            raise ValueError("Rubric scoring requires a configured Scorer and judge backend")
        if self.validate_output(prediction):
            return 0.0
        if self.kind == "classification" or self.metric == "exact":
            return float(prediction.strip() == target.strip())
        if self.metric == "normalized_exact":
            return float(" ".join(prediction.casefold().split()) == " ".join(target.casefold().split()))
        if self.metric == "token_f1":
            a, b = Counter(re.findall(r"\w+", prediction.casefold())), Counter(re.findall(r"\w+", target.casefold()))
            size = sum(a.values()) + sum(b.values())
            return 2 * sum((a & b).values()) / size if size else float(prediction.strip() == target.strip())
        if self.kind == "numeric":
            return float(abs(strict_json(prediction) - strict_json(target)) <= self.numeric_tolerance)
        a, b = strict_json(prediction), strict_json(target)
        if self.metric == "json_exact":
            return float(canonical(a) == canonical(b))
        # Path/value leaves: extra keys are penalized as well as missing ones. Types remain distinct.
        def leaves(obj, path=()):
            if isinstance(obj, dict) and obj:
                return {item for k, v in obj.items() for item in leaves(v, path + ("key:" + k,))}
            if isinstance(obj, list) and obj:
                return {item for i, v in enumerate(obj) for item in leaves(v, path + ("index:" + str(i),))}
            return {(path, canonical(obj))}
        sa, sb = leaves(a), leaves(b)
        return 2 * len(sa & sb) / (len(sa) + len(sb))
