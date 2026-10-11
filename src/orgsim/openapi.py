"""Restricted OpenAPI 3.x -> draft simulator manifest importer.

This imports *contracts*, never inferred organizational business semantics.
Unsupported/non-CRUD endpoints are placed in review_notes, not fabricated.
"""
from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

from .spec import load_spec, SpecError


def _resolve_local(schema: Any, document: dict, seen: frozenset[str] = frozenset()) -> Any:
    if isinstance(schema, list):
        return [_resolve_local(item, document, seen) for item in schema]
    if not isinstance(schema, dict):
        return schema
    if "$ref" in schema:
        ref = schema["$ref"]
        if not isinstance(ref, str) or not ref.startswith("#/"):
            raise SpecError("External OpenAPI references are unsupported; inline/resolve them before import")
        if ref in seen:
            raise SpecError(f"Cyclic OpenAPI reference: {ref}")
        value = document
        for segment in ref[2:].split("/"):
            key = segment.replace("~1", "/").replace("~0", "~")
            try:
                value = value[key]
            except (KeyError, TypeError):
                raise SpecError(f"Unresolved OpenAPI reference: {ref}")
        merged = {**_resolve_local(value, document, seen | {ref}),
                  **{k: v for k, v in schema.items() if k != "$ref"}}
        return _resolve_local(merged, document, seen | {ref})
    return {k: _resolve_local(v, document, seen) for k, v in schema.items()}


def _request_schema(op: dict, doc: dict) -> dict | None:
    body = _resolve_local(op.get("requestBody", {}), doc)
    content = body.get("content", {}) if isinstance(body, dict) else {}
    schema = content.get("application/json", {}).get("schema")
    return _resolve_local(schema, doc) if schema is not None else None


def _response_schema(op: dict, doc: dict) -> dict | None:
    for code in ("200", "201", "default"):
        response = _resolve_local(op.get("responses", {}).get(code, {}), doc)
        schema = response.get("content", {}).get("application/json", {}).get("schema") if isinstance(response, dict) else None
        if schema:
            schema = _resolve_local(schema, doc)
            if schema.get("type") == "array":
                return schema.get("items")
            return schema
    return None


def _response_contract(op: dict, doc: dict) -> dict | None:
    """Record a declared response shape; not observed behavior or a guarantee."""
    for code in ("200", "201", "202", "204"):
        if code not in op.get("responses", {}):
            continue
        response = _resolve_local(op["responses"][code], doc)
        raw = response.get("content", {}).get("application/json", {}).get("schema", {})
        raw = _resolve_local(raw, doc)
        typ = raw.get("type") if isinstance(raw, dict) else None
        kind = ("number" if typ in {"integer", "number"} else
                "string" if typ == "string" else
                "object" if typ == "object" else
                "array" if typ == "array" else "null" if code == "204" else "unknown")
        return {"status": int(code), "kind": kind, "evidence": "declared_openapi_only"}
    return None


def _merge_property(a: dict, b: dict) -> dict:
    """Draft storage accepts input/output representations of a property.

    This deliberately DOES NOT claim conversion from one representation to the
    other. The read-response serializer is an independent reconstruction task.
    """
    a, b = copy.deepcopy(a), copy.deepcopy(b)
    types_a = a.get("type")
    types_b = b.get("type")
    if types_a and types_b and types_a != types_b:
        both = []
        for t in (types_a if isinstance(types_a, list) else [types_a]) + (types_b if isinstance(types_b, list) else [types_b]):
            if t not in both:
                both.append(t)
        return {"type": both}
    return a  # request schema wins when the type is compatible


def import_openapi(paths: list[str | Path], *, system: str, organization: str = "Unreviewed organization", base_path: str = "") -> dict:
    """Import common OpenAPI resources while separating input and stored state.

    A generated manifest remains unreviewed. Request schemas, response types,
    server defaults and business rules are separate things; never merge their
    required fields or silently assume response normalization.
    """
    resources: dict = {}
    notes: list[str] = []
    for path in paths:
        doc = load_spec(path)
        if not isinstance(doc.get("openapi"), str) or not doc["openapi"].startswith("3."):
            raise SpecError(f"Expected OpenAPI 3.x document: {path}")
        for route, path_item in doc.get("paths", {}).items():
            segments = [s for s in route.split("/") if s]
            if base_path:
                prefix = [s for s in base_path.split("/") if s]
                if segments[:len(prefix)] != prefix:
                    notes.append(f"Skipped {route}: does not begin with configured base path {base_path}")
                    continue
                segments = segments[len(prefix):]
            else:
                while segments and (segments[0].lower() in {"api", "rest"} or re.fullmatch(r"v[0-9]+", segments[0].lower())):
                    segments.pop(0)
            if not segments:
                continue
            resource = re.sub(r"[^a-z0-9_]+", "_", segments[0].lower().replace("-", "_"))
            if not resource:
                continue
            is_detail = len(segments) == 2 and segments[1].startswith("{") and segments[1].endswith("}")
            is_collection = len(segments) == 1
            for verb, op in path_item.items():
                if verb.lower() not in {"get", "post", "put", "patch", "delete"} or not isinstance(op, dict):
                    continue
                action = {("get", True): "list", ("get", False): "get", ("post", True): "create",
                          ("put", False): "update", ("patch", False): "update", ("delete", False): "delete"}.get((verb.lower(), is_collection))
                if not (is_collection or is_detail) or (verb.lower() == "post" and not is_collection):
                    action = None
                if not action:
                    notes.append(f"Unsupported custom endpoint: {verb.upper()} {route} (requires explicit adapter)")
                    continue
                try:
                    if action == "list":
                        response = _resolve_local(op.get("responses", {}).get("200", {}), doc)
                        raw = _resolve_local(response.get("content", {}).get("application/json", {}).get("schema", {}), doc)
                        if not isinstance(raw, dict) or raw.get("type") != "array":
                            notes.append(f"Unsupported collection response: GET {route} (expected array; inspect real response)")
                            continue
                    if action == "create":
                        schema = _request_schema(op, doc)
                    elif action == "update":
                        schema = _request_schema(op, doc)
                    else:
                        schema = _response_schema(op, doc)
                    response_spec = _response_contract(op, doc)
                except (SpecError, TypeError, AttributeError) as exc:
                    notes.append(f"Unsupported schema for {verb.upper()} {route}: {type(exc).__name__} ({str(exc)[:120]})")
                    continue
                if action in {"create", "list", "get"} and not isinstance(schema, dict):
                    notes.append(f"Incomplete schema for {verb.upper()} {route}; imported only API shape, fields may be absent")
                rc = resources.setdefault(resource, {
                    "id_field": "id",
                    "schema": {"type": "object", "properties": {"id": {"type": "string"}},
                               "required": ["id"], "additionalProperties": True},
                    "allow": {}, "guards": {}, "foreign_keys": [], "response_contract": {}
                })
                rc["allow"][action] = ["test_agent"]
                if response_spec is not None:
                    rc["response_contract"][action] = response_spec
                if action in {"create", "update"} and isinstance(schema, dict) and schema.get("type") == "object":
                    rc[f"{action}_input_schema"] = copy.deepcopy(schema)
                if isinstance(schema, dict) and schema.get("type") == "object":
                    for field, field_schema in schema.get("properties", {}).items():
                        if field not in rc["schema"]["properties"]:
                            rc["schema"]["properties"][field] = copy.deepcopy(field_schema)
                        elif field == "id" and rc["schema"]["properties"]["id"].get("type") == "string":
                            rc["schema"]["properties"]["id"] = copy.deepcopy(field_schema)
                        else:
                            rc["schema"]["properties"][field] = _merge_property(rc["schema"]["properties"][field], field_schema)
            if "get" in path_item and is_collection:
                notes.append(f"Verify list response envelope for GET {route}")
    for resource, rc in resources.items():
        rc["schema"]["properties"].setdefault("id", {"type": "string"})
        notes.append(f"{system}.{resource}: verify input/output type conversions, business constraints, default fields, uniqueness, permissions and side effects")
    return {"version": "0.1", "organization": organization, "reviewed": False,
            "systems": {system: {"resources": resources}}, "integrations": [],
            "scenarios": [], "review_notes": notes, "origin": "OpenAPI contract import (NOT a verified behavioral clone)"}
