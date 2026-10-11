"""Manifest loading, static validation and registry generation.

The compiler does not infer business invariants from natural-language descriptions.
Unverified OpenAPI imports remain drafts until a reviewer explicitly approves them.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator


class SpecError(ValueError):
    pass


def load_spec(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise SpecError(f"Manifest not found: {path}")
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SpecError("Manifest must be a mapping")
    return data


def _check_ref(expr: Any, context: str) -> None:
    """Reject unsupported expression syntax; never execute embedded code."""
    if not isinstance(expr, dict) or len(expr) != 1:
        raise SpecError(f"{context}: expression must be a single-key mapping")
    k, v = next(iter(expr.items()))
    if k == "literal":
        return
    if k == "path":
        if not isinstance(v, str) or not v or not v.startswith(("input.", "record.", "event.", "before.", "after.", "steps.")):
            raise SpecError(f"{context}: path must start with input., record., event., before., after., or steps.")
    elif k in {"negate"}:
        _check_ref(v, context)
    elif k in {"add", "mul"}:
        if not isinstance(v, list) or len(v) < 2:
            raise SpecError(f"{context}: {k} requires at least two operands")
        for item in v:
            _check_ref(item, context)
    elif k == "lookup":
        if not isinstance(v, dict) or not {"system", "resource", "id"} <= v.keys():
            raise SpecError(f"{context}: lookup requires system, resource, id")
        _check_ref(v["id"], context)
        if "field" in v and not isinstance(v["field"], str):
            raise SpecError(f"{context}: lookup field must be a string")
    elif k == "count":
        if not isinstance(v, dict) or not {"system", "resource"} <= v.keys():
            raise SpecError(f"{context}: count requires system and resource")
        for val in v.get("where", {}).values():
            _check_ref(val, context)
    else:
        raise SpecError(f"{context}: unsupported expression {k!r}")


def _check_condition(cond: Any, context: str) -> None:
    if not isinstance(cond, dict) or len(cond) != 1:
        raise SpecError(f"{context}: condition must contain exactly one operator")
    k, v = next(iter(cond.items()))
    if k in {"all", "any"}:
        if not isinstance(v, list) or not v:
            raise SpecError(f"{context}: {k} requires nonempty list")
        for c in v:
            _check_condition(c, context)
    elif k == "not":
        _check_condition(v, context)
    elif k == "exists":
        _check_ref(v, context)
    elif k == "compare":
        if not isinstance(v, dict) or not {"left", "right", "op"} <= v.keys():
            raise SpecError(f"{context}: compare needs left, op, right")
        if v["op"] not in {"eq", "ne", "lt", "le", "gt", "ge", "in", "not_in"}:
            raise SpecError(f"{context}: unsupported comparator")
        _check_ref(v["left"], context)
        _check_ref(v["right"], context)
    else:
        raise SpecError(f"{context}: unsupported condition {k!r}")


def _validate_schema(schema: dict, context: str) -> None:
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise SpecError(f"Invalid JSON schema for {context}: {exc}") from exc
    if schema.get("type") != "object":
        raise SpecError(f"{context}: resource schema type must be object")
    # Runtime is deliberately offline: remote JSON-schema refs aren't permitted.
    def scan(obj: Any) -> None:
        if isinstance(obj, dict):
            if "$ref" in obj:
                raise SpecError(f"{context}: $ref must be resolved before compilation")
            for value in obj.values():
                scan(value)
        elif isinstance(obj, list):
            for item in obj:
                scan(item)
    scan(schema)


def _verify_targets(tree: Any, systems: dict, context: str) -> None:
    """Static check that declarative lookups/counts name existing resources/fields."""
    if isinstance(tree, dict):
        for key, item in tree.items():
            if key in {"lookup", "count"} and isinstance(item, dict):
                system, resource = item.get("system"), item.get("resource")
                if system not in systems or resource not in systems[system].get("resources", {}):
                    raise SpecError(f"{context}: unknown referenced resource {system}.{resource}")
                if key == "lookup" and "field" in item:
                    prop = item["field"].split(".")[0]
                    if prop not in systems[system]["resources"][resource]["schema"]["properties"]:
                        raise SpecError(f"{context}: unknown lookup field {system}.{resource}.{item['field']}")
                if key == "count":
                    props = systems[system]["resources"][resource]["schema"]["properties"]
                    for prop in item.get("where", {}):
                        if prop not in props:
                            raise SpecError(f"{context}: unknown count filter {system}.{resource}.{prop}")
            _verify_targets(item, systems, context)
    elif isinstance(tree, list):
        for item in tree:
            _verify_targets(item, systems, context)


def validate_spec(spec: dict[str, Any], *, allow_draft: bool = False) -> None:
    if spec.get("version") != "0.1":
        raise SpecError("Manifest version must be '0.1'")
    if not isinstance(spec.get("organization"), str) or not spec["organization"].strip():
        raise SpecError("organization is required")
    if not spec.get("reviewed", False) and not allow_draft:
        raise SpecError("Unreviewed manifest. Review business policies then set reviewed: true; --allow-draft is for contract-only exploration.")
    systems = spec.get("systems")
    if not isinstance(systems, dict) or not systems:
        raise SpecError("systems must be a nonempty mapping")
    for sys_id, system in systems.items():
        if not isinstance(system, dict) or not isinstance(system.get("resources"), dict):
            raise SpecError(f"{sys_id}: resources mapping required")
        for name, conf in system["resources"].items():
            context = f"{sys_id}.{name}"
            if not isinstance(conf, dict):
                raise SpecError(f"{context}: resource must be an object")
            schema = conf.get("schema", {})
            _validate_schema(schema, context)
            for input_action in ('create', 'update'):
                if f'{input_action}_input_schema' in conf:
                    _validate_schema(conf[f'{input_action}_input_schema'], f'{context}.{input_action}_input_schema')
            id_field = conf.get("id_field", "id")
            if id_field not in schema.get("properties", {}):
                raise SpecError(f"{context}: id_field {id_field!r} is absent in schema")
            allowed_actions = {"create", "get", "list", "update", "delete"}
            for action, roles in conf.get("allow", {}).items():
                if action not in allowed_actions or not isinstance(roles, list) or not all(isinstance(r, str) for r in roles):
                    raise SpecError(f"{context}: invalid roles for action {action}")
            for action, guards in conf.get("guards", {}).items():
                if action not in {"create", "update", "delete"} or not isinstance(guards, list):
                    raise SpecError(f"{context}: invalid guards for {action}")
                for condition in guards:
                    _check_condition(condition, f"{context}.{action}")
                    _verify_targets(condition, systems, f"{context}.{action}")
            for action, effects in conf.get("effects", {}).items():
                if action not in {"create", "update", "delete"} or not isinstance(effects, list):
                    raise SpecError(f"{context}: invalid effects for {action}")
                for effect in effects:
                    if effect.get("kind") != "adjust" or not {"system", "resource", "id"} <= effect.get("target", {}).keys():
                        raise SpecError(f"{context}: only adjust effects with target system/resource/id are supported")
                    _check_ref(effect["target"]["id"], context)
                    _check_ref(effect["by"], context)
                    _verify_targets(effect, systems, context)
                    target = effect["target"]
                    if target["system"] not in systems or target["resource"] not in systems[target["system"]].get("resources", {}):
                        raise SpecError(f"{context}: unknown effect resource {target['system']}.{target['resource']}")
                    field = effect.get("field")
                    target_schema = systems[target["system"]]["resources"][target["resource"]]["schema"]
                    if field not in target_schema.get("properties", {}):
                        raise SpecError(f"{context}: unknown effect field {field}")
            for action, project in conf.get('response_projection', {}).items():
                if action not in {'get', 'list'} or not isinstance(project, dict):
                    raise SpecError(f'{context}: response_projection supports get/list only')
                fields = project.get('fields')
                if not isinstance(fields, dict) or len(fields) > 500:
                    raise SpecError(f'{context}: projection fields must be mapping of at most 500 fields')
                if type(project.get('include_base', False)) is not bool:
                    raise SpecError(f'{context}: include_base must be boolean')
                for out_name, expression in fields.items():
                    if not isinstance(out_name, str) or not out_name:
                        raise SpecError(f'{context}: projection field name invalid')
                    _check_ref(expression, context)
                    _verify_targets(expression, systems, context)
            for fk in conf.get("foreign_keys", []):
                if not {"field", "system", "resource"} <= fk.keys():
                    raise SpecError(f"{context}: foreign_keys entry incomplete")
                if fk["field"] not in schema.get("properties", {}):
                    raise SpecError(f"{context}: foreign key column absent: {fk['field']}")
                if fk["system"] not in systems or fk["resource"] not in systems[fk["system"]].get("resources", {}):
                    raise SpecError(f"{context}: unknown foreign key target {fk['system']}.{fk['resource']}")
            for fields in conf.get("unique", []):
                if not isinstance(fields, list) or not fields or any(f not in schema["properties"] for f in fields):
                    raise SpecError(f"{context}: invalid unique field group")
    from .workflow import validate_workflows
    validate_workflows(spec)
    from .lookup import validate_lookups
    validate_lookups(spec)
    for idx, rule in enumerate(spec.get("integrations", [])):
        if not isinstance(rule.get("event"), str) or not isinstance(rule.get("target"), dict):
            raise SpecError(f"integrations[{idx}]: event and target required")
        target = rule["target"]
        if target.get("system") not in systems or target.get("resource") not in systems[target["system"]]["resources"]:
            raise SpecError(f"integrations[{idx}]: invalid target")
        if rule.get("action", "upsert") != "upsert":
            raise SpecError(f"integrations[{idx}]: only upsert currently supported")
        parts = rule["event"].split(".")
        if len(parts) != 3 or parts[2] not in {"created", "updated", "deleted"} or parts[0] not in systems or parts[1] not in systems[parts[0]].get("resources", {}):
            raise SpecError(f"integrations[{idx}]: source event must reference a known resource and action")
        if not isinstance(rule.get("mapping"), dict):
            raise SpecError(f"integrations[{idx}]: mapping required")
        properties = systems[target["system"]]["resources"][target["resource"]]["schema"].get("properties", {})
        id_field = systems[target["system"]]["resources"][target["resource"]].get("id_field", "id")
        if id_field not in rule["mapping"] or set(rule["mapping"]) - set(properties):
            raise SpecError(f"integrations[{idx}]: mapping must supply ID and may only contain target fields")
        for val in rule["mapping"].values():
            _check_ref(val, f"integrations[{idx}]")
            _verify_targets(val, systems, f"integrations[{idx}]")
        if not isinstance(rule.get("delay_ticks", 0), int) or rule.get("delay_ticks", 0) < 0:
            raise SpecError(f"integrations[{idx}]: invalid delay_ticks")
    for idx, scenario in enumerate(spec.get("scenarios", [])):
        if not {"id", "task", "seed_calls", "checks"} <= scenario.keys():
            raise SpecError(f"scenarios[{idx}]: missing id, task, seed_calls, checks")
        for check in scenario["checks"]:
            _check_condition(check, f"scenarios[{idx}].checks")
            _verify_targets(check, systems, f"scenarios[{idx}].checks")


def build_registry(spec: dict[str, Any]) -> dict[str, Any]:
    tools = []
    edges = []
    for system, sc in sorted(spec["systems"].items()):
        for resource, rc in sorted(sc["resources"].items()):
            for action, roles in sorted(rc.get("allow", {}).items()):
                if not roles:
                    continue
                id_field = rc.get("id_field", "id")
                props = rc["schema"].get("properties", {})
                if action == "create":
                    input_schema = copy.deepcopy(rc.get("create_input_schema", rc["schema"]))
                    input_schema["required"] = [f for f in input_schema.get("required", []) if f != id_field]
                elif action == "update":
                    patch_schema = {"type": "object", "properties": {k: copy.deepcopy(v) for k, v in props.items() if k != id_field},
                                    "additionalProperties": False, "minProperties": 1}
                    input_schema = {"type": "object", "properties": {"id": {"type": "string"}, "patch": patch_schema},
                                    "required": ["id", "patch"], "additionalProperties": False}
                elif action == "list":
                    input_schema = {"type": "object", "properties": {"where": {"type": "object"},
                                    "limit": {"type": "integer", "minimum": 1, "maximum": 1000}}, "additionalProperties": False}
                else:
                    input_schema = {"type": "object", "properties": {"id": {"type": "string"}},
                                    "required": ["id"], "additionalProperties": False}
                tools.append({
                    "name": f"{system}.{resource}.{action}",
                    "description": f"{action.capitalize()} {resource} in {system}; subject to configured business policies",
                    "system": system, "resource": resource, "action": action,
                    "roles": roles,
                    "input_schema": input_schema,
                    "mutates": action in {"create", "update", "delete"},
                })
            for fk in rc.get("foreign_keys", []):
                edges.append({"kind": "foreign_key", "from": f"{system}.{resource}", "to": f"{fk['system']}.{fk['resource']}", "field": fk["field"]})
    from .workflow import workflow_registry
    workflow_tools, workflow_edges = workflow_registry(spec)
    tools.extend(workflow_tools)
    edges.extend(workflow_edges)
    from .lookup import lookup_registry
    tools.extend(lookup_registry(spec))
    for integration in spec.get("integrations", []):
        edges.append({"kind": "event", "from": integration["event"], "to": f"{integration['target']['system']}.{integration['target']['resource']}", "delay_ticks": integration.get("delay_ticks", 0)})
    return {"organization": spec["organization"], "reviewed": bool(spec.get("reviewed")), "tools": tools, "dependency_edges": edges}
