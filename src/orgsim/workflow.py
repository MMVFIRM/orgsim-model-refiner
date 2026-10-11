"""Deterministic, declarative workflow transitions (research-only).

No eval, Python code loading, network, or dynamic tool dispatch. An action model
is an *unverified hypothesis* until separately checked against the real system.
All action steps are committed in one SQLite transaction by SimulationEngine.
"""
from __future__ import annotations

import copy
from typing import Any

from .spec import SpecError, _check_ref, _check_condition, _validate_schema, _verify_targets


SAFE_NAME = 'workflow'


def validate_workflows(spec: dict[str, Any]) -> None:
    systems = spec['systems']
    for system, sc in systems.items():
        if 'workflow' in sc.get('resources', {}) and sc.get('workflows'):
            raise SpecError(f'{system}: workflow is a reserved resource name')
        workflows = sc.get('workflows', {})
        if not isinstance(workflows, dict):
            raise SpecError(f'{system}: workflows must be a mapping')
        for name, conf in workflows.items():
            context = f'{system}.workflow.{name}'
            if not isinstance(name, str) or not name or '.' in name or not name.replace('_', '').isalnum():
                raise SpecError(f'{context}: invalid workflow name')
            if not isinstance(conf, dict):
                raise SpecError(f'{context}: workflow must be an object')
            roles = conf.get('roles', [])
            if not isinstance(roles, list) or not roles or not all(isinstance(r, str) and r for r in roles):
                raise SpecError(f'{context}: explicit nonempty roles required')
            if conf.get('evidence', 'hypothesis') not in {'hypothesis', 'observed_partial', 'reviewed'}:
                raise SpecError(f'{context}: invalid evidence label')
            http = conf.get('http')
            if http is not None:
                if (not isinstance(http, dict) or http.get('method') not in {'POST', 'PUT', 'PATCH', 'DELETE'}
                    or not isinstance(http.get('path'), str) or not http['path'].startswith('/')
                    or http['path'].startswith('//') or '..' in http['path']):
                    raise SpecError(f'{context}: invalid HTTP route mapping')
            schema = conf.get('input_schema', {'type': 'object', 'additionalProperties': True})
            _validate_schema(schema, context + '.input_schema')
            preconditions = conf.get('preconditions', [])
            if not isinstance(preconditions, list):
                raise SpecError(f'{context}: preconditions must be a list')
            for item in preconditions:
                _check_condition(item, context)
                _verify_targets(item, systems, context)
            steps = conf.get('steps')
            if not isinstance(steps, list) or not steps:
                raise SpecError(f'{context}: nonempty steps required; unsupported actions must not be executable')
            if len(steps) > 40:
                raise SpecError(f'{context}: workflow exceeds 40 steps')
            for idx, step in enumerate(steps):
                ctx = f'{context}.steps[{idx}]'
                if not isinstance(step, dict) or step.get('kind') not in {'update', 'create', 'adjust'}:
                    raise SpecError(f'{ctx}: supported step kinds are update/create/adjust')
                sid, resource = step.get('system'), step.get('resource')
                if sid not in systems or resource not in systems[sid].get('resources', {}):
                    raise SpecError(f'{ctx}: unknown step target {sid}.{resource}')
                properties = systems[sid]['resources'][resource]['schema']['properties']
                kind = step['kind']
                if kind != 'create':
                    if 'id' not in step:
                        raise SpecError(f'{ctx}: missing record id expression')
                    _check_ref(step['id'], ctx)
                if kind in {'create', 'update'}:
                    values = step.get('values')
                    if not isinstance(values, dict) or not values:
                        raise SpecError(f'{ctx}: nonempty values mapping required')
                    if kind == 'update' and systems[sid]['resources'][resource].get('id_field', 'id') in values:
                        raise SpecError(f'{ctx}: ID cannot be updated')
                    for field, value in values.items():
                        if field not in properties:
                            raise SpecError(f'{ctx}: unknown output field {field}')
                        _check_ref(value, ctx)
                        _verify_targets(value, systems, ctx)
                if kind == 'adjust':
                    if step.get('field') not in properties:
                        raise SpecError(f'{ctx}: adjustment field not in resource schema')
                    if 'by' not in step:
                        raise SpecError(f'{ctx}: adjustment needs by expression')
                    _check_ref(step['by'], ctx)
                    _verify_targets(step['by'], systems, ctx)
                if 'require' in step:
                    if not isinstance(step['require'], list):
                        raise SpecError(f'{ctx}: require must be list')
                    for item in step['require']:
                        _check_condition(item, ctx)
                        _verify_targets(item, systems, ctx)
            if 'result' in conf:
                _check_ref(conf['result'], context)
                _verify_targets(conf['result'], systems, context)


def workflow_registry(spec: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    tools, edges = [], []
    for system, sc in sorted(spec['systems'].items()):
        for name, conf in sorted(sc.get('workflows', {}).items()):
            tool = f'{system}.workflow.{name}'
            tools.append({
                'name': tool,
                'description': conf.get('description', f'Workflow action {name}; unverified until independently tested'),
                'system': system, 'resource': 'workflow', 'action': name,
                'roles': conf['roles'], 'input_schema': copy.deepcopy(conf.get('input_schema', {'type': 'object'})),
                'mutates': True, 'kind': 'workflow', 'evidence': conf.get('evidence', 'hypothesis'),
            })
            for step in conf['steps']:
                edges.append({'kind': 'workflow_effect', 'from': tool,
                              'to': f'{step["system"]}.{step["resource"]}', 'effect': step['kind']})
    return tools, edges
