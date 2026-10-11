"""Dolibarr Swagger 2.0 -> untrusted OpenAPI 3.x contract + coverage diagnostics.

Contract translation is intentionally separate from behavioral inference. A
Swagger type is *not* evidence that a policy, workflow or side effect exists.
"""
from __future__ import annotations

import copy
from typing import Any

from .spec import SpecError

_METHODS = {'get', 'post', 'put', 'patch', 'delete'}


def _rewrite_refs(obj: Any) -> Any:
    if isinstance(obj, list):
        return [_rewrite_refs(x) for x in obj]
    if not isinstance(obj, dict):
        return obj
    out: dict = {}
    for k, v in obj.items():
        if k == '$ref':
            if not isinstance(v, str) or not v.startswith('#/'):
                raise SpecError('External Swagger references are not supported')
            mapping = {'#/definitions/': '#/components/schemas/',
                       '#/parameters/': '#/components/parameters/',
                       '#/responses/': '#/components/responses/'}
            out[k] = next((new + v[len(old):] for old, new in mapping.items() if v.startswith(old)), v)
        else:
            out[k] = _rewrite_refs(v)
    return out


def _resolve_parameters(doc: dict, params: list[dict]) -> list[dict]:
    out = []
    for param in params:
        if not isinstance(param, dict):
            continue
        if '$ref' in param:
            ref = param['$ref']
            if not isinstance(ref, str) or not ref.startswith('#/parameters/'):
                raise SpecError(f'Unsupported Swagger parameter reference: {ref}')
            key = ref.split('/')[-1].replace('~1', '/').replace('~0', '~')
            if key not in doc.get('parameters', {}):
                raise SpecError(f'Missing Swagger parameter: {ref}')
            param = doc['parameters'][key]
            if '$ref' in param:
                raise SpecError(f'Nested parameter reference needs manual resolution: {ref}')
        out.append(copy.deepcopy(param))
    return out


def swagger2_to_openapi3(swagger: dict) -> dict:
    """Translate Swagger 2.0 constructs needed by OrgSim's resource importer.

    Preserves all paths including custom workflow paths, so the importer can
    report rather than hallucinate their behavior. Other non-JSON/form operations
    are retained as contract operations but are not given simulated behavior.
    """
    if str(swagger.get('swagger')) != '2.0':
        raise SpecError('Expected a Swagger 2.0 document exported from Dolibarr API explorer')
    if not isinstance(swagger.get('paths'), dict):
        raise SpecError('Malformed Dolibarr Swagger: paths must be an object, not a list')
    result = {
        'openapi': '3.0.3',
        'info': copy.deepcopy(swagger.get('info', {'title': 'Dolibarr API', 'version': 'unknown'})),
        'paths': {},
        'components': {'schemas': _rewrite_refs(swagger.get('definitions', {}))},
        'x-original-format': 'Swagger 2.0',
    }
    for route, path_item in swagger.get('paths', {}).items():
        if not isinstance(path_item, dict):
            continue
        converted: dict = {}
        inherited = _resolve_parameters(swagger, path_item.get('parameters', []))
        for verb, op in path_item.items():
            if verb.lower() not in _METHODS or not isinstance(op, dict):
                continue
            parameters = _resolve_parameters(swagger, inherited + op.get('parameters', []))
            entry = {'summary': op.get('summary', ''),
                     'description': op.get('description', ''),
                     'responses': {}}
            if op.get('operationId'):
                entry['operationId'] = op['operationId']
            query_params = []
            for p in parameters:
                if p.get('in') == 'body':
                    if 'requestBody' in entry:
                        raise SpecError(f'Multiple body parameters for {verb.upper()} {route}')
                    entry['requestBody'] = {
                        'required': bool(p.get('required', False)),
                        'content': {'application/json': {'schema': _rewrite_refs(p.get('schema', {}))}}
                    }
                elif p.get('in') in {'path', 'query', 'header'}:
                    schema = _rewrite_refs(p.get('schema', {k: p[k] for k in (
                        'type', 'format', 'enum', 'default', 'items', 'minimum', 'maximum') if k in p}))
                    query_params.append({
                        'name': p.get('name', ''),
                        'in': p['in'],
                        'required': bool(p.get('required', False) or p['in'] == 'path'),
                        'schema': schema or {'type': 'string'},
                        'description': p.get('description', ''),
                    })
                else:
                    # Swagger formData is deliberately not treated as JSON.
                    entry.setdefault('x-unconverted-parameters', []).append(p)
            if query_params:
                entry['parameters'] = query_params
            for code, resp in op.get('responses', {}).items():
                if '$ref' in resp:
                    ref = resp['$ref']
                    if not ref.startswith('#/responses/'):
                        raise SpecError('Unsupported response ref')
                    key = ref.split('/')[-1]
                    resp = swagger.get('responses', {}).get(key, {})
                converted_resp = {'description': resp.get('description', '')}
                if isinstance(resp.get('schema'), dict):
                    converted_resp['content'] = {'application/json': {
                        'schema': _rewrite_refs(resp['schema'])
                    }}
                entry['responses'][str(code)] = converted_resp
            converted[verb.lower()] = entry
        if converted:
            result['paths'][route] = converted
    return result


def contract_coverage(openapi: dict, manifest: dict, *, system: str = 'dolibarr') -> dict:
    """Coverage is lexical API mapping, not semantic correctness."""
    offered = []
    expected_maps: list[dict] = []
    custom = []
    resources = manifest.get('systems', {}).get(system, {}).get('resources', {})
    for route, spec in sorted(openapi.get('paths', {}).items()):
        segments = [s for s in route.strip('/').split('/') if s]
        for method, op in spec.items():
            if method not in _METHODS:
                continue
            offered.append(f'{method.upper()} {route}')
            action = None
            if len(segments) == 1:
                action = {'get': 'list', 'post': 'create'}.get(method)
            elif len(segments) == 2 and segments[1].startswith('{') and segments[1].endswith('}'):
                action = {'get': 'get', 'put': 'update', 'patch': 'update', 'delete': 'delete'}.get(method)
            resource = segments[0].replace('-', '_') if segments else ''
            if resource in resources and action in resources[resource].get('allow', {}):
                if action == 'list':
                    r = op.get('responses', {}).get('200', {})
                    if r.get('content', {}).get('application/json', {}).get('schema', {}).get('type') != 'array':
                        custom.append(f'{method.upper()} {route}')
                        continue
                expected_maps.append({'method': method.upper(), 'path': route,
                                      'tool': f'{system}.{resource}.{action}'})
            else:
                custom.append(f'{method.upper()} {route}')
    return {
        'format': 'orgsim-contract-coverage-v1',
        'evidence': 'API surface ONLY; no real-system behavior was tested',
        'api_operations': len(offered),
        'mapped_operations': len(expected_maps),
        'unimplemented_operations': len(offered) - len(expected_maps),
        'mapped_fraction': round(len(expected_maps) / len(offered), 4) if offered else 0,
        'mapped': expected_maps,
        'not_reconstructed': sorted(set(custom)),
        'resource_count': len(resources),
        'review_notes': manifest.get('review_notes', []),
    }


def model_operation_catalog(swagger: dict) -> list[dict]:
    """Generate a value-free operation catalog for model-side reconstruction.

    Contract signatures/descriptions are unverified hints, not policy evidence.
    Unlike a tool-name-only coverage report this includes action endpoints and
    declared request properties. It copies no example payloads, default values,
    live hostnames, Swagger security schemes or API keys.
    """
    from .openapi import _resolve_local
    document = swagger2_to_openapi3(swagger)
    catalog = []
    def schema_hint(schema: Any) -> dict:
        if not isinstance(schema, dict):
            return {'type': 'unspecified', 'fields': []}
        try:
            resolved = _resolve_local(schema, document)
        except (SpecError, RecursionError, TypeError):
            return {'type': 'unresolved', 'fields': []}
        if not isinstance(resolved, dict):
            return {'type': 'unspecified', 'fields': []}
        typ = resolved.get('type', 'unspecified')
        fields = []
        if typ == 'array':
            item_hint = schema_hint(resolved.get('items', {}))
            return {'type': 'array', 'item': item_hint}
        if typ == 'object' or 'properties' in resolved:
            for field, val in sorted(resolved.get('properties', {}).items()):
                if len(fields) >= 250:  # conservative size bound per endpoint
                    break
                fields.append({'name': field,
                               'type': val.get('type', 'unspecified') if isinstance(val, dict) else 'unspecified',
                               'required': field in resolved.get('required', [])})
        return {'type': typ, 'fields': fields}
    for path, item in sorted(document.get('paths', {}).items()):
        for method, op in item.items():
            if method not in _METHODS:
                continue
            parameters = []
            for param in op.get('parameters', []):
                parameters.append({'name': param.get('name'),
                                   'in': param.get('in'), 'required': bool(param.get('required')),
                                   'type': param.get('schema', {}).get('type', 'unspecified')})
            requests = op.get('requestBody', {}).get('content', {}).get('application/json', {}).get('schema', {})
            responses = {}
            for status_code, response in op.get('responses', {}).items():
                if str(status_code).startswith(('2', '4')):
                    responses[status_code] = schema_hint(response.get('content', {}).get('application/json', {}).get('schema', {}))
            catalog.append({
                'method': method.upper(), 'path': path,
                'operation_id': op.get('operationId', ''),
                'summary': str(op.get('summary') or '')[:300],
                'description': str(op.get('description') or '')[:500],
                'parameters': parameters,
                'request': schema_hint(requests),
                'responses': responses,
            })
    return catalog
