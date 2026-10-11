"""Query-by-field adapters for observable alternate-key endpoints.

A lookup model is evidence-labelled and cannot mutate state. Requires explicit
schema fields and permissions; never interprets arbitrary SQL expressions.
"""
from __future__ import annotations

import copy

from .spec import SpecError, _validate_schema


def validate_lookups(spec: dict) -> None:
    for sys_id, system in spec['systems'].items():
        if 'lookup' in system.get('resources', {}) and system.get('lookups'):
            raise SpecError('lookup is a reserved resource name')
        lookups = system.get('lookups', {})
        if not isinstance(lookups, dict):
            raise SpecError(f'{sys_id}: lookups must be a mapping')
        for name, conf in lookups.items():
            context = f'{sys_id}.lookup.{name}'
            if not isinstance(name, str) or not name or '.' in name or not name.replace('_','').isalnum():
                raise SpecError(f'{context}: invalid name')
            if not isinstance(conf, dict):
                raise SpecError(f'{context}: config must be mapping')
            resource, field = conf.get('resource'), conf.get('field')
            if resource not in system.get('resources', {}):
                raise SpecError(f'{context}: target resource missing')
            if field not in system['resources'][resource]['schema'].get('properties', {}):
                raise SpecError(f'{context}: field not present in declared schema')
            if not isinstance(conf.get('arg'), str) or not conf['arg'] or '.' in conf['arg']:
                raise SpecError(f'{context}: argument name required')
            http = conf.get('http')
            if http is not None:
                if (not isinstance(http, dict) or http.get('method') != 'GET'
                    or not isinstance(http.get('path'), str) or not http['path'].startswith('/')
                    or http['path'].startswith('//') or '..' in http['path']):
                    raise SpecError(f'{context}: invalid HTTP lookup mapping')
            roles = conf.get('roles')
            if not isinstance(roles, list) or not roles or any(not isinstance(x,str) for x in roles):
                raise SpecError(f'{context}: explicit roles required')
            if conf.get('evidence','hypothesis') not in {'hypothesis','observed_partial','reviewed'}:
                raise SpecError(f'{context}: unsupported evidence')
            _validate_schema(conf.get('input_schema', {'type':'object', 'properties':{conf['arg']:{'type':'string'}},
                                                        'required':[conf['arg']]}), context)


def lookup_registry(spec: dict) -> list[dict]:
    result = []
    for sys_id, system in sorted(spec['systems'].items()):
        for name, conf in sorted(system.get('lookups', {}).items()):
            result.append({'name': f'{sys_id}.lookup.{name}', 'description': conf.get('description',f'Lookup {conf["resource"]} by {conf["field"]}'),
                           'system': sys_id, 'resource':'lookup', 'action': name,
                           'kind':'lookup', 'roles': conf['roles'], 'mutates':False,
                           'evidence': conf.get('evidence','hypothesis'),
                           'input_schema': copy.deepcopy(conf.get('input_schema', {
                               'type':'object', 'properties':{conf['arg']:{'type':'string'}},'required':[conf['arg']]}))})
    return result
