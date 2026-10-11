"""Dolibarr reconstruction Gate 1: capture, import, and compare.

No remote activity at import time. Nothing writes to Dolibarr unless the human
explicitly confirms a disposable sandbox and enables write probes.
"""
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
from string import Template
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler
from typing import Any

import yaml
from .compiler import compile_spec
from .dolibarr_contract import contract_coverage, swagger2_to_openapi3, model_operation_catalog
from .engine import SimError, SimulationEngine
from .openapi import import_openapi
from .spec import SpecError, load_spec, build_registry

MAX_BODY = 12_000_000
CONFIRM = 'I-OWN-THIS-DISPOSABLE-DOLIBARR-SANDBOX'


def _local_host(host: str) -> bool:
    if host.lower() == 'localhost':
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError('HTTP redirect blocked: do not follow external ERP redirects automatically')


class DolibarrClient:
    def __init__(self, base_url: str, api_key: str | None, *, transport=None):
        root = base_url.rstrip('/')
        parsed = urlparse(root)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError('URL must be an HTTP(S) Dolibarr server without embedded credentials')
        if parsed.query or parsed.fragment:
            raise ValueError('Place API keys in DOLIBARR_API_KEY, never in URL parameters')
        if parsed.scheme == 'http' and not _local_host(parsed.hostname):
            raise ValueError('Remote Dolibarr requires HTTPS; HTTP is allowed only on loopback')
        if root.endswith('/api/index.php'):
            self.root = root
        else:
            self.root = root + '/api/index.php'
        self.api_key = api_key
        self.transport = transport or build_opener(NoRedirect())

    @property
    def loopback(self) -> bool:
        return _local_host(urlparse(self.root).hostname or '')

    def request(self, method: str, path: str, data: Any = None) -> tuple[int, Any]:
        method = method.upper()
        if method not in {'GET', 'POST', 'PUT', 'PATCH', 'DELETE'}:
            raise ValueError('Unsupported HTTP method')
        if not isinstance(path, str) or not path.startswith('/') or path.startswith('//') or '\\' in path:
            raise ValueError('API path must begin with one slash and cannot change the host')
        parsed_path = urlsplit(path)
        if any(x in {'.', '..'} for x in parsed_path.path.split('/')) or parsed_path.fragment:
            raise ValueError('Invalid or redirected API path')
        # urljoin(path with leading slash) otherwise discards the Dolibarr index.php prefix
        uri = self.root + parsed_path.path + (('?' + parsed_path.query) if parsed_path.query else '')
        headers = {'Accept': 'application/json'}
        if self.api_key:
            headers['DOLAPIKEY'] = self.api_key
        payload = None
        if data is not None:
            payload = json.dumps(data, separators=(',', ':')).encode('utf-8')
            headers['Content-Type'] = 'application/json'
        req = Request(uri, headers=headers, data=payload, method=method)
        try:
            with self.transport.open(req, timeout=25) as resp:
                status = resp.status
                body = resp.read(MAX_BODY + 1)
        except HTTPError as exc:
            status = exc.code
            body = exc.read(MAX_BODY + 1)
        if len(body) > MAX_BODY:
            raise ValueError('Remote response exceeds safety limit')
        if not body:
            return status, None
        try:
            return status, json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Never embed arbitrary server HTML/error pages in reports.
            return status, {'_non_json_response': True}


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def capture_swagger(client: DolibarrClient, out: Path) -> dict:
    status, raw = client.request('GET', '/explorer/swagger.json')
    # Some Restler versions also expose a /root document.
    if status in {404, 400}:
        status, raw = client.request('GET', '/explorer/swagger.json/root')
    if status != 200 or not isinstance(raw, dict):
        raise ValueError(f'Dolibarr Swagger discovery failed (HTTP {status}); check REST module, API key and permissions')
    if raw.get('swagger') != '2.0':
        raise ValueError(f'Expected Swagger 2.0; got {raw.get("swagger")!r}')
    # Swagger may contain host, path and example values, so treat export as sensitive.
    _json(out, raw)
    return {'captured': str(out), 'sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
            'operations': sum(sum(method in {'get', 'post', 'put', 'patch', 'delete'}
                                  for method in item) for item in raw.get('paths', {}).values()),
            'note': 'Local capture of the target API contract; retain privately'}


def reconstruct(swagger: dict, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    api = swagger2_to_openapi3(swagger)
    openapi_path = output_dir / 'normalized_openapi.json'
    _json(openapi_path, api)
    spec = import_openapi([openapi_path], system='dolibarr', organization='Dolibarr disposable test tenant')
    manifest_path = output_dir / 'draft_manifest.yaml'
    manifest_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding='utf-8')
    artifacts = output_dir / 'simulator'
    compile_spec(spec, artifacts, allow_draft=True)
    coverage = contract_coverage(api, spec)
    _json(output_dir / 'coverage.json', coverage)
    return {'manifest': str(manifest_path), 'compile_dir': str(artifacts),
            'coverage': {k: coverage[k] for k in ('api_operations', 'mapped_operations',
                        'unimplemented_operations', 'mapped_fraction', 'resource_count')},
            'assurance': 'contract_only_unreviewed; not a validated Dolibarr replica'}


def _expand(obj: Any, variables: dict[str, Any]) -> Any:
    # Keep numeric captures numeric when they occupy the entire value.
    if isinstance(obj, str):
        whole = re.fullmatch(r'\$\{([A-Za-z_][A-Za-z_0-9]*)\}', obj)
        if whole:
            key = whole.group(1)
            if key not in variables:
                raise ValueError(f'Missing capture variable: {key}')
            return variables[key]
        return Template(obj).substitute({k: str(v) for k, v in variables.items()})
    if isinstance(obj, dict):
        return {k: _expand(v, variables) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand(v, variables) for v in obj]
    return obj


def _kind(item: Any) -> str:
    if isinstance(item, dict): return 'object'
    if isinstance(item, list): return 'array'
    if isinstance(item, bool): return 'boolean'
    if isinstance(item, (float, int)): return 'number'
    if isinstance(item, str): return 'string'
    if item is None: return 'null'
    return 'unknown'


def _shape(item: Any) -> dict:
    """Only data shape is logged; never retain actual customer/order values."""
    kind = _kind(item)
    out: dict = {'kind': kind}
    if kind == 'array':
        out['length'] = len(item)
        if item and isinstance(item[0], dict):
            out['first_keys'] = sorted(item[0].keys())
            out['first_types'] = {k: _kind(v) for k, v in item[0].items()}
    elif kind == 'object':
        out['keys'] = sorted(item.keys())
        out['types'] = {k: _kind(v) for k, v in item.items()}
    return out



def _field_coverage(live_shape: dict, sim_shape: dict) -> dict:
    """Read/list key and type metrics. Key overlap is not behavioral fidelity."""
    lk = 'keys' if live_shape.get('kind') == 'object' else 'first_keys'
    tk = 'types' if live_shape.get('kind') == 'object' else 'first_types'
    if live_shape.get('kind') != sim_shape.get('kind') or live_shape.get('kind') not in {'object', 'array'}:
        return {}
    if lk not in live_shape or lk not in sim_shape:
        return {}
    live_keys, sim_keys = set(live_shape[lk]), set(sim_shape[lk])
    shared = live_keys & sim_keys
    lt, st = live_shape.get(tk, {}), sim_shape.get(tk, {})
    type_matches = sum(lt.get(k) == st.get(k) for k in shared)
    return {
        'field_precision': len(shared) / len(sim_keys) if sim_keys else None,
        'field_type_match_rate': type_matches / len(shared) if shared else None,
        'fields_missing': len(live_keys - sim_keys),
        'fields_extra': len(sim_keys - live_keys),
        'fields_with_type_disagreement': len(shared) - type_matches,
    }


def _semantic_value_metrics(live: Any, simulated: Any, case: dict) -> dict:
    """Evaluate operator-selected business fields without exposing record values.

    No name/reference equality is assumed between independently generated IDs.
    For multi-row arrays an explicit, unique match_items_by key is required.
    """
    fields = case.get('compare_fields')
    if fields is None:
        return {}
    if (not isinstance(fields, list) or not fields or len(fields) > 50 or
        not all(isinstance(k, str) and re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', k) for k in fields) or
        len(fields) != len(set(fields))):
        raise ValueError('compare_fields must be a unique list of 1..50 simple field names')
    if isinstance(live, dict) and isinstance(simulated, dict):
        pairs = [(live, simulated)]
    elif isinstance(live, list) and isinstance(simulated, list):
        if len(live) != len(simulated):
            return {'semantic_status': 'unavailable_count_mismatch', 'semantic_compared': 0}
        if len(live) == 0:
            return {'semantic_status': 'unavailable_empty', 'semantic_compared': 0}
        if len(live) == 1 and isinstance(live[0], dict) and isinstance(simulated[0], dict):
            pairs = [(live[0], simulated[0])]
        else:
            key = case.get('match_items_by')
            if not isinstance(key, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', key):
                return {'semantic_status': 'unavailable_alignment_key_required', 'semantic_compared': 0}
            def keyed(items: list) -> dict | None:
                mapping = {}
                for item in items:
                    if not isinstance(item, dict) or key not in item or not isinstance(item[key], (str, int, float)):
                        return None
                    encoded = _kind(item[key]), str(item[key])
                    if encoded in mapping:
                        return None
                    mapping[encoded] = item
                return mapping
            left, right = keyed(live), keyed(simulated)
            if left is None or right is None or set(left) != set(right):
                return {'semantic_status': 'unavailable_unaligned_rows', 'semantic_compared': 0}
            pairs = [(left[k], right[k]) for k in left]
    else:
        return {'semantic_status': 'unavailable_nonrecord', 'semantic_compared': 0}
    compared = len(fields) * len(pairs)
    matches = sum((f in x and f in y and _kind(x[f]) == _kind(y[f]) and x[f] == y[f])
                  for x, y in pairs for f in fields)
    return {'semantic_status': 'compared', 'semantic_compared': compared,
            'semantic_matched': matches,
            'semantic_match_rate': matches / compared if compared else None}


def summarize_discovery_gaps(report: dict) -> dict:
    """Extract type/key hypotheses from *discovery only*; no values or inferred rules.

    Rejected holdout provenance remains rejected even if it is relabeled by
    a command-line argument: source report phase is checked in the file.
    """
    if report.get('phase') != 'discovery':
        raise ValueError('Only discovery-phase reports may be summarized for reconstruction')
    result = []
    for probe in report.get('probes', []):
        live, sim = probe.get('live_shape', {}), probe.get('sim_shape', {})
        lk = 'keys' if live.get('kind') == 'object' else 'first_keys'
        tk = 'types' if live.get('kind') == 'object' else 'first_types'
        if live.get('kind') not in ('object', 'array') or lk not in live:
            continue
        live_keys, sim_keys = set(live.get(lk, [])), set(sim.get(lk, []))
        live_types, sim_types = live.get(tk, {}), sim.get(tk, {})
        result.append({'probe_id': probe.get('id'), 'response_kind': live['kind'],
                       'missing_from_simulator': [
                           {'field': k, 'observed_kind': live_types.get(k, 'unknown')}
                           for k in sorted(live_keys - sim_keys)],
                       'type_conflicts': [
                           {'field': k, 'observed_kind': live_types.get(k, 'unknown'),
                            'simulated_kind': sim_types.get(k, 'unknown')}
                           for k in sorted(live_keys & sim_keys) if live_types.get(k) != sim_types.get(k)],
                       'simulator_only_fields': sorted(sim_keys - live_keys),
                       'verified_values': 0})
    return {'format': 'orgsim-discovery-shape-gaps-v1', 'phase': 'discovery',
            'evidence': 'observed response field names/types only; no field values or workflow rules',
            'warning': 'Field stubs and null padding cannot establish behavioral fidelity.',
            'cases': result}

def _captured_id(item: Any) -> Any:
    if isinstance(item, (str, int)):
        return item
    if isinstance(item, dict):
        for k in ('id', 'rowid'):
            if item.get(k) is not None:
                return item[k]
    raise ValueError('API response does not include a usable record ID')


def _mapped_tools(spec: dict) -> set[str]:
    return {t['name'] for t in build_registry(spec)['tools']}


def _path_matches_tool(method: str, path: str, tool: str, spec: dict | None = None) -> bool:
    """Prevent a configured probe from invoking an unrelated real ERP endpoint."""
    parts = tool.split('.')
    if len(parts) != 3 or parts[0] != 'dolibarr':
        return False
    sys_id, resource, action = parts
    if resource in {'workflow', 'lookup'}:
        if not spec:
            return False
        key = 'workflows' if resource == 'workflow' else 'lookups'
        conf = spec.get('systems', {}).get(sys_id, {}).get(key, {}).get(action, {})
        http = conf.get('http', {})
        if method != http.get('method') or not isinstance(http.get('path'), str):
            return False
        expected = http['path'].strip('/').split('/')
        actual = urlsplit(path).path.strip('/').split('/')
        if len(expected) != len(actual):
            return False
        return all((x.startswith('{') and x.endswith('}') and bool(y)) or x == y
                   for x, y in zip(expected, actual))
    segments = urlsplit(path).path.strip('/').split('/')
    if not segments or segments[0].replace('-', '_') != resource:
        return False
    expected = {
        'list': ('GET', 1), 'get': ('GET', 2),
        'create': ('POST', 1), 'update': ('PUT', 2),
        'delete': ('DELETE', 2),
    }
    if action not in expected:
        return False
    verb, count = expected[action]
    return method == verb and len(segments) == count



def _response_from_contract(result: Any, rc: dict, action: str) -> tuple[int, Any]:
    """Adapt only shapes explicitly declared in the source OpenAPI contract.

    This does NOT synthesize server-generated read fields, errors, or business rules.
    """
    info = rc.get('response_contract', {}).get(action, {})
    status = info.get('status', 200)
    kind = info.get('kind')
    if action == 'create' and isinstance(result, dict):
        rid = result.get(rc.get('id_field', 'id'))
        if rid is not None and kind == 'number':
            return status, (int(rid) if isinstance(rid, str) and rid.isdecimal() else rid)
        if rid is not None and kind == 'string':
            return status, str(rid)
    if action == 'list' and isinstance(result, list) and not result and info.get('empty_status'):
        return int(info['empty_status']), {'_simulated_empty_list_error': True}
    return status, result


def run_probes(spec: dict, suite: dict, client: DolibarrClient, *, phase: str,
               allow_writes: bool = False, confirm: str = '') -> dict:
    """Run matched probes on a live tenant and an independent empty simulator.

    A differential match is only a match for measured dimensions, not proof of
    correct business behavior. Never run discovery cases during holdout.
    """
    if phase not in {'discovery', 'holdout'}:
        raise ValueError('phase must be discovery or holdout')
    if allow_writes and confirm != CONFIRM:
        raise ValueError('Write probes require the exact disposable-sandbox confirmation phrase')
    if allow_writes and not client.loopback:
        raise ValueError('Write probes are restricted to localhost; use a local disposable sandbox')
    selected = [item for item in suite['cases'] if item.get('phase') == phase]
    if not selected:
        raise ValueError(f'No probes configured for phase: {phase}')
    tools = _mapped_tools(spec)
    # Run the simulator under a role that models the actual sandbox API user.
    # Scenario fixture seeding remains explicitly separate from the agent role.
    sim_role = suite.get('sim_role', 'test_agent')
    if not isinstance(sim_role, str) or sim_role not in {r for t in build_registry(spec)['tools'] for r in t['roles']}:
        raise ValueError('Suite sim_role must be a declared simulator role')
    source_sha = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    records: list[dict] = []
    vars_live: dict[str, Any] = {}
    vars_sim: dict[str, Any] = {}
    sim = SimulationEngine(spec, allow_draft=True)
    seed_report = {'configured': 0, 'applied': 0, 'failures': []}
    try:
        # Synthetic fixture injection is performed using normal simulator tools,
        # never by copying hidden holdout answers or bypassing schema validation.
        seed_calls = suite.get('sim_seed_calls', [])
        if not isinstance(seed_calls, list):
            raise ValueError('sim_seed_calls must be a list')
        seed_report['configured'] = len(seed_calls)
        for index, seed in enumerate(seed_calls):
            tool_name = seed.get('tool') if isinstance(seed, dict) else None
            if tool_name not in tools or not tool_name.endswith('.create'):
                seed_report['failures'].append({'index': index, 'error_code': 'UNMAPPED_SEED_TOOL'})
                continue
            try:
                sim.call(tool_name, seed.get('args', {}), role='test_agent', actor='scenario_setup')
                seed_report['applied'] += 1
            except SimError as exc:
                seed_report['failures'].append({'index': index, 'error_code': exc.code,
                                                'validation': exc.diagnostics})
        for case in selected:
            cid = case['id']
            method = case['method'].upper()
            tool = case['tool']
            if tool not in tools:
                records.append({'id': cid, 'result': 'unmapped', 'reason': 'Tool was not reconstructed'})
                continue
            if method not in {'GET', 'POST', 'PUT', 'PATCH', 'DELETE'}:
                records.append({'id': cid, 'result': 'skipped', 'reason': 'Unsupported method'})
                continue
            # PATCH is allowed only when the manifest exposes update and the path is a detail route.
            route_for_validation = case['path']
            if not _path_matches_tool('PUT' if method == 'PATCH' else method, route_for_validation, tool, spec):
                records.append({'id': cid, 'result': 'skipped', 'reason': 'Tool/HTTP path or method mismatch'})
                continue
            if method != 'GET' and not allow_writes:
                records.append({'id': cid, 'result': 'write_blocked', 'reason': 'Requires --allow-writes and sandbox confirmation'})
                continue
            try:
                missing_dependencies = [v for v in re.findall(r'\$\{([A-Za-z_][A-Za-z_0-9]*)\}',
                                                               str(case['path']) + json.dumps(case.get('args', {})))
                                        if v not in vars_live or v not in vars_sim]
                if missing_dependencies:
                    records.append({'id': cid, 'result': 'dependency_unavailable',
                                    'reason': 'Prior captured ID unavailable'})
                    continue
                path = _expand(case['path'], vars_live)
                live_args = _expand(case.get('body'), vars_live) if 'body' in case else None
                sim_args = _expand(case.get('args', case.get('body', {})), vars_sim)
                if method == 'POST' and live_args is not None and 'args' in case and sim_args != live_args:
                    records.append({'id': cid, 'result': 'skipped', 'reason': 'Create request body differs between systems'})
                    continue
                live_status, live_body = client.request(method, path, live_args)
                if live_status >= 500:
                    records.append({'id': cid, 'result': 'reference_server_error',
                                    'live_status': live_status, 'reason': 'Not scored as reconstruction parity'})
                    continue
                sim_error = None
                try:
                    sim_result = sim.call(tool, sim_args, role=sim_role, actor='reconstruction_gate')
                    resource_name, action_name = tool.split('.')[1:]
                    if resource_name == 'workflow':
                        rc = spec['systems']['dolibarr']['workflows'][action_name]
                        sim_status, sim_body = _response_from_contract(sim_result['result'], rc, 'invoke')
                    elif resource_name == 'lookup':
                        rc = spec['systems']['dolibarr']['lookups'][action_name]
                        sim_status, sim_body = _response_from_contract(sim_result['result'], rc, 'get')
                    else:
                        rc = spec['systems']['dolibarr']['resources'][resource_name]
                        sim_status, sim_body = _response_from_contract(sim_result['result'], rc, action_name)
                except SimError as ex:
                    sim_status = ex.status
                    sim_body = ex.as_dict()
                    sim_error = {'code': ex.code}
                    if ex.diagnostics:
                        sim_error['validation'] = ex.diagnostics
                live_shape = _shape(live_body)
                sim_shape = _shape(sim_body)
                outcome_match = (200 <= live_status < 300) == (200 <= sim_status < 300)
                # For negative cases preserve exact HTTP status, positive may return 200/201/204.
                status_match = outcome_match if 200 <= live_status < 300 and 200 <= sim_status < 300 else live_status == sim_status
                shape_match = live_shape['kind'] == sim_shape['kind'] if outcome_match and live_status < 300 else None
                field_recall = None
                if shape_match and live_shape['kind'] == 'object':
                    live_keys, sim_keys = set(live_shape['keys']), set(sim_shape['keys'])
                    field_recall = len(live_keys & sim_keys) / len(live_keys) if live_keys else 1.0
                    shape_match = live_shape['keys'] == sim_shape['keys'] and live_shape['types'] == sim_shape['types']
                if shape_match and live_shape['kind'] == 'array' and live_shape['length'] and sim_shape['length']:
                    live_keys, sim_keys = set(live_shape.get('first_keys', [])), set(sim_shape.get('first_keys', []))
                    field_recall = len(live_keys & sim_keys) / len(live_keys) if live_keys else 1.0
                    shape_match = (live_shape.get('first_keys') == sim_shape.get('first_keys') and
                                   live_shape.get('first_types') == sim_shape.get('first_types'))
                count_match = None
                if live_shape['kind'] == 'array' and sim_shape['kind'] == 'array':
                    count_match = live_shape['length'] == sim_shape['length']
                # Do not count success based on status alone when output shapes differ.
                matched = bool(status_match and (shape_match is None or shape_match))
                if count_match is False:
                    matched = False
                field_coverage = _field_coverage(live_shape, sim_shape)
                semantic = _semantic_value_metrics(live_body, sim_body, case)
                # If the suite explicitly asks for business-field equality,
                # a shape-only match must not be counted as success.
                if semantic and (semantic.get('semantic_status') != 'compared' or
                                 semantic.get('semantic_matched') != semantic.get('semantic_compared')):
                    matched = False
                trivial = ((live_status == sim_status == 404) or
                           (live_shape['kind'] == sim_shape['kind'] == 'array' and
                            live_shape['length'] == sim_shape['length'] == 0))
                row = {'id': cid, 'result': 'match' if matched else 'mismatch',
                       'evidence_tier': 'trivial' if trivial else 'nontrivial_shape',
                       'live_status': live_status, 'sim_status': sim_status,
                       'status_match': status_match, 'structural_shape_match': shape_match,
                       'field_recall': field_recall, 'list_count_match': count_match,
                       'live_shape': live_shape, 'sim_shape': sim_shape,
                       **field_coverage, **semantic}
                if sim_error:
                    row['sim_error'] = sim_error
                if (method == 'GET' and tool.endswith('.list') and count_match is False
                    and not seed_report['configured']):
                    row['interpretation'] = 'INITIAL_STATE_UNALIGNED_OR_LIST_BEHAVIOR_UNKNOWN'
                capture = case.get('capture_id_as')
                if capture and live_status < 300 and sim_status < 300:
                    try:
                        vars_live[capture] = _captured_id(live_body)
                        vars_sim[capture] = _captured_id(sim_body)
                        row['captured_id'] = True
                    except ValueError:
                        row['captured_id'] = False
                records.append(row)
            except (ValueError, KeyError, TypeError, URLError, OSError, RuntimeError) as exc:
                # No response bodies nor credential-bearing URLs are logged.
                records.append({'id': cid, 'result': 'error', 'reason': type(exc).__name__})
    finally:
        sim.close()
    scored = [r for r in records if r['result'] in {'match', 'mismatch'}]
    matched = sum(r['result'] == 'match' for r in scored)
    informative = [r for r in scored if r.get('evidence_tier') != 'trivial']
    semantic_scored = [r for r in records if r.get('semantic_status') == 'compared']
    semantic_count = sum(r['semantic_compared'] for r in semantic_scored)
    semantic_matched = sum(r['semantic_matched'] for r in semantic_scored)
    return {'format': 'dolibarr-gate-1-v1', 'phase': phase,
            'evidence': 'LIVE_DOLIBARR' if scored else 'NO_LIVE_COMPARISONS',
            'manifest_sha256': source_sha,
            'sim_role': sim_role,
            'tests_configured': len(selected), 'tests_scored': len(scored),
            'tests_matched': matched, 'match_rate': matched / len(scored) if scored else None,
            'semantic_fields_scored': semantic_count,
            'semantic_fields_matched': semantic_matched,
            'semantic_field_match_rate': (semantic_matched / semantic_count if semantic_count else None),
            'nontrivial_tests_scored': len(informative),
            'nontrivial_tests_matched': sum(r['result'] == 'match' for r in informative),
            'sim_seed': seed_report,
            'state_alignment': ('synthetic_fixture_applied_unverified_against_live' if seed_report['configured']
                                else 'independent_empty_simulator_live_state_unverified'),
            'caveat': ('404 parity and two empty lists are trivial evidence. '
                       'Matches measure status, output shape/field types and list length. '
                       'Equal simulated states and business-rule fidelity have NOT been established. '
                       'Run on a fresh and controlled sandbox; keep holdouts unseen during reconstruction.'),
            'probes': records}


def make_model_bundle(manifest: dict, coverage: dict, discovery: dict | None = None, *,
                      swagger: dict | None = None) -> dict:
    """Assemble only *permitted* reconstruction evidence for a model.

    Hidden holdout outcomes and real data values are never included. The model
    is asked to return a valid manifest or a proposed safe probe plan, not code.
    """
    if discovery is not None and discovery.get('phase') != 'discovery':
        raise ValueError('Cannot include holdout results in a model-facing reconstruction bundle')
    if manifest.get('reviewed'):
        raise ValueError('Use an unreviewed imported manifest, not a manually certified one')
    return {
        'format': 'orgsim-blind-reconstruction-input-v1',
        'goal': 'Predict Dolibarr tool behavior and constraints from allowed evidence.',
        'rules': [
            'No Dolibarr source code or external documentation during this blind phase.',
            'Do not invent constraints as facts: label every unverified hypothesis.',
            'Provide an OrgSim v0.1 manifest using ONLY the supported declarative policy DSL.',
            'Preserve reviewed:false until independently verified.',
            'Never make API calls to production or to a shared demo.',
            'Do not access or infer the separate held-out test suite.',
        ],
        'candidate_baseline_manifest': manifest,
        'api_coverage': {k: coverage.get(k) for k in
                         ('api_operations', 'mapped_operations', 'unimplemented_operations',
                          'resource_count', 'not_reconstructed')},
        'discovery_results': (discovery or {}).get('probes', []),
        'candidate_output': 'Complete YAML/JSON orgsim manifest and explicit uncertain/unsupported list',
        'evidence_limit': 'A contract alone does not identify workflow policy, defaults, or side effects',
        'operation_catalog': model_operation_catalog(swagger) if swagger is not None else None,
        'operation_catalog_source': ('Swagger 2.0 fields/types/descriptions only; no observed behavior'
                                     if swagger is not None else 'NOT_PROVIDED'),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description='Dolibarr Gate 1: benchmark real ERP reconstruction')
    sub = p.add_subparsers(dest='command', required=True)
    capt = sub.add_parser('capture', help='Fetch live Swagger metadata (GET only)')
    capt.add_argument('--base-url', required=True)
    capt.add_argument('--out', required=True)
    comp = sub.add_parser('reconstruct', help='Compile a draft from saved Swagger 2.0')
    comp.add_argument('--swagger', required=True)
    comp.add_argument('--out-dir', required=True)
    gaps = sub.add_parser('shape-gaps', help='Summarize discovery-only response-field gaps without values')
    gaps.add_argument('--discovery', required=True)
    gaps.add_argument('--out', required=True)
    bundle = sub.add_parser('bundle', help='Prepare model input without leaking holdout cases')
    bundle.add_argument('--manifest', required=True)
    bundle.add_argument('--coverage', required=True)
    bundle.add_argument('--discovery', help='Optional discovery-phase results only')
    bundle.add_argument('--swagger', help='Optional saved Swagger 2.0: add a value-free operation catalog')
    bundle.add_argument('--out', required=True)
    run = sub.add_parser('probe', help='Run discovery/holdout probes against live and simulated APIs')
    run.add_argument('--base-url', required=True)
    run.add_argument('--manifest', required=True)
    run.add_argument('--suite', required=True)
    run.add_argument('--phase', choices=('discovery', 'holdout'), required=True)
    run.add_argument('--out', required=True)
    run.add_argument('--allow-writes', action='store_true')
    run.add_argument('--confirm', default='')
    args = p.parse_args(argv)
    try:
        if args.command == 'reconstruct':
            output = reconstruct(load_spec(args.swagger), Path(args.out_dir))
        elif args.command == 'shape-gaps':
            output = summarize_discovery_gaps(load_spec(args.discovery))
            _json(Path(args.out), output)
        elif args.command == 'bundle':
            output_bundle = make_model_bundle(load_spec(args.manifest), load_spec(args.coverage),
                                               load_spec(args.discovery) if args.discovery else None,
                                               swagger=load_spec(args.swagger) if args.swagger else None)
            _json(Path(args.out), output_bundle)
            output = {'bundle': args.out,
                      'discovery_probes': len(output_bundle['discovery_results']),
                      'holdout_probes': 0}
        else:
            client = DolibarrClient(args.base_url, os.getenv('DOLIBARR_API_KEY'))
            if args.command == 'capture':
                output = capture_swagger(client, Path(args.out))
            else:
                spec = load_spec(args.manifest)
                suite = load_spec(args.suite)
                output = run_probes(spec, suite, client, phase=args.phase,
                                    allow_writes=args.allow_writes, confirm=args.confirm)
                _json(Path(args.out), output)
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return 0
    except (ValueError, SpecError, SimError, OSError, URLError, RuntimeError) as exc:
        # Strip any potentially credential-bearing text from runtime transport errors.
        print(f'Gate failed: {type(exc).__name__}: {str(exc)[:400] if not isinstance(exc, URLError) else "network unavailable"}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
