"""Franzen/EDL-inspired black-box environment learning primitives for OrgSim.

Reusable across unfamiliar APIs. Discovery observations are stored as salted
hashes/shape metrics only; not raw customer records, API tokens, or requests.
A blind holdout must NEVER enter this memory. There is no network connectivity.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import heapq
import json
from pathlib import Path
import re
import sqlite3
from typing import Any


def _stable(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False, default=str)


def _hash(obj: Any) -> str:
    return sha256(_stable(obj).encode('utf-8')).hexdigest()


def _type(x: Any) -> str:
    if x is None: return 'null'
    if isinstance(x, bool): return 'boolean'
    if isinstance(x, (int, float)): return 'number'
    if isinstance(x, str): return 'string'
    if isinstance(x, list): return 'array'
    if isinstance(x, dict): return 'object'
    return 'unknown'


_ID = re.compile(r'(?:^id$|_id$|^fk_|^socid$|^rowid$|^.+_ids$)', re.I)
_TIME = re.compile(r'(?:^tms$|timestamp|^date_|_date$|created_at|updated_at)', re.I)
# Only explicitly state-bearing fields retain their values in the generalized
# signature. Arbitrary names, emails, IDs, descriptions and notes are elided.
_STATE_FIELDS = frozenset({'status', 'statut', 'state', 'stage', 'client', 'type',
                          'role', 'priority', 'qty', 'quantity', 'available', 'stock',
                          'balance', 'total', 'is_paid', 'validated', 'draft'})


def normalize_state(value: Any, *, generalize: bool = True) -> Any:
    """ID/timestamp-invariant state representation; does not claim equivalence.

    Exact and generalized signatures are both hashes. Use generalized matches
    only to *propose* transfers, not as proof of identical hidden state.
    """
    if not generalize:
        return value
    if isinstance(value, dict):
        out = {}
        for key, item in sorted(value.items()):
            if _TIME.search(key):
                continue
            if _ID.search(key):
                out[key] = f'<{_type(item)}:id>' if item is not None else None
            elif key.lower() in _STATE_FIELDS:
                out[key] = normalize_state(item, generalize=True)
            elif isinstance(item, (list, dict)):
                out[key] = normalize_state(item, generalize=True)
            else:
                out[key] = f'<{_type(item)}>'
        return out
    if isinstance(value, list):
        return sorted((normalize_state(x, generalize=True) for x in value), key=_stable)
    return value


def state_signatures(value: Any) -> dict[str, str]:
    return {'exact_sha256': _hash(value), 'generalized_sha256': _hash(normalize_state(value))}


def safe_shape(value: Any, *, max_fields: int = 300) -> dict:
    result = {'kind': _type(value)}
    if isinstance(value, dict):
        result['fields'] = {k: _type(v) for k, v in sorted(value.items())[:max_fields]}
        result['field_count'] = len(value)
    elif isinstance(value, list):
        result['length'] = len(value)
        if value and isinstance(value[0], dict):
            result['first_fields'] = {k: _type(v) for k, v in sorted(value[0].items())[:max_fields]}
    return result


def structural_delta(before: Any, after: Any, *, max_paths: int = 250) -> dict:
    """ARC animation-diff analogue: field paths that changed, not their values.

    List positions are compared index-wise; rearrangements are consequently
    marked as changes even when rows are semantically equivalent. The result
    is evidence to inspect, not proof of a business side effect.
    """
    changed, added, removed = [], [], []
    def scan(old: Any, new: Any, path: str, depth: int) -> None:
        if len(changed) + len(added) + len(removed) >= max_paths:
            return
        if depth > 10:
            if old != new:
                changed.append(path or '<root>')
            return
        if isinstance(old, dict) and isinstance(new, dict):
            for key in sorted(old.keys() | new.keys()):
                label = f'{path}.{key}' if path else str(key)
                if key not in old: added.append(label)
                elif key not in new: removed.append(label)
                else: scan(old[key],new[key],label,depth+1)
        elif isinstance(old, list) and isinstance(new, list):
            if len(old)!=len(new):
                changed.append(f'{path}.length')
            for idx,(a,b) in enumerate(zip(old[:100],new[:100])):
                scan(a,b,f'{path}[{idx}]',depth+1)
        elif old!=new:
            changed.append(path or '<root>')
    scan(before,after,'',0)
    return {'changed_paths':changed[:max_paths], 'added_paths':added[:max_paths],
            'removed_paths':removed[:max_paths],
            'counts':{'changed':len(changed),'added':len(added),'removed':len(removed)},
            'values_recorded':False}


class EvidenceLedger:
    """SQLite discovery memory. Field VALUES and action arguments are never stored."""
    def __init__(self, db_path: str | Path = ':memory:'):
        self.db = sqlite3.connect(str(db_path))
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS observations (
          id INTEGER PRIMARY KEY, phase TEXT NOT NULL, operation TEXT NOT NULL,
          role TEXT NOT NULL, args_sha256 TEXT NOT NULL,
          state_exact TEXT NOT NULL, state_generalized TEXT NOT NULL,
          outcome TEXT NOT NULL, status INTEGER NOT NULL, output_shape TEXT NOT NULL,
          next_exact TEXT NOT NULL, changed INTEGER NOT NULL, source TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS observations_equiv ON observations(operation,role,state_generalized);
        CREATE TABLE IF NOT EXISTS hypotheses (
          key TEXT PRIMARY KEY, operation TEXT NOT NULL, state_generalized TEXT NOT NULL,
          expected_outcome TEXT NOT NULL, evidence TEXT NOT NULL, status TEXT NOT NULL
        );
        ''')

    def close(self) -> None:
        self.db.close()

    def record(self, *, operation: str, args: Any, before: Any, after: Any,
               status: int, response: Any, role: str = 'test_agent',
               phase: str = 'discovery', source: str = 'sandbox_observation') -> dict:
        if phase != 'discovery':
            raise ValueError('Holdouts and evaluation evidence cannot enter reconstruction memory')
        if source not in {'sandbox_observation', 'sanitized_discovery', 'synthetic_fixture'}:
            raise ValueError('Evidence source must be a permitted discovery source')
        if not isinstance(operation, str) or not operation:
            raise ValueError('operation required')
        if type(status) is not int or not 100 <= status <= 599:
            raise ValueError('Valid HTTP-like status required')
        before_sig, after_sig = state_signatures(before), state_signatures(after)
        outcome = 'success' if 200 <= status < 300 else 'rejected'
        changed = before_sig['exact_sha256'] != after_sig['exact_sha256']
        if outcome == 'success' and not changed:
            outcome = 'no_change'  # read-only success or true no-op; not a failure
        with self.db:
            self.db.execute('''INSERT INTO observations(
                phase,operation,role,args_sha256,state_exact,state_generalized,
                outcome,status,output_shape,next_exact,changed,source)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)''',
                (phase, operation, role, _hash(args), before_sig['exact_sha256'],
                 before_sig['generalized_sha256'], outcome, status,
                 _stable({'response': safe_shape(response), 'state_delta':structural_delta(before,after)}),
                 after_sig['exact_sha256'], int(changed), source))
            self._reconcile(operation, before_sig['generalized_sha256'], outcome)
        return {'state_signature': before_sig, 'next_signature': after_sig,
                'outcome': outcome, 'status': status, 'state_changed': changed,
                'response_shape': safe_shape(response), 'transition_delta': structural_delta(before, after)}

    def _reconcile(self, operation: str, signature: str, outcome: str) -> None:
        self.db.execute('''UPDATE hypotheses SET status = CASE
            WHEN expected_outcome=? THEN CASE WHEN status='falsified' THEN status ELSE 'supported_provisional' END
            ELSE 'falsified' END
            WHERE operation=? AND (state_generalized=? OR state_generalized='*')''',
            (outcome, operation, signature))

    def add_hypothesis(self, *, key: str, operation: str, expected_outcome: str,
                       signature: str = '*', evidence: str = 'hypothesis') -> None:
        if expected_outcome not in {'success', 'rejected', 'no_change'}:
            raise ValueError('Unknown expected outcome')
        if evidence not in {'hypothesis', 'contract', 'inferred_dependency', 'sanitized_discovery'}:
            raise ValueError('Evidence label must be unverified')
        with self.db:
            self.db.execute('INSERT INTO hypotheses(key,operation,state_generalized,expected_outcome,evidence,status) VALUES(?,?,?,?,?,?)',
                            (key, operation, signature, expected_outcome, evidence, 'unverified'))

    def hypotheses(self) -> list[dict]:
        return [dict(r) for r in self.db.execute('SELECT * FROM hypotheses ORDER BY key')]

    def stats(self, operation: str, state: Any, *, role: str = 'test_agent') -> dict:
        sig = state_signatures(state)
        rows = self.db.execute('''SELECT outcome,changed,status FROM observations
            WHERE operation=? AND role=? AND state_generalized=?''',
            (operation, role, sig['generalized_sha256'])).fetchall()
        counts = Counter(r['outcome'] for r in rows)
        statuses = sorted(set(int(r['status']) for r in rows))
        # Multiple outcomes in equivalent *generalized* states can indicate
        # hidden inputs or missing state dimensions; never average them away.
        return {'observations': len(rows), 'outcomes': dict(counts),
                'status_codes': statuses, 'conflict': len(counts) > 1,
                'state': sig}

    def decision_gate(self, operation: str, state: Any, *, role: str = 'test_agent',
                      planned_mutations: int = 1) -> dict:
        s = self.stats(operation, state, role=role)
        n = s['observations']
        # Read-only probes remain available; mutating batches halt at decision
        # boundaries if evidence is absent, contradictory, or indicates rejection.
        if n == 0:
            code = 'VERIFY_UNSEEN_STATE'
        elif s['conflict']:
            code = 'VERIFY_CONTRADICTORY_EVIDENCE'
        elif s['outcomes'].get('rejected', 0):
            code = 'VERIFY_KNOWN_REJECTION'
        elif planned_mutations > 1 and n < 2:
            code = 'VERIFY_BEFORE_LONG_COMMIT'
        else:
            code = 'NO_KNOWN_CONFLICT'
        return {'decision': 'verify' if code != 'NO_KNOWN_CONFLICT' else 'proceed_provisional',
                'reason': code, 'evidence': s,
                'note': 'Discovery evidence is not proof of real-ERP business rules'}

    def redundant_probe(self, operation: str, state: Any, args: Any,
                        *, role: str = 'test_agent') -> bool:
        sig = state_signatures(state)
        found = self.db.execute('''SELECT count(*) FROM observations WHERE operation=? AND role=?
                        AND state_exact=? AND args_sha256=?''',
                        (operation, role, sig['exact_sha256'], _hash(args))).fetchone()
        return bool(found and found[0])


_UNSAFE = re.compile(r'login|password|token|reset|generate|\/setup\/|\/download|\/upload|\/builddoc|\/merge|\/enable|\/disable|\/setgroup|\/setpassword', re.I)
_FOCUS = ('/thirdparties', '/products', '/orders', '/invoices')
_VERBS = frozenset({'get', 'post', 'put', 'patch', 'delete'})


def classify_operation(method: str, route: str, summary: str = '') -> dict:
    """No Dolibarr-specific operation implementations; general route taxonomy."""
    method = method.upper()
    parts = route.strip('/').split('/')
    risk = 'blocked_sensitive' if _UNSAFE.search(route) or method in {'DELETE', 'PATCH'} else (
        'mutating' if method != 'GET' else 'read')
    if method == 'GET':
        kind = 'lookup' if len(parts) > 1 and (parts[-1].startswith('{') or parts[-2] in {'ref', 'email', 'barcode'}) else 'query'
    elif method == 'POST' and len(parts) == 1:
        kind = 'create'
    elif method in {'PUT', 'PATCH'} and len(parts) == 2:
        kind = 'update'
    elif method == 'DELETE' and len(parts) == 2:
        kind = 'delete'
    elif method in {'POST', 'PUT', 'DELETE'}:
        kind = 'workflow_or_relationship'
    else:
        kind = 'special'
    # A no-op-looking endpoint could still mutate state, regardless of HTTP verb.
    if re.search(r'\/(validate|close|reopen|settodraft|settopaid|settounpaid|setinvoiced)(?:/|$)', route, re.I):
        kind = 'lifecycle_transition'
    return {'method': method, 'path': route, 'summary': summary, 'kind': kind,
            'risk': risk, 'implementation': 'UNVERIFIED_NO_EXECUTABLE_MODEL'}


def catalog_from_swagger(swagger: dict) -> list[dict]:
    catalog = []
    for path, routes in sorted(swagger.get('paths', {}).items()):
        if not isinstance(routes, dict):
            continue
        for method, spec in sorted(routes.items()):
            if method.lower() in _VERBS and isinstance(spec, dict):
                item = classify_operation(method, path, spec.get('summary', ''))
                item['operation_id'] = spec.get('operationId', '')
                item['params'] = [p.get('name') for p in spec.get('parameters', [])
                                  if isinstance(p, dict) and p.get('in') in {'path', 'query'}]
                catalog.append(item)
    return catalog


def prioritize_probes(catalog: list[dict], observed_operations: set[str] | None = None,
                      *, budget: int = 15) -> dict:
    """Suggest *requests for a HUMAN-APPROVED test suite*; never execute them.

    Only allow read-only discovery in the four enabled business namespaces.
    Additional unsafe-looking GET routes are excluded. Calls with path arguments
    require existing synthetic fixture IDs and are flagged, not filled in.
    """
    if not 1 <= budget <= 100:
        raise ValueError('budget 1..100 required')
    observed = observed_operations or set()
    choices, withheld = [], Counter()
    for op in catalog:
        path = op['path']
        key = f'{op["method"]} {path}'
        if op['method'] != 'GET' or op['risk'] != 'read':
            withheld['unsafe_or_mutating'] += 1
            continue
        if not path.startswith(_FOCUS):
            withheld['outside_focus'] += 1
            continue
        if _UNSAFE.search(path):
            withheld['sensitive'] += 1
            continue
        vars_ = re.findall(r'\{(.*?)\}', path)
        # Broad range of resource/selector endpoints; prioritize previously
        # unobserved operations and diverse endpoint families.
        score = 5 if key not in observed else 0
        score += 4 if not vars_ else 2
        score += 2 if op['kind'] == 'query' else 1
        score -= len(vars_)
        choices.append((score, key, vars_, op))
    choices.sort(key=lambda x: (-x[0], x[1]))
    out, families = [], Counter()
    for score, key, vars_, op in choices:
        family = op['path'].strip('/').split('/')[0]
        if len(out) >= budget:
            break
        if families[family] >= max(2, (budget + 2) // 3):
            continue
        families[family] += 1
        out.append({'method': 'GET', 'path_template': op['path'],
                    'requires_synthetic_fixtures': vars_, 'operation_id': op.get('operation_id'),
                    'information_priority': score,
                    'request_status': 'PROPOSED_NOT_EXECUTED',
                    'note': 'Read-only permission and risk must still be confirmed by a human'})
    return {'format': 'orgsim-probe-priorities-v1', 'probes': out,
            'excluded_counts': dict(withheld), 'live_calls_performed': 0}


def shortest_verified_plan(start: str, goal: str, transitions: list[dict],
                           *, allow_hypotheses: bool = False, max_states: int = 1000) -> dict:
    """ARC Math analogue: Dijkstra over *explicit* transitions, never assumed rules."""
    adjacency: dict[str, list[tuple[float, str, str]]] = defaultdict(list)
    for t in transitions:
        if t.get('evidence') != 'verified' and not allow_hypotheses:
            continue
        cost = t.get('cost', 1)
        if type(cost) not in (int, float) or cost <= 0:
            raise ValueError('Transition costs must be positive')
        if not all(isinstance(t.get(k), str) for k in ('from', 'to', 'action')):
            raise ValueError('Transition from/to/action must be strings')
        adjacency[t['from']].append((float(cost), t['to'], t['action']))
    q = [(0., start, ())]
    best = {start: 0.}
    while q and len(best) <= max_states:
        cost, node, path = heapq.heappop(q)
        if cost > best.get(node, float('inf')):
            continue
        if node == goal:
            return {'status': 'path_found', 'cost': cost, 'actions': list(path),
                    'hypothetical': allow_hypotheses}
        for step_cost, next_state, action in adjacency[node]:
            new = cost + step_cost
            if new < best.get(next_state, float('inf')):
                best[next_state] = new
                heapq.heappush(q, (new, next_state, path + (action,)))
    return {'status': 'unknown_transition_model', 'actions': [],
            'note': 'No route can be established without verified transitions'}
