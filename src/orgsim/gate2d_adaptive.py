"""Gate 2D -- adaptive, read-only black-box enterprise investigation.

The investigator is intentionally NOT a policy-reconstruction certification system.
It gathers development observations through a finite allowlist of GET probes,
commits shape-only predictions before oracle requests, and produces unreviewed
response profiles. Record values/fixture IDs/API keys are never serialized.

The default policy is a deterministic evidence-based scout; optional explicit
opt-in invokes a model as an *advisor* using only sanitized metadata. The model
can select from allowed probe IDs, but cannot supply URLs or HTTP methods.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from typing import Any, Protocol
from urllib.parse import urlparse

from .dolibarr_gate import DolibarrClient, _local_host
from .gate2b_readonly import _ID, _REF

FORMAT = 'orgsim-gate2d-adaptive-discovery-v1'
FAMILIES = ('thirdparties', 'products', 'orders', 'invoices')
FIELD = re.compile(r'^[A-Za-z_][A-Za-z0-9_]{0,63}$')
KINDS = {'array', 'object', 'number', 'string', 'boolean', 'null', 'unknown'}
FIXTURE_TYPES = {
    'thirdparty_id': 'thirdparties', 'product_id': 'products',
    'order_id': 'orders', 'invoice_id': 'invoices',
    'product_ref': 'products',
}
BASE_LIMIT = 3
MAX_PROBES = 32


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def hash_json(value: Any) -> str:
    return sha256(canonical(value).encode('utf-8')).hexdigest()


def type_name(value: Any) -> str:
    if value is None: return 'null'
    if isinstance(value, bool): return 'boolean'
    if isinstance(value, (int, float)): return 'number'
    if isinstance(value, str): return 'string'
    if isinstance(value, dict): return 'object'
    if isinstance(value, list): return 'array'
    return 'unknown'


def safe_shape(value: Any) -> dict:
    """Return schema, never scalar values; cap key and output size."""
    kind = type_name(value)
    out: dict[str, Any] = {'kind': kind}
    if kind == 'array':
        out['length'] = len(value)
        if value:
            out['first_item_kind'] = type_name(value[0])
            if isinstance(value[0], dict):
                filtered = {k: v for k, v in value[0].items() if isinstance(k, str) and FIELD.fullmatch(k)}
                keys = sorted(filtered)[:220]
                out.update({'first_keys': keys,
                            'first_types': {k: type_name(filtered[k]) for k in keys},
                            'keys_truncated': len(filtered) > len(keys)})
    elif kind == 'object':
        filtered = {k: v for k, v in value.items() if isinstance(k, str) and FIELD.fullmatch(k)}
        keys = sorted(filtered)[:220]
        out.update({'keys': keys, 'types': {k: type_name(filtered[k]) for k in keys},
                    'keys_truncated': len(filtered) > len(keys)})
    return out


def validate_fixtures(fixtures: dict | None) -> dict[str, str]:
    if fixtures is None: return {}
    if not isinstance(fixtures, dict) or set(fixtures) - set(FIXTURE_TYPES):
        raise ValueError('Only recognized synthetic fixture identifiers are allowed')
    clean = {}
    for name, raw in fixtures.items():
        if raw is None: continue
        if isinstance(raw, bool) or not isinstance(raw, (int, str)):
            raise ValueError('Fixture identifiers must be decimal IDs or safe product refs')
        value = str(raw)
        if name == 'product_ref':
            if not _REF.fullmatch(value): raise ValueError('Unsafe product reference')
        elif not _ID.fullmatch(value):
            raise ValueError('Fixture IDs must be 1-12 decimal digits')
        clean[name] = value
    return clean


def get_live_identifiers(family: str, status: int, payload: Any) -> dict[str, str]:
    """Ephemeral: leverage a real list result to investigate an existing item.

    Only use documented ID/ref fields and conservative validators. Returned
    IDs stay in process memory and NEVER enter events/reports/model prompts.
    """
    if status != 200 or not isinstance(payload, list) or not payload:
        return {}
    item = payload[0]
    if not isinstance(item, dict): return {}
    output = {}
    family_to_key = {v: k for k, v in FIXTURE_TYPES.items() if k != 'product_ref'}
    key = family_to_key.get(family)
    ident = item.get('id')
    if key and type(ident) in (int, str) and _ID.fullmatch(str(ident)):
        output[key] = str(ident)
    if family == 'products':
        ref = item.get('ref')
        if isinstance(ref, str) and _REF.fullmatch(ref):
            output['product_ref'] = ref
    return output


@dataclass(frozen=True)
class Probe:
    id: str
    family: str
    category: str
    path: str
    template: str
    weight: int
    anchor: str | None = None

    def public(self) -> dict:
        return {'id': self.id, 'family': self.family, 'category': self.category,
                'method': 'GET', 'path_template': self.template,
                'expected_information_rank': self.weight,
                'anchor': self.anchor}


def probe(id: str, family: str, category: str, path: str, weight: int,
          *, anchor: str | None = None, template: str | None = None) -> Probe:
    return Probe(id, family, category, path, template or path, weight, anchor)


def build_candidates(fixtures: dict, seen: dict[str, dict]) -> tuple[list[Probe], dict[str, str]]:
    """Rebuild finite allowlist from current observations; no model-supplied paths."""
    out: list[Probe] = []
    skipped: dict[str, str] = {}
    for family in FAMILIES:
        if family + '_list' not in seen:
            out.append(probe(family + '_list', family, 'anchor',
                             f'/{family}?limit={BASE_LIMIT}', 1000))
    if any(f + '_list' not in seen for f in FAMILIES):
        return out, skipped  # Establish one baseline for each family first.

    counts = {f: (seen[f + '_list']['shape'].get('length', 0)
                  if seen[f + '_list']['status'] == 200 else 0) for f in FAMILIES}
    for family in FAMILIES:
        n = counts[family]
        if n == 0:
            skipped[family] = 'no_observed_records_or_non_200_anchor; skip derivative reads'
            continue
        anchor = family + '_list'
        out.append(probe(family + '_limit1', family, 'limit', f'/{family}?limit=1', 45, anchor=anchor))
        if n >= 2:
            out.append(probe(family + '_page1', family, 'page',
                             f'/{family}?limit=1&page=1', 40, anchor=anchor))
        else:
            skipped[family + '_page1'] = 'one_observed_record; low-value pagination'
        if family == 'thirdparties':
            out.append(probe('thirdparties_properties', family, 'projection',
                             '/thirdparties?limit=1&properties=id,name,client', 92, anchor=anchor))
        if family == 'products':
            out.append(probe('products_properties', family, 'projection',
                             '/products?limit=1&properties=id,ref,label', 92, anchor=anchor))
            out.append(probe('products_ids_only', family, 'ids_only',
                             '/products?limit=1&ids_only=1', 81, anchor=anchor))

    for fixture_key, family in FIXTURE_TYPES.items():
        if family in skipped or fixture_key not in fixtures: continue
        value = fixtures[fixture_key]
        anchor = family + '_list'
        if fixture_key == 'product_ref':
            out.append(probe('product_by_ref', family, 'alternate_key',
                             '/products/ref/' + value, 138,
                             template='/products/ref/{ref}', anchor=anchor))
            continue
        out.append(probe(family + '_detail', family, 'individual',
                         f'/{family}/{value}', 150,
                         template=f'/{family}/{{id}}', anchor=anchor))
        if family == 'thirdparties':
            for name, subpath in [('outstanding_orders', 'outstandingorders'),
                                  ('outstanding_invoices', 'outstandinginvoices')]:
                out.append(probe('thirdparties_' + name, family, 'related',
                                 f'/thirdparties/{value}/{subpath}', 122,
                                 template=f'/thirdparties/{{id}}/{subpath}', anchor=anchor))
        if family == 'products':
            out.extend([
                probe('products_stock', family, 'related', f'/products/{value}/stock', 120,
                      template='/products/{id}/stock', anchor=anchor),
                probe('products_categories', family, 'related', f'/products/{value}/categories', 100,
                      template='/products/{id}/categories', anchor=anchor),
            ])
        if family == 'orders':
            out.extend([
                probe('orders_lines', family, 'related', f'/orders/{value}/lines', 145,
                      template='/orders/{id}/lines', anchor=anchor),
                probe('orders_shipment', family, 'related', f'/orders/{value}/shipment', 112,
                      template='/orders/{id}/shipment', anchor=anchor),
            ])
        if family == 'invoices':
            out.extend([
                probe('invoices_lines', family, 'related', f'/invoices/{value}/lines', 145,
                      template='/invoices/{id}/lines', anchor=anchor),
                probe('invoices_payments', family, 'related', f'/invoices/{value}/payments', 118,
                      template='/invoices/{id}/payments', anchor=anchor),
            ])
    done = set(seen)
    return [p for p in out if p.id not in done], skipped


def score_probe(p: Probe, seen: dict) -> float:
    family_usage = sum(o['family'] == p.family for o in seen.values())
    novelty = 24.0 / (family_usage + 1)
    return p.weight + novelty


def choose_probe(candidates: list[Probe], seen: dict) -> Probe:
    if not candidates: raise ValueError('No available probes')
    # Before all four anchors exist, prioritize only missing anchors.
    return sorted(candidates, key=lambda p: (-score_probe(p, seen), p.id))[0]


def _predict(p: Probe, seen: dict) -> dict:
    """Conservative prediction; every prediction is hypothesis, never certification."""
    anchor = seen.get(p.anchor or '')
    base = {'status': None, 'kind': None, 'length': None, 'first_keys': None}
    if p.category == 'anchor':
        base['kind'] = 'array'  # inferred from the collection contract, unverified
    if anchor is not None:
        sh = anchor['shape']
        if anchor['status'] == 200:
            base['status'] = 200
        if p.category in {'limit', 'page', 'projection', 'ids_only'}:
            base['kind'] = sh.get('kind')
            n = sh.get('length')
            if isinstance(n, int):
                if p.category == 'page': base['length'] = min(max(0, n-1), 1)
                else: base['length'] = min(n, 1)
            if p.category == 'projection':
                wanted = (('id', 'name', 'client') if p.family == 'thirdparties'
                          else ('id', 'ref', 'label'))
                observed_keys = set(sh.get('first_keys', []))
                base['first_keys'] = sorted(k for k in wanted if k in observed_keys)
        elif p.category in {'individual', 'alternate_key'}:
            base['kind'] = 'object'  # unverified, extrapolated from list element
    return {k: v for k, v in base.items() if v is not None}


def validate_prediction(value: dict) -> dict:
    """A model can predict outcomes but may not supply arbitrary data or effects."""
    if not isinstance(value, dict) or set(value) - {'status', 'kind', 'length', 'first_keys'}:
        raise ValueError('Invalid prediction keys')
    clean = {}
    if 'status' in value:
        if type(value['status']) is not int or not 100 <= value['status'] <= 599:
            raise ValueError('Invalid prediction status')
        clean['status'] = value['status']
    if 'kind' in value:
        if value['kind'] not in KINDS:
            raise ValueError('Invalid prediction kind')
        clean['kind'] = value['kind']
    if 'length' in value:
        if type(value['length']) is not int or not 0 <= value['length'] <= 1000:
            raise ValueError('Invalid prediction length')
        clean['length'] = value['length']
    if 'first_keys' in value:
        keys = value['first_keys']
        if not isinstance(keys, list) or len(keys) > 220 or any(
            not isinstance(k, str) or not FIELD.fullmatch(k) for k in keys):
            raise ValueError('Invalid predicted field names')
        clean['first_keys'] = sorted(set(keys))
    return clean


def score_prediction(pred: dict, status: int, shape: dict) -> dict:
    expected = {'status': status, 'kind': shape.get('kind'),
                'length': shape.get('length'),
                'first_keys': shape.get('first_keys')}
    checks = {}
    for k, value in pred.items():
        if k == 'first_keys':
            checks[k] = expected.get(k) is not None and value == expected[k]
        else:
            checks[k] = value == expected.get(k)
    return {'checked_fields': len(checks), 'matched_fields': sum(checks.values()),
            'exact_match_on_predicted_fields': bool(checks) and all(checks.values()),
            'field_matches': checks}


def make_model_prompt(candidates: list[Probe], seen: dict) -> dict:
    """Metadata only. Never transmit URLs, credentials, payload values or fixture IDs."""
    compact = [{'probe_id': pid, 'family': ob['family'], 'category': ob['category'],
                'status': ob['status'], 'kind': ob['shape']['kind'],
                'length': ob['shape'].get('length'),
                'first_keys': ob['shape'].get('first_keys', [])[:12]}
               for pid, ob in seen.items()]
    return {'environment': 'READ_ONLY_SANITIZED_ENTERPRISE_DISCOVERY',
            'purpose': 'Choose the next most informative safe GET probe and predict its HTTP response shape',
            'rules': ['Choose an existing probe_id only', 'No invented API paths',
                      'Use only observed shapes; all predictions remain unverified'],
            'observations': compact[-18:],
            'allowed_probes': [p.public() for p in candidates[:24]],
            'output': {'probe_id': 'one allowed id', 'prediction':
                       {'status': 'optional HTTP integer', 'kind': 'optional response kind',
                        'length': 'optional list length', 'first_keys': 'optional field name list'}}}


class ProbeAdvisor(Protocol):
    def choose(self, prompt: dict) -> dict: ...


class OpenAIProbeAdvisor:
    """Explicit opt-in only. Sends shape metadata to OpenAI; never ERP values."""
    def __init__(self, model: str):
        from urllib.request import Request, build_opener
        from .dolibarr_gate import NoRedirect
        if not isinstance(model, str) or not 1 <= len(model) <= 120 or any(c.isspace() for c in model):
            raise ValueError('Invalid model name')
        self._key = os.environ.get('OPENAI_API_KEY')
        if not self._key: raise ValueError('OPENAI_API_KEY is required for explicit model mode')
        self.model = model
        self._request = Request
        self._opener = build_opener(NoRedirect())

    def choose(self, prompt: dict) -> dict:
        instructions = (
            'Return ONLY one strict JSON object with keys probe_id and prediction. '
            'Choose probe_id exactly from allowed_probes. Prediction may contain only '
            'status integer, kind, length integer, or first_keys array. '
            'Do not include any URL, path, raw record values, or rationale. '
            'If uncertain, use an empty prediction object. Your answer is a hypothesis, not verification.'
        )
        body = {'model': self.model, 'store': False, 'max_output_tokens': 350,
                'input': [{'role': 'system', 'content': instructions},
                          {'role': 'user', 'content': canonical(prompt)}]}
        req = self._request('https://api.openai.com/v1/responses',
                            method='POST', data=canonical(body).encode('utf-8'),
                            headers={'Content-Type': 'application/json',
                                     'Authorization': 'Bearer ' + self._key})
        with self._opener.open(req, timeout=70) as resp:
            data = resp.read(1_000_001)
        if len(data) > 1_000_000: raise ValueError('Model output too large')
        result = json.loads(data)
        content = result.get('output_text') or ''.join(
            p.get('text', '') for item in result.get('output', [])
            for p in item.get('content', []) if p.get('type') == 'output_text')
        if not isinstance(content, str) or not content:
            raise ValueError('Model did not return a JSON message')
        if content.lstrip().startswith('```'):
            content = content.strip().split('\n', 1)[-1].rsplit('```', 1)[0].strip()
        answer = json.loads(content)
        if not isinstance(answer, dict): raise ValueError('Model must return an object')
        return answer


def validate_advice(advice: dict, candidates: list[Probe]) -> tuple[Probe, dict]:
    if not isinstance(advice, dict) or set(advice) != {'probe_id', 'prediction'}:
        raise ValueError('Model advice must contain exactly probe_id and prediction')
    choices = {p.id: p for p in candidates}
    probe_id = advice['probe_id']
    if not isinstance(probe_id, str) or probe_id not in choices:
        raise ValueError('Model selected an unapproved probe')
    return choices[probe_id], validate_prediction(advice['prediction'])


class EvidenceLog:
    """Append-only, hash-chained, fsynced precommit and observation events."""
    def __init__(self, folder: Path, *, enabled: bool):
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)
        self.file = folder / 'events.jsonl'
        if self.file.exists():
            raise FileExistsError('Output directory has an existing evidence log; choose a fresh directory')
        self.fp = self.file.open('x', encoding='utf-8')
        self.last = '0' * 64
        self.enabled = enabled
        self.count = 0

    def append(self, event: dict):
        self.count += 1
        record = {'seq': self.count, 'prev_sha256': self.last, **event}
        digest = hash_json(record)
        record['sha256'] = digest
        self.fp.write(canonical(record) + '\n')
        self.fp.flush()
        os.fsync(self.fp.fileno())
        self.last = digest

    def close(self):
        if not self.fp.closed: self.fp.close()


@dataclass
class Investigator:
    client: Any
    fixtures: dict = field(default_factory=dict)
    budget: int = 20
    auto_ids: bool = True
    advisor: ProbeAdvisor | None = None
    model_calls_limit: int = 8
    seen: dict = field(default_factory=dict)
    live_ids: dict = field(default_factory=dict)

    def __post_init__(self):
        if not getattr(self.client, 'loopback', False):
            raise ValueError('Gate 2D is loopback-only')
        if type(self.budget) is not int or not 4 <= self.budget <= MAX_PROBES:
            raise ValueError(f'Probe budget must be between 4 and {MAX_PROBES}')
        if type(self.model_calls_limit) is not int or not 0 <= self.model_calls_limit <= 16:
            raise ValueError('Model advice budget must be 0..16')
        self.fixtures = validate_fixtures(self.fixtures)
        self.live_ids = dict(self.fixtures)

    def run(self, *, execute: bool, outdir: str | Path) -> dict:
        out = Path(outdir)
        if out.exists() and (out / 'events.jsonl').exists():
            raise FileExistsError('Evidence already exists; use a new --out directory')
        # No network call on dry-run, and no fake prediction score.
        report = {'format': FORMAT, 'phase': 'discovery',
                  'evidence': 'LIVE_DOLIBARR' if execute else 'DRY_RUN',
                  'independent_holdout': False,
                  'read_only': True, 'live_writes': 0, 'raw_values_saved': False,
                  'credentials_saved': False, 'fixture_values_saved': False,
                  'budget_max': self.budget, 'operations_executed': 0,
                  'model_provider_calls': 0, 'model_enabled': self.advisor is not None,
                  'probes': [], 'skipped': {}, 'stop_reason': None}
        if not execute:
            out.mkdir(parents=True, exist_ok=True)
            report['initial_plan'] = [p.public() for p in build_candidates(self.live_ids, {})[0]]
            report['stop_reason'] = 'dry_run_only; subsequent plan is observation-dependent'
            self._save_outputs(out, report, None)
            return report
        # Validate expected transport and log BEFORE the first oracle request.
        ledger = EvidenceLog(out, enabled=True)
        try:
            ledger.append({'event': 'run_started', 'format': FORMAT,
                           'budget': self.budget,
                           'fixture_types': sorted(self.fixtures),
                           'auto_ids': self.auto_ids,
                           'model_mode': self.advisor is not None})
            for i in range(self.budget):
                candidates, skipped = build_candidates(self.live_ids, self.seen)
                report['skipped'].update(skipped)
                if not candidates:
                    report['stop_reason'] = 'no_informative_approved_probes_available'
                    break
                chosen = choose_probe(candidates, self.seen)
                predicted = validate_prediction(_predict(chosen, self.seen))
                predictor = 'symbolic_hypothesis'
                if self.advisor and i >= 4 and report['model_provider_calls'] < self.model_calls_limit:
                    # No model is allowed to see fixture IDs, endpoint values, or a holdout.
                    advice = self.advisor.choose(make_model_prompt(candidates, self.seen))
                    report['model_provider_calls'] += 1
                    chosen, predicted = validate_advice(advice, candidates)
                    predictor = 'model_hypothesis'
                # Commit predictions durably BEFORE issuing any API read.
                ledger.append({'event': 'prediction_committed', 'index': i,
                               'probe_id': chosen.id, 'family': chosen.family,
                               'path_template': chosen.template,
                               'source': predictor, 'prediction': predicted})
                try:
                    status, payload = self.client.request('GET', chosen.path)
                except Exception as exc:
                    # Exception messages can contain URLs or credential material.
                    ledger.append({'event': 'probe_transport_error', 'index': i,
                                   'probe_id': chosen.id, 'error_type': type(exc).__name__})
                    report['stop_reason'] = 'transport_error:' + type(exc).__name__
                    break
                shape = safe_shape(payload)
                evidence = {'family': chosen.family, 'category': chosen.category,
                            'status': status, 'shape': shape}
                self.seen[chosen.id] = evidence
                if self.auto_ids and chosen.category == 'anchor':
                    for key, val in get_live_identifiers(chosen.family, status, payload).items():
                        self.live_ids.setdefault(key, val)
                payload = None  # Don't hold raw values longer than necessary.
                result = score_prediction(predicted, status, shape)
                row = {'id': chosen.id, 'family': chosen.family,
                       'category': chosen.category, 'method': 'GET',
                       'path_template': chosen.template, 'predictor': predictor,
                       'prediction': predicted, 'status': status,
                       'shape': shape, 'prediction_score': result}
                ledger.append({'event': 'observation_recorded', 'index': i,
                               'probe_id': chosen.id, 'status': status,
                               'shape': shape, 'prediction_score': result})
                report['probes'].append(row)
                report['operations_executed'] += 1
            else:
                report['stop_reason'] = 'budget_exhausted'
            if report['stop_reason'] is None:
                report['stop_reason'] = 'unknown'
            ledger.append({'event': 'run_finished', 'reads': report['operations_executed'],
                           'model_calls': report['model_provider_calls'],
                           'stop_reason': report['stop_reason']})
        finally:
            ledger.close()
        self._save_outputs(out, report, ledger.last)
        return report

    @staticmethod
    def _save_outputs(out: Path, report: dict, event_sha: str | None):
        out.mkdir(parents=True, exist_ok=True)
        metrics = {k: {'predicted_fields': 0, 'correct_fields': 0,
                       'exact_predictions': 0, 'predictions': 0}
                   for k in ('symbolic_hypothesis', 'model_hypothesis')}
        for row in report['probes']:
            m = metrics[row['predictor']]
            s = row['prediction_score']
            m['predicted_fields'] += s['checked_fields']
            m['correct_fields'] += s['matched_fields']
            m['exact_predictions'] += int(s['exact_match_on_predicted_fields'])
            m['predictions'] += 1
        report['prediction_metrics'] = metrics
        body = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + '\n'
        (out / 'discovery_report.json').write_bytes(body.encode('utf-8'))  # stable SHA across Windows newline translation
        # Build response profiles for later *unreviewed* manifest reconstruction.
        profiles: dict = {}
        for row in report['probes']:
            profiles[row['id']] = {'method': 'GET', 'path_template': row['path_template'],
                                   'observed_status': row['status'],
                                   'observed_shape': row['shape'],
                                   'verification': 'observed_once_not_behaviorally_verified'}
        candidate = {'format': 'orgsim-gate2d-response-profile-v1',
                     'reviewed': False, 'behavioral_clone': False,
                     'source': 'discovery_only_no_holdout',
                     'response_profiles': profiles,
                     'unverified_relationship_hypotheses': [
                         {'from': 'orders.socid', 'to': 'thirdparties.id',
                          'evidence': 'API field naming; not tested by Gate 2D'},
                         {'from': 'invoice.socid', 'to': 'thirdparties.id',
                          'evidence': 'API naming; not tested by Gate 2D'},
                         {'from': 'order_lines.fk_product', 'to': 'products.id',
                          'evidence': 'API naming; not tested by Gate 2D'}]}
        candidate_txt = json.dumps(candidate, indent=2, sort_keys=True) + '\n'
        (out / 'candidate_read_model_unreviewed.json').write_bytes(candidate_txt.encode('utf-8'))
        (out / 'FREEZE.json').write_text(json.dumps({
            'format': 'orgsim-gate2d-freeze-v1',
            'phase': 'discovery', 'independent_holdout': False,
            'live_writes': 0,
            'raw_values_saved': False,
            'report_sha256': sha256(body.encode('utf-8')).hexdigest(),
            'candidate_sha256': sha256(candidate_txt.encode('utf-8')).hexdigest(),
            'evidence_chain_sha256': event_sha,
            'prediction_precommit_enabled': event_sha is not None,
            'candidate_reviewed': False,
        }, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description='Gate 2D: adaptive GET-only local Dolibarr investigator; dry run default')
    p.add_argument('--base-url', default='http://127.0.0.1:8088')
    p.add_argument('--fixtures', help='Optional JSON of synthetic fixture IDs and safe product reference')
    p.add_argument('--budget', type=int, default=20)
    p.add_argument('--no-auto-ids', action='store_true',
                   help='Do not inspect in-memory list IDs to investigate details')
    p.add_argument('--execute-readonly', action='store_true', help='Only GETs to the local sandbox')
    p.add_argument('--execute-model', action='store_true', help='Opt-in: sends sanitized metadata to OpenAI API')
    p.add_argument('--model', default='gpt-6')
    p.add_argument('--model-calls', type=int, default=8)
    p.add_argument('--out', default='build/gate2d_dryrun')
    args = p.parse_args(argv)
    url = urlparse(args.base_url)
    if url.scheme not in ('http', 'https') or not _local_host(url.hostname or ''):
        p.error('Only an HTTP(S) loopback Dolibarr sandbox is permitted')
    if url.path.rstrip('/') not in ('', '/api/index.php') or url.query or url.fragment or url.username:
        p.error('Only base URLs at root or /api/index.php are supported; no credentials in the URL')
    if args.execute_model and not args.execute_readonly:
        p.error('--execute-model requires --execute-readonly')
    fixtures = validate_fixtures(json.loads(Path(args.fixtures).read_text(encoding='utf-8'))
                                 if args.fixtures else {})
    advisor = OpenAIProbeAdvisor(args.model) if args.execute_model else None
    key = os.environ.get('DOLIBARR_API_KEY') if args.execute_readonly else None
    if args.execute_readonly and not key: p.error('DOLIBARR_API_KEY required in local process environment')
    try:
        client = DolibarrClient(args.base_url, key)
        investigator = Investigator(client, fixtures=fixtures, budget=args.budget,
                                    auto_ids=not args.no_auto_ids,
                                    advisor=advisor, model_calls_limit=args.model_calls)
        report = investigator.run(execute=args.execute_readonly, outdir=args.out)
    except (ValueError, FileExistsError) as exc:
        # Only sanitized tool-generated exceptions are expected here.
        p.error(type(exc).__name__ + ': ' + str(exc))
    print(json.dumps({'phase': report['phase'], 'evidence': report['evidence'],
                      'reads': report['operations_executed'],
                      'model_calls': report['model_provider_calls'], 'writes': 0,
                      'stop_reason': report['stop_reason'],
                      'report': str(Path(args.out) / 'discovery_report.json')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
