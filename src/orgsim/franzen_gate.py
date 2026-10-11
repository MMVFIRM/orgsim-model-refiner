"""Offline Franzen-to-OrgSim Gate 2 preflight: catalog, probe planner, blind brief.

Consumes local files only. No requests to Dolibarr; no holdout contamination.
This does not call an LLM or fabricate missing workflow semantics.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

from .exploration import catalog_from_swagger, prioritize_probes
from .spec import load_spec


def _save(path: Path, data: Any) -> str:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return sha256(path.read_bytes()).hexdigest()


def workflow_hypotheses(catalog: list[dict]) -> list[dict]:
    result = []
    for op in catalog:
        if op['kind'] not in {'lifecycle_transition', 'workflow_or_relationship'}:
            continue
        path = op['path']
        family = path.strip('/').split('/')[0]
        if family not in {'orders', 'invoices', 'products', 'thirdparties'}:
            continue
        result.append({
            'id': op.get('operation_id') or f'{op["method"]} {path}',
            'http': {'method': op['method'], 'path': path},
            'action_family': family,
            'candidate_kind': op['kind'],
            'description_from_contract': op.get('summary', ''),
            'known_parameters': op.get('params', []),
            'evidence': 'contract_only_hypothesis',
            'preconditions': 'UNKNOWN', 'postconditions': 'UNKNOWN',
            'side_effects': 'UNKNOWN', 'error_behavior': 'UNKNOWN',
            'state_transition': 'UNKNOWN',
            'executable': False,
            'next_step': 'Infer a candidate using permitted discovery, compile into the declarative workflow DSL, and check in a separate sandbox',
        })
    return result


def make_reconstruction_brief(catalog: list[dict], manifest: dict | None,
                              discovery: dict | None) -> dict:
    if discovery is not None and discovery.get('phase') != 'discovery':
        raise ValueError('Only explicitly labeled discovery evidence may enter reconstruction brief')
    actions = workflow_hypotheses(catalog)
    counts = Counter(c['kind'] for c in catalog)
    by_resource = Counter(c['path'].strip('/').split('/')[0] for c in catalog)
    return {
        'format': 'orgsim-franzen-reconstruction-brief-v1',
        'phase': 'discovery',
        'rules': [
            'Do not read held-out cases, Dolibarr source code, or external documentation.',
            'Operate only on the exported Swagger contract and approved discovery evidence.',
            'Do not claim unobserved business constraints or default values as verified.',
            'Emit a declarative OrgSim manifest; no Python snippets, eval, or network tools.',
            'Mark the entire candidate reviewed:false; label workflow evidence hypothesis.',
            'Prioritize predicted state transitions, preconditions and rejection behavior.',
            'Never treat a matching HTTP 404, empty array, or response padding as a workflow pass.',
        ],
        'counts': {'api_operations': len(catalog), 'by_kind': dict(counts),
                   'by_first_path_component': dict(by_resource),
                   'hypothesized_special_operations': len(actions)},
        'known_reconstruction_limits': [
            'Ordinary CRUD is not a full business workflow.',
            'Swagger return types can conflict with observed runtime values.',
            'Current discovery contains only customer and product read/list behavior.',
            'No live evidence exists for orders/invoices transitions or side effects.',
            'Business-value fidelity and state-transition fidelity must be scored separately.',
        ],
        'candidate_manifest_sha256': sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest() if manifest else None,
        'candidate_manifest_reviewed': manifest.get('reviewed') if manifest else None,
        'discovery_probes_available': len(discovery.get('discovery_results', discovery.get('probes', []))) if discovery else 0,
        'heldout_probes_available_to_model': 0,
        'workflow_blueprints': actions,
        'desired_model_output': {
            'manifest_version': '0.1',
            'system_workflows': 'Declarative actions, guards and steps; separate evidence labels',
            'uncertain': 'Explicitly unsupported actions and undecidable transitions',
            'evidence_ledger': 'Hypothesis-to-observation claims, never heldout data',
        },
    }


def analyze(swagger: dict, manifest: dict | None = None, discovery: dict | None = None,
            *, outdir: Path, budget: int = 18) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    catalog = catalog_from_swagger(swagger)
    brief = make_reconstruction_brief(catalog, manifest, discovery)
    plan = prioritize_probes(catalog, budget=budget)
    hashes = {
        'catalog.json': _save(outdir / 'catalog.json', catalog),
        'workflow_hypotheses.json': _save(outdir / 'workflow_hypotheses.json', brief['workflow_blueprints']),
        'probe_priorities.json': _save(outdir / 'probe_priorities.json', plan),
        'model_reconstruction_brief.json': _save(outdir / 'model_reconstruction_brief.json', brief),
    }
    marker = {'format': 'orgsim-offline-freeze-v1', 'source': 'user-provided sanitized Swagger plus discovery evidence',
              'swagger_sha256': sha256(json.dumps(swagger, sort_keys=True).encode()).hexdigest(),
              'manifest_sha256': brief['candidate_manifest_sha256'],
              'no_live_calls': True, 'no_holdouts_seen': True, 'artifacts_sha256': hashes,
              'note': 'No blind behavioral evaluation has been performed'}
    _save(outdir / 'FREEZE.json', marker)
    return {'api_operations': len(catalog), 'proposed_workflows': len(brief['workflow_blueprints']),
            'suggested_read_probes': len(plan['probes']), 'outdir': str(outdir)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Offline environment-reconstruction preflight')
    parser.add_argument('--swagger', required=True, help='Local sanitized Swagger 2.0 JSON file')
    parser.add_argument('--manifest', help='Optional local OrgSim draft manifest')
    parser.add_argument('--discovery', help='Optional discovery-phase report, never holdout')
    parser.add_argument('--out', required=True, help='Destination folder for offline evidence')
    parser.add_argument('--budget', type=int, default=18)
    args = parser.parse_args(argv)
    swagger = json.loads(Path(args.swagger).read_text(encoding='utf-8'))
    manifest = load_spec(args.manifest) if args.manifest else None
    discovery = json.loads(Path(args.discovery).read_text(encoding='utf-8')) if args.discovery else None
    print(json.dumps(analyze(swagger, manifest, discovery, outdir=Path(args.out), budget=args.budget), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
