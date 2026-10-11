"""Offline check that Gate 2D prediction commits precede recorded observations.

Verifies local event ordering and artifact digests. A local hash chain is
integrity evidence against accidental modification, NOT an external timestamp
or independent proof that a remote service responded at a particular moment.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path

from .gate2d_adaptive import canonical, hash_json


def verify(folder: str | Path) -> dict:
    path=Path(folder)
    freeze=json.loads((path/'FREEZE.json').read_text(encoding='utf-8'))
    report_data=(path/'discovery_report.json').read_bytes()
    candidate_data=(path/'candidate_read_model_unreviewed.json').read_bytes()
    report=json.loads(report_data)
    expected=freeze.get('report_sha256')==sha256(report_data).hexdigest()
    expected &= freeze.get('candidate_sha256')==sha256(candidate_data).hexdigest()
    prev='0'*64
    expected_index=0
    pending=None
    pred_events=ob_events=0
    with (path/'events.jsonl').open(encoding='utf-8') as f:
        for line_number,line in enumerate(f,1):
            item=json.loads(line)
            if item.get('seq')!=line_number or item.get('prev_sha256')!=prev:
                raise ValueError('Event numbering or previous hash is invalid')
            saved=item['sha256'];unsig=dict(item);del unsig['sha256']
            if saved!=hash_json(unsig):raise ValueError('Event hash does not verify')
            prev=saved
            event=item.get('event')
            if event=='prediction_committed':
                if pending is not None:raise ValueError('New prediction without completed previous read')
                if item.get('index')!=expected_index:raise ValueError('Prediction index skipped')
                pending=(item['probe_id'],item['index'])
                pred_events+=1
            if event in {'observation_recorded','probe_transport_error'}:
                if pending != (item.get('probe_id'),item.get('index')):
                    raise ValueError('An oracle result was logged without a preceding matching prediction')
                if event=='observation_recorded': ob_events+=1
                pending=None;expected_index+=1
    if prev!=freeze.get('evidence_chain_sha256'):
        raise ValueError('Chain freeze mismatch')
    if pred_events!=expected_index or ob_events!=report.get('operations_executed'):
        raise ValueError('Evidence counts mismatch')
    if any(row.get('method')!='GET' for row in report.get('probes',[])):
        raise ValueError('Non-GET method found in report')
    if not expected:raise ValueError('Report or candidate digest mismatch')
    return {'status':'PASS_LOCAL_ARTIFACT_INTEGRITY_ONLY',
            'sha_chain_valid':True, 'precommit_events':pred_events,
            'observations_recorded':ob_events,
            'live_behavioral_fidelity_verified':False,
            'external_timestamp_verified':False,
            'independent_holdout':False}


def main(argv=None)->int:
    ap=argparse.ArgumentParser(description='Verify Gate 2D local event order and digests')
    ap.add_argument('directory',help='Directory containing Gate 2D report, candidate, evidence, FREEZE')
    args=ap.parse_args(argv)
    print(json.dumps(verify(args.directory),indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
