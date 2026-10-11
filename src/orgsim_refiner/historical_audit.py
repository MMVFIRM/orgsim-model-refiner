"""Read the disclosed hard-transfer archive WITHOUT inventing its missing oracle.

No re-training against private thresholds, no claimed blind score. This is a
regression audit of tenant isolation and existing file commitments only.
"""
from __future__ import annotations
import zipfile,hashlib
from pathlib import Path
from collections import Counter
from .util import loads,digest,write_json

def audit_archive(bundle:Path,out:Path):
    with zipfile.ZipFile(bundle) as z:
        required={'sealed_study.json','predictions.jsonl','STUDY_FREEZE.json','HARD_TRANSFER_REPORT.json'}
        names=set(z.namelist())
        if not required<=names:raise ValueError('Expected the disclosed hard-transfer archive')
        for name in required:
            if z.getinfo(name).file_size>64_000_000:raise ValueError('Archive member too large')
        data={n:z.read(n) for n in required}
    study=loads(data['sealed_study.json'].decode());freeze=loads(data['STUDY_FREEZE.json'].decode());rep=loads(data['HARD_TRANSFER_REPORT.json'].decode())
    checks={'sealed_matches':hashlib.sha256(data['sealed_study.json']).hexdigest()==freeze['sealed_study_sha256'],
            'freeze_matches':hashlib.sha256(data['STUDY_FREEZE.json']).hexdigest()==rep['study_freeze_sha256'],
            'predictions_match':hashlib.sha256(data['predictions.jsonl']).hexdigest()==rep['holdout_prediction_sha256']}
    if not all(checks.values()):raise ValueError('Disclosed archive hash mismatch')
    preds=[loads(s) for s in data['predictions.jsonl'].decode().splitlines() if s.strip()]
    training={r['id']:r for r in study['train']}
    correct_tenant=0;wrong_tenant=0;unknown=[];mixed=0
    for p in preds:
        if p['arm']!='same_tenant_retrieval':continue
        bad=0
        for rid in p.get('retrieved_ids',[]):
            r=training.get(rid)
            if r is None:unknown.append(rid);continue
            tenant=r.get('tenant') or rid.split('-train-')[0]
            if tenant==p['tenant']:correct_tenant+=1
            else:wrong_tenant+=1;bad+=1
        mixed+=bad>0
    result={'format':'orgsim-refiner-disclosed-archive-regression-v1','checks':checks,'predictions':len(preds),
            'unique_case_arm_pairs':len({(p['id'],p['arm']) for p in preds}),
            'retrieval_same_tenant_ids':correct_tenant,'retrieval_other_tenant_ids':wrong_tenant,'mixed_sets':mixed,
            'unknown_retrieved_ids':unknown,'new_provider_calls':0,'oracle_synthesis_attempted':False,
            'limitations':['Retrospective regression only; disclosed inputs are not fresh holdouts.',
                           'Original f1–f6 executable definitions and actual prompt bodies are absent.',
                           'Logged retrieved IDs do not independently prove which examples reached the provider.']}
    write_json(out,result);return result
