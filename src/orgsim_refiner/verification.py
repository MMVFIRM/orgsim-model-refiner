from __future__ import annotations
from pathlib import Path
from .util import read_json,digest,file_digest
from .artifact import PolicyArtifact
from .evidence import EvidenceLedger
from .workspace import read_rows

def verify_study(root:Path,*,check_current_source=True):
    root=Path(root);spec=read_json(root/'STUDY_FREEZE.json');freeze=read_json(root/'FINAL_FREEZE.json');report=read_json(root/'REPORT.json')
    if digest(spec)!=freeze['study_sha256'] or digest(report)!=freeze['report_sha256']:raise ValueError('Study/report digest mismatch')
    if check_current_source:
        for name,h in spec['source_sha256'].items():
            if Path(name).name!=name or file_digest(Path(__file__).parent/name)!=h:raise ValueError('Learner source changed after study freeze: '+name)
    for name,h in freeze.get('artifact_file_sha256',{}).items():
        rel=Path(name)
        if rel.is_absolute() or '..' in rel.parts or file_digest(root/rel)!=h:raise ValueError('Study artifact changed: '+name)
    checked=0
    for env in sorted(root.glob('tenant_*')):
        if not env.is_dir():continue
        pol=PolicyArtifact.load(env/'POLICY.json');result=read_json(env/'RESULT.json')
        if pol.fingerprint!=result['policy_sha256']:raise ValueError('Policy result mismatch')
        if result!=report['tenants_detail'][checked]:raise ValueError('Tenant result differs from aggregate report')
        if digest(read_json(env/'SEALED_EVALUATOR.json'))!=result['sealed_evaluator_sha256']:raise ValueError('Sealed data changed')
        if digest(read_json(env/'EFFECTS.json'))!=result['effects_sha256']:raise ValueError('Effect program changed')
        evidence=read_rows(env/'TRAINING_EVIDENCE.jsonl')
        from .util import digest as dh
        from .candidates import replay
        h=dh([r.to_dict() for r in sorted(evidence,key=lambda x:x.sample_hash)])
        if h!=pol.data['evidence']['training_sha256']:raise ValueError('Training evidence changed')
        if pol.data['ast'] is not None and replay(pol.data['ast'],pol.schema,evidence):raise ValueError('Program fails training replay')
        ledger=EvidenceLedger(env/'EVIDENCE.sqlite');verified=ledger.verify()
        if verified!=result['evidence']:raise ValueError('Ledger/result mismatch')
        events=ledger.events();commits=[e for e in events if e['kind']=='predictions_frozen'];scores=[e for e in events if e['kind']=='scoring_completed']
        sealed_events=[e for e in events if e['kind']=='tenant_data_sealed']
        artifacts=[e for e in events if e['kind']=='learned_artifacts_frozen']
        if len(sealed_events)!=1 or sealed_events[0]['seq']!=1 or sealed_events[0]['payload']['sealed_sha256']!=result['sealed_evaluator_sha256']:raise ValueError('Missing sealed-data precommit')
        if len(artifacts)!=1 or artifacts[0]['payload']['effects_sha256']!=result['effects_sha256']:raise ValueError('Artifact commitment mismatch')
        if commits and artifacts[0]['seq']>=commits[0]['seq']:raise ValueError('Artifacts not frozen before prediction')
        model_events=[e for e in events if e['kind']=='model_predictions_frozen']
        if (env/'MODEL_PREDICTIONS.json').exists():
            if len(model_events)!=1 or digest(read_json(env/'MODEL_PREDICTIONS.json'))!=model_events[0]['payload']['sha256']:raise ValueError('Model commitment mismatch')
            if scores and model_events[0]['seq']>=scores[0]['seq']:raise ValueError('Model predictions not committed before score')
        if len(commits)!=1 or len(scores)!=1 or commits[0]['seq']>=scores[0]['seq']:raise ValueError('Missing prediction-before-score commitment')
        if digest(read_json(env/'PREDICTIONS.json'))!=commits[0]['payload']['predictions_sha256']:raise ValueError('Predictions modified')
        sealed=read_json(env/'SEALED_EVALUATOR.json');train={r.sample_hash for r in evidence}
        for split in ('validation','holdout','stress'):
            from .types import Observation
            if any(Observation.from_dict(r).sample_hash in train for r in sealed['splits'][split]):raise ValueError('Learning/evaluation overlap')
        checked+=1
    if checked!=spec['tenants']:raise ValueError('Incomplete tenant results')
    return {'verified':True,'tenants':checked,'scope':'Local source, evidence and file consistency, not independent attestation.',
            'externally_timestamped':False,'new_provider_calls':0,'dolibarr_calls':0}
