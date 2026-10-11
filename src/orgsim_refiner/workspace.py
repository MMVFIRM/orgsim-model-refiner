"""Durable, network-free oracle inbox for an external simulator/operator.

Resuming recomputes the deterministic learning schedule against retained
observations. It never overwrites an outcome or invokes a real ERP write.
This is NEW development work, not a way to amend the frozen Dolibarr study.
"""
from __future__ import annotations
from pathlib import Path
from .types import Scope,Schema,Observation
from .learner import ActivePolicyLearner,LearnConfig
from .evidence import TrainingOracle,EvidenceLedger
from .artifact import PolicyArtifact
from .util import read_json,write_json,canonical,loads,digest,file_digest

class PendingObservation(Exception):
    def __init__(self,request_id):self.request_id=request_id;super().__init__('An observed answer is required')

def read_rows(path):
    p=Path(path)
    if p.stat().st_size>64_000_000:raise ValueError('Evidence file too large')
    return [Observation.from_dict(loads(s)) for s in p.read_text('utf-8').splitlines() if s.strip()]

def initialize(root,schema,scope,training,validation,*,budget=8192,config=None):
    root=Path(root)
    if root.exists():raise FileExistsError('Workspace already exists')
    if not 0<=budget<=100000:raise ValueError('Invalid oracle budget')
    config=config or LearnConfig()
    seen=set()
    for split,rows in (('train',training),('validation',validation)):
        for r in rows:
            if r.scope!=scope or r.split!=split or not r.usable():raise ValueError('Invalid workspace evidence')
            schema.validate(r.sample)
            if r.sample_hash in seen:raise ValueError('Duplicate/split-overlapping input')
            seen.add(r.sample_hash)
    root.mkdir(parents=True)
    plan={'format':'orgsim-refiner-oracle-inbox-v1','schema':schema.to_dict(),'scope':scope.to_dict(),
          'config':config.to_dict(),'budget':budget,'training':[r.to_dict() for r in training],
          'validation':[r.to_dict() for r in validation],'automatic_erp_writes':False}
    h=write_json(root/'private'/'PLAN.json',plan)
    write_json(root/'FREEZE.json',{'plan_sha256':h,'externally_timestamped':False})
    EvidenceLedger(root/'private'/'EVIDENCE.sqlite').append('workspace_created',{'plan_sha256':h})
    return {'root':str(root),'plan_sha256':h,'automatic_erp_writes':0}

def _plan(root):
    root=Path(root);p=read_json(root/'private'/'PLAN.json')
    if digest(p)!=read_json(root/'FREEZE.json')['plan_sha256']:raise ValueError('Workspace plan changed')
    return p,Schema.from_dict(p['schema']),Scope(**p['scope'])

def learn(root):
    root=Path(root)
    if (root/'POLICY.json').exists():raise FileExistsError('Policy already frozen; start no new learning in this workspace')
    plan,schema,scope=_plan(root);ledger=EvidenceLedger(root/'private'/'EVIDENCE.sqlite')
    def callback(sample):
        reqid=digest({'scope':scope.to_dict(),'sample':sample})
        answer=root/'private'/'answers'/f'{reqid}.json'
        if answer.exists():return Observation.from_dict(read_json(answer))
        pending=root/'private'/'requests'/f'{reqid}.json'
        if not pending.exists():
            write_json(pending,{'request_id':reqid,'scope':scope.to_dict(),'sample':sample,
                'automatic_erp_writes':False,'note':'Only an external observed outcome can answer this request. Do not invent labels.'})
            ledger.append('pending_query',{'request_id':reqid,'sample_sha256':digest(sample)})
        raise PendingObservation(reqid)
    training=[Observation.from_dict(x) for x in plan['training']]
    validation=[Observation.from_dict(x) for x in plan['validation']]
    oracle=TrainingOracle(schema,scope,callback,budget=plan['budget'],forbidden_hashes={r.sample_hash for r in validation})
    learner=ActivePolicyLearner(schema,scope,LearnConfig(**plan['config']))
    try:policy=learner.fit(training,oracle)
    except PendingObservation as pending:
        return {'status':'awaiting_observation','request_id':pending.request_id,'automatic_erp_writes':0}
    policy=learner.calibrate(policy,validation);policy.save(root/'POLICY.json')
    ledger.append('policy_frozen',{'artifact_sha256':policy.fingerprint,'ready':policy.ready})
    return {'status':'frozen','ready':policy.ready,'artifact_sha256':policy.fingerprint,'automatic_erp_writes':0}

def submit(root,request_id,answer:Observation,*,receipt:Path|None=None,diagnostic:Path|None=None):
    root=Path(root);_,schema,scope=_plan(root)
    if (root/'POLICY.json').exists():raise ValueError('Cannot add evidence to a frozen policy')
    if len(request_id)!=64 or any(c not in '0123456789abcdef' for c in request_id):raise ValueError('Invalid request id')
    request=read_json(root/'private'/'requests'/f'{request_id}.json')
    if answer.scope!=scope or digest(answer.sample)!=digest(request['sample']):raise ValueError('Answer/request scope or input mismatch')
    schema.validate(answer.sample)
    if answer.split not in ('active','audit'):raise ValueError('Answer must belong to training-query phase')
    if answer.provenance not in ('SIMULATOR_OBSERVED','REAL_OPERATOR_CORROBORATED'):raise ValueError('Do not submit model guesses')
    if answer.provenance=='REAL_OPERATOR_CORROBORATED':
        if receipt is None or diagnostic is None:raise ValueError('Real labels need separate response and diagnostic evidence')
        hashes=[]
        for p in (Path(receipt),Path(diagnostic)):
            if not p.is_file() or not 0<p.stat().st_size<=1_000_000:raise ValueError('Missing/oversized local evidence')
            hashes.append(file_digest(p))
        if hashes[0]==hashes[1]:raise ValueError('Separate evidence artifacts required')
        ev=answer.evidence or {}
        if not ev.get('operator_correlated') or ev.get('outcome_category') not in ('SUCCESS','BUSINESS_REFUSAL','UNKNOWN'):
            raise ValueError('Explicit operator correlation/outcome category required')
        if answer.before is None or answer.after is None:raise ValueError('Before/after state evidence required')
        if answer.label=='REJECT' and (ev['outcome_category']!='BUSINESS_REFUSAL' or answer.before!=answer.after):raise ValueError('No-op/access/unknown/error is not business refusal')
        if answer.label=='APPROVE' and (ev['outcome_category']!='SUCCESS' or answer.before==answer.after):raise ValueError('Approval needs observed transition')
        answer=Observation.from_dict({**answer.to_dict(),'evidence':{**ev,'receipt_sha256':hashes[0],'diagnostic_sha256':hashes[1],
                                              'not_independently_authenticated':True}})
    write_json(root/'private'/'answers'/f'{request_id}.json',answer.to_dict())
    EvidenceLedger(root/'private'/'EVIDENCE.sqlite').append('observed_answer',{'request_id':request_id,'answer_sha256':digest(answer.to_dict()),
        'provenance':answer.provenance,'label':answer.label})
    return {'accepted_into_inbox':True,'request_id':request_id,'automatic_erp_writes':0,
            'note':'File correlation is operator-attested, not independent validation of the business reason.'}
