from __future__ import annotations
import copy,json,sqlite3,random
from pathlib import Path
import pytest
from orgsim_refiner import Scope,Field,Schema,Observation,ActivePolicyLearner,LearnConfig,PolicyArtifact
from orgsim_refiner import dsl
from orgsim_refiner.evidence import EvidenceSet,TrainingOracle,EvidenceLedger,EvidenceError,QueryBudgetExceeded
from orgsim_refiner.retrieval import TenantMemory
from orgsim_refiner.candidates import CandidateBank,replay,compile_partition
from orgsim_refiner.artifact import Partition
from orgsim_refiner.util import write_json,read_json,canonical,digest,loads
from orgsim_refiner.effects import EffectModel
from orgsim_refiner.runtime import PolicySimulator
from orgsim_refiner.benchmark import SyntheticTenant,dataset,SCHEMA,run_study
from orgsim_refiner.verification import verify_study

S=Scope('tenant-A','approve','cfg-1')
SMALL=Schema((Field('x','int',0,15),Field('segment','enum',values=('a','b'))))

def obs(x,y,split='train',scope=S):return Observation(scope,{'x':x,'segment':'a'},y,split)

@pytest.fixture(scope='module')
def fitted():
    t=SyntheticTenant(987112,'xor');sp=dataset(t,98712,{'train':16,'validation':32,'holdout':16})
    oracle=TrainingOracle(SCHEMA,t.scope,t.observe,budget=4096,forbidden_hashes={r.sample_hash for k,rs in sp.items() if k!='train' for r in rs})
    learner=ActivePolicyLearner(SCHEMA,t.scope,LearnConfig(seed=8132,audit_samples=16))
    p=learner.fit(sp['train'],oracle);p=learner.calibrate(p,sp['validation'])
    assert p.ready
    return t,sp,oracle,learner,p

@pytest.mark.parametrize('op,a,b,want',[('and',True,False,False),('or',False,True,True),('xor',True,True,False),('eq',2,2,True),('lt',2,3,True),('ge',2,3,False)])
def test_dsl_operators(op,a,b,want):
    n=dsl.binary(op,dsl.constant(a),dsl.constant(b));dsl.validate(n,set());assert dsl.evaluate(n,{})==want

def test_dsl_unknown_and_short_circuit():
    n=dsl.binary('and',dsl.field('missing'),dsl.constant(False));dsl.validate(n,{'missing'})
    assert dsl.evaluate(n,{}) is False
    assert dsl.evaluate(dsl.binary('xor',dsl.field('missing'),dsl.constant(True)),{}) is None

def test_atleast_and_if():
    n={'op':'atleast','k':2,'args':[dsl.constant(True),dsl.constant(True),dsl.field('x')]};dsl.validate(n,{'x'})
    assert dsl.evaluate(n,{}) is True
    expr={'op':'if','condition':n,'then':dsl.constant(False),'else':dsl.constant(True)}
    dsl.validate(expr,{'x'});assert not dsl.evaluate(expr,{})

@pytest.mark.parametrize('bad',[{'op':'exec','code':'rm -rf anything'}, {'op':'field','name':'__class__'}, {'op':'and','args':[dsl.constant('yes')]}, {'op':'le','args':[]}, {'op':'const','value':float('nan')}, {'op':'const','value':1,'extra':True}])
def test_dsl_rejects_bad_programs(bad):
    with pytest.raises(ValueError):dsl.validate(bad,{'x'})

def test_deep_ast_refused():
    n=dsl.constant(True)
    for _ in range(80):n={'op':'not','args':[n]}
    with pytest.raises(ValueError):dsl.validate(n,set())

@pytest.mark.parametrize('sample',[{'x':True,'segment':'a'},{'x':16,'segment':'a'},{'x':2,'segment':'bad'},{'x':2},{'x':2,'segment':'a','extra':1}])
def test_schema_fail_closed(sample):
    with pytest.raises(ValueError):SMALL.validate(sample)

def test_strict_json():
    with pytest.raises(ValueError):loads('{"a":1,"a":2}')
    with pytest.raises(ValueError):loads('{"a":NaN}')

def test_portable_bytes_and_no_overwrite(tmp_path):
    p=tmp_path/'artifact.json';h=write_json(p,{'message':'a\nb','unicode':'ü'})
    assert b'\r\n' not in p.read_bytes();assert digest(read_json(p))==h
    with pytest.raises(FileExistsError):write_json(p,{'b':2})

def test_isolation_before_ranking():
    other=Scope('tenant-B','approve','cfg-1');rows=[obs(0,'APPROVE'),obs(15,'REJECT',scope=other)]
    m=TenantMemory(SMALL,rows);got=m.retrieve(S,{'x':15,'segment':'a'},8)
    assert len(got)==1 and got[0].scope==S
    assert m.retrieve(Scope('none','approve','cfg-1'),{'x':15,'segment':'a'})==[]

def test_configuration_isolation():
    m=TenantMemory(SMALL,[obs(0,'APPROVE')])
    assert m.vote(Scope('tenant-A','approve','cfg-2'),{'x':0,'segment':'a'})=='UNKNOWN'

def test_no_pseudo_labels_or_holdouts():
    e=EvidenceSet(SMALL,S)
    for split in ('validation','holdout','stress'):
        with pytest.raises(EvidenceError):e.add(obs(0,'APPROVE',split))
    for prov in ('MODEL_PSEUDOLABEL','UNVERIFIED','LLM_OUTPUT'):
        with pytest.raises(EvidenceError):e.add(Observation(S,{'x':0,'segment':'a'},'APPROVE',provenance=prov))

def test_conflicts_quarantine():
    e=EvidenceSet(SMALL,S);e.add(obs(0,'APPROVE'))
    with pytest.raises(EvidenceError):e.add(obs(0,'REJECT'))
    assert e.conflicted

def test_evidence_defensive_copy():
    sample={'x':0,'segment':'a'};e=EvidenceSet(SMALL,S);e.add(Observation(S,sample,'APPROVE'));sample['x']=4
    assert e.rows[0].sample['x']==0

def test_oracle_excludes_holdout_and_budget():
    calls=[]
    def cb(x):calls.append(x);return Observation(S,x,'APPROVE')
    x={'x':1,'segment':'a'};q=TrainingOracle(SMALL,S,cb,budget=1,forbidden_hashes={digest(x)})
    with pytest.raises(EvidenceError):q.query(x)
    assert not calls
    q.query({'x':0,'segment':'a'});q.query({'x':0,'segment':'a'});assert len(calls)==1
    with pytest.raises(QueryBudgetExceeded):q.query({'x':2,'segment':'a'})
    q.close()
    with pytest.raises(EvidenceError):q.query({'x':0,'segment':'a'})

def test_oracle_rejects_scope_mismatch():
    q=TrainingOracle(SMALL,S,lambda x:Observation(Scope('b','a','c'),x,'APPROVE'),budget=1)
    with pytest.raises(EvidenceError):q.query({'x':0,'segment':'a'})

def test_oracle_detects_drift_on_repeat():
    ys=iter(['APPROVE','REJECT']);q=TrainingOracle(SMALL,S,lambda x:Observation(S,x,next(ys)),budget=2)
    q.query({'x':0,'segment':'a'})
    with pytest.raises(EvidenceError):q.query({'x':0,'segment':'a'},force_repeat=True)

def test_ledger_hashes_and_append_only(tmp_path):
    l=EvidenceLedger(tmp_path/'a.sqlite');l.append('x',{'a':'line\nbreak'});l.append('y',{'n':2})
    assert l.verify()['events']==2
    with sqlite3.connect(l.path) as con:
        with pytest.raises(sqlite3.IntegrityError):con.execute("UPDATE events SET kind='modified'")
        con.execute('DROP TRIGGER events_no_update');con.execute("UPDATE events SET kind='modified' WHERE seq=1")
    with pytest.raises(EvidenceError):l.verify()

def test_reject_inconsistent_candidate():
    b=CandidateBank(SMALL).fit([obs(0,'APPROVE'),obs(15,'REJECT')],[dsl.constant(True)])
    assert b.rejections[0]['reason']=='contradicts_evidence'
    assert all(not replay(a,SMALL,[obs(0,'APPROVE'),obs(15,'REJECT')]) for a in b.asts)

def test_partition_uses_learned_boundaries_not_midpoints():
    schema=Schema((Field('x','int',0,30),));part=Partition(schema,{'x':[7]})
    ast=compile_partition(part,{'0':'APPROVE','1':'REJECT'})
    assert dsl.label(ast,{'x':6})=='APPROVE' and dsl.label(ast,{'x':7})=='REJECT'
    assert dsl.label(ast,{'x':10})=='REJECT'

def test_nonmonotone_interval():
    schema=Schema((Field('x','int',0,15),Field('n','int',0,99)))
    scope=Scope('nonmono','act','v1');fn=lambda x:'REJECT' if 5<=x['x']<=8 else 'APPROVE'
    train=[Observation(scope,{'x':x,'n':0},fn({'x':x})) for x in (0,6,15)]
    vals=[Observation(scope,{'x':i%16,'n':90+i%10},fn({'x':i%16}),'validation') for i in range(32)]
    q=TrainingOracle(schema,scope,lambda x:Observation(scope,x,fn(x)),budget=700,forbidden_hashes={r.sample_hash for r in vals})
    l=ActivePolicyLearner(schema,scope,LearnConfig(audit_samples=16));pol=l.fit(train,q);pol=l.calibrate(pol,vals)
    assert pol.ready and pol.partition.cuts['x']==[5,9]
    assert all(pol.predict({'x':i,'n':80},scope)['decision']==fn({'x':i}) for i in range(16))

def test_fitted_raw_inputs_not_oracle_bits(fitted):
    t,sp,q,l,p=fitted
    assert p.ready and all(p.predict(r.sample,t.scope)['decision']==r.label for r in sp['holdout'])
    assert p.data['evidence']['training_replay_mismatches']==0 and q.closed
    assert all('thresholds' not in r.sample for r in l.evidence.rows)

def test_artifact_integrity_and_immutable_views(fitted,tmp_path):
    t,sp,q,l,p=fitted;path=tmp_path/'p.json';p.save(path);assert PolicyArtifact.load(path).fingerprint==p.fingerprint
    view=p.data;view['ready']=False;assert p.ready
    partition=p.partition;partition.cuts['credit'].append(30);assert p.partition.cuts['credit']!=partition.cuts['credit']
    bad=p.to_dict();bad['ready']=False
    with pytest.raises(ValueError):PolicyArtifact(bad)

def test_artifact_refuses_cross_tenant_and_missing_feature(fitted):
    t,sp,q,l,p=fitted;x=sp['holdout'][0].sample
    assert p.predict(x,Scope('other',t.scope.operation,t.scope.configuration))['decision']=='UNKNOWN'
    z=dict(x);z.pop('stock');assert p.predict(z,t.scope)['decision']=='UNKNOWN'

def test_calibration_not_automatically_learning(fitted):
    t,sp,q,l,p=fitted
    bad=[Observation(r.scope,r.sample,'REJECT' if r.label=='APPROVE' else 'APPROVE','validation') for r in sp['validation']]
    n=len(l.evidence.rows);r=l.calibrate(p,bad)
    assert not r.ready and len(l.evidence.rows)==n

def test_budget_failure_is_abstention_not_success():
    q=TrainingOracle(SMALL,S,lambda x:Observation(S,x,'APPROVE'),budget=0)
    l=ActivePolicyLearner(SMALL,S,LearnConfig(audit_samples=4));p=l.fit([obs(0,'APPROVE'),obs(15,'REJECT')],q)
    assert not p.ready and p.predict({'x':5,'segment':'a'},S)['decision']=='UNKNOWN'

def test_effects_separate_and_roundtrip(fitted):
    t,sp,q,l,p=fitted;e=EffectModel(t.scope).fit(l.evidence.rows,sp['validation'])
    assert e.ready
    loaded=EffectModel.from_dict(e.to_dict())
    from orgsim_refiner.util import flatten
    for r in sp['holdout']:
        if r.label=='APPROVE':assert loaded.predict(r.before,r.action,t.scope)==flatten(r.after)
    assert loaded.predict(sp['holdout'][0].before,{},t.scope) is None

def test_effects_bad_ast_rejected(fitted):
    t,sp,q,l,p=fitted;e=EffectModel(t.scope).fit(l.evidence.rows,sp['validation']).to_dict();e['programs']['x']=[{'op':'shell','args':[]}]
    with pytest.raises(ValueError):EffectModel.from_dict(e)

def test_simulated_outcomes_never_truth():
    with pytest.raises(EvidenceError):EvidenceSet(SMALL,S).add(Observation(S,{'x':3,'segment':'a'},'APPROVE',provenance='MODEL_PSEUDOLABEL'))

def test_integrated_freeze_and_verify(tmp_path):
    out=tmp_path/'study';r=run_study(out,tenants=1,train=16,validation=32,test=16,stress=16,audit_samples=16)
    assert r['new_llm_calls']==0 and r['dolibarr_calls']==0
    assert verify_study(out)['verified']
    with pytest.raises(FileExistsError):run_study(out,tenants=1)
