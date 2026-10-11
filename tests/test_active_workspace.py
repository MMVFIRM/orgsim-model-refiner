from __future__ import annotations
from pathlib import Path
import json
import pytest
from orgsim_refiner.types import Scope,Field,Schema,Observation
from orgsim_refiner.learner import LearnConfig
from orgsim_refiner.workspace import initialize,learn,submit
from orgsim_refiner.util import read_json,write_json,canonical
from orgsim_refiner.artifact import PolicyArtifact
from orgsim_refiner.effects import EffectModel
from orgsim_refiner.runtime import PolicySimulator
from orgsim_refiner.evidence import EvidenceSet,EvidenceError
from orgsim_refiner import dsl
S=Scope('workspace','approval','v1')
SCHEMA=Schema((Field('x','int',0,31),))

def observed(x,split='active'):
    return Observation(S,{'x':x},'APPROVE' if x<16 else 'REJECT',split,'SIMULATOR_OBSERVED',f'{split}-{x}')

def test_inbox_resume_is_append_only(tmp_path):
    root=tmp_path/'inbox'
    local_schema=Schema((Field('x','int',0,31),Field('z','bool')))
    def row(x,z,split):return Observation(S,{'x':x,'z':z},'APPROVE' if x<16 else 'REJECT',split,'SIMULATOR_OBSERVED')
    initialize(root,local_schema,S,[row(0,False,'train'),row(31,False,'train')],
               [row(x,True,'validation') for x in (3,7,24,29)],budget=100,
               config=LearnConfig(audit_samples=4,min_validation=4,min_validation_per_class=2))
    requested=set()
    for _ in range(120):
        result=learn(root)
        if result['status']=='frozen':break
        rid=result['request_id'];assert rid not in requested
        requested.add(rid);req=read_json(root/'private'/'requests'/f'{rid}.json')
        ans=row(req['sample']['x'],req['sample']['z'],'active');submit(root,rid,ans)
        with pytest.raises(FileExistsError):submit(root,rid,ans)
    assert result['status']=='frozen'
    # A finite domain may exhaust distinct audit inputs; no forced promotion.
    p=PolicyArtifact.load(root/'POLICY.json')
    assert p.data['evidence']['training_replay_mismatches']==0
    with pytest.raises(FileExistsError):learn(root)
    assert len(requested)<=100

def test_real_inbox_requires_independent_files_and_explicit_category(tmp_path):
    root=tmp_path/'inbox';initialize(root,SCHEMA,S,[observed(0,'train'),observed(31,'train')],[observed(7,'validation'),observed(29,'validation')])
    res=learn(root);rid=res['request_id'];req=read_json(root/'private'/'requests'/f'{rid}.json')
    r=Observation(S,req['sample'],'REJECT','active','REAL_OPERATOR_CORROBORATED',before={'status':0},after={'status':0})
    with pytest.raises(ValueError):submit(root,rid,r)
    a=tmp_path/'receipt';b=tmp_path/'diagnostic';a.write_text('HTTP response');b.write_text('local log')
    with pytest.raises(ValueError):submit(root,rid,r,receipt=a,diagnostic=b)

def test_workspace_plan_tampering_detected(tmp_path):
    root=tmp_path/'inbox';initialize(root,SCHEMA,S,[observed(0,'train'),observed(31,'train')],[observed(7,'validation')])
    p=root/'private'/'PLAN.json';v=read_json(p);v['budget']=0;p.write_text(canonical(v))
    with pytest.raises(ValueError):learn(root)

def test_runtime_emits_pseudolabel_and_atomic_copy():
    p=PolicyArtifact.create(scope=S.to_dict(),schema=SCHEMA.to_dict(),cuts={'x':[16]},table={'0':'APPROVE','1':'REJECT'},
         ast=dsl.binary('lt',dsl.field('x'),dsl.constant(16)),ready=True,reason='test_double',evidence={},calibration={},assumptions=['test'])
    effects=EffectModel(S);effects.ready=True;effects.programs={'x':[dsl.field('before.x')],'count':[dsl.binary('add',dsl.field('before.count'),dsl.constant(1))]}
    runtime=PolicySimulator(p,effects,{'x':1,'count':0},{'x':'state.x'})
    result=runtime.step({},S);assert result['committed'] and runtime.state['count']==1
    assert result['provenance']=='MODEL_PSEUDOLABEL' and result['real_erp_write'] is False
    result=runtime.step({},Scope('different','approval','v1'));assert not result['committed'] and runtime.state['count']==1
    with pytest.raises(EvidenceError):EvidenceSet(SCHEMA,S).add(Observation(S,{'x':1},'APPROVE',provenance='MODEL_PSEUDOLABEL'))
    effects.ready=False;result=runtime.step({},S);assert result['decision']=='UNKNOWN' and runtime.state['count']==1

def test_cli_plan_offline(capsys):
    from orgsim_refiner.cli import main
    assert main(['plan','--tenants','8'])==0
    result=json.loads(capsys.readouterr().out)
    assert result['maximum_provider_attempts']==0 and result['actual_erp_calls']==0

def test_model_integrated_study_records_usage(tmp_path):
    from model_refiner.backend import ModelBackend
    from model_refiner.types import ModelResult
    from orgsim_refiner.model_bridge import BudgetBackend
    from orgsim_refiner.benchmark import run_study
    from orgsim_refiner.verification import verify_study
    class Fake(ModelBackend):
        def cache_identity(self):return {'test_double':True,'max_output_tokens':4096}
        def generate(self,*,model,messages):return ModelResult(text='APPROVE',input_tokens=10,output_tokens=1,metadata={})
    root=tmp_path/'study';b=BudgetBackend(Fake(),12)
    r=run_study(root,tenants=1,train=16,validation=32,test=16,stress=16,budget=2048,audit_samples=16,
                model='test-double-not-a-model-result',backend=b,model_cases=4)
    assert r['new_llm_calls']==12 and r['model_scores']['llm_zero_shot']['cases']==4
    assert verify_study(root)['verified']
    assert read_json(root/'PROVIDER_RECEIPTS.json')['usage']['provider_attempts']==12

def test_metrics_never_silently_truncates_predictions():
    from orgsim_refiner.benchmark import metrics
    with pytest.raises(ValueError):metrics([observed(0)],[])
