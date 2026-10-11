from __future__ import annotations
import json
import pytest
from model_refiner.backend import ModelBackend
from model_refiner.types import ModelResult
from orgsim_refiner.types import Scope,Field,Schema,Observation
from orgsim_refiner.artifact import PolicyArtifact
from orgsim_refiner.model_bridge import BudgetBackend,compare_models,propose_ast
from orgsim_refiner.util import canonical
from orgsim_refiner import dsl

S=Scope('a','op','cfg');schema=Schema((Field('x','int',0,10),))

def policy():
    ast=dsl.binary('lt',dsl.field('x'),dsl.constant(5))
    return PolicyArtifact.create(scope=S.to_dict(),schema=schema.to_dict(),cuts={'x':[5]},table={'0':'APPROVE','1':'REJECT'},ast=ast,evidence={},calibration={},ready=True,reason='test_double',assumptions=['test_double_only'])
class FakeBackend(ModelBackend):
    def __init__(self,text='APPROVE',metadata=None):self.text=text;self.metadata=metadata or {};self.prompts=[]
    def cache_identity(self):return {'test_double':True,'max_output_tokens':4096}
    def generate(self,*,model,messages):
        self.prompts.append(messages);return ModelResult(text=self.text,input_tokens=20,output_tokens=2,metadata=self.metadata)

def train():
    return [Observation(S,{'x':0},'APPROVE',source_id='a-train-0'),Observation(S,{'x':10},'REJECT',source_id='a-train-1'),
            Observation(Scope('b','op','cfg'),{'x':3},'REJECT',source_id='b-train')]

def test_native_bridge_scope_and_raw_separation():
    f=FakeBackend();b=BudgetBackend(f,3);r=compare_models(schema,S,policy(),train(),[{'x':7}],b,model='test')
    assert b.calls==3 and r['predictions']['llm_executable_context']==['APPROVE']
    assert r['predictions']['program_verified']==['REJECT'] and r['receipts'][0]['verifier_overrode_model']
    m=r['receipts'][0]['arms']['llm_tenant_retrieval'];assert m['retrieved_scope_keys']==[S.key,S.key]
    assert 'b-train' not in canonical(f.prompts) and 'test_double' in canonical(b.cache_identity())
    assert all('target' not in canonical(m) for m in f.prompts)

def test_no_true_label_in_model_interface():
    f=FakeBackend();b=BudgetBackend(f,3)
    with pytest.raises(ValueError):compare_models(schema,S,policy(),train(),[{'x':0}],b,model='test')
    assert b.calls==0

def test_incomplete_is_recorded_as_unknown_not_repaired():
    f=FakeBackend(metadata={'incomplete':True,'status':'incomplete'});b=BudgetBackend(f,3)
    r=compare_models(schema,S,policy(),train(),[{'x':7}],b,model='test')
    assert b.calls==3 and r['predictions']['llm_executable_context']==['UNKNOWN']
    assert r['receipts'][0]['arms']['llm_executable_context']['status']=='incomplete'

def test_budget_hard_ceiling():
    b=BudgetBackend(FakeBackend(),1);b.generate(model='x',messages=[])
    with pytest.raises(RuntimeError):b.generate(model='x',messages=[])
    assert b.calls==1

def test_proposer_must_fit_training():
    b=BudgetBackend(FakeBackend(canonical(dsl.constant(True))),1)
    r=propose_ast(schema,S,train()[:2],b,model='test')
    assert not r['accepted'] and r['reason']=='training_contradiction'

def test_proposer_executable_and_scoped():
    a=dsl.binary('lt',dsl.field('x'),dsl.constant(5));b=BudgetBackend(FakeBackend(canonical(a)),1)
    r=propose_ast(schema,S,train()[:2],b,model='test')
    assert r['accepted'] and r['not_certified_for_unseen_cases']
    with pytest.raises(ValueError):propose_ast(schema,S,train(),b,model='test')
