"""Regression tests: executable policies, OOD abstention, matched-evidence evaluation."""
import copy
import json
from pathlib import Path
import pytest

from simlearn.oracle import OPS,Oracle,sample_input
from simlearn.dataset import generate_environment
from simlearn.structured import (Program,fit_structured,neutral,counterfactual,
                                 ModelRefinerPolicyAdapter,Membership,first_rejection)
from simlearn.structured_study import run_environment,run_ood,evaluate
from simlearn.structured_cli import main as structured_main


def test_intervention_catalog_only_contains_public_data():
    for op in OPS:
        s=neutral(op)
        assert s['operation']==op
        assert set(s['state'])=={'customer','product','order','invoice'}
        assert all('policy' not in json.dumps(s) for _ in (1,))

@pytest.mark.parametrize('op',OPS)
def test_program_recovers_hidden_rules_from_only_oracle_response(op):
    o=Oracle(20261213)
    try:
        def black_box(sample):
            outcome=o.execute(sample)
            return outcome
        p,examples=fit_structured(black_box,op,seed=3,max_queries=160,audit_count=20)
        assert p.audit_ready and p.audit_mismatches==0
        assert p.queries==len(examples) and p.queries<=160
        assert p.evidence_hash and p.guards
        assert 'Policy(' not in json.dumps(p.to_dict())
        for sample in [counterfactual(op,**axis) for axis in (
            ([{'qty':7,'amount':120},{'qty':1,'amount':0}] if op=='validate_order' else
             [{'payment':80,'count':5},{'payment':1,'count':0}] if op=='pay_invoice' else
             [{'qty':12,'reserved':9},{'qty':1,'reserved':0}] if op=='reserve_stock' else
             [{'qty':12,'amount':120,'shipped':20},{'qty':1,'amount':0,'shipped':5}]))]:
            assert p.classify(sample)==o.execute(sample)['decision']
    finally:o.close()


def test_fail_closed_when_outside_rule_grammar():
    r=run_ood(31945,max_probes=160,audit_count=24,eval_per_op=40)
    for d in r['per_operation'].values():
        assert not d['audit_ready']
        assert 0.0<d['coverage']<1.0
        assert d['incorrect_accepts']==0


def test_model_refiner_proposal_cannot_override_frozen_policy():
    o=Oracle(1021)
    try:
        p,_=fit_structured(o.execute,'validate_order',seed=9,max_queries=160,audit_count=16)
        a=ModelRefinerPolicyAdapter({'validate_order':p})
        reject=counterfactual('validate_order',qty=20,amount=200)
        assert a.decide(reject,'ACCEPT')['decision']=='REJECT'
        assert a.decide(reject,'ACCEPT')['executable'] is False
        assert a.decide(counterfactual('pay_invoice'),'ACCEPT')['decision']=='ABSTAIN'
        assert not a.decide(reject,'ACCEPT')['model_override_authorized']
        scope=ModelRefinerPolicyAdapter({'validate_order':p},tenant_fingerprint='tenant-A')
        assert scope.decide(counterfactual('validate_order'),'ACCEPT',tenant_fingerprint='tenant-B')['decision']=='ABSTAIN'
        with pytest.raises(ValueError):a.decide(reject,'GRANT')
    finally:o.close()


def test_program_roundtrip_and_reject_malformed_fields():
    p=Program('reserve_stock',[{'kind':'sum_upper','expression':'product.reserved+action.reserve_qty','max':9}],
              audit_ready=True,audit_total=50,evidence_hash='x')
    assert Program.from_dict(p.to_dict()).to_dict()==p.to_dict()
    malformed=p.to_dict()
    malformed['guards'][0]['expression']='os.system'
    with pytest.raises(ValueError,match='Unknown feature'):Program.from_dict(malformed)
    malformed=p.to_dict();malformed['guards'][0]['max']=True
    with pytest.raises(ValueError,match='Invalid threshold'):Program.from_dict(malformed)


def test_query_budget_and_unknown_provenance():
    m=Membership(lambda s:{'decision':'ACCEPT','provenance':'NOT_ORACLE'},'validate_order',max_queries=20)
    with pytest.raises(ValueError,match='provenance'):m.ask(counterfactual('validate_order',qty=2,amount=30))
    assert len(m.rows)==0
    assert first_rejection(lambda x:x>=6,0,10)==6
    assert first_rejection(lambda x:False,0,10) is None


def test_program_abstains_if_uncertified():
    p=Program('validate_order',[],audit_ready=False)
    assert p.classify(counterfactual('validate_order'))=='ABSTAIN'


def test_single_environment_matched_evidence_and_roundtrip(tmp_path):
    data=tmp_path/'dataset'
    generate_environment(seed=1617,outdir=data,train=40,router=12,validation=16,test=32,stress=32)
    report=run_environment(data,tmp_path/'policy',seed=1617,max_probes=160,audit_count=16)
    assert report['provider_model_calls']==0
    assert report['matched_probe_evidence_for_tabular']
    assert report['queries_in_active_adaptation']>0
    assert all(p['ready'] for p in report['audits_by_operation'].values())
    for split in ('test','stress'):
        assert report['scores'][split]['structured_program']['accuracy_all_cases']>=.99
        assert report['scores'][split]['structured_program']['false_accepts']==0
    p=json.loads((tmp_path/'policy'/'MODEL_REFINER_PROGRAM_MEMORY.json').read_text())
    for op,row in p['policies'].items():
        assert Program.from_dict(row).operation==op


def test_full_reproduces_and_tamper_detection(tmp_path):
    root=tmp_path/'run'
    args=['full','--environments','1','--seed','4421',
          '--train','16','--router','8','--validation','8','--test','16','--stress','16',
          '--audit-samples','8','--ood-seeds','0','--out',str(root)]
    assert structured_main(args)==0
    assert structured_main(['verify','--root',str(root)])==0
    with pytest.raises(FileExistsError):structured_main(args)
    with (root/'env1_policy'/'MODEL_REFINER_PROGRAM_MEMORY.json').open('a') as f:f.write('corrupt')
    with pytest.raises(ValueError,match='program integrity mismatch'):
        structured_main(['verify','--root',str(root)])


def test_optional_native_model_refiner_dryrun_is_no_network(tmp_path,monkeypatch):
    from simlearn.model_assisted import run
    data=tmp_path/'data'
    freeze=generate_environment(seed=781,outdir=data,train=16,router=8,validation=8,test=8,stress=8)
    run_environment(data,tmp_path/'policy',seed=781,max_probes=160,audit_count=8)
    import simlearn.model_assisted as ma
    monkeypatch.setattr(ma,'OpenAIResponsesBackend',lambda **kw: (_ for _ in ()).throw(AssertionError('backend created')))
    plan=run(data_dir=data,memory_path=tmp_path/'policy'/'MODEL_REFINER_PROGRAM_MEMORY.json',
             outdir=tmp_path/'would_not_create',limit=8)
    assert plan['max_provider_attempts_without_repair']==16
    assert plan['provider_calls']==0
    assert not (tmp_path/'would_not_create').exists()


def test_native_model_refiner_predictor_uses_program_recipe_with_mock(tmp_path):
    from simlearn.model_assisted import run
    from simlearn.refiner_runner import MockERPBackend
    data=tmp_path/'data'
    generate_environment(seed=957,outdir=data,train=16,router=8,validation=8,test=8,stress=8)
    run_environment(data,tmp_path/'policy',seed=957,max_probes=160,audit_count=8)
    result=run(data_dir=data,memory_path=tmp_path/'policy'/'MODEL_REFINER_PROGRAM_MEMORY.json',
               outdir=tmp_path/'mock',limit=8,backend=MockERPBackend())
    assert result['model_provider_attempts']==16
    assert result['native_model_refiner_fit_performed'] is False
    assert result['inference_uses_native_model_refiner_Predictor_and_Recipe'] is True
    assert result['metrics']['policy_verified']['false_accepts']==0
    assert all(r['policy_verified']['tenant_binding_matched'] for r in result['inference_receipts'])


def test_native_model_refiner_requires_explicit_key_before_call(tmp_path,monkeypatch):
    from simlearn.model_assisted import run
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    data=tmp_path/'data'
    generate_environment(seed=907,outdir=data,train=16,router=8,validation=8,test=8,stress=8)
    run_environment(data,tmp_path/'policy',seed=907,max_probes=160,audit_count=8)
    with pytest.raises(ValueError,match='OPENAI_API_KEY'):
        run(data_dir=data,memory_path=tmp_path/'policy'/'MODEL_REFINER_PROGRAM_MEMORY.json',
            outdir=tmp_path/'blocked',limit=8,execute_model=True)
    assert not (tmp_path/'blocked').exists()


def test_wrong_tenant_program_memory_cannot_be_run(tmp_path):
    from simlearn.model_assisted import plan
    data=tmp_path/'data'
    generate_environment(seed=141,outdir=data,train=16,router=8,validation=8,test=8,stress=8)
    other=tmp_path/'other'
    generate_environment(seed=142,outdir=other,train=16,router=8,validation=8,test=8,stress=8)
    run_environment(data,tmp_path/'policy',seed=141,max_probes=160,audit_count=8)
    with pytest.raises(ValueError,match='different synthetic tenant'):
        plan(other,tmp_path/'policy'/'MODEL_REFINER_PROGRAM_MEMORY.json',limit=8)
