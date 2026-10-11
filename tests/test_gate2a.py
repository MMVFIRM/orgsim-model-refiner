"""Gate 2A synthetic oracle: discovery isolation, manifested transitions, API safety."""
from copy import deepcopy
import json
import pytest
from orgsim.engine import SimulationEngine, SimError
from orgsim.gate2a_lab import (
    public_operations, reference_state, features_for, candidate_atoms, fit_model,
    predict_model, run_gate, _SyntheticOracle, _make_pool, run_arm, _scores,
)
from orgsim.gate2a_export import model_to_manifest
from orgsim.gate2b_readonly import build_readonly_plan, run_readonly
from orgsim.dolibarr_gate import DolibarrClient


def test_public_examples_are_valid_across_secret_settings():
    # A valid recorded example is included in both arms' observation budgets.
    for seed in range(40):
        oracle=_SyntheticOracle(seed)
        for op in public_operations():
            status,after,_=oracle.query(op,reference_state(op))
            assert status==200 and after!=reference_state(op)


def test_candidate_grammar_is_public_and_expresses_relational_features():
    op=public_operations()[0]
    features=features_for(op,reference_state(op))
    assert features['stock_minus_qty']==12
    assert any(a.field=='stock_minus_qty' and a.value==1 and a.relation=='ge'
               for a in candidate_atoms(op))


def test_holdout_is_rejected_by_evidence_ledger():
    from orgsim.exploration import EvidenceLedger
    led=EvidenceLedger()
    try:
        with pytest.raises(ValueError):
            led.record(operation='orders.validate',args={},before={},after={},status=200,
                       response={},source='synthetic_fixture',phase='holdout')
    finally:led.close()


def test_effect_induction_recovers_numeric_offset_with_two_successes():
    op=next(x for x in public_operations() if x.name=='invoices.payment')
    oracle=_SyntheticOracle(3)
    observations=[]
    for pay in [10,20,30]:
        s=reference_state(op);s['payment']=pay
        code,after,_=oracle.query(op,s)
        assert code==200
        observations.append({'before':s,'after':after,'features':features_for(op,s),'accepted':True})
    model=fit_model(op,observations+[{'before':dict(reference_state(op),status='paid'),
                                     'after':dict(reference_state(op),status='paid'),
                                     'features':features_for(op,dict(reference_state(op),status='paid')),
                                     'accepted':False}])
    assert model['effects']['due']['kind']=='subtract'
    assert model['effects']['payment_count']=={'kind':'offset','from':'payment_count','delta':1}


def test_manifest_compiler_validates_and_runs_predicted_workflow():
    op=public_operations()[0]
    proposal={'rule':[{'field':'status','op':'eq','value':'draft'},
                      {'field':'stock_minus_qty','op':'ge','value':0},
                      {'field':'credit_minus_amount','op':'ge','value':0}],
              'effects':{'status':{'kind':'literal','value':'confirmed'},
                         'stock':{'kind':'subtract','from':'stock','source':'qty'}}}
    spec=model_to_manifest(op,proposal)
    assert spec['reviewed'] is False
    assert spec['systems']['lab']['workflows']['run_action']['evidence']=='hypothesis'
    e=SimulationEngine(spec,allow_draft=True)
    try:
        good=reference_state(op)
        rid=e.call('lab.cases.create',good,role='test_agent')['result']['id']
        result=e.call('lab.workflow.run_action',{'id':str(rid)},role='test_agent')
        got=e.get('lab','cases',str(rid))
        assert got['status']=='confirmed' and got['stock']==12
        bad=reference_state(op);bad['stock']=0
        bid=e.call('lab.cases.create',bad,role='test_agent')['result']['id']
        audits=len(e.audit())
        with pytest.raises(SimError):
            e.call('lab.workflow.run_action',{'id':str(bid)},role='test_agent')
        assert e.get('lab','cases',str(bid))['status']=='draft'
        assert e.get('lab','cases',str(bid))['stock']==0
        assert len(e.audit())==audits
    finally:e.close()


def test_unsupported_learned_effect_not_silently_accepted():
    op=public_operations()[0]
    with pytest.raises(ValueError,match='Effects are unidentifiable'):
        model_to_manifest(op,{'rule':[{'field':'status','op':'eq','value':'draft'}],
                              'effects':{'status':{'kind':'literal','value':'confirmed'}}})
    with pytest.raises(ValueError,match='no identifiable'):
        model_to_manifest(op,{'rule':None,'effects':{}})


def test_budget_match_and_offline_holdout_only(tmp_path):
    g=run_gate(seeds=2,budget=8,holdout_size=70,outdir=tmp_path)
    assert g['live_calls']==0 and g['total_discovery_queries_per_arm']==80
    by=g['by_operation']; assert len(by)==10
    raw=(tmp_path/'report.json').read_text()
    assert 'private_rules' not in raw
    rows=json.loads((tmp_path/'per_seed_metrics.json').read_text())
    assert len(rows)==20 and all(x['queries']==8 for x in rows)
    assert all(x['coverage']<=1.0 for x in rows)
    assert all('before' not in x and 'after' not in x for x in rows)


def test_readonly_probe_plan_limits_and_no_credentials(tmp_path):
    plan=build_readonly_plan({'thirdparty_id':6,'product_ref':'P-TEST'},limit=2)
    assert len(plan)==6 and all(p['method']=='GET' for p in plan)
    for bad in ({'product_ref':'x/../secrets'},{'thirdparty_id':'1?admin=1'},{'host':'other'}):
        with pytest.raises(ValueError):build_readonly_plan(bad)
    class NoNetworkTransport:
        def open(self,*args,**kwargs):raise AssertionError('Dry-run accessed network')
    client=DolibarrClient('http://127.0.0.1:8088','secret',transport=NoNetworkTransport())
    report=run_readonly(client,plan,execute=False,outdir=tmp_path)
    assert report['operations_executed']==0 and report['writes_executed']==0
    assert 'secret' not in (tmp_path/'discovery_report.json').read_text()
    with pytest.raises(ValueError):run_readonly(client,[{'method':'POST','category':'collection'}],execute=False,outdir=tmp_path)
    with pytest.raises(ValueError):run_readonly(DolibarrClient('https://example.org','secret'),plan,execute=False,outdir=tmp_path)


def test_execute_readonly_no_writes_and_no_response_values(tmp_path):
    class StubClient:
        loopback=True
        def __init__(self):self.calls=[]
        def request(self,method,path):
            self.calls.append((method,path));return 200,[{'id':'1','name':'PRIVATE-SYNTHETIC'}]
    fake=StubClient()
    p=build_readonly_plan(limit=2)
    report=run_readonly(fake,p,execute=True,outdir=tmp_path)
    assert len(fake.calls)==4 and all(method=='GET' for method,_ in fake.calls)
    assert report['operations_executed']==4
    assert 'PRIVATE-SYNTHETIC' not in (tmp_path/'discovery_report.json').read_text()


def test_learned_workflow_compiles_and_reports_unsafe_false_accepts(tmp_path):
    from orgsim.gate2a_compile_smoke import smoke
    report=smoke(seed=20261231,budget=20,withheld=70,out=tmp_path)
    assert report['live_dolibarr_calls']==0
    assert report['true_positives']==report['effects_correct_given_true_positives']
    assert report['correct_accept']+report['correct_reject']+report['false_accept']+report['false_reject']==70
    assert report['safety_gate'] in {'FAIL_FALSE_ACCEPTS','NO_FALSE_ACCEPTS_IN_THIS_SMALL_TEST'}
    manifest=json.loads((tmp_path/'learned_unreviewed_manifest.json').read_text())
    assert manifest['reviewed'] is False
