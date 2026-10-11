"""Gate 2C counterexample search, shield, model-validation, runtime invariants."""
from copy import deepcopy
import json
import pytest

from orgsim.gate2a_lab import _SyntheticOracle,public_operations,reference_state
from orgsim.gate2c_counterexamples import (
    _InteractionOracle,_candidate_axes,_applies,fit_audited_model,run_cegis_arm,
    run_gate2c,predict_audited,
)
from orgsim.gate2c_export import certified_synthetic_manifest
from orgsim.gate2c_model_loop import (
    public_prompt, validate_model_proposal, safeguarded_prediction,
    model_gate, main as model_main,
)
from orgsim.gate2c_runtime import replay
from orgsim.engine import SimulationEngine, SimError


def test_critical_status_not_omitted_as_output_field():
    axes={a.name for a in _candidate_axes(public_operations()[0])}
    assert {'status','stock_minus_qty','credit_minus_amount'} <= axes


def test_counterfactual_audit_recovers_conjunctions_on_new_secret_thresholds():
    for idx,op in enumerate(public_operations()):
        result=run_cegis_arm(op,_SyntheticOracle(55221),budget=20,seed=idx+30)
        audit=result['model']['audit']
        assert audit['ready_for_offline_execution'], (op.name,audit)
        assert audit['training_conflicts']==0
        assert audit['unidentified_effects']==[]
        assert all(x['status'] in {'enumerated','adjacent_boundary','all_values_observed'}
                   for x in audit['critical_axes'])


def test_incomplete_evidence_abstains_instead_of_executing():
    op=public_operations()[0]
    r=run_cegis_arm(op,_SyntheticOracle(55221),budget=8,seed=13)
    assert r['model']['audit']['ready_for_offline_execution'] is False
    assert predict_audited(op,r['model'],reference_state(op))['decision']=='abstain'
    with pytest.raises(ValueError,match='Unverified critical dimensions'):
        certified_synthetic_manifest(op,r['model'])


def test_discovered_cross_feature_conflict_invalidates_audit():
    op=next(op for op in public_operations() if op.name=='orders.validate')
    r=run_cegis_arm(op,_InteractionOracle(55221),budget=32,seed=104)
    assert r['model']['audit']['training_conflicts']>0
    assert r['model']['audit']['ready_for_offline_execution'] is False
    assert predict_audited(op,r['model'],reference_state(op))['decision']=='abstain'


def test_certified_export_runs_only_synthetic_and_preserves_policy_on_rejection():
    op=next(x for x in public_operations() if x.name=='orders.validate')
    model=run_cegis_arm(op,_SyntheticOracle(88117),budget=20,seed=111)['model']
    spec=certified_synthetic_manifest(op,model)
    assert spec['reviewed'] is False
    w=spec['systems']['lab']['workflows']['run_action']
    assert w['evidence']=='hypothesis'
    assert w['roles']==['test_agent']
    sim=SimulationEngine(spec,allow_draft=True)
    try:
        state=reference_state(op)
        rec_id=str(sim.call('lab.cases.create',state,role='test_agent')['result']['id'])
        sim.call('lab.workflow.run_action',{'id':rec_id},role='test_agent')
        assert sim.get('lab','cases',rec_id)['status']=='confirmed'
        bad=dict(state,status='closed')
        rid=str(sim.call('lab.cases.create',bad,role='test_agent')['result']['id'])
        with pytest.raises(SimError):
            sim.call('lab.workflow.run_action',{'id':rid},role='test_agent')
        assert sim.get('lab','cases',rid)['status']=='closed'
        with pytest.raises(SimError):
            sim.call('lab.workflow.run_action',{'id':rec_id},role='readonly')
    finally:
        sim.close()


def test_offline_confirmatory_synthetic_gate_and_adversarial_abstention(tmp_path):
    basic=run_gate2c(seeds=2,budgets=(8,20,32),holdout_size=80,freeze_seed=20320913,
                     outdir=tmp_path/'simple')
    assert basic['live_dolibarr_calls']==0
    assert basic['safety_gate']=='PASS_OFFLINE_SYNTHETIC_ONLY'
    assert basic['overall'][0]['mean_coverage']==0
    assert all(r['total_false_accepts']==0 for r in basic['overall'])
    assert basic['overall'][-1]['aggregate_positive_accept_recall']>=.8
    adverse=run_gate2c(seeds=2,budgets=(20,32),holdout_size=80,freeze_seed=20320913,
                       outdir=tmp_path/'adversarial',oracle_family='interaction')
    assert adverse['live_dolibarr_calls']==0
    assert adverse['overall'][-1]['total_false_accepts']==0
    assert adverse['overall'][-1]['mean_coverage']<.8
    assert adverse['safety_gate']=='FAIL'  # fail closed != successful reconstruction


def test_compiled_synthetic_runtime_gate(tmp_path):
    doc=replay(seeds=(20271323,),budget=20,heldout=80,outdir=tmp_path)
    assert doc['pass_offline_synthetic']
    assert doc['false_accepts']==doc['false_rejects']==0
    assert doc['true_accepts']==doc['effects_correct_on_accepted_truth']
    assert doc['all_manifest_reviewed_false']
    assert doc['live_dolibarr_calls']==0


def test_model_prompt_leaks_no_hidden_oracle_rules_or_ids():
    op=public_operations()[0]
    result=run_cegis_arm(op,_SyntheticOracle(993),budget=20,seed=24)
    txt=json.dumps(public_prompt(op,result['observations']))
    assert 'private_rules' not in txt
    assert 'record_id' not in txt
    assert 'ORGSIM_ADMIN_TOKEN' not in txt
    assert 'DOLAPIKEY' not in txt
    assert 'heldout_states' not in txt


def test_model_output_validation_rejects_unregistered_features_and_bad_effects():
    op=public_operations()[0]
    observed=run_cegis_arm(op,_SyntheticOracle(2212),budget=20,seed=2)['observations']
    bad={'rule':[{'field':'secret_admin','op':'eq','value':True}],
         'effects':{'status':{'kind':'literal','value':'confirmed'},
                    'stock':{'kind':'subtract','from':'stock','source':'qty'}}}
    with pytest.raises(ValueError,match='unregistered'):
        validate_model_proposal(op,bad,observed)
    bad['rule']=[{'field':'status','op':'eq','value':'draft'}]
    with pytest.raises(ValueError,match='contradicts'):
        validate_model_proposal(op,bad,observed)


def test_model_cannot_override_risk_shield():
    op=public_operations()[0]
    model=run_cegis_arm(op,_SyntheticOracle(1177),budget=8,seed=2)['model']
    forged={'rule':[{'field':'status','op':'eq','value':'draft'}],
            'effects':{'status':{'kind':'literal','value':'confirmed'},
                       'stock':{'kind':'subtract','from':'stock','source':'qty'}}}
    assert safeguarded_prediction(op,forged,model,reference_state(op))['decision']=='abstain'


def test_model_loop_with_test_double_is_not_reported_as_real_model(tmp_path):
    class IntentionallyWrongModel:
        def complete_json(self,prompt):
            return {'rule':[{'field':'nonexistent','op':'eq','value':1}],
                    'effects':{},'uncertainties':[]}
    doc=model_gate(IntentionallyWrongModel(),seeds=1,budget=20,heldout=80,
                   freeze_seed=20270930,outdir=tmp_path)
    assert doc['real_model_calls']==0 and doc['backend_invocations']==10
    assert all(v['false_accepts']==0 for v in doc['by_arm'].values())
    assert all(v['model_valid']==0 for v in doc['by_arm'].values())
    assert not (tmp_path/'oracle_private_rules.json').exists()


def test_model_runner_dryrun_never_contacts_provider(capsys):
    assert model_main([])==0
    assert 'DRY RUN' in capsys.readouterr().out


def test_gate2c_dolibarr_fixed_get_only_plan_and_no_secret_values(tmp_path):
    from orgsim.gate2c_dolibarr_readonly import bounded_differential_plan,differential_readonly
    plan=bounded_differential_plan({'thirdparty_id':2,'product_ref':'P-TEST'},budget=20)
    assert len(plan)<=20 and all(p['method']=='GET' for p in plan)
    class FakeClient:
        loopback=True
        def __init__(self):self.calls=[]
        def request(self,method,path):
            self.calls.append((method,path))
            return 200,[{'id':'2','name':'ULTRA_PRIVATE_FIELD_VALUE'}]
    fake=FakeClient()
    result=differential_readonly(fake,plan,execute=True,outdir=tmp_path)
    assert result['writes_executed']==0
    assert len(fake.calls)==len(plan)
    assert result['comparisons']
    assert 'ULTRA_PRIVATE_FIELD_VALUE' not in (tmp_path/'discovery_report.json').read_text()
    assert all(m=='GET' for m,_ in fake.calls)
    with pytest.raises(ValueError,match='bounded GET'):
        differential_readonly(fake,[{'method':'POST','path':'/thirdparties','category':'collection'}],outdir=tmp_path)
    with pytest.raises(ValueError,match='Unapproved'):
        differential_readonly(fake,[{'method':'GET','path':'/users/info','category':'discovery_variant'}],outdir=tmp_path)


def test_dolibarr_gate2c_dryrun_makes_zero_network_requests(tmp_path):
    from orgsim.gate2c_dolibarr_readonly import bounded_differential_plan,differential_readonly
    class NoNetwork:
        loopback=True
        def request(self,*args,**kwargs):raise AssertionError('Dry-run contacted ERP')
    result=differential_readonly(NoNetwork(),bounded_differential_plan(budget=16),
                                 execute=False,outdir=tmp_path)
    assert result['operations_executed']==0
    assert result['evidence']=='DRY_RUN'
