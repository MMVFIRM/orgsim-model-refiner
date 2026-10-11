"""Synthetic-only verification of Franzen-style OrgSim capabilities.

No external ERP, no heldout data, no requirements on Dolibarr semantics.
"""
from copy import deepcopy
import json

import pytest

from orgsim.dolibarr_gate import _path_matches_tool
from orgsim.engine import SimulationEngine, SimError
from orgsim.exploration import (
    EvidenceLedger, catalog_from_swagger, classify_operation, normalize_state,
    prioritize_probes, shortest_verified_plan, state_signatures,
)
from orgsim.franzen_gate import analyze, make_reconstruction_brief
from orgsim.spec import SpecError, build_registry, validate_spec


def schema(properties, required=()):
    return {'type':'object', 'properties':{'id':{'type':'string'}, **properties},
            'required':['id', *required], 'additionalProperties':False}


def lab_spec():
    return {
        'version':'0.1','organization':'Synthetic unknown-ERP workflow oracle',
        'reviewed':False,
        'systems':{'erp':{
            'resources':{
                'customers':{'schema':schema({'name':{'type':'string'}}, ['name']),
                             'allow':{'create':['test_agent'], 'get':['test_agent']}},
                'products':{'schema':schema({'ref':{'type':'string'},'stock':{'type':'integer','minimum':0}}, ['ref','stock']),
                            'allow':{'create':['test_agent'], 'get':['test_agent'], 'list':['test_agent']},
                            'response_projection':{'get':{'include_base':False,'fields':{
                                'ref':{'path':'record.ref'}, 'stock_left':{'path':'record.stock'}}}}},
                'orders':{'schema':schema({'customer_id':{'type':'string'},'product_id':{'type':'string'},
                    'qty':{'type':'integer','minimum':1},'status':{'type':'string'}}, ['customer_id','product_id','qty','status']),
                    'foreign_keys':[{'field':'customer_id','system':'erp','resource':'customers'},
                                    {'field':'product_id','system':'erp','resource':'products'}],
                    'allow':{'create':['test_agent'], 'get':['test_agent']}},
                'invoices':{'schema':schema({'order_id':{'type':'string'},'status':{'type':'string'}},['order_id','status']),
                    'foreign_keys':[{'field':'order_id','system':'erp','resource':'orders'}],
                    'allow':{'get':['test_agent']}}
            },
            'lookups':{
                'product_by_ref':{'resource':'products','field':'ref','arg':'ref',
                                  'roles':['test_agent'],'http':{'method':'GET','path':'/products/ref/{ref}'},
                                  'evidence':'hypothesis'}},
            'workflows':{
                'validate_order':{'description':'Validate order and reserve stock (SYNTHETIC ONLY)',
                    'roles':['test_agent'],'evidence':'hypothesis',
                    'http':{'method':'POST','path':'/orders/{id}/validate'},
                    'input_schema':{'type':'object','properties':{'id':{'type':'string'}},
                                    'required':['id'], 'additionalProperties':False},
                    'preconditions':[{'compare':{'left':{'lookup':{'system':'erp','resource':'orders',
                                        'id':{'path':'input.id'}, 'field':'status'}},'op':'eq','right':{'literal':'draft'}}}],
                    'steps':[
                      {'kind':'update','system':'erp','resource':'orders','id':{'path':'input.id'},
                       'values':{'status':{'literal':'validated'}},
                       'require':[{'compare':{'left':{'path':'record.status'},'op':'eq','right':{'literal':'draft'}}}]},
                      {'kind':'adjust','system':'erp','resource':'products',
                       'id':{'lookup':{'system':'erp','resource':'orders','id':{'path':'input.id'},'field':'product_id'}},
                       'field':'stock','by':{'negate':{'lookup':{'system':'erp','resource':'orders',
                                                       'id':{'path':'input.id'},'field':'qty'}}}},
                    ],
                    'result':{'path':'steps.0.after.status'},
                    'response_contract':{'invoke':{'status':200,'kind':'string'}}},
                'invoice_from_order':{'roles':['test_agent'],'evidence':'hypothesis',
                    'http':{'method':'POST','path':'/invoices/createfromorder/{orderid}'},
                    'input_schema':{'type':'object','properties':{'id':{'type':'string'}},
                                    'required':['id']},
                    'preconditions':[{'compare':{
                        'left':{'lookup':{'system':'erp','resource':'orders','id':{'path':'input.id'}, 'field':'status'}},
                        'op':'eq','right':{'literal':'validated'}}}],
                    'steps':[{'kind':'create','system':'erp','resource':'invoices',
                              'values':{'order_id':{'path':'input.id'},'status':{'literal':'draft'}}}],
                    'result':{'path':'after.id'}}
            }
        }}, 'integrations':[], 'scenarios':[]
    }


def setup(engine, *, stock, quantity=4):
    engine.call('erp.customers.create', {'id':'1','name':'Synthetic Customer'}, role='test_agent')
    engine.call('erp.products.create', {'id':'2','ref':'SYN-001','stock':stock}, role='test_agent')
    engine.call('erp.orders.create', {'id':'3','customer_id':'1','product_id':'2',
                                     'qty':quantity,'status':'draft'}, role='test_agent')


def test_workflow_success_atomic_and_event_audit():
    spec=lab_spec()
    eng=SimulationEngine(spec, allow_draft=True)
    setup(eng, stock=10)
    try:
        result=eng.call('erp.workflow.validate_order',{'id':'3'},role='test_agent')
        assert result['result']=='validated'
        assert result['evidence']=='hypothesis' and result['steps_executed']==2
        assert eng.get('erp','orders','3')['status']=='validated'
        assert eng.get('erp','products','2')['stock']==6
        assert any(x['name']=='erp.orders.updated' for x in eng.events()) is False  # no integrations
        inv=eng.call('erp.workflow.invoice_from_order',{'id':'3'},role='test_agent')
        assert inv['result'].startswith('erp_invoices_')
        assert eng.get('erp','invoices',inv['result'])['order_id']=='3'
        assert all(r['ok'] for r in eng.tool_trace())
    finally:eng.close()


def test_rollback_when_second_step_overdraws_stock():
    eng=SimulationEngine(lab_spec(), allow_draft=True)
    setup(eng, stock=2)
    prior_audits=len(eng.audit())
    try:
        with pytest.raises(SimError) as exc:
            eng.call('erp.workflow.validate_order', {'id':'3'}, role='test_agent')
        assert exc.value.code=='SCHEMA_VIOLATION'
        assert eng.get('erp','orders','3')['status']=='draft'
        assert eng.get('erp','products','2')['stock']==2
        assert len(eng.audit())==prior_audits  # no partial workflow audit
        assert eng.tool_trace()[-1]['ok'] is False
    finally: eng.close()


def test_workflow_forbidden_and_unknown_actions():
    eng=SimulationEngine(lab_spec(), allow_draft=True)
    setup(eng, stock=6)
    try:
        for name, role, code in [('erp.workflow.validate_order','viewer','FORBIDDEN'),
                                  ('erp.workflow.imaginary','test_agent','UNKNOWN_TOOL')]:
            with pytest.raises(SimError) as exc:
                eng.call(name, {'id':'3'}, role=role)
            assert exc.value.code==code
        assert eng.get('erp','orders','3')['status']=='draft'
    finally:eng.close()


def test_invalid_workflow_dsl_refused_before_runtime():
    spec=lab_spec()
    spec['systems']['erp']['workflows']['validate_order']['steps'][0]['values']['evil']= {'literal':1}
    with pytest.raises(SpecError): validate_spec(spec, allow_draft=True)
    spec=lab_spec()
    spec['systems']['erp']['workflows']['validate_order']['steps']=[{'kind':'python','code':'import os'}]
    with pytest.raises(SpecError): validate_spec(spec, allow_draft=True)


def test_no_placeholder_workflow_compiles():
    spec=lab_spec()
    spec['systems']['erp']['workflows']['validate_order'].pop('steps')
    with pytest.raises(SpecError): validate_spec(spec, allow_draft=True)


def test_alternate_key_lookup_and_value_projection():
    eng=SimulationEngine(lab_spec(),allow_draft=True)
    setup(eng,stock=9)
    try:
        result=eng.call('erp.lookup.product_by_ref', {'ref':'SYN-001'},role='test_agent')
        assert result['result']=={'ref':'SYN-001','stock_left':9}
        with pytest.raises(SimError) as err:
            eng.call('erp.lookup.product_by_ref', {'ref':'INVALID'},role='test_agent')
        assert err.value.status == 404
        assert eng.call('erp.products.get',{'id':'2'},role='test_agent')['result']==result['result']
        tools=[t['name'] for t in build_registry(lab_spec())['tools']]
        assert 'erp.workflow.validate_order' in tools and 'erp.lookup.product_by_ref' in tools
    finally:eng.close()


def test_harness_route_binding_is_strict():
    sp=lab_spec()
    assert _path_matches_tool('POST','/orders/3/validate','erp.workflow.validate_order',sp) is False  # not Dolibarr namespace
    ds=deepcopy(sp); ds['systems']['dolibarr']=ds['systems'].pop('erp')
    assert _path_matches_tool('POST','/orders/3/validate','dolibarr.workflow.validate_order',ds)
    assert not _path_matches_tool('POST','/orders/3/close','dolibarr.workflow.validate_order',ds)
    assert not _path_matches_tool('DELETE','/orders/3/validate','dolibarr.workflow.validate_order',ds)
    assert _path_matches_tool('GET','/products/ref/SYN-001','dolibarr.lookup.product_by_ref',ds)
    assert not _path_matches_tool('POST','/products/ref/SYN-001','dolibarr.lookup.product_by_ref',ds)


def test_normalized_equivalence_preserves_business_state():
    a={'orders':[{'id':'120','name':'Synthetic A','status':'draft','date_creation':'foo','qty':3}]}
    b={'orders':[{'id':'901','name':'Synthetic B','status':'draft','date_creation':'bar','qty':3}]}
    c={'orders':[{'id':'901','name':'Synthetic B','status':'validated','date_creation':'bar','qty':3}]}
    assert state_signatures(a)['exact_sha256'] != state_signatures(b)['exact_sha256']
    assert state_signatures(a)['generalized_sha256']==state_signatures(b)['generalized_sha256']
    assert state_signatures(a)['generalized_sha256']!=state_signatures(c)['generalized_sha256']


def test_discovery_memory_conflict_verification_and_no_redundancy():
    led=EvidenceLedger()
    before={'orders':[{'id':'1','status':'draft'}]}
    after={'orders':[{'id':'1','status':'validated'}]}
    try:
        g=led.decision_gate('POST /orders/{id}/validate',before,planned_mutations=3)
        assert g['decision']=='verify' and g['reason']=='VERIFY_UNSEEN_STATE'
        led.record(operation='POST /orders/{id}/validate',args={'id':'1'},before=before,after=after,
                   status=200,response={'status':'validated'})
        assert led.redundant_probe('POST /orders/{id}/validate',before,{'id':'1'})
        g=led.decision_gate('POST /orders/{id}/validate', before, planned_mutations=3)
        assert g['reason']=='VERIFY_BEFORE_LONG_COMMIT'
        led.record(operation='POST /orders/{id}/validate',args={'id':'2'},before={
                    'orders':[{'id':'2','status':'draft'}]},after={'orders':[{'id':'2','status':'draft'}]},
                   status=400,response={'error':'synthetic rejection'})
        g=led.decision_gate('POST /orders/{id}/validate',before,planned_mutations=3)
        assert g['reason']=='VERIFY_CONTRADICTORY_EVIDENCE'
        with pytest.raises(ValueError):
            led.record(operation='GET /secret',args={},before={},after={},status=200,response={},phase='holdout')
    finally:led.close()


def test_hypotheses_falsified_not_automatically_verified():
    led=EvidenceLedger()
    try:
        led.add_hypothesis(key='h1',operation='POST /orders/{id}/close',expected_outcome='success')
        led.record(operation='POST /orders/{id}/close',args={'id':'1'},before={'status':'draft'},
                   after={'status':'draft'},status=400,response={})
        assert led.hypotheses()[0]['status']=='falsified'
        led.record(operation='POST /orders/{id}/close',args={'id':'1'},before={'status':'draft'},
                   after={'status':'closed'},status=200,response={})
        assert led.hypotheses()[0]['status']=='falsified'
    finally:led.close()


def test_safe_probe_selection_and_math_not_hallucinated():
    doc={'paths':{
        '/login':{'get':{'summary':'Login'}},
        '/users/1/setPassword':{'get':{'summary':'Set password'}},
        '/products':{'get':{'summary':'List products'},'post':{'summary':'Create product'}},
        '/products/ref/{ref}':{'get':{'summary':'Get by ref'}},
        '/orders/{id}/validate':{'post':{'summary':'Validate order'}},
        '/invoices':{'get':{'summary':'List invoices'}},
        '/thirdparties':{'get':{'summary':'List customers'}},
        '/setup/modules/{id}/enable':{'put':{'summary':'Danger'}}
    }}
    c=catalog_from_swagger(doc)
    plan=prioritize_probes(c,budget=8)
    assert all(p['method']=='GET' and '/login' not in p['path_template'] and '/users' not in p['path_template'] for p in plan['probes'])
    assert all(p['request_status']=='PROPOSED_NOT_EXECUTED' for p in plan['probes'])
    assert shortest_verified_plan('draft','validated',[{'action':'validate','from':'draft','to':'validated',
                        'evidence':'hypothesis'}])['status']=='unknown_transition_model'
    assert shortest_verified_plan('draft','validated',[{'action':'validate','from':'draft','to':'validated',
                        'evidence':'verified'}])['actions']==['validate']


def test_offline_gate_analyze_freeze(tmp_path):
    doc={'paths':{'/orders/{id}/validate':{'post':{'operationId':'validateOrder','summary':'Validate order'}},
                  '/products':{'get':{'operationId':'listProducts','summary':'List products'}}}}
    report=analyze(doc,lab_spec(),{'phase':'discovery','probes':[{'id':'x'}]},outdir=tmp_path)
    assert report['api_operations']==2 and report['proposed_workflows']==1
    data=json.loads((tmp_path/'model_reconstruction_brief.json').read_text())
    assert data['heldout_probes_available_to_model']==0
    assert all(not x['executable'] for x in data['workflow_blueprints'])
    assert json.loads((tmp_path/'FREEZE.json').read_text())['no_live_calls']
    with pytest.raises(ValueError):
        make_reconstruction_brief([],lab_spec(),{'phase':'holdout','probes':[]})


def test_observation_driven_guard_induction_requires_contrast():
    from orgsim.induction import infer_guard
    examples=[]
    for status in ['draft','draft','validated','validated','closed','closed']:
        examples.append({'features':{'record.status':status},
                         'outcome':'success' if status=='draft' else 'rejected'})
    found=infer_guard(examples,allowed_features=['record.status'])
    assert found['status']=='provisional_guard'
    assert found['training_accuracy']==1.0
    assert found['hypothesis']=={'compare':{'left':{'path':'record.status'},'op':'eq',
                                           'right':{'literal':'draft'}}}
    assert found['evidence']=='discovery_only_unverified'
    insufficient=infer_guard([{'features':{'record.status':'draft'},'outcome':'success'}],
                             allowed_features=['record.status'])
    assert insufficient['status']=='insufficient_contrast'
    with pytest.raises(ValueError):
        infer_guard(examples, allowed_features=['input.customer_email']) if False else infer_guard(examples, allowed_features=['email'])


def test_structural_transition_diff_exposes_paths_not_customer_values():
    from orgsim.exploration import structural_delta
    before={'orders':[{'id':'1','status':'draft','secret':'PRIVATE-CUSTOMER'}]}
    after={'orders':[{'id':'1','status':'validated','secret':'PRIVATE-CUSTOMER'}], 'invoices':[{'id':'2'}]}
    report=structural_delta(before,after)
    assert 'orders[0].status' in report['changed_paths']
    assert 'invoices' in report['added_paths']
    assert 'PRIVATE-CUSTOMER' not in json.dumps(report)
    assert report['values_recorded'] is False


def test_existing_candidate_readonly_identity_and_three_lookup_hypotheses():
    from pathlib import Path
    from orgsim.spec import load_spec
    path=Path(__file__).resolve().parents[1] / 'gate_dolibarr/gate2_reconstruction/model_candidate_v020_unreviewed.json'
    spec=load_spec(path)
    assert not spec['reviewed']
    eng=SimulationEngine(spec,allow_draft=True)
    try:
        ans=eng.call('dolibarr.thirdparties.list',role='orgsim_gate_readonly')
        assert ans['ok'] and ans['result']==[]
        with pytest.raises(SimError) as exc:
            eng.call('dolibarr.thirdparties.create',{'name':'Fictional'},role='orgsim_gate_readonly')
        assert exc.value.status==403
        assert all('orgsim_gate_readonly' in t['roles']
                   for t in build_registry(spec)['tools'] if t.get('kind')=='lookup')
    finally:eng.close()
