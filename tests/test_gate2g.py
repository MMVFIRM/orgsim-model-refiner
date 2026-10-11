"""Gate 2G: shared metric rubric, conditional inference, in-memory GET-only oracle."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit,parse_qs
import pytest

from orgsim.gate2g_readonly import (prepare,Evaluator,verify,QUERIES,BY_ID,
    _score,_metrics,_check_pred,_swagger_ops,_read_frozen)
from orgsim.conditional_read import ConditionalReadModel
from orgsim.interaction_learning import (run_interaction_arm,fit_interaction_model,
     predict_interaction,score_interaction)
from orgsim.gate2a_lab import public_operations,_make_pool
from orgsim.gate2c_counterexamples import _InteractionOracle
from orgsim.gate2g_causal import VariableVetoOracle,OutOfGrammarOracle,run as run_causal

ROOT=Path(__file__).resolve().parents[1]
DISCOVERY=ROOT/'gate_dolibarr/gate2f_inputs/gate2d_discovery_report.json'
SWAGGER=ROOT/'gate_dolibarr/gate2f_inputs/sanitized_live_swagger.json'
DEVELOPMENT=ROOT/'gate_dolibarr/gate2g_inputs/gate2f_development_results.json'


class SimulatedTenant:
    loopback=True
    def __init__(self):
        self.calls=[]
        self.data={
            'thirdparties':[{'id':str(6000+i),'name':f'SECRET_CUST_{i}',
                  'ref':f'SECRET_CREF_{i}','client':'1'} for i in range(3)],
            'products':[{'id':'7000','ref':'SECRET_PROD','label':'SECRET_LABEL',
                  'type':'0','price_ttc':'19.00'}],
            'orders':[{'id':str(8000+i),'socid':str(6000+i),'ref':f'SECRET_OREF_{i}',
                  'status':'1','total_ht':'12.00'} for i in range(2)],
            'invoices':[{'id':'9000','ref':'SECRET_IREF','status':'1','socid':'6000',
                  'lines':[{'id':'9999'}]}]
        }

    def request(self,method,path):
        assert method=='GET' and path.startswith('/') and not path.startswith('//')
        self.calls.append((method,path))
        route=urlsplit(path);sp=route.path.strip('/').split('/')
        family=sp[0];assert family in self.data
        args=parse_qs(route.query)
        if len(sp)==1:
            records=list(self.data[family])
            if args.get('category')==['1'] or args.get('status')==['cancelled']:
                records=[]
            if args.get('sortorder')==['DESC']:records.reverse()
            limit=int(args.get('limit',['20'])[0]);page=int(args.get('page',['0'])[0])
            records=records[page*limit:(page+1)*limit]
            if args.get('ids_only')==['1']:return 200,[x['id'] for x in records]
            if args.get('withLines')==['false']:
                records=[{k:v for k,v in rec.items() if k!='lines'} for rec in records]
            if 'properties' in args:
                props=args['properties'][0].split(',')
                records=[{k:rec[k] for k in props if k in rec} for rec in records]
            return 200,records
        if sp[1]=='ref':
            records=[r for r in self.data[family] if r.get('ref')==sp[2]]
            if len(sp)>3 and sp[3]=='variants':return 200,[]
            return (200,records[0]) if records else (404,{'error':'PRIVATE'})
        rec=next((x for x in self.data[family] if x['id']==sp[1]),None)
        if rec is None:return 404,{'error':'PRIVATE'}
        if len(sp)==2:
            rec=dict(rec)
            if args.get('withLines')==['false']:rec.pop('lines',None)
            return 200,rec
        name=sp[2]
        if name=='outstandingorders':
            return 200,{'opened':1,'refs':['SECRET_REF']}
        if name in ('supplier_categories','selling_multiprices'):
            return 403,{'error':{'private':'DO_NOT_LOG'}}
        if name=='fixedamountdiscounts':return 200,[]
        if name=='notifications':return 200,[]
        if name=='representatives':return 200,[]
        if name=='subproducts':return 200,[]
        if name=='contacts':return 200,[]
        return 404,{'error':'PRIVATE'}


def test_validated_dolibarr_get_templates_are_distinct_from_revealed_gate2f():
    assert len(QUERIES)==24 and len({q.route for q in QUERIES})==24
    c=_swagger_ops(json.loads(SWAGGER.read_text()))
    assert len(c)==24
    old=json.loads(DEVELOPMENT.read_text())
    old_routes={x['route_template'] for x in old['cases']}
    assert all(q.route not in old_routes for q in QUERIES)


def test_shared_truth_denominator_not_inflated_by_extra_predictions():
    truth={'status':200,'kind':'array','length':0,'field_types':{}}
    empty=_score({'kind':'array','status':200},truth)
    bad=_score({'kind':'array','status':200,'field_types':{'id':'string'}},truth)
    assert empty['observable_fields']==bad['observable_fields']==3
    assert empty['coverage']==bad['coverage']
    assert empty['excess_claims']==0 and bad['excess_claims']==1
    assert bad['wrong_claims']==1 and bad['utility'] < empty['utility']
    rows=[{'scores':{'a':empty,'b':bad}}]
    assert _metrics(rows,'a')['shared_denominator_observable']==_metrics(rows,'b')['shared_denominator_observable']
    assert _metrics(rows,'b')['correct_target_recall']==_metrics(rows,'a')['correct_target_recall']


def test_type_union_is_not_claimed_as_exact_information():
    q=BY_ID['customer_outstanding_mode']
    result=_check_pred(q,{'field_types':{'refs':['array','object']},'status':200})
    assert result['field_types']['refs']==['array','object']
    s=_score(result,{'status':200,'kind':'object','field_types':{'refs':'array'}})
    assert s['field_checks']['field_types.refs'] and s['information_score']<s['utility']
    with pytest.raises(ValueError):_check_pred(q,{'field_types':{'refs':['object','object']}})
    with pytest.raises(ValueError):_check_pred(q,{'field_types':{'refs':['array','string','object','boolean']}})


def test_conditional_abstains_from_fields_on_empty_filters_and_pagination():
    model=ConditionalReadModel(json.loads(DISCOVERY.read_text()), json.loads(DEVELOPMENT.read_text()))
    catalog=_swagger_ops(json.loads(SWAGGER.read_text()))
    empty=model.predict(BY_ID['invoices_status_cancelled'],catalog['invoices_status_cancelled'])
    assert 'field_types' not in empty and 'item_keys' not in empty
    page=model.predict(BY_ID['orders_page2'],catalog['orders_page2'])
    assert page['length']==0 and 'field_types' not in page and 'item_keys' not in page
    suppressed=model.predict(BY_ID['invoices_suppress_lines'],catalog['invoices_suppress_lines'])
    assert 'lines' not in suppressed['field_types'] and 'lines' not in suppressed['item_keys']
    detail=model.predict(BY_ID['invoice_detail_suppress_lines'],catalog['invoice_detail_suppress_lines'])
    assert 'lines' not in detail.get('field_types',{})
    poly=model.predict(BY_ID['customer_outstanding_mode'],catalog['customer_outstanding_mode'])
    assert set(poly['field_types']['refs'])=={'array','object'}


def test_preparation_three_arms_and_provenance_is_reproducible(tmp_path):
    f=tmp_path/'freeze'
    data=prepare(DISCOVERY,SWAGGER,f,development=DEVELOPMENT)
    assert data['planned_cases']==24
    assert data['arms']==['contract','symbolic','conditional']
    candidate,plan,fr=_read_frozen(f)
    assert candidate['source_development_sha256']==hashlib.sha256(DEVELOPMENT.read_bytes()).hexdigest()
    assert fr['model_provider_calls']==fr['live_reads']==fr['live_writes']==0
    assert plan['fixed_scoring_rubric'] is True
    with pytest.raises(FileExistsError):prepare(DISCOVERY,SWAGGER,f,development=DEVELOPMENT)


def test_dry_run_network_disabled_and_verification(tmp_path):
    frozen=tmp_path/'freeze';prepare(DISCOVERY,SWAGGER,frozen,development=DEVELOPMENT)
    class NoNetwork:
        loopback=True
        def request(self,*args):raise AssertionError('Network request in dry-run')
    report=Evaluator(NoNetwork(),frozen,budget=32).run(tmp_path/'preview',execute=False)
    assert report['operations_executed']==0
    assert verify(frozen,tmp_path/'preview')['status']=='PASS_OFFLINE_PREVIEW'


def test_full_simulated_live_read_frozen_chain_and_privacy(tmp_path):
    frozen=tmp_path/'freeze';prepare(DISCOVERY,SWAGGER,frozen,development=DEVELOPMENT)
    fake=SimulatedTenant()
    report=Evaluator(fake,frozen,budget=32).run(tmp_path/'evaluation',execute=True)
    assert report['cases_scored']==24
    assert report['operations_executed']==28 and len(fake.calls)==28
    assert report['live_writes']==0 and report['model_provider_calls_during_evaluation']==0
    assert verify(frozen,tmp_path/'evaluation')['reads_verified']==28
    assert all(q=='GET' for q,_ in fake.calls)
    for text in (tmp_path/'evaluation'/'holdout_report.json').read_text(),(tmp_path/'evaluation'/'events.jsonl').read_text():
        assert 'SECRET_' not in text and 'DO_NOT_LOG' not in text
        assert '6000' not in text and '8000' not in text and '9000' not in text
    metric=report['scores'];d={a:x['shared_denominator_observable'] for a,x in metric.items()}
    assert len(set(d.values()))==1
    assert metric['conditional']['excess_claims']<=metric['symbolic']['excess_claims']


def test_model_prepare_mock_and_evaluate_not_called_again(tmp_path):
    class Advisor:
        calls=0
        def predict_many(self,queries,evidence):
            self.calls+=1
            assert len(queries)==24
            assert 'SECRET' not in json.dumps(evidence)
            return {q.id:{'status':200} for q in queries}
    advisor=Advisor();f=tmp_path/'f'
    prepare(DISCOVERY,SWAGGER,f,model_advisor=advisor,development=DEVELOPMENT)
    report=Evaluator(SimulatedTenant(),f,budget=32).run(tmp_path/'run',execute=True)
    assert advisor.calls==1 and report['model_provider_calls_during_evaluation']==0
    assert set(report['arms'])=={'contract','symbolic','conditional','model'}
    assert verify(f,tmp_path/'run')['reads_verified']==28


def test_freeze_and_event_integrity_tampering_detected(tmp_path):
    f=tmp_path/'f';prepare(DISCOVERY,SWAGGER,f,development=DEVELOPMENT)
    Evaluator(SimulatedTenant(),f,budget=32).run(tmp_path/'run',execute=True)
    candidate=f/'candidate_frozen.json';bytes_=candidate.read_bytes();candidate.write_bytes(bytes_+b' ')
    with pytest.raises(ValueError):verify(f,tmp_path/'run')
    candidate.write_bytes(bytes_)
    ev=tmp_path/'run'/'events.jsonl'
    ev.write_bytes(ev.read_bytes().replace(b'prediction_committed',b'prediction_uncommitted',1))
    with pytest.raises(ValueError):verify(f,tmp_path/'run')


def test_hidden_interaction_generative_learning_and_abstention():
    op=public_operations()[0];o=_InteractionOracle(2047)
    run=run_interaction_arm(op,o,budget=48,seed=97)
    assert run['queries']<=48
    assert run['model']['reviewed'] is False
    pool=_make_pool(op,881,n=160)
    truth=[(s,*o.query(op,s,phase='evaluation')[:2]) for s in pool]
    result=score_interaction(op,run['model'],truth)
    assert result['false_accepts']==0 and result['effect_exact']==1.0
    # A rule outside the declared grammar should not be automatically trusted.
    adversary=OutOfGrammarOracle(2047)
    r=run_interaction_arm(op,adversary,budget=48,seed=97)
    assert r['model']['candidate_count']==0 or not r['model']['audited'] or r['model']['reviewed'] is False


def test_offline_causal_comparison_retains_failed_out_of_grammar_gate(tmp_path):
    r=run_causal(seeds=2,budget=48,holdout=80,freeze_seed=20291204,outdir=tmp_path/'causal')
    assert r['live_reads']==r['live_writes']==0 and r['model_in_loop'] is False
    assert len(r['results'])==9
    for family in ('known_interaction','variable_threshold','outside_grammar'):
        rows=[x for x in r['results'] if x['family']==family]
        assert set(x['arm'] for x in rows)=={'uniform','counterexample_conjunction','interaction_audited'}
        assert len(set(x['cases'] for x in rows))==1
    inside=[x for x in r['results'] if x['family']=='known_interaction' and x['arm']=='interaction_audited'][0]
    assert inside['false_accepts']==0 and inside['coverage']>=.8
    outside=[x for x in r['results'] if x['family']=='outside_grammar' and x['arm']=='interaction_audited'][0]
    assert outside['gate']=='FAIL'
    folder=tmp_path/'causal'
    freeze=json.loads((folder/'CAUSAL_FREEZE.json').read_text())
    assert freeze['report_sha256']==hashlib.sha256((folder/'causal_report.json').read_bytes()).hexdigest()
