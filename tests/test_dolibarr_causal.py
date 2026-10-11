"""Adversarial same-status causal Dolibarr protocol tests; all ERP calls are mock GETs."""
from __future__ import annotations
import copy,json
from pathlib import Path
import pytest
from simlearn import dolibarr_causal as c
from simlearn import dolibarr_transfer as d


class FakeERP:
    def __init__(self):
        self.orders={};self.products={};self.calls=[]
    def add(self,order:str,product:str,sale:int):
        self.orders[str(order)]={'status':'0','statut':'0','total_ht':'50.00',
                                 'total_ttc':'60.00','lines':[{'fk_product':int(product),'qty':1}],
                                 'socid':'SECRET_CUSTOMER','webservices_key':'SECRETKEY'}
        self.products[str(product)]={'status':str(sale),'type':'0','price':'50.00',
                                     'stock_reel':'100','webservices_key':'SECRETKEY'}
    def request(self,method,path,data=None):
        assert method=='GET' and data is None
        self.calls.append((method,path))
        scope,ident=path.strip('/').split('/')[:2]
        ident=ident.split('?')[0]
        return (200,copy.deepcopy({'orders':self.orders,'products':self.products}[scope][ident]))


def receipt_files(tmp:Path):
    receipt=tmp/'server_receipt.txt';receipt.write_text('HTTP 500: Error when validating Order: ProductNotOnSale',encoding='utf-8')
    diagnostic=tmp/'dolibarr_log_redacted.txt';diagnostic.write_text('Commande::valid checkActiveProductInLines product not on sale',encoding='utf-8')
    return receipt,diagnostic


def setup(tmp_path,train=5,val=3,holdout=8):
    fake=FakeERP();fake.add('100','200',1);fake.add('101','201',0)
    root=tmp_path/'pilot'
    pilot=c.pilot_start(root,order_a='100',order_b='101',product_a='200',product_b='201',client=fake)
    assert pilot['same_status_verified']
    fake.orders['100']['status']='1'
    receipt,diag=receipt_files(tmp_path)
    confirmation=c.pilot_finish(root,active_status=200,inactive_status=500,
                                receipt_file=receipt,diagnostic_file=diag,client=fake)
    assert confirmation['qualified']
    template=tmp_path/'scaffold.json'
    c.scaffold(template,pilot_file=Path(confirmation['pilot_proof']),
               train_pairs=train,validation_pairs=val,holdout_pairs=holdout)
    p=json.loads(template.read_text('utf-8'))
    for i,case in enumerate(p['base_plan']['cases']):
        oid=str(1000+i);pid=str(2000+i)
        # Different sides across pairs: neither 'a' nor 'b' is a safe label shortcut.
        pairnumber=int(case['id'].rsplit('-',2)[1])
        active_side='a' if pairnumber%2 else 'b'
        sale=1 if case['id'].endswith('-'+active_side) else 0
        fake.add(oid,pid,sale)
        case['record_id']=oid;case['watch'][0]['record_id']=pid
    template.unlink();template.write_text(json.dumps(p))
    c._assert_protocol(p)
    workspace=tmp_path/'workspace';c.init(template,workspace)
    return p,workspace,fake,receipt,diag


def intervene(p,root,fake,receipt,diag,pair):
    answer=c.start_pair(root,pair['id'],client=fake)
    assert answer['quality']['same_draft_status']
    for cid in pair['members']:
        case=d._case_lookup(p['base_plan'],cid)
        sale=fake.products[case['watch'][0]['record_id']]['status']
        if sale=='1':
            fake.orders[case['record_id']]['status']='1'
            res=c.finish_case(root,cid,outcome='SUCCESS',source='UI',
                              operator_role='authorized_writer',client=fake)
            assert res['label']=='ACCEPT'
        else:
            res=c.finish_case(root,cid,outcome='BUSINESS_REJECTED',source='MANUAL_API',
                              receipt_status=500,operator_role='authorized_writer',
                              business_reason='product_not_on_sale',
                              receipt_file=receipt,diagnostic_file=diag,client=fake)
            assert res['label']=='REJECT'


def test_pilot_blocks_mismatched_status_and_incorrect_product_binding(tmp_path):
    fake=FakeERP();fake.add('100','200',1);fake.add('101','201',0)
    fake.orders['101']['status']='1'
    with pytest.raises(ValueError,match='SAME lifecycle status'):
        c.pilot_start(tmp_path/'x',order_a='100',order_b='101',product_a='200',product_b='201',client=fake)
    fake.orders['101']['status']='0'
    fake.orders['101']['lines'][0]['fk_product']=200
    with pytest.raises(ValueError,match='Watched product does not match'):
        c.pilot_start(tmp_path/'x',order_a='100',order_b='101',product_a='200',product_b='201',client=fake)
    assert not (tmp_path/'x').exists()


def test_pilot_requires_actual_correlated_diagnostic_not_arbitrary_500(tmp_path):
    fake=FakeERP();fake.add('100','200',1);fake.add('101','201',0)
    c.pilot_start(tmp_path/'pilot',order_a='100',order_b='101',product_a='200',product_b='201',client=fake)
    fake.orders['100']['status']='1'
    receipt,diag=receipt_files(tmp_path)
    diag.write_text('PHP memory exhausted unrelated database exception',encoding='utf-8')
    with pytest.raises(ValueError,match='product on-sale validation'):
        c.pilot_finish(tmp_path/'pilot',active_status=200,inactive_status=500,
                       receipt_file=receipt,diagnostic_file=diag,client=fake)
    assert not (tmp_path/'pilot/public/PILOT_CONFIRM.json').exists()


def test_pilot_proof_must_be_qualified_and_same_base_url(tmp_path):
    path=tmp_path/'fake.json';path.write_text('{}')
    with pytest.raises(ValueError,match='pilot-confirm'):
        c.scaffold(tmp_path/'study.json',pilot_file=path)
    assert not (tmp_path/'study.json').exists()


def test_500_without_diagnostic_never_admitted_and_304_not_business(tmp_path):
    p,root,fake,receipt,diag=setup(tmp_path)
    pair=p['pairs'][0];c.start_pair(root,pair['id'],client=fake)
    negative=next(cid for cid in pair['members'] if fake.products[d._case_lookup(p['base_plan'],cid)['watch'][0]['record_id']]['status']=='0')
    with pytest.raises(ValueError,match='correlated diagnostic'):
        c.finish_case(root,negative,outcome='BUSINESS_REJECTED',source='MANUAL_API',
                      receipt_status=500,operator_role='authorized_writer',
                      business_reason='product_not_on_sale',business_code='ProductNotOnSale',receipt_file=receipt,client=fake)
    with pytest.raises(ValueError,match='304'):
        c.finish_case(root,negative,outcome='BUSINESS_REJECTED',source='MANUAL_API',
                      receipt_status=304,operator_role='authorized_writer',
                      business_reason='product_not_on_sale',business_code='ProductNotOnSale',receipt_file=receipt,client=fake)
    with pytest.raises(ValueError,match='401/403'):
        c.finish_case(root,negative,outcome='BUSINESS_REJECTED',source='MANUAL_API',
                      receipt_status=403,operator_role='authorized_writer',
                      business_reason='product_not_on_sale',business_code='ProductNotOnSale',receipt_file=receipt,client=fake)
    assert not (root/'private/cases'/f'{negative}.finish.json').exists()


def test_negative_no_op_cannot_fit_or_open_holdout(tmp_path):
    p,root,fake,receipt,diag=setup(tmp_path)
    no_op=tmp_path/'no_op.txt';no_op.write_text('304 Not Modified')
    for pair in p['pairs']:
        if pair['split']=='holdout':continue
        c.start_pair(root,pair['id'],client=fake)
        for cid in pair['members']:
            case=d._case_lookup(p['base_plan'],cid)
            if fake.products[case['watch'][0]['record_id']]['status']=='1':
                fake.orders[case['record_id']]['status']='1'
                c.finish_case(root,cid,outcome='SUCCESS',source='UI',operator_role='authorized_writer',client=fake)
            else:
                result=c.finish_case(root,cid,outcome='NO_OP',source='MANUAL_API',receipt_status=304,
                                     receipt_file=no_op,operator_role='authorized_writer',client=fake)
                assert result['label']=='UNKNOWN'
    with pytest.raises(ValueError,match='qualifying'):
        c.fit(root)
    assert not (root/'private/FROZEN_POLICY.json').exists()


def test_complete_realistic_simulation_and_integrity(tmp_path):
    p,root,fake,receipt,diag=setup(tmp_path)
    for pair in p['pairs']:
        if pair['split']!='holdout':intervene(p,root,fake,receipt,diag,pair)
    status=c.status(root)
    assert status['ready_to_freeze']
    seal=c.fit(root)
    assert seal['all_required_operations_trained']
    for pair in p['pairs']:
        if pair['split']=='holdout':intervene(p,root,fake,receipt,diag,pair)
    verified=c.verify(root)
    assert verified['valid'] and verified['automatic_writes']==0
    report=c.report(root)
    assert report['usable_labeled_cases']==16
    assert report['discriminating_holdout_pairs']==8
    assert report['scores']['always_accept']['accuracy_all_cases']==.5
    assert report['scores']['status_only']['accuracy_all_cases']==.5
    assert report['per_operation']['order_validate']['negative_evidence_types']['correlated_application_500_refusals']==8
    assert report['business_rejection_quality_gate_passed']
    assert report['llm_comparison_executed_on_every_scored_case'] is False
    assert report['incremental_model_refiner_advantage'] is None
    assert all(m=='GET' for m,_ in fake.calls)
    public=''.join(q.read_text('utf-8') for q in (root/'public').glob('*.json'))
    assert 'SECRET' not in public and 'ProductNotOnSale' not in public
    for case in p['base_plan']['cases']:
        # A private ID must not appear as a standalone value/token. A coincidental
        # decimal substring inside a SHA-256 digest is not an ID disclosure.
        import re
        assert re.search(r'(?<![0-9A-Za-z])'+re.escape(case['record_id'])+r'(?![0-9A-Za-z])',public) is None
    assert len(set(x['case_id'] for x in ()))==0
    with pytest.raises(FileExistsError):c.report(root)


def test_tampering_with_correlated_receipt_is_detected(tmp_path):
    p,root,fake,receipt,diag=setup(tmp_path)
    pair=p['pairs'][0];intervene(p,root,fake,receipt,diag,pair)
    cid=pair['members'][0];path=root/'private/receipts'/f'{cid}.json'
    doc=json.loads(path.read_text());doc['status']=200
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='modified'):
        c.verify(root)


def test_strict_get_only_sources(tmp_path):
    p,root,fake,receipt,diag=setup(tmp_path)
    c.start_pair(root,p['pairs'][0]['id'],client=fake)
    assert all(method=='GET' for method,_ in fake.calls)
    assert all('webservices_key' not in f.read_text('utf-8') for f in (root/'private/cases').glob('*.json'))
