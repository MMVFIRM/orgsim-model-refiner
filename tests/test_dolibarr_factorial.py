"""One-shot factorial evaluation: fake HTTP GET oracle, NEVER real ERP writes."""
from __future__ import annotations
import csv, json, pathlib, tempfile
import pytest
from simlearn import dolibarr_factorial as f
from simlearn import dolibarr_transfer as legacy
from model_refiner.backend import ModelBackend
from model_refiner.types import ModelResult

class FakeDolibarr:
    def __init__(self):self.orders={};self.products={};self.methods=[]
    def fixture(self,order,product,cell,qty=4):
        st=qty if cell[1]=='1' else max(0,qty-2)
        self.orders[str(order)]={'status': '0','total_ht':'20','total_ttc':'24',
            'lines':[{'fk_product':str(product),'qty':str(qty)}]}
        self.products[str(product)]={'status':cell[0], 'stock_reel':str(st),
                                    'price':'5','type':'0','tva_tx':'20'}
    def request(self,method,path):
        self.methods.append((method,path))
        assert method=='GET', 'Harness made an ERP WRITE!'
        rest=path.split('?',1)[0].split('/')
        family,oid=rest[1],rest[2]
        table=self.orders if family=='orders' else self.products
        if oid not in table:return 404,{'error':'not found'}
        return 200,dict(table[oid])
    def action(self,order,product,cell,qty=4):
        assert cell in f.CELLS
        if cell=='11':
            self.orders[order]['status']='1';self.products[product]['stock_reel']=str(int(self.products[product]['stock_reel'])-qty)
            return 'SUCCESS'
        return 'BUSINESS_REJECTED'

class MockProvider(ModelBackend):
    def __init__(self):self.calls=0
    def generate(self,*,model,messages):
        self.calls+=1
        return ModelResult(text='REJECT',input_tokens=80,output_tokens=1,metadata={'status':'completed'})


def evidence(tmp):
    doc={}
    for which in ('sale','stock'):
        receipt=tmp/(which+'_receipt.txt');receipt.write_text('Error when validating Order: application rule',encoding='utf8')
        diagnostic=tmp/(which+'_diagnostic.txt')
        diagnostic.write_text('checkActiveProductInLines product not on sale' if which=='sale' else
                              'Not enough stock at warehouse; insufficient stock quantity',encoding='utf8')
        doc[which]=(receipt,diagnostic)
    return doc


def pilot(tmp,fake):
    ev=evidence(tmp)
    f.pilot_template(tmp/'pilot.csv')
    with (tmp/'pilot.csv').open('w',encoding='utf8',newline='') as h:
        w=csv.writer(h,lineterminator='\n');w.writerow(['case_id','expected_cell','order_id','product_id'])
        for i,cell in enumerate(f.STUDY_CELLS):
            fake.fixture(10+i,20+i,cell)
            w.writerow(['pilot-'+cell,cell,str(10+i),str(20+i)])
    f.pilot_start(tmp/'pilot.csv',tmp/'pilot_run',client=fake)
    fake.action('10','20','11')
    q=f.pilot_finish(tmp/'pilot_run',success_status=200,sale_status=500,stock_status=500,
      sale_receipt=ev['sale'][0],sale_diagnostic=ev['sale'][1],
      stock_receipt=ev['stock'][0],stock_diagnostic=ev['stock'][1],client=fake)
    assert q['qualified']
    return tmp/'pilot_run/public/PILOT_CONFIRM.json',ev


def create_study(tmp,fake,*,train_blocks=4,validation_blocks=2,holdout_blocks=12):
    proof,ev=pilot(tmp,fake)
    f.scaffold(proof,tmp/'study.json',train_blocks=train_blocks,validation_blocks=validation_blocks,holdout_blocks=holdout_blocks)
    with (tmp/'study_IDS.csv').open(encoding='utf8',newline='') as h:
        rr=list(csv.DictReader(h))
    with (tmp/'study_IDS.csv').open('w',encoding='utf8',newline='') as h:
        ww=csv.DictWriter(h,fieldnames=list(rr[0]),lineterminator='\n');ww.writeheader()
        for i,row in enumerate(rr):
            order,prod=str(100+i),str(1000+i)
            row['order_id']=order;row['product_id']=prod
            fake.fixture(order,prod,row['on_sale']+row['stock_sufficient'])
            ww.writerow(row)
    f.bind(tmp/'study.json',tmp/'study_IDS.csv',tmp/'study_PRIVATE.json')
    f.init(tmp/'study_PRIVATE.json',tmp/'run')
    return tmp/'run',ev


def finish_one(root,fake,ev,cid,cell):
    p=f.read(root/'private/FACTORIAL_PLAN.json')
    c=next(x for x in p['cases'] if x['id']==cid)
    outcome=fake.action(c['record_id'],c['watch'][0]['record_id'],cell)
    kws={'client':fake}
    if outcome=='BUSINESS_REJECTED':
        reason='sale' if cell[0]=='0' else 'stock'
        # Each operator action must have a distinct, independently saved
        # redacted local receipt, even when the error category is identical.
        receipt=ev[reason][0].with_name(cid+'_response.txt')
        receipt.write_text(ev[reason][0].read_text()+'\nrequest '+cid,encoding='utf8')
        diagnostic=ev[reason][1].with_name(cid+'_diagnostic.txt')
        diagnostic.write_text(ev[reason][1].read_text()+'\nrequest '+cid,encoding='utf8')
        kws.update({'receipt_file':receipt,'diagnostic_file':diagnostic})
    return f.finish_case(root,cid,outcome=outcome,http_status=200 if outcome=='SUCCESS' else 500,**kws)


def finish_split(root,fake,ev,split,*,with_provider=None):
    p=f.read(root/'private/FACTORIAL_PLAN.json')
    for b in p['blocks']:
        if b['split']!=split:continue
        f.start_block(root,b['id'],client=fake,execute_model=bool(with_provider),share_sanitized_fields=bool(with_provider),backend=with_provider)
        for m in b['members']:
            fin=finish_one(root,fake,ev,m['case_id'],m['cell'])
            assert fin['classified']==('ACCEPT' if m['cell']=='11' else 'REJECT')


def test_full_factorial_one_shot(tmp_path):
    fake=FakeDolibarr();root,ev=create_study(tmp_path,fake)
    # Verify no mixed/00 cases in evidence exposed to the compiler.
    finish_split(root,fake,ev,'train')
    finish_split(root,fake,ev,'validation')
    q=f.fit(root)
    assert q['learned_program']=='sale_and_stock'
    assert q['stock_effect_formula_supported']
    bank=f.read(root/'private/TRAIN_BANK.json')
    assert all(r['cell']!='00' for r in bank['train']+bank['validation'])
    finish_split(root,fake,ev,'holdout')
    v=f.verify(root);assert v['verified']
    r=f.report(root)
    assert r['holdout_total']==48
    assert r['holdout_qualified']==48
    assert r['scores']['structured_program']['correct']==48
    assert r['scores']['structured_program']['false_accepts']==0
    assert r['scores']['always_reject']['accuracy_all_cases']==.75
    assert r['policy_decision_gate']
    assert r['exclusive_incremental_adaptation_gate'] is False
    assert r['non_status_stock_effect_scored']==12
    assert r['non_status_stock_effect_accuracy']==1
    assert r['cells']['00']['rejected']==12
    assert r['model_provider_usage']['logical_calls']==0
    assert all(method=='GET' for method,_ in fake.methods)
    freeze=f.read(root/'public/FACTORIAL_AUDIT_FREEZE.json')
    assert freeze['report_sha256']==legacy.digest(r)
    with pytest.raises(FileExistsError):f.report(root)


def test_model_refiner_opt_in_comparison(tmp_path):
    fake=FakeDolibarr();root,ev=create_study(tmp_path,fake)
    finish_split(root,fake,ev,'train');finish_split(root,fake,ev,'validation');f.fit(root)
    provider=MockProvider()
    finish_split(root,fake,ev,'holdout',with_provider=provider)
    assert provider.calls==12*4*4
    result=f.report(root)
    assert result['model_comparison_executed_on_all_holdout_cases']
    assert result['scores']['llm_baseline']['accuracy_all_cases']==.75
    assert result['scores']['llm_policy']['false_rejects']==12
    assert result['exclusive_incremental_adaptation_gate'] is False
    assert result['policy_vs_retrieval']['delta']==0
    assert result['model_provider_usage']['logical_calls']==provider.calls


def test_bad_factor_status_and_leakage(tmp_path):
    fake=FakeDolibarr();root,ev=create_study(tmp_path,fake)
    plan=f.read(root/'private/FACTORIAL_PLAN.json')
    hold=next(b for b in plan['blocks'] if b['split']=='holdout')
    with pytest.raises(ValueError,match='Fit and freeze'):
        f.start_block(root,hold['id'],client=fake)
    train=next(b for b in plan['blocks'] if b['split']=='train')
    c=train['members'][0]['case_id'];x=next(z for z in plan['cases'] if z['id']==c)
    fake.products[x['watch'][0]['record_id']]['status']='x'
    with pytest.raises(ValueError,match='sale status'):
        f.start_block(root,train['id'],client=fake)
    assert not (root/'private/blocks'/(train['id']+'.start.json')).exists()


def test_all_positive_or_uncorroborated_errors_block_freeze(tmp_path):
    fake=FakeDolibarr();root,ev=create_study(tmp_path,fake)
    with pytest.raises(ValueError,match='All training and validation'):
        f.fit(root)
    train=next(b for b in f.read(root/'private/FACTORIAL_PLAN.json')['blocks'] if b['split']=='train')
    f.start_block(root,train['id'],client=fake)
    bad=next(m for m in train['members'] if m['cell']=='01')
    with pytest.raises(ValueError,match='redacted receipt'):
        f.finish_case(root,bad['case_id'],outcome='BUSINESS_REJECTED',http_status=500,client=fake)
    with pytest.raises(ValueError,match='specific stated business refusal'):
        p=tmp_path/'wronglog.txt';p.write_text('Generic Python service unavailable')
        f.triage_500(ev['sale'][0],p,reason='product_not_on_sale')


def test_tamper_detected(tmp_path):
    fake=FakeDolibarr();root,ev=create_study(tmp_path,fake)
    finish_split(root,fake,ev,'train')
    path=next((root/'private'/'blocks').glob('*.start.json'))
    doc=f.read(path);doc['facts'][next(iter(doc['facts']))]['status']='99'
    path.write_text(json.dumps(doc),encoding='utf8')
    with pytest.raises(ValueError,match='modified'):
        f.verify(root)


def test_csv_binding_rejects_duplicates_and_assignment_tampering(tmp_path):
    fake=FakeDolibarr();proof,ev=pilot(tmp_path,fake)
    f.scaffold(proof,tmp_path/'study.json',train_blocks=4,validation_blocks=2,holdout_blocks=12)
    p=tmp_path/'study_IDS.csv'
    with p.open(newline='',encoding='utf8') as h:rows=list(csv.DictReader(h))
    for i,row in enumerate(rows):row.update(order_id=str(100+i),product_id=str(200+i))
    rows[1]['product_id']=rows[0]['product_id']
    with p.open('w',newline='',encoding='utf8') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    with pytest.raises(ValueError,match='distinct fresh'):
        f.bind(tmp_path/'study.json',p,tmp_path/'private.json')
    rows[1]['product_id']='909';rows[0]['on_sale']='9'
    with p.open('w',newline='',encoding='utf8') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    with pytest.raises(ValueError,match='assignment altered'):
        f.bind(tmp_path/'study.json',p,tmp_path/'private.json')


def test_missing_stock_pilot_cannot_qualify(tmp_path):
    fake=FakeDolibarr();ev=evidence(tmp_path)
    f.pilot_template(tmp_path/'pilot.csv')
    with (tmp_path/'pilot.csv').open('w',encoding='utf8',newline='') as h:
        w=csv.writer(h,lineterminator='\n');w.writerow(['case_id','expected_cell','order_id','product_id'])
        for i,cell in enumerate(f.STUDY_CELLS):
            fake.fixture(10+i,20+i,cell);w.writerow(['pilot-'+cell,cell,10+i,20+i])
    f.pilot_start(tmp_path/'pilot.csv',tmp_path/'pilot_run',client=fake)
    fake.action('10','20','11')
    # Stock insufficient also succeeds: strict stage cannot proceed.
    fake.orders['12']['status']='1'
    with pytest.raises(ValueError,match='no watched changes'):
        f.pilot_finish(tmp_path/'pilot_run',success_status=200,sale_status=500,stock_status=500,
          sale_receipt=ev['sale'][0],sale_diagnostic=ev['sale'][1],
          stock_receipt=ev['stock'][0],stock_diagnostic=ev['stock'][1],client=fake)
    assert not (tmp_path/'pilot_run/public/PILOT_CONFIRM.json').exists()


def test_disjoint_groups_and_00_absent_until_holdout(tmp_path):
    fake=FakeDolibarr();root,ev=create_study(tmp_path,fake)
    p=f.read(root/'private/FACTORIAL_PLAN.json')
    train=[m['cell'] for b in p['blocks'] if b['split'] in ('train','validation') for m in b['members']]
    test=[m['cell'] for b in p['blocks'] if b['split']=='holdout' for m in b['members']]
    assert '00' not in train
    assert {c: test.count(c) for c in f.CELLS}=={c:12 for c in f.CELLS}
    assert len({c['record_id'] for c in p['cases']})==len(p['cases'])


class PolicyOnlyMock(ModelBackend):
    """Tests the comparison scorer; NOT a real-model efficacy result."""
    def __init__(self):self.calls=0
    def generate(self,*,model,messages):
        self.calls+=1
        is_policy=(self.calls%4)==0
        # The fake fixtures use qty=4, stock=4 (enough) or stock=2.
        # Only the fourth condition receives the compiled operator.
        text='REJECT'
        if is_policy:
            combined=' '.join(m.get('content','') for m in messages)
            if '"status_on_sale":1' in combined and '"stock_reel":"4"' in combined:
                text='ACCEPT'
        return ModelResult(text=text,input_tokens=80,output_tokens=1,metadata={'status':'completed'})


def test_preregistered_retrieval_superiority_gate_only_with_actual_delta(tmp_path):
    fake=FakeDolibarr();root,ev=create_study(tmp_path,fake)
    finish_split(root,fake,ev,'train');finish_split(root,fake,ev,'validation');f.fit(root)
    provider=PolicyOnlyMock()
    finish_split(root,fake,ev,'holdout',with_provider=provider)
    r=f.report(root)
    assert provider.calls==192
    assert r['scores']['llm_policy']['correct']==48
    assert r['scores']['llm_real_retrieval']['correct']==36
    assert r['policy_vs_retrieval']['delta']==.25
    assert r['policy_vs_retrieval']['block_bootstrap_ci95'][0]>0
    assert r['exclusive_incremental_adaptation_gate']
    assert r['non_status_stock_effect_exact']==12


def test_cross_tenant_policy_file_is_rejected_before_holdout(tmp_path):
    fake=FakeDolibarr();root,ev=create_study(tmp_path,fake)
    finish_split(root,fake,ev,'train');finish_split(root,fake,ev,'validation');f.fit(root)
    p=root/'private'/'FROZEN_POLICY.json'
    changed=f.read(p);changed['tenant_fingerprint']='0'*64
    p.write_text(json.dumps(changed),encoding='utf8')
    hold=next(b for b in f.read(root/'private/FACTORIAL_PLAN.json')['blocks'] if b['split']=='holdout')
    with pytest.raises(ValueError,match='another tenant or plan'):
        f.start_block(root,hold['id'],client=fake)


def test_http_stock_reason_with_generic_correlated_stock_rollback_is_accepted(tmp_path):
    """Dolibarr's HTTP body, not its local log, may hold the stock reason."""
    receipt=tmp_path/'writer_stock_response.txt'
    receipt.write_text('Error when validating Order: Not enough stock for this product; negative stock disallowed', encoding='utf8')
    diagnostic=tmp_path/'dolibarr_stock_log.txt'
    diagnostic.write_text('2026-10-11T08:00:00 StockMouvement::create attempted stock movement\n'
                          '2026-10-11T08:00:00 DB::rollback transaction rollback', encoding='utf8')
    triaged=f.triage_500(receipt,diagnostic,reason='insufficient_stock')
    assert triaged['evidence_mode']=='HTTP_REASON_PLUS_CORRELATED_LOCAL_ROLLBACK'
    assert triaged['reason_observed_in']=='HTTP_RESPONSE'
    assert triaged['local_diagnostic_support']=='stock_attempt_and_rollback'


def test_http_sale_reason_with_local_validation_rollback_is_accepted(tmp_path):
    receipt=tmp_path/'sale_response.txt'
    receipt.write_text('Error when validating Order: Product disabled for sale',encoding='utf8')
    diagnostic=tmp_path/'sale_diagnostic.txt'
    diagnostic.write_text('2026-10-11 Commande::valid validation request failed\n'
                          '2026-10-11 Database transaction rollback',encoding='utf8')
    out=f.triage_500(receipt,diagnostic,reason='product_not_on_sale')
    assert out['evidence_mode']=='HTTP_REASON_PLUS_CORRELATED_LOCAL_ROLLBACK'
    assert out['reason']=='product_not_on_sale'


def test_api_reason_does_not_excuse_unrelated_local_log(tmp_path):
    receipt=tmp_path/'r.txt';receipt.write_text('Error when validating Order: Not enough stock',encoding='utf8')
    diag=tmp_path/'d.txt'
    for text in ('Generic web server error',
                 'transaction rollback',
                 'StockMouvement::create successfully created stock movement'):
        diag.write_text(text,encoding='utf8')
        with pytest.raises(ValueError,match='rollback|stock movement'):
            f.triage_500(receipt,diag,reason='insufficient_stock')


def test_generic_api_500_and_generic_rollback_rejected(tmp_path):
    receipt=tmp_path/'r.txt';receipt.write_text('Error when validating Order: Internal error 500',encoding='utf8')
    diag=tmp_path/'d.txt';diag.write_text('StockMouvement create attempted stock movement\nDatabase rollback',encoding='utf8')
    with pytest.raises(ValueError,match='Neither response nor diagnostic'):
        f.triage_500(receipt,diag,reason='insufficient_stock')


def test_mismatched_refusal_reason_rejected(tmp_path):
    receipt=tmp_path/'r.txt';receipt.write_text('Error when validating Order: Not enough stock',encoding='utf8')
    diag=tmp_path/'d.txt';diag.write_text('Order validation attempted\nDatabase rollback',encoding='utf8')
    with pytest.raises(ValueError,match='Neither response nor diagnostic'):
        f.triage_500(receipt,diag,reason='product_not_on_sale')


def test_existing_failed_pilot_can_be_resumed_read_only(tmp_path):
    """After updating the harness, do not repeat the original writer actions."""
    fake=FakeDolibarr();ev=evidence(tmp_path)
    f.pilot_template(tmp_path/'pilot.csv')
    with (tmp_path/'pilot.csv').open('w',encoding='utf8',newline='') as handle:
        writer=csv.writer(handle,lineterminator='\n')
        writer.writerow(['case_id','expected_cell','order_id','product_id'])
        for i,cell in enumerate(f.STUDY_CELLS):
            fake.fixture(200+i,300+i,cell)
            writer.writerow(['pilot-'+cell,cell,200+i,300+i])
    f.pilot_start(tmp_path/'pilot.csv',tmp_path/'pilot_run',client=fake)
    fake.action('200','300','11')  # one authorized writer action outside GET-only harness
    stock_response=tmp_path/'actual_stock_response.txt'
    stock_response.write_text('Error when validating Order: Not enough stock, negative stock not allowed',encoding='utf8')
    stock_diag=tmp_path/'actual_stock_log.txt'
    stock_diag.write_text('StockMouvement::create attempted stock movement\nDolibarr DB transaction rollback',encoding='utf8')
    out=f.pilot_finish(tmp_path/'pilot_run',success_status=200,sale_status=500,stock_status=500,
                       sale_receipt=ev['sale'][0],sale_diagnostic=ev['sale'][1],
                       stock_receipt=stock_response,stock_diagnostic=stock_diag,client=fake)
    assert out['qualified']
    proof=f.read(tmp_path/'pilot_run/public/PILOT_CONFIRM.json')
    assert proof['rejection_evidence_hashes']['stock']['reason_observed_in']=='HTTP_RESPONSE'
    assert proof['rejection_evidence_hashes']['stock']['local_diagnostic_support']=='stock_attempt_and_rollback'
    assert fake.methods and all(method=='GET' for method,_ in fake.methods)
    with pytest.raises(FileExistsError):
        f.pilot_finish(tmp_path/'pilot_run',success_status=200,sale_status=500,stock_status=500,
                       sale_receipt=ev['sale'][0],sale_diagnostic=ev['sale'][1],
                       stock_receipt=stock_response,stock_diagnostic=stock_diag,client=fake)
