"""Gate 2D adaptive investigation regressions: behavioral utility, safety, no leakage."""
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from orgsim.gate2d_adaptive import (
    EvidenceLog, Investigator, Probe, build_candidates, choose_probe,
    get_live_identifiers, make_model_prompt, safe_shape, score_prediction,
    validate_advice, validate_fixtures, validate_prediction,
)


class FakeLocalERP:
    loopback = True

    def __init__(self, *, populated=False, empty=False, on_request=None):
        self.calls = []
        self.on_request = on_request
        self.db = {
            'thirdparties': [] if empty else [
                {'id': str(7395829134+i), 'name': f'SECRET_CUSTOMER_{i}',
                 'client': '1', 'country_code': 'US',
                 **{f'additional_{x}': None for x in range(16)}}
                for i in range(3)],
            'products': [] if empty else [
                {'id': '7395829123', 'ref': 'SECRET-PRODUCT-ABC',
                 'label': 'ULTRA_SECRET_PRODUCT_NAME',
                 'price': '5.50', 'type': '0',
                 **{f'extra_{x}': None for x in range(12)}}],
            'orders': ([{'id': '7395829001', 'socid': '7395829134',
                         'status': '1', 'ref': 'ORDER_SECRET', 'total_ht': '55.00'}]
                       if populated else []),
            'invoices': ([{'id': '7395829002', 'socid': '7395829134',
                           'status': '0', 'ref': 'INVOICE_SECRET', 'total_ht': '55.00'}]
                         if populated else []),
        }

    def request(self, method, path):
        assert method == 'GET', 'Mutations should never be possible'
        self.calls.append((method, path))
        if self.on_request: self.on_request(method, path)
        parts = urlsplit(path)
        bits = parts.path.strip('/').split('/')
        family = bits[0]
        assert family in self.db, path
        qs = parse_qs(parts.query)
        if len(bits) == 1:
            records = self.db[family]
            limit = int(qs.get('limit', ['3'])[0])
            page = int(qs.get('page', ['0'])[0])
            selected = records[page*limit:(page+1)*limit]
            if qs.get('ids_only') == ['1']:
                return 200, [i['id'] for i in selected]
            if 'properties' in qs:
                wanted = qs['properties'][0].split(',')
                selected = [{k: r[k] for k in wanted if k in r} for r in selected]
            return 200, selected
        if family == 'products' and bits[1] == 'ref':
            for rec in self.db['products']:
                if rec['ref'] == bits[-1]: return 200, rec
            return 404, {'error': 'PRIVATE'}
        rec = next((r for r in self.db[family] if r['id'] == bits[1]), None)
        if not rec: return 404, {'error': 'SECRET_ERROR'}
        if len(bits) == 2: return 200, rec
        if family == 'products' and bits[-1] == 'stock':
            return 200, {'stock_reel': '27', 'stock_theorique': '29'}
        if family == 'products' and bits[-1] == 'categories':
            return 200, []
        if family == 'thirdparties' and bits[-1].startswith('outstanding'):
            return 200, {'total': '0'}
        if family == 'orders' and bits[-1] == 'lines':
            return 200, [{'id': '7395829003', 'fk_product': '7395829123', 'qty': '2'}]
        if family == 'orders' and bits[-1] == 'shipment':
            return 200, []
        if family == 'invoices' and bits[-1] == 'lines':
            return 200, [{'id': '7395829004', 'qty': '2'}]
        if family == 'invoices' and bits[-1] == 'payments':
            return 200, []
        raise AssertionError('Unexpected path: '+path)


def _run(tmp_path, *, populated=False, empty=False, budget=20, advisor=None,
         auto_ids=True, on_request=None, fixtures=None):
    oracle = FakeLocalERP(populated=populated, empty=empty, on_request=on_request)
    inv = Investigator(oracle, budget=budget, advisor=advisor, auto_ids=auto_ids,
                       fixtures=fixtures or {})
    output = tmp_path / 'run'
    report = inv.run(execute=True, outdir=output)
    return oracle, report, output


def test_dryrun_no_network_and_four_public_anchors(tmp_path):
    class NoNetwork:
        loopback = True
        def request(self, *args): raise AssertionError('No network in dry-run')
    result = Investigator(NoNetwork(), fixtures={'thirdparty_id': 123}, budget=20).run(
        execute=False, outdir=tmp_path)
    assert result['operations_executed'] == 0
    assert sorted(p['id'] for p in result['initial_plan']) == [
        'invoices_list', 'orders_list', 'products_list', 'thirdparties_list']
    assert '123' not in (tmp_path/'discovery_report.json').read_text()
    assert not (tmp_path/'events.jsonl').exists()


def test_adaptive_prefers_entities_and_skips_empty_order_invoice_pages(tmp_path):
    client, result, output = _run(tmp_path, budget=16)
    ids = {p['id'] for p in result['probes']}
    assert len(client.calls) == result['operations_executed']
    assert result['operations_executed'] <= 16
    assert 'products_detail' in ids and 'thirdparties_detail' in ids
    assert 'product_by_ref' in ids
    assert 'orders_page1' not in ids and 'invoices_page1' not in ids
    assert 'products_page1' not in ids
    assert result['skipped']['orders'].startswith('no_observed')
    assert result['skipped']['invoices'].startswith('no_observed')
    assert result['live_writes'] == 0
    assert all(method == 'GET' for method, _ in client.calls)
    assert result['stop_reason'] in {'budget_exhausted','no_informative_approved_probes_available'}


def test_populated_lifecycle_entities_are_explored(tmp_path):
    _, result, _ = _run(tmp_path, populated=True, budget=32)
    ids = {p['id'] for p in result['probes']}
    assert {'orders_detail', 'orders_lines', 'invoices_detail', 'invoices_lines'} <= ids
    assert 'orders_page1' not in ids and 'invoices_page1' not in ids
    assert result['operations_executed'] <= 32


def test_empty_environment_stops_early(tmp_path):
    _, result, _ = _run(tmp_path, empty=True, budget=32)
    assert result['operations_executed'] == 4
    assert result['stop_reason'] == 'no_informative_approved_probes_available'
    assert all(p['category'] == 'anchor' for p in result['probes'])


def test_persist_only_shape_fields_not_ids_refs_values_or_passwords(tmp_path):
    _, result, folder = _run(tmp_path, budget=20,
                             fixtures={'thirdparty_id': '7395829134',
                                       'product_ref': 'SECRET-PRODUCT-ABC'})
    for name in ['discovery_report.json','candidate_read_model_unreviewed.json',
                 'FREEZE.json', 'events.jsonl']:
        data = (folder / name).read_text()
        assert 'SECRET' not in data
        assert '7395829134' not in data
        assert '7395829123' not in data
        assert 'ULTRA_SECRET' not in data
        assert 'DOLAPIKEY' not in data
        assert 'credential' not in data.lower() or name == 'discovery_report.json' or name=='FREEZE.json'
    assert result['raw_values_saved'] is False


def test_predict_before_observe_and_chained_log(tmp_path):
    output = tmp_path / 'run'
    snapshots = []
    def inspect(_method, _path):
        text = (output/'events.jsonl').read_text()
        events = [json.loads(line) for line in text.splitlines()]
        assert events[-1]['event'] == 'prediction_committed'
        snapshots.append(events[-1]['probe_id'])
    client, result, folder = _run(tmp_path, budget=12, on_request=inspect)
    assert len(snapshots) == len(client.calls)
    events = [json.loads(x) for x in (folder/'events.jsonl').read_text().splitlines()]
    from orgsim.gate2d_adaptive import hash_json
    previous = '0'*64
    for event in events:
        assert event['prev_sha256'] == previous
        digest = event['sha256']
        without_digest = dict(event)
        del without_digest['sha256']
        assert hash_json(without_digest) == digest
        previous = digest
    assert json.loads((folder/'FREEZE.json').read_text())['evidence_chain_sha256'] == previous
    assert result['operations_executed'] == len(snapshots)


def test_prediction_score_counts_abstentions_not_successes():
    assert score_prediction({}, 200, {'kind':'array'})['exact_match_on_predicted_fields'] is False
    assert score_prediction({'status':200, 'kind':'array','length':1}, 200,
                            {'kind':'array','length':0})['matched_fields']==2
    with pytest.raises(ValueError):
        validate_prediction({'field_values':{'id':'top_secret'}})
    with pytest.raises(ValueError):
        validate_prediction({'first_keys':['$password']})


def test_model_can_only_choose_approved_probe_and_no_raw_values(tmp_path):
    prompts=[]
    class HonestAdvisor:
        def choose(self, prompt):
            prompts.append(prompt)
            return {'probe_id': prompt['allowed_probes'][0]['id'],
                    'prediction': {'status':200}}
    _, report, folder = _run(tmp_path, budget=10, advisor=HonestAdvisor())
    assert report['model_provider_calls'] == 6
    assert any(p['predictor'] == 'model_hypothesis' for p in report['probes'])
    prompt=json.dumps(prompts)
    assert '7395829134' not in prompt and 'SECRET' not in prompt
    assert all('path_template' in x for x in prompts[0]['allowed_probes'])
    assert 'unreviewed' in (folder/'candidate_read_model_unreviewed.json').name


def test_model_illegal_route_and_extra_keys_rejected_before_http(tmp_path):
    class MaliciousAdvisor:
        def choose(self, prompt):
            return {'probe_id':'delete_all', 'prediction':{'kind':'array'}}
    client=FakeLocalERP()
    inv=Investigator(client, advisor=MaliciousAdvisor(), budget=12)
    with pytest.raises(ValueError, match='unapproved probe'):
        inv.run(execute=True, outdir=tmp_path)
    assert len(client.calls)==4 # Anchor GETs only; no fifth request.
    assert all(method=='GET' for method,path in client.calls)
    bad={'probe_id':'products_list','prediction':{'kind':'array'},'path':'/users/info'}
    with pytest.raises(ValueError,match='exactly'):
        validate_advice(bad,[Probe('products_list','products','anchor','/products', '/products',1)])


def test_validate_fixtures_and_no_remote_tenant(tmp_path):
    for bad in ({'thirdparty_id':'../../admin'}, {'product_ref':'../a'},
                {'user_id':'100'}, {'thirdparty_id':True}):
        with pytest.raises(ValueError): validate_fixtures(bad)
    class Remote:
        loopback=False
    with pytest.raises(ValueError,match='loopback'):
        Investigator(Remote())


def test_readonly_transport_exception_does_not_store_server_error(tmp_path):
    class ErrorClient:
        loopback=True
        def request(self,method,path):
            raise RuntimeError('TOP_SECRET_TOKEN at '+path)
    folder=tmp_path/'failed'
    rep=Investigator(ErrorClient(),budget=8).run(execute=True,outdir=folder)
    assert rep['operations_executed']==0
    assert rep['stop_reason']=='transport_error:RuntimeError'
    assert 'TOP_SECRET' not in (folder/'events.jsonl').read_text()
    assert 'TOP_SECRET' not in (folder/'discovery_report.json').read_text()


def test_prevent_overwrite_of_existing_evidence(tmp_path):
    client=FakeLocalERP()
    inv=Investigator(client,budget=8)
    inv.run(execute=True,outdir=tmp_path)
    before=(tmp_path/'events.jsonl').read_bytes()
    with pytest.raises(FileExistsError,match='already exists'):
        Investigator(client,budget=8).run(execute=True,outdir=tmp_path)
    assert (tmp_path/'events.jsonl').read_bytes()==before


def test_gate2d_artifact_verifier_detects_tampering(tmp_path):
    from orgsim.gate2d_verify import verify
    _, _, folder = _run(tmp_path, budget=12)
    verified = verify(folder)
    assert verified['sha_chain_valid']
    with (folder/'events.jsonl').open('a') as f:
        f.write('{"seq":100,"event":"forged"}\n')
    with pytest.raises((ValueError,KeyError), match='hash|Event|number'):
        verify(folder)


def test_gate2d_offline_comparison_never_contacts_dolibarr(tmp_path):
    from orgsim.gate2d_lab import evaluate
    out = evaluate(budget=12,outdir=tmp_path)
    assert out['real_dolibarr_calls'] == 0
    assert out['external_model_calls'] == 0
    assert len(out['cases']) == 6
    rows=[r for r in out['cases'] if r['adaptive']['arm']=='adaptive_with_ids']
    assert next(r for r in rows if r['oracle_family']=='empty')['adaptive']['reads']==4
    assert all(r['adaptive']['reads']<=12 for r in rows)
    assert any(r['adaptive']['novel_nonempty_response_signatures'] >
               r['fixed']['novel_nonempty_response_signatures'] for r in rows)
