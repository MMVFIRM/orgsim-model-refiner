"""Offline harness tests. FakeTransport is NOT a real Dolibarr installation."""
import io
import json
from pathlib import Path
from urllib.error import HTTPError

import pytest
import yaml

from orgsim.dolibarr_contract import contract_coverage, swagger2_to_openapi3
from orgsim.dolibarr_gate import (CONFIRM, DolibarrClient, capture_swagger,
                                  reconstruct, run_probes)
from orgsim.engine import SimulationEngine
from orgsim.openapi import import_openapi
from orgsim.spec import SpecError, load_spec

FIXTURE = Path(__file__).resolve().parents[1] / 'gate_dolibarr' / 'fixtures'


class FakeResponse(io.BytesIO):
    def __init__(self, status, value):
        super().__init__(json.dumps(value).encode())
        self.status = status


class FakeTransport:
    """A small controlled offline reference, intentionally not actual ERP semantics."""
    def __init__(self):
        self.thirdparties = {}
        self.products = {}
        self.calls = []

    def open(self, req, timeout=25):
        assert timeout <= 25
        assert req.get_header('Dolapikey') == 'unit-test-key'
        assert 'DOLAPIKEY' not in req.full_url
        from urllib.parse import urlsplit
        path = urlsplit(req.full_url).path.split('/api/index.php', 1)[1]
        method = req.get_method()
        self.calls.append((method, path))
        status, body = 200, None
        if path == '/explorer/swagger.json':
            body = load_spec(FIXTURE / 'illustrative_swagger2.json')
        elif method == 'GET' and path == '/thirdparties':
            body = list(self.thirdparties.values())
        elif method == 'GET' and path == '/products':
            body = list(self.products.values())
        elif method == 'POST' and path == '/thirdparties':
            value = json.loads(req.data)
            i = 1 + len(self.thirdparties)
            self.thirdparties[i] = {'id': i, **value}
            body = i  # Deliberate response-shape difference from OrgSim
        elif method == 'GET' and path.startswith('/thirdparties/'):
            pk = int(path.split('/')[-1])
            status, body = ((200, self.thirdparties[pk]) if pk in self.thirdparties else (404, {'error': 'missing'}))
        elif method == 'GET' and path.startswith('/products/'):
            status, body = 404, {'error': 'missing'}
        else:
            status, body = 404, {'error': 'unknown'}
        if status >= 400:
            raise HTTPError(req.full_url, status, 'failed', {}, io.BytesIO(json.dumps(body).encode()))
        return FakeResponse(status, body)


def get_fixture():
    return load_spec(FIXTURE / 'illustrative_swagger2.json')


def test_swagger2_converter_preserves_unimplemented_operations_and_local_refs(tmp_path):
    oai = swagger2_to_openapi3(get_fixture())
    assert oai['openapi'] == '3.0.3'
    assert oai['paths']['/thirdparties']['get']['responses']['200']['content']['application/json']['schema']['items']['$ref'] == '#/components/schemas/Thirdparty'
    assert oai['paths']['/thirdparties']['post']['requestBody']['content']['application/json']['schema']['$ref'] == '#/components/schemas/Thirdparty'
    assert '/orders/{id}/validate' in oai['paths']
    f = tmp_path / 'normalized.json'
    f.write_text(json.dumps(oai))
    draft = import_openapi([f], system='dolibarr')
    assert 'login' not in draft['systems']['dolibarr']['resources']
    assert draft['systems']['dolibarr']['resources']['thirdparties']['schema']['properties']['id']['type'] == 'integer'
    coverage = contract_coverage(oai, draft)
    assert coverage['mapped_operations'] == 7
    assert coverage['api_operations'] == 9
    assert 'POST /orders/{id}/validate' in coverage['not_reconstructed']


def test_swagger2_unsupported_external_ref():
    swagger = get_fixture()
    swagger['definitions']['Product']['properties']['unexpected'] = {'$ref': 'http://external/evil.json'}
    with pytest.raises(SpecError, match='External'):
        swagger2_to_openapi3(swagger)


def test_dolibarr_reconstruct_compiles_unreviewed_manifest_and_integral_ids(tmp_path):
    report = reconstruct(get_fixture(), tmp_path)
    assert report['coverage']['resource_count'] == 2
    assert report['assurance'].startswith('contract_only_unreviewed')
    spec = load_spec(tmp_path / 'draft_manifest.yaml')
    assert not spec['reviewed']
    with _engine_context(spec) as engine:
        result = engine.call('dolibarr.thirdparties.create', {'name': 'Mock Customer', 'client': 1}, role='test_agent')
        assert result['result']['id'] == 1
        assert isinstance(result['result']['id'], int)
        got = engine.call('dolibarr.thirdparties.get', {'id': 1}, role='test_agent')
        assert got['result']['name'] == 'Mock Customer'


class _engine_context:
    def __init__(self, spec): self.sim = SimulationEngine(spec, allow_draft=True)
    def __enter__(self): return self.sim
    def __exit__(self, *_): self.sim.close()


def test_fake_server_discovery_and_holdout_with_write_block(tmp_path):
    reconstruct(get_fixture(), tmp_path)
    spec = load_spec(tmp_path / 'draft_manifest.yaml')
    suite = load_spec(FIXTURE / 'example_probes.yaml')
    fake = FakeTransport()
    client = DolibarrClient('http://127.0.0.1:8088', 'unit-test-key', transport=fake)
    discover = run_probes(spec, suite, client, phase='discovery')
    assert discover['tests_scored'] == 2
    assert discover['match_rate'] == 1.0  # Trivial empty-array shape parity only
    holdout = run_probes(spec, suite, client, phase='holdout')
    assert holdout['tests_scored'] == 2
    assert holdout['tests_matched'] == 2  # Only identical 404 statuses
    assert {v['result'] for v in holdout['probes']} == {'match', 'write_blocked', 'dependency_unavailable'}
    assert not any(method != 'GET' for method, path in fake.calls)


def test_fake_server_contract_declared_id_shape_and_id_capture(tmp_path):
    reconstruct(get_fixture(), tmp_path)
    spec = load_spec(tmp_path / 'draft_manifest.yaml')
    suite = load_spec(FIXTURE / 'example_probes.yaml')
    fake = FakeTransport()
    client = DolibarrClient('http://localhost:8088', 'unit-test-key', transport=fake)
    with pytest.raises(ValueError, match='confirmation'):
        run_probes(spec, suite, client, phase='holdout', allow_writes=True)
    report = run_probes(spec, suite, client, phase='holdout', allow_writes=True, confirm=CONFIRM)
    assert report['tests_scored'] == 4
    assert report['tests_matched'] == 4
    create = next(x for x in report['probes'] if x['id'] == 'create-thirdparty')
    assert create['result'] == 'match'  # Swagger declares numeric create ID; adapter projects it.
    assert create['live_shape'] == create['sim_shape'] == {'kind': 'number'}
    assert create['captured_id']
    assert fake.thirdparties[1]['name'] == 'OrgSim Gate Temporary Customer'
    assert not any('Customer' in json.dumps(r) for r in report['probes'])


def test_client_security_and_capture(tmp_path):
    with pytest.raises(ValueError, match='HTTPS'):
        DolibarrClient('http://not-local.example', 'k')
    with pytest.raises(ValueError, match='credentials'):
        DolibarrClient('http://user:pass@localhost:8088', 'k')
    with pytest.raises(ValueError, match='never in URL'):
        DolibarrClient('https://example.com?DOLAPIKEY=stolen', 'k')
    fake = FakeTransport()
    client = DolibarrClient('http://localhost:8088', 'unit-test-key', transport=fake)
    out = tmp_path / 'swagger.json'
    capture = capture_swagger(client, out)
    assert capture['operations'] == 9
    assert 'unit-test-key' not in out.read_text()


def test_mismatched_path_is_rejected_without_remote_calls(tmp_path):
    reconstruct(get_fixture(), tmp_path)
    spec = load_spec(tmp_path / 'draft_manifest.yaml')
    fake = FakeTransport()
    suite = {'cases': [{'id': 'bad', 'phase': 'holdout', 'method': 'GET',
                        'path': '/setup/configuration',
                        'tool': 'dolibarr.thirdparties.get', 'args': {'id': 1}}]}
    report = run_probes(spec, suite, DolibarrClient('http://localhost:8088', 'unit-test-key', transport=fake), phase='holdout')
    assert report['tests_scored'] == 0
    assert report['probes'][0]['result'] == 'skipped'
    assert not fake.calls


def test_model_bundle_rejects_holdout_leak(tmp_path):
    from orgsim.dolibarr_gate import make_model_bundle
    reconstruct(get_fixture(), tmp_path)
    manifest = load_spec(tmp_path / 'draft_manifest.yaml')
    coverage = load_spec(tmp_path / 'coverage.json')
    with pytest.raises(ValueError, match='holdout'):
        make_model_bundle(manifest, coverage, {'phase': 'holdout', 'probes': [{'id': 'secret'}]})
    result = make_model_bundle(manifest, coverage, {'phase': 'discovery', 'probes': [{'id': 'public'}]})
    assert result['discovery_results'] == [{'id': 'public'}]
    assert 'holdout_results' not in result


def test_import_separates_create_request_types_from_read_response_types(tmp_path):
    # A real API can accept an integer flag and return a string. An imported
    # request must not be overwritten by the response-only representation.
    swagger = get_fixture()
    swagger['definitions']['Thirdparty']['properties']['client'] = {'type': 'string'}
    swagger['paths']['/thirdparties']['post']['parameters'][0]['schema'] = {
        'type': 'object', 'required': ['name'], 'additionalProperties': False,
        'properties': {'name': {'type': 'string'}, 'client': {'type': 'integer'}}
    }
    report = reconstruct(swagger, tmp_path)
    assert report['coverage']['mapped_operations'] == 7
    spec = load_spec(tmp_path / 'draft_manifest.yaml')
    rc = spec['systems']['dolibarr']['resources']['thirdparties']
    assert rc['create_input_schema']['properties']['client']['type'] == 'integer'
    # Internal storage can contain either representation, without claiming
    # to synthesize the GET response's string conversion.
    assert set(rc['schema']['properties']['client']['type']) == {'integer', 'string'}
    with _engine_context(spec) as engine:
        result = engine.call('dolibarr.thirdparties.create', {'name': 'Fixture', 'client': 1}, role='test_agent')
        assert result['result']['client'] == 1
        with pytest.raises(Exception) as exc:
            engine.call('dolibarr.thirdparties.create', {'name': 'Bad', 'client': 'invalid'}, role='test_agent')
        assert getattr(exc.value, 'code') == 'REQUEST_SCHEMA_VIOLATION'
        assert exc.value.diagnostics['path'] == 'client'
        assert exc.value.diagnostics['validator'] == 'type'


def test_synthetic_seed_reconciles_list_lengths_and_never_writes_reference(tmp_path):
    reconstruct(get_fixture(), tmp_path)
    spec = load_spec(tmp_path / 'draft_manifest.yaml')
    suite = {'sim_seed_calls': [{'tool': 'dolibarr.thirdparties.create',
                                 'args': {'id': 1, 'name': 'Fixture Customer', 'client': 1}}],
             'cases': [{'id': 'seeded-list', 'phase': 'discovery', 'method': 'GET',
                        'path': '/thirdparties', 'tool': 'dolibarr.thirdparties.list'}]}
    fake = FakeTransport()
    fake.thirdparties[1] = {'id': 1, 'name': 'Fixture Customer', 'client': 1}
    report = run_probes(spec, suite, DolibarrClient('http://localhost:8088', 'unit-test-key', transport=fake), phase='discovery')
    assert report['sim_seed']['applied'] == 1
    assert report['sim_seed']['failures'] == []
    assert report['tests_matched'] == 1
    assert fake.calls == [('GET', '/thirdparties')]


def test_import_keeps_response_only_required_fields_out_of_create_validation(tmp_path):
    swagger = get_fixture()
    swagger['paths']['/thirdparties']['post']['parameters'][0]['schema'] = {
        'type': 'object', 'properties': {'name': {'type': 'string'}}, 'required': ['name']
    }
    swagger['definitions']['Thirdparty']['required'].append('date_creation')
    swagger['definitions']['Thirdparty']['properties']['date_creation'] = {'type': 'string'}
    reconstruct(swagger, tmp_path)
    spec = load_spec(tmp_path / 'draft_manifest.yaml')
    with _engine_context(spec) as engine:
        result = engine.call('dolibarr.thirdparties.create', {'name': 'Fixture'}, role='test_agent')
        assert result['result']['id'] == 1
        assert 'date_creation' not in result['result']


def test_imported_create_does_not_require_server_generated_id(tmp_path):
    swagger = get_fixture()
    swagger['definitions']['Thirdparty']['required'].append('id')
    reconstruct(swagger, tmp_path)
    spec = load_spec(tmp_path / 'draft_manifest.yaml')
    with _engine_context(spec) as engine:
        row = engine.call('dolibarr.thirdparties.create', {'name': 'Fixture', 'client': 1}, role='test_agent')
        assert row['result']['id'] == 1


def test_numeric_post_response_overrides_opaque_string_id_generation(tmp_path):
    """Regression for observed Dolibarr POST 200 number, GET id as string."""
    swagger = get_fixture()
    swagger['definitions']['Thirdparty']['properties']['id'] = {'type': 'string'}
    reconstruct(swagger, tmp_path)
    spec = load_spec(tmp_path / 'draft_manifest.yaml')
    rc = spec['systems']['dolibarr']['resources']['thirdparties']
    assert rc['schema']['properties']['id']['type'] == 'string'
    assert rc['response_contract']['create']['kind'] == 'number'
    from orgsim.dolibarr_gate import _response_from_contract
    with _engine_context(spec) as sim:
        first = sim.call('dolibarr.thirdparties.create', {'name': 'Fixture 1', 'client': 1}, role='test_agent')['result']
        assert first['id'] == '1'
        status, result = _response_from_contract(first, rc, 'create')
        assert status == 200 and result == 1 and isinstance(result, int)
        assert sim.call('dolibarr.thirdparties.get', {'id': result}, role='test_agent')['result']['id'] == '1'
        # A manually supplied, integral ID should move the allocation cursor.
        sim.call('dolibarr.thirdparties.create', {'name': 'Fixture 12', 'id': '12'}, role='test_agent')
        third = sim.call('dolibarr.thirdparties.create', {'name': 'Fixture 13'}, role='test_agent')['result']
        assert third['id'] == '13'


def test_generic_id_allocation_skips_explicit_opaque_collision():
    import copy
    from orgsim.spec import load_spec
    spec = copy.deepcopy(load_spec('examples/demo_org.yaml'))
    with _engine_context(spec) as sim:
        # Choose a resource with a generic opaque string identity; its default
        # allocator must not override existing explicitly seeded identities.
        system, resource = next((s, r) for s, sc in spec['systems'].items()
                                for r, rc in sc['resources'].items()
                                if rc['schema']['properties'][rc.get('id_field', 'id')].get('type') == 'string'
                                and 'create' in rc.get('allow', {}))
        rc = spec['systems'][system]['resources'][resource]
        assert sim._numeric_identity_mode(system, resource) == 'opaque_string'
        first_id = f'{system}_{resource}_000001'
        # DB-level fixture setup bypasses business rules; only exercises ID collision logic.
        with sim._transaction():
            sim.db.execute('INSERT INTO entities(system,resource,id,body) VALUES(?,?,?,?)',
                           (system, resource, first_id, json.dumps({rc.get('id_field','id'): first_id})))
            allocated = sim._next_id(system, resource)
        assert allocated == f'{system}_{resource}_000002'


def test_discovery_shape_gap_summary_not_holdout():
    from orgsim.dolibarr_gate import summarize_discovery_gaps
    report = {'phase': 'discovery', 'probes': [{
        'id': 'read-customer',
        'live_shape': {'kind': 'object', 'keys': ['id', 'name', 'status', 'list'],
                       'types': {'id': 'string', 'name': 'string', 'status': 'string', 'list': 'array'}},
        'sim_shape': {'kind': 'object', 'keys': ['id', 'name', 'extra'],
                      'types': {'id': 'number', 'name': 'string', 'extra': 'null'}}}]}
    gaps = summarize_discovery_gaps(report)
    assert gaps['phase'] == 'discovery'
    assert [x['field'] for x in gaps['cases'][0]['missing_from_simulator']] == ['list', 'status']
    assert gaps['cases'][0]['type_conflicts'] == [{'field': 'id', 'observed_kind': 'string',
                                                  'simulated_kind': 'number'}]
    assert gaps['cases'][0]['verified_values'] == 0
    assert 'super-secret' not in json.dumps(gaps)
    with pytest.raises(ValueError, match='discovery'):
        summarize_discovery_gaps({'phase': 'holdout', 'probes': report['probes']})


def test_semantic_value_metrics_compare_selected_fields_only():
    from orgsim.dolibarr_gate import _semantic_value_metrics
    fields = {'compare_fields': ['name', 'status'], 'match_items_by': 'name'}
    live = [{'name': 'Synthetic A', 'status': '0', 'secret': 'DO-NOT-LOG'},
            {'name': 'Synthetic B', 'status': '1'}]
    sim = [{'name': 'Synthetic B', 'status': '0'}, {'name': 'Synthetic A', 'status': '0'}]
    result = _semantic_value_metrics(live, sim, fields)
    assert result['semantic_status'] == 'compared'
    assert result['semantic_compared'] == 4
    assert result['semantic_matched'] == 3
    assert 'Synthetic' not in str(result)
    assert 'DO-NOT-LOG' not in str(result)
    assert _semantic_value_metrics(live, sim, {'compare_fields': ['status']})['semantic_status'] == 'unavailable_alignment_key_required'


def test_gate_scores_semantic_mismatches_without_output_values(tmp_path):
    reconstruct(get_fixture(), tmp_path)
    spec = load_spec(tmp_path / 'draft_manifest.yaml')
    fake = FakeTransport()
    fake.thirdparties[1] = {'id': 1, 'name': 'Synthetic A', 'client': 1}
    suite = {
        'sim_seed_calls': [{'tool': 'dolibarr.thirdparties.create',
                            'args': {'id': 1, 'name': 'Synthetic A', 'client': 2}}],
        'cases': [{'id': 'semantic-difference', 'phase': 'discovery', 'method': 'GET',
                   'path': '/thirdparties/1', 'tool': 'dolibarr.thirdparties.get', 'args': {'id': 1},
                   'compare_fields': ['name', 'client']}]
    }
    result = run_probes(spec, suite, DolibarrClient('http://localhost:8088', 'unit-test-key', transport=fake), phase='discovery')
    assert result['tests_scored'] == 1 and result['tests_matched'] == 0
    assert result['semantic_fields_scored'] == 2
    assert result['semantic_fields_matched'] == 1
    assert result['probes'][0]['structural_shape_match'] is True
    assert result['probes'][0]['semantic_match_rate'] == 0.5
    assert 'Synthetic A' not in json.dumps(result)


def test_model_catalog_preserves_unmapped_action_signatures_without_examples():
    from orgsim.dolibarr_contract import model_operation_catalog
    swagger = get_fixture()
    swagger['host'] = 'private.internal.invalid'
    swagger['definitions']['Thirdparty']['properties']['name']['example'] = 'SECRET-NEVER-COPY'
    catalog = model_operation_catalog(swagger)
    assert len(catalog) == 9
    action = next(x for x in catalog if x['path'] == '/orders/{id}/validate')
    assert action['method'] == 'POST'
    assert action['summary'] == ''
    create = next(x for x in catalog if x['path'] == '/thirdparties' and x['method'] == 'POST')
    assert any(x['name'] == 'name' for x in create['request']['fields'])
    encoded = json.dumps(catalog)
    assert 'SECRET-NEVER-COPY' not in encoded
    assert 'private.internal.invalid' not in encoded


def test_model_bundle_can_include_saved_swagger_catalog_without_holdout(tmp_path):
    from orgsim.dolibarr_gate import make_model_bundle
    reconstruct(get_fixture(), tmp_path)
    manifest = load_spec(tmp_path / 'draft_manifest.yaml')
    coverage = load_spec(tmp_path / 'coverage.json')
    bundle = make_model_bundle(manifest, coverage, {'phase': 'discovery', 'probes': []}, swagger=get_fixture())
    assert len(bundle['operation_catalog']) == 9
    assert 'Swagger' in bundle['operation_catalog_source']
    assert bundle['discovery_results'] == []
    assert 'holdout_results' not in bundle
