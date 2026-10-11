"""v0.9 operational safety tests; all HTTP calls are mocked/offline."""
from __future__ import annotations
import datetime as dt
import io
import json
import urllib.error
from pathlib import Path

import pytest
from orgsim_refiner.erp.safety import (
  _origin,ACKNOWLEDGMENT,ClonePermit,DolibarrReadOnly,
  ConnectionSafetyError,AmbiguousWriteError,request_json)
from orgsim_refiner.erp.dolibarr_live import DolibarrLive,_refusal_kind,LiveError


class FakeResponse:
    def __init__(self, obj, status=200):self.payload=json.dumps(obj).encode();self.status=status
    def __enter__(self):return self
    def __exit__(self,*a):pass
    def read(self,n=None):return self.payload if n is None else self.payload[:n]

class FakeOpener:
    def __init__(self):self.requests=[]
    def open(self,req,timeout):
        self.requests.append(req)
        return FakeResponse({'statut': '0', 'id': 1})


def permit_file(tmp_path, *, origin='http://127.0.0.1:8189', ref='ORGSIM-CLONE-ABCDEF123456', label='ORGSIM-test-private-sentinel-abc123xyz', expires_hours=6):
    path=tmp_path/'permit.json'
    obj={'format':'orgsim-disposable-clone-permit-v1','acknowledgment':ACKNOWLEDGMENT,
        'target_origin':origin,'marker_ref':ref,'marker_label':label,'max_mutations':2,
        'expires_at':(dt.datetime.now(dt.timezone.utc)+dt.timedelta(hours=expires_hours)).isoformat()}
    path.write_text(json.dumps(obj))
    return path

@pytest.mark.parametrize('url',[
    'http://example.com', 'https://user:secret@example.com',
    'https://example.com/?token=abc', 'file:///etc/shadow', 'http://127.0.0.1:8189/evil',
])
def test_unsafe_origins_rejected(url):
    with pytest.raises(ConnectionSafetyError):_origin(url)

def test_explicit_remote_read_requires_https():
    assert _origin('https://erp.example.com',remote_read=True)=='https://erp.example.com'
    with pytest.raises(ConnectionSafetyError):_origin('http://erp.example.com',remote_read=True)
    with pytest.raises(ConnectionSafetyError):_origin('https://erp.example.com',write=True)

def test_read_only_client_get_no_methods_and_families():
    opener=FakeOpener()
    client=DolibarrReadOnly('https://erp.example.com','secret',allow_remote_read=True,opener=opener)
    assert client.get('orders',1)['statut']=='0'
    assert opener.requests[0].get_method()=='GET'
    assert opener.requests[0].get_header('Dolapikey')=='secret'
    with pytest.raises(ConnectionSafetyError):client.get('stockmovements',1)
    with pytest.raises(ConnectionSafetyError):client.get('orders',0)
    assert len(opener.requests)==1

def test_no_credential_redirect_or_path_traversal():
    for path in ('//other.example.com/x', '/orders/%2e%2e', '/orders/1?x=2', '/../../etc'):
        with pytest.raises(ConnectionSafetyError):request_json('http://127.0.0.1','key','GET',path,opener=FakeOpener())

def test_no_write_without_disposable_permit():
    client=DolibarrLive('http://127.0.0.1:8189','key')
    with pytest.raises(LiveError,match='Mutation blocked'):
        client._call('POST','/orders',{})

def test_remote_or_parallel_live_oracle_refused():
    with pytest.raises(LiveError):DolibarrLive('https://erp.example.com','x',allow_non_loopback=True)
    with pytest.raises(LiveError):DolibarrLive('http://127.0.0.1','x',unsafe_parallel=True)

def test_permit_scoped_expiring_and_marker_required(tmp_path,monkeypatch):
    f=permit_file(tmp_path)
    with pytest.raises(ConnectionSafetyError):ClonePermit.from_file(f,'http://127.0.0.1:9999')
    assert ClonePermit.from_file(f,'http://127.0.0.1:8189').max_mutations==2
    monkeypatch.setattr('orgsim_refiner.erp.safety.request_json',lambda *a,**kw:(200,{'ref':'wrong','label':'wrong'}))
    with pytest.raises(ConnectionSafetyError):ClonePermit.from_file(f,'http://127.0.0.1:8189').verify_sentinel('k')
    f.write_text(json.dumps({**json.loads(f.read_text()),'expires_at':'2025-01-01T00:00:00+00:00'}))
    with pytest.raises(ConnectionSafetyError):ClonePermit.from_file(f,'http://127.0.0.1:8189')

def test_write_budget_and_no_retry_after_ambiguous_network_error(tmp_path,monkeypatch):
    f=permit_file(tmp_path)
    monkeypatch.setattr(ClonePermit,'verify_sentinel',lambda *a,**kw:True)
    asked=[]
    def fail(*args,**kw):
        asked.append((args[2],args[3]));raise urllib.error.URLError('timeout')
    monkeypatch.setattr('orgsim_refiner.erp.dolibarr_live.request_json',fail)
    cl=DolibarrLive('http://127.0.0.1:8189','secret',clone_permit=str(f))
    with pytest.raises(AmbiguousWriteError,match='DO NOT RETRY'):cl._call('POST','/orders',{})
    assert len(asked)==1
    assert cl._mutation_count==1

def test_specific_business_reason_required():
    assert _refusal_kind('Internal Server Error: Error when validating Order') is None
    assert _refusal_kind('Database connection failed') is None
    assert _refusal_kind('Error when validating Order: not enough stock')=='insufficient_stock'
    assert _refusal_kind('Product not for sale')=='not_for_sale'

def test_state_effect_report_never_assumes_unchanged_on_reject():
    from pathlib import Path
    script=(Path(__file__).resolve().parents[1]/'tools'/'erp_replicate.py').read_text()
    assert 'actual_state = obs.after' in script
    assert "obs.after if obs.label == 'APPROVE' else before_of(x)" not in script

def test_live_entrypoints_require_explicit_clone_permit():
    root=Path(__file__).resolve().parents[1]/'tools'
    for name in ('erp_experiment.py','erp_replicate.py','erp_drift_check.py'):
        code=(root/name).read_text()
        assert '--clone-permit' in code
        assert 'clone_permit=a.clone_permit' in code
