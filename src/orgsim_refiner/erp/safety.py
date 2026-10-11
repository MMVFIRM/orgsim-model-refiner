"""Network-safe Dolibarr read client and explicit disposable-clone write permit.

This is a *client* safeguard, not permission granted by Dolibarr. Always restrict
server-side API rights independently. Never embed keys in a permit or reports.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

LOCAL_HOSTS = frozenset({'127.0.0.1', 'localhost', '::1'})
ACKNOWLEDGMENT = 'I CONFIRM THIS IS A DISPOSABLE CLONE'
RESOURCE = re.compile(r'^/(?:orders|products|thirdparties|invoices)/[1-9][0-9]*$')
MARKER = re.compile(r'^ORGSIM-CLONE-[A-Z0-9]{12,48}$')


class ConnectionSafetyError(RuntimeError):
    pass


class AmbiguousWriteError(ConnectionSafetyError):
    """The mutation might have committed. DO NOT retry automatically."""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _origin(url: str, *, remote_read: bool = False, write: bool = False) -> str:
    u = urllib.parse.urlsplit(url)
    if u.scheme not in ('http', 'https') or not u.hostname or u.username or u.password or u.query or u.fragment:
        raise ConnectionSafetyError('URL requires an http(s) origin without credentials, query or fragment')
    if u.path not in ('', '/'):
        raise ConnectionSafetyError('Use the Dolibarr server origin, not an arbitrary API path')
    if u.hostname not in LOCAL_HOSTS:
        if write or not remote_read or u.scheme != 'https':
            raise ConnectionSafetyError('Writes must target loopback; remote reads require explicit opt-in and TLS')
    if u.port is not None and not (1 <= u.port <= 65535):
        raise ConnectionSafetyError('Bad port')
    host = '[' + u.hostname + ']' if ':' in u.hostname else u.hostname.lower()
    return f'{u.scheme}://{host}' + (f':{u.port}' if u.port else '')


def _opener():
    # No environment proxy or implicit cross-origin redirect when sending an API key.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())


def request_json(base: str, key: str, method: str, path: str, *, body=None,
                 timeout: float = 20, max_bytes: int = 1024 * 1024,
                 opener=None) -> tuple[int, object]:
    if not key or '\n' in key or '\r' in key:
        raise ConnectionSafetyError('API key must be supplied securely via environment')
    if path.startswith('//') or '..' in path or '?' in path or '#' in path or '%' in path or '\\' in path or not path.startswith('/'):
        raise ConnectionSafetyError('Unsafe API path')
    if not re.fullmatch(r'/[A-Za-z0-9/_-]+', path):
        raise ConnectionSafetyError('Unsupported API path characters')
    payload = None if body is None else json.dumps(body, separators=(',', ':')).encode('utf-8')
    req = urllib.request.Request(base + '/api/index.php' + path, method=method.upper(), data=payload,
        headers={'DOLAPIKEY': key, 'Accept': 'application/json', 'Content-Type': 'application/json'})
    handler = opener if opener is not None else _opener()
    try:
        with handler.open(req, timeout=timeout) as result:
            data = result.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise ConnectionSafetyError('Response exceeded configured size cap')
            return int(result.status), _decode(data)
    except urllib.error.HTTPError as exc:
        # No retry. Redirect is a refusal, never forward credentials.
        if 300 <= exc.code < 400:
            raise ConnectionSafetyError('Redirect refused: API credential never forwarded') from None
        data = exc.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ConnectionSafetyError('Error response exceeded configured size cap') from None
        return int(exc.code), _decode(data)


def _decode(data: bytes):
    if not data:
        return None
    try:
        return json.loads(data.decode('utf-8'))
    except (ValueError, UnicodeError):
        # Never send a whole HTML error page or response back to public reports.
        return {'raw': data[:400].decode('utf-8', errors='replace')}


class DolibarrReadOnly:
    """Allow-listed GET of known record types only. No method argument exists."""
    def __init__(self, origin: str, api_key: str, *, allow_remote_read: bool = False,
                 timeout: float = 20, opener=None):
        self.origin = _origin(origin, remote_read=allow_remote_read)
        self._key = api_key
        self._timeout = timeout
        self._opener = opener

    def get(self, family: str, record_id: int) -> dict:
        if family not in ('orders', 'products', 'thirdparties', 'invoices'):
            raise ConnectionSafetyError('Read-only family not allow-listed')
        if type(record_id) is not int or record_id < 1:
            raise ConnectionSafetyError('Record ID must be a positive integer')
        path = f'/{family}/{record_id}'
        assert RESOURCE.fullmatch(path)
        status, result = request_json(self.origin, self._key, 'GET', path,
                                      timeout=self._timeout, opener=self._opener)
        if status != 200 or not isinstance(result, dict):
            raise ConnectionSafetyError(f'GET failed, status {status}; no mutation attempted')
        return result


@dataclass(frozen=True)
class ClonePermit:
    target_origin: str
    expires_at: dt.datetime
    marker_ref: str
    marker_label: str
    max_mutations: int

    @classmethod
    def from_file(cls, filename: str | Path, target_origin: str):
        obj = json.loads(Path(filename).read_text('utf-8'))
        if obj.get('format') != 'orgsim-disposable-clone-permit-v1' or obj.get('acknowledgment') != ACKNOWLEDGMENT:
            raise ConnectionSafetyError('Explicit disposable-clone acknowledgment missing')
        target = _origin(target_origin, write=True)
        if obj.get('target_origin') != target:
            raise ConnectionSafetyError('Permit target and requested clone do not match')
        try:
            exp = dt.datetime.fromisoformat(obj['expires_at'].replace('Z', '+00:00'))
        except (ValueError, TypeError, KeyError):
            raise ConnectionSafetyError('Permit expiration is invalid') from None
        now = dt.datetime.now(dt.timezone.utc)
        if exp.tzinfo is None or not now < exp <= now + dt.timedelta(hours=24):
            raise ConnectionSafetyError('Permit must be unexpired and expire within 24h')
        ref = obj.get('marker_ref', '')
        label = obj.get('marker_label', '')
        if not isinstance(ref, str) or not MARKER.fullmatch(ref):
            raise ConnectionSafetyError('Clone marker reference must be a dedicated ORGSIM-CLONE marker')
        if not isinstance(label, str) or len(label) < 24 or len(label) > 128:
            raise ConnectionSafetyError('Unique clone marker label is missing')
        limit = obj.get('max_mutations')
        if type(limit) is not int or not 1 <= limit <= 100000:
            raise ConnectionSafetyError('Explicit mutation ceiling 1..100000 required')
        return cls(target, exp, ref, label, limit)

    def assert_current(self):
        if dt.datetime.now(dt.timezone.utc) >= self.expires_at:
            raise ConnectionSafetyError('Disposable-clone permit expired; stop writes')

    def verify_sentinel(self, api_key: str, *, opener=None):
        """Operator must first create the unique product marker in the clone, NOT production.

        A marker alone is not an independent clone attestation: combine with an
        isolated Docker/network profile and write-only-on-clone account.
        """
        self.assert_current()
        status, payload = request_json(self.target_origin, api_key, 'GET',
            '/products/ref/' + self.marker_ref, opener=opener)
        if status != 200 or not isinstance(payload, dict) or payload.get('ref') != self.marker_ref or payload.get('label') != self.marker_label:
            raise ConnectionSafetyError('Disposable-clone sentinel not confirmed; no writes allowed')
        return True
