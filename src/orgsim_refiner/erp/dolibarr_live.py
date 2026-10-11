"""Mutating Dolibarr research oracle: explicitly authorized DISPOSABLE clones only.

Never point this oracle at production. A localhost URL does not itself prove
safety. It requires an expiring permit, a private clone-only sentinel product,
and independent server-side write restrictions. No ambiguous POST is retried.
"""
from __future__ import annotations

import itertools
import re
import threading
import time
import urllib.error
from ..types import Observation
from .orders import SCOPE
from .safety import (
    _origin, ClonePermit, ConnectionSafetyError, AmbiguousWriteError, request_json,
)


class LiveError(ConnectionSafetyError):
    pass


def _refusal_kind(message: str) -> str | None:
    s = message.casefold()
    if any(w in s for w in ('not enough stock', 'insufficient stock', 'negative stock', 'not sufficient stock')):
        return 'insufficient_stock'
    if any(w in s for w in ('not for sale', 'not on sale', 'unavailable for sale',
                            'not available for sale', 'not authorized for sale', 'not allowed for sale')):
        return 'not_for_sale'
    return None


def _error_message(payload) -> str:
    if not isinstance(payload, dict):
        return ''
    err = payload.get('error')
    if isinstance(err, dict):
        return str(err.get('message') or '')[:500]
    if isinstance(err, str):
        return err[:500]
    return str(payload.get('message') or '')[:500]


class DolibarrLive:
    def __init__(self, base_url: str, api_key: str, *, warehouse_id: int = 1,
                 allow_non_loopback: bool = False, timeout: float = 30,
                 unsafe_parallel: bool = False, capture_state: bool = True,
                 clone_permit: str | None = None):
        if allow_non_loopback:
            raise LiveError('Mutating oracle refuses non-loopback targets even with an override')
        if unsafe_parallel:
            raise LiveError('Parallel live transactions are disabled due to observed Dolibarr races')
        try:
            self.origin = _origin(base_url, write=True)
        except ConnectionSafetyError as exc:
            raise LiveError(str(exc)) from exc
        self.base = self.origin + '/api/index.php'
        self._key = api_key
        self._permit = ClonePermit.from_file(clone_permit, self.origin) if clone_permit else None
        self._permit_checked = False
        self._mutation_count = 0
        self.warehouse = warehouse_id
        if type(warehouse_id) is not int or warehouse_id < 1:
            raise LiveError('Warehouse ID must be a positive integer')
        self.timeout = timeout
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._cust_lock = threading.Lock()
        self._serial = threading.Lock()
        self._run = format(int(time.time()), 'x')
        self.customer = None
        self.calls = 0
        self.capture_state = capture_state

    def _call(self, method, path, body=None):
        method = method.upper()
        if method not in ('GET', 'POST'):
            raise LiveError('Research oracle permits only GET and explicitly authorized POST')
        if method == 'POST':
            if not re.fullmatch(r'/(?:thirdparties|products|stockmovements|orders|orders/[1-9][0-9]*/validate)', path):
                raise LiveError('Mutation endpoint not permitted by disposable-lab oracle')
            if self._permit is None:
                raise LiveError('Mutation blocked: provide an expiring --clone-permit for an isolated lab clone')
            self._permit.assert_current()
            if not self._permit_checked:
                self._permit.verify_sentinel(self._key)
                self._permit_checked = True
            if self._mutation_count >= self._permit.max_mutations:
                raise LiveError('Mutation budget exhausted; further writes blocked')
            self._mutation_count += 1   # charge BEFORE making the request, including ambiguous attempts
        with self._lock:
            self.calls += 1
        try:
            return request_json(self.origin, self._key, method, path, body=body, timeout=self.timeout)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            if method == 'POST':
                raise AmbiguousWriteError('Write outcome unknown: stop and reconcile in clone; DO NOT RETRY') from exc
            raise LiveError('Read failed; no mutation attempted') from exc

    def _uid(self):
        with self._lock:
            return f'{self._run}{next(self._ids):06d}'

    def ensure_customer(self):
        with self._cust_lock:
            if self.customer is None:
                s, cid = self._call('POST', '/thirdparties',
                    {'name': f'ORGSIM-LAB-CUSTOMER-{self._run}', 'client': 1, 'code_client': '-1'})
                if s != 200:
                    raise LiveError(f'Clone customer create failed (HTTP {s}); no retry')
                self.customer = cid
            return self.customer

    def _product(self, tosell: bool, service: bool, stock: int):
        uid = self._uid()
        s, pid = self._call('POST', '/products', {
            'ref': f'LAB{uid}', 'label': 'lab', 'price': 10, 'price_base_type': 'HT',
            'status': int(tosell), 'status_buy': 1, 'type': 1 if service else 0})
        if s != 200:
            raise LiveError(f'Clone product create failed (HTTP {s}); no retry')
        if not service and stock > 0:
            s, _ = self._call('POST', '/stockmovements', {
                'product_id': pid, 'warehouse_id': self.warehouse, 'qty': stock, 'movementcode': 'LAB'})
            if s != 200:
                raise LiveError(f'Clone stock movement failed (HTTP {s}); no retry')
        return pid

    def _stock(self, pid):
        if pid is None:
            return 0
        s, value = self._call('GET', f'/products/{int(pid)}')
        if s != 200 or not isinstance(value, dict):
            raise LiveError('Product stock verification unavailable; cannot certify effect')
        try:
            return int(float(value.get('stock_reel', 0) or 0))
        except (ValueError, TypeError, OverflowError) as exc:
            raise LiveError('Product stock value unreadable') from exc

    def run_order(self, x: dict):
        with self._serial:
            return self._run_order(x)

    def _run_order(self, x: dict):
        cust = self.ensure_customer()
        pa = self._product(x['sale_a'], x['kind_a'] == 'service', x['stock_a'])
        lines = [(pa, x['qty_a1'])]
        if x['qty_a2'] > 0:
            lines.append((pa, x['qty_a2']))
        pb = None
        if x['qty_b'] > 0:
            pb = self._product(x['sale_b'], False, x['stock_b'])
            lines.append((pb, x['qty_b']))
        s, oid = self._call('POST', '/orders', {'socid': cust, 'date': 1760000000,
            'lines': [{'fk_product': p, 'qty': q, 'subprice': 10, 'tva_tx': 0} for p, q in lines]})
        if s != 200:
            raise LiveError(f'Order create failed (HTTP {s}); no retry')
        s0, o0 = self._call('GET', f'/orders/{oid}')
        if s0 != 200 or not isinstance(o0, dict):
            raise LiveError('Cannot read new order before validation')
        before_status = str(o0.get('statut'))
        if before_status != '0':
            raise LiveError('Order is not draft before validation')
        before_stock = None
        if self.capture_state:
            before_stock = {'stock_a': self._stock(pa) if x['kind_a'] == 'good' else 0,
                            'stock_b': self._stock(pb)}
        # ONE validation attempt. Even a server HTTP 500 may reflect a committed write.
        s, result = self._call('POST', f'/orders/{oid}/validate', {'idwarehouse': self.warehouse})
        s1, o1 = self._call('GET', f'/orders/{oid}')
        if s1 != 200 or not isinstance(o1, dict):
            raise LiveError('Post-write order status cannot be read; unknown outcome')
        after_status = str(o1.get('statut'))
        message = _error_message(result) if s != 200 else ''
        after_stock = None
        if self.capture_state:
            after_stock = {'stock_a': self._stock(pa) if x['kind_a'] == 'good' else 0,
                           'stock_b': self._stock(pb)}
        return {
            'state_before': before_stock, 'state_after': after_stock,
            'order_id': oid, 'http_status': s, 'status_before': before_status,
            'status_after': after_status, 'message': message, 'reason': _refusal_kind(message),
            'validate_attempts': 1,
        }

    def callback(self, x: dict) -> Observation:
        r = self.run_order(x)
        ok = (r['http_status'] == 200 and r['status_before'] == '0' and r['status_after'] == '1')
        refused = (r['http_status'] == 500 and r['reason'] is not None and
                   r['status_before'] == '0' and r['status_after'] == '0')
        # Refusals are only accepted if corroborating before/after product states agree.
        if refused and (r['state_before'] is None or r['state_after'] is None or
                        r['state_before'] != r['state_after']):
            raise LiveError('Rejection not verified: stock state changed or is unknown')
        if not ok and not refused:
            raise LiveError(f'Unclassified business/technical outcome: HTTP {r["http_status"]}; no retry')
        before = after = action = None
        if r['state_before'] is not None and r['state_after'] is not None:
            before = {'order_status': int(r['status_before']), **r['state_before']}
            after = {'order_status': int(r['status_after']), **r['state_after']}
            action = {k: x[k] for k in ('kind_a', 'qty_a1', 'qty_a2', 'qty_b')}
        return Observation(SCOPE, x, 'APPROVE' if ok else 'REJECT',
            provenance='REAL_API_VERIFIED', source_id=f'order:{r["order_id"]}',
            evidence={k: v for k, v in r.items() if k not in ('state_before', 'state_after')},
            before=before, after=after, action=action)
