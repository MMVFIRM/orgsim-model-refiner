"""v0.8: oblique integer learner, AST artifacts, and the live-oracle safety rules."""
from __future__ import annotations
import random
import pytest
from orgsim_refiner.types import Scope, Schema, Field, Observation
from orgsim_refiner.evidence import TrainingOracle, EvidenceError
from orgsim_refiner.util import digest
from orgsim_refiner.erp import orders
from orgsim_refiner.erp.dolibarr_live import DolibarrLive, LiveError
from orgsim_refiner.expr_learner import ExprPolicyLearner, ExprConfig


def _learn(schema, truth, *, seed=1, budget=6000, noise=None):
    sc = Scope('t', 'op', 'c')

    def cb(x):
        y = truth(x)
        if noise and int(digest(x)[:8], 16) % 100 < noise:
            y = not y
        return Observation(sc, x, 'APPROVE' if y else 'REJECT', provenance='SIMULATOR_OBSERVED')
    rng = random.Random(seed)
    hold = [schema.sample(rng) for _ in range(1500)]
    cal = list({digest(x): x for x in (schema.sample(rng) for _ in range(100))}.values())
    reserved = {digest(x) for x in hold + cal}
    learner = ExprPolicyLearner(schema, sc, ExprConfig(seed=seed + 1000))
    oracle = TrainingOracle(schema, sc, cb, budget=budget, forbidden_hashes=reserved)
    first = schema.sample(rng)
    while digest(first) in reserved:
        first = schema.sample(rng)
    art = learner.fit([cb(first)], oracle)
    seen = {r.sample_hash for r in learner.evidence.rows}
    cal_rows = [Observation.from_dict({**cb(x).to_dict(), 'split': 'validation'}) for x in cal if digest(x) not in seen]
    art = learner.calibrate(art, cal_rows)
    acc = sum(art.predict(x, sc, advisory=True)['decision'] == ('APPROVE' if truth(x) else 'REJECT') for x in hold) / len(hold)
    return learner, art, acc, oracle


def test_learns_sum_versus_field_diagonal():
    schema = Schema((Field('stock', 'int', 0, 40), Field('q1', 'int', 1, 20), Field('q2', 'int', 0, 20), Field('ok', 'bool')))
    learner, art, acc, oracle = _learn(schema, lambda x: x['ok'] and x['stock'] >= x['q1'] + x['q2'])
    assert art.ready and learner.converged
    assert acc >= 0.999
    assert art.data['ast'] is not None and not art.data['table']      # executable AST, not a region table
    assert oracle.calls < 2500


def test_learns_credit_limit_with_product_term():
    schema = Schema((Field('price', 'int', 1, 30), Field('qty', 'int', 1, 20), Field('out', 'int', 0, 300), Field('lim', 'int', 0, 600)))
    learner, art, acc, _ = _learn(schema, lambda x: x['out'] + x['price'] * x['qty'] <= x['lim'])
    assert art.ready and acc >= 0.999


def test_dolibarr_order_twin_is_learned_exactly():
    learner, art, acc, oracle = _learn(orders.SCHEMA, lambda x: orders.twin_decision(x) == 'APPROVE')
    assert art.ready and acc >= 0.999


def test_noisy_oracle_is_never_promoted():
    schema = Schema((Field('a', 'int', 0, 30), Field('b', 'int', 0, 30)))
    learner, art, acc, _ = _learn(schema, lambda x: x['a'] + x['b'] <= 30, noise=8, budget=1500)
    assert not art.ready                       # label noise must fail closed
    pred = art.predict({'a': 1, 'b': 1}, Scope('t', 'op', 'c'))
    assert pred['decision'] == 'UNKNOWN'       # unpromoted policies answer UNKNOWN, not guesses


def test_reserved_inputs_are_never_queried():
    schema = Schema((Field('a', 'int', 0, 20), Field('b', 'int', 0, 20)))
    sc = Scope('t', 'op', 'c')
    asked = []

    def cb(x):
        asked.append(digest(x))
        return Observation(sc, x, 'APPROVE' if x['a'] <= x['b'] else 'REJECT', provenance='SIMULATOR_OBSERVED')
    rng = random.Random(0)
    reserved = {digest(schema.sample(rng)) for _ in range(300)}
    learner = ExprPolicyLearner(schema, sc, ExprConfig(seed=2))
    oracle = TrainingOracle(schema, sc, cb, budget=3000, forbidden_hashes=reserved)
    learner.fit([], oracle)
    assert not (set(asked) & reserved)


# ---- live oracle safety (no network: the HTTP layer is replaced by a fake ERP) -------------
class FakeERP(DolibarrLive):
    def __init__(self, *, glitch=0, always_draft=False):
        super().__init__('http://127.0.0.1:1', 'k')
        self.n = 0
        self.glitch = glitch
        self.always_draft = always_draft
        self.stock, self.sale, self.orders, self.status = {}, {}, {}, {}

    def _call(self, method, path, body=None):
        if path == '/thirdparties':
            return 200, 1
        if path == '/products':
            self.n += 1
            self.stock[self.n] = 0
            self.sale[self.n] = body['status']
            self.kind = body['type']
            return 200, self.n
        if path == '/stockmovements':
            self.stock[body['product_id']] += body['qty']
            return 200, 1
        if path == '/orders':
            self.n += 1
            self.orders[self.n] = body['lines']
            self.status[self.n] = 0
            return 200, self.n
        if method == 'GET':
            if path.startswith('/products/'):
                pid=int(path.rsplit('/',1)[1])
                return 200, {'stock_reel': self.stock[pid]}
            oid = int(path.rsplit('/', 1)[1])
            return 200, {'statut': self.status[oid]}
        if path.endswith('/validate'):
            oid = int(path.split('/')[2])
            if self.glitch > 0:
                self.glitch -= 1
                return 500, {'raw': 'technical error'}
            if self.always_draft:
                return 200, {}
            need = {}
            for ln in self.orders[oid]:
                need[ln['fk_product']] = need.get(ln['fk_product'], 0) + ln['qty']
            if any(not self.sale[p] or self.stock[p] < q for p, q in need.items()):
                return 500, {'error': {'message': 'Internal Server Error: Error when validating Order: not enough stock'}}
            for p,q in need.items():
                self.stock[p]-=q
            self.status[oid] = 1
            return 200, {}
        raise AssertionError(path)


SAMPLE = {'sale_a': True, 'kind_a': 'good', 'stock_a': 5, 'qty_a1': 2, 'qty_a2': 0, 'sale_b': True, 'stock_b': 0, 'qty_b': 0}


def test_live_oracle_refuses_non_loopback():
    with pytest.raises(LiveError):
        DolibarrLive('http://erp.example.com', 'k')


def test_live_label_needs_receipt_and_state():
    erp = FakeERP()
    assert erp.callback(SAMPLE).label == 'APPROVE'
    assert erp.callback({**SAMPLE, 'stock_a': 1}).label == 'REJECT'


def test_transient_fault_is_not_retried_and_not_labelled():
    erp = FakeERP(glitch=2)
    with pytest.raises(LiveError):
        erp.callback(SAMPLE)
    assert erp.glitch == 1    # only ONE write attempt; unknown outcomes need reconciliation


def test_http_200_with_draft_order_is_never_an_approval():
    erp = FakeERP(always_draft=True)
    with pytest.raises(LiveError):
        erp.callback(SAMPLE)


# ---- effects: multi-field sums and categorical conditionals -------------------------------
def _effect_rows(n, split, seed):
    from orgsim_refiner.erp.orders import SCOPE
    rng = random.Random(seed)
    rows = []
    while len(rows) < n:
        kind = rng.choice(['good', 'service'])
        q1, q2, stock = rng.randint(1, 9), rng.randint(0, 9), rng.randint(20, 40)
        x = {'k': kind, 'q1': q1, 'q2': q2, 's': stock}
        after = {'s': stock - q1 - q2 if kind == 'good' else stock, 'status': 1}
        rows.append(Observation(SCOPE, x, 'APPROVE', split=split, provenance='SIMULATOR_OBSERVED',
                                before={'s': stock, 'status': 0}, after=after, action={'k': kind, 'q1': q1, 'q2': q2}))
        rows[-1] = Observation.from_dict({**rows[-1].to_dict(), 'sample': {**x, 'u': len(rows) + seed * 1000}})
    return rows


def test_effect_model_learns_multi_line_sum_with_service_exception():
    from orgsim_refiner.effects import EffectModel
    from orgsim_refiner.erp.orders import SCOPE
    m = EffectModel(SCOPE).fit(_effect_rows(40, 'train', 1), _effect_rows(20, 'validation', 2))
    assert m.ready, m.certificate
    good = m.predict({'s': 30, 'status': 0}, {'k': 'good', 'q1': 4, 'q2': 7}, SCOPE)
    svc = m.predict({'s': 30, 'status': 0}, {'k': 'service', 'q1': 4, 'q2': 7}, SCOPE)
    assert good == {'s': 19, 'status': 1} and svc == {'s': 30, 'status': 1}
