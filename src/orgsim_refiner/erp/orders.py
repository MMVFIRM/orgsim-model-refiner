"""Dolibarr sales-order validation as a learning task.

Features describe a one- or two-product order: the primary product (sale flag,
kind, stock, up to two lines of it) and an optional second product.
qty_a2==0 means no second line of product A; qty_b==0 means no product B.
"""
from __future__ import annotations
from ..types import Scope, Schema, Field, Observation

SCOPE = Scope('dolibarr-lab', 'order.validate', 'stock_on_validate=1;negative_stock=disallowed')
SCHEMA = Schema((
    Field('sale_a', 'bool'),
    Field('kind_a', 'enum', values=('good', 'service')),
    Field('stock_a', 'int', 0, 40),
    Field('qty_a1', 'int', 1, 20),
    Field('qty_a2', 'int', 0, 20),
    Field('sale_b', 'bool'),
    Field('stock_b', 'int', 0, 40),
    Field('qty_b', 'int', 0, 20),
))


def twin_decision(x: dict) -> str:
    """Rule observed on Dolibarr 24.0.1 (see lab probes); used only as a fast twin."""
    ok = x['sale_a'] and (x['kind_a'] == 'service' or x['stock_a'] >= x['qty_a1'] + x['qty_a2'])
    if x['qty_b'] > 0:
        ok = ok and x['sale_b'] and x['stock_b'] >= x['qty_b']
    return 'APPROVE' if ok else 'REJECT'


def twin_callback(x: dict) -> Observation:
    return Observation(SCOPE, x, twin_decision(x), provenance='SIMULATOR_OBSERVED', source_id='twin')
