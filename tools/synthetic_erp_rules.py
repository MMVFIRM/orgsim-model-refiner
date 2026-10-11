"""Additional ERP-style rules for stress-testing the learner on the fast twin.

Each task = (name, Schema, truth(x)). They deliberately go beyond axis-aligned
regions: credit limits (product + sum), discount floors, tier-dependent
branches, two-field ratios, and a line-total cap.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from orgsim_refiner.types import Schema, Field


def S(*f):
    return Schema(tuple(f))

TASKS = {}

TASKS['credit_limit'] = (S(Field('price', 'int', 1, 30), Field('qty', 'int', 1, 20), Field('outstanding', 'int', 0, 300),
                          Field('credit_limit', 'int', 0, 600), Field('blocked', 'bool')),
    lambda x: (not x['blocked']) and x['outstanding'] + x['price'] * x['qty'] <= x['credit_limit'])

TASKS['discount_floor'] = (S(Field('price', 'int', 10, 100), Field('discount', 'int', 0, 60), Field('min_price', 'int', 5, 100),
                            Field('role', 'enum', values=('clerk', 'manager', 'admin'))),
    lambda x: x['role'] == 'admin' or (x['price'] - x['discount'] * x['price'] // 100 >= x['min_price'] if x['role'] == 'manager'
                                         else x['price'] - x['price'] // 10 >= x['min_price'] and x['discount'] <= 10))

TASKS['tier_branches'] = (S(Field('tier', 'enum', values=('std', 'gold', 'plat')), Field('amount', 'int', 0, 200),
                           Field('open_invoices', 'int', 0, 12), Field('limit', 'int', 0, 400), Field('hold', 'bool')),
    lambda x: (not x['hold']) and (x['amount'] <= x['limit'] if x['tier'] == 'std'
                                   else x['amount'] <= x['limit'] + 50 if x['tier'] == 'gold'
                                   else x['open_invoices'] < 10 or x['amount'] <= x['limit'])
                                   and x['open_invoices'] <= 8 + (3 if x['tier'] == 'plat' else 0))

TASKS['three_line_stock'] = (S(Field('stock', 'int', 0, 60), Field('q1', 'int', 1, 15), Field('q2', 'int', 0, 15), Field('q3', 'int', 0, 15),
                              Field('on_sale', 'bool')),
    lambda x: x['on_sale'] and x['stock'] >= x['q1'] + x['q2'] + x['q3'])

TASKS['ratio_cap'] = (S(Field('requested', 'int', 0, 100), Field('limit', 'int', 1, 100), Field('used', 'int', 0, 100), Field('flag', 'bool')),
    lambda x: (not x['flag']) and x['requested'] + x['used'] <= 2 * x['limit'])
