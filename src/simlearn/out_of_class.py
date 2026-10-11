"""Adversarial synthetic ERP policy family not representable by the monotone grammar.

Evaluation-only. Its private rule extensions are never supplied to the learner.
Each observation still comes from an *executed* OrgSim SQLite workflow.
"""
from __future__ import annotations
from .oracle import Oracle,policy_for,make_manifest,cmp,loc,lit,inp,both,neither,digest
from orgsim.engine import SimulationEngine

class NonMonotoneOracle(Oracle):
    def __init__(self,seed:int):
        self.policy=policy_for(seed)
        self.spec=make_manifest(self.policy,identifier=f'adversarial-synthetic-{seed}')
        O=lambda f:loc('orders',f,'o1')
        I=lambda f:loc('invoices',f,'i1')
        P=lambda f:loc('products',f,'p1')
        extra={
            'validate_order':neither(both(cmp(O('qty'),'eq',lit(3)),cmp(O('amount'),'ge',lit(10)))),
            'pay_invoice':neither(both(cmp(I('payment_count'),'eq',lit(0)),cmp(inp('payment'),'le',lit(4)))),
            'reserve_stock':neither(both(cmp(inp('reserve_qty'),'eq',lit(2)),cmp(P('reserved'),'eq',lit(1)))),
            'close_order':neither(both(cmp(O('amount'),'eq',lit(0)),cmp(O('shipped'),'eq',lit(7)))),
        }
        for op,guard in extra.items():
            self.spec['systems']['erp']['workflows'][op]['preconditions'].append(guard)
        self.manifest_sha256=digest(self.spec)
        self.engine=SimulationEngine(self.spec,allow_draft=True)
        self.queries=0
