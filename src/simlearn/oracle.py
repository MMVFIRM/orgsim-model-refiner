"""Procedural OrgSim workflow oracles. These are NOT observations of Dolibarr.

All decisions and effects are measured by invoking the actual transactional
OrgSim interpreter. The candidate learners receive inputs and labeled examples,
never the hidden manifest or Policy object.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import random
from typing import Any

from orgsim.engine import SimError, SimulationEngine

OPS = ('validate_order', 'pay_invoice', 'reserve_stock', 'close_order')
RESOURCES = ('customers', 'products', 'orders', 'invoices')


def canonical(x: Any) -> str:
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def digest(x: Any) -> str:
    return sha256(canonical(x).encode('utf-8')).hexdigest()


def lit(v): return {'literal': v}
def inp(key): return {'path': f'input.{key}'}
def loc(resource, field, entity_id):
    return {'lookup': {'system':'erp', 'resource':resource, 'id':lit(entity_id), 'field':field}}
def cmp(left, op, right): return {'compare': {'left':left, 'op':op, 'right':right}}
def both(*xs): return {'all':list(xs)}
def neither(x): return {'not':x}
def step_adj(resource, field, amount, entity_id):
    return {'kind':'adjust','system':'erp','resource':resource,'id':lit(entity_id),'field':field,'by':amount}
def schema(fields):
    return {'type':'object','properties':{'id':{'type':'string'},**{f:{'type':t} for f,t in fields.items()}},
            'required':['id',*fields],'additionalProperties':False}

def res(fields):
    return {'schema':schema(fields), 'allow':{'create':['seed'], 'get':['seed','agent'], 'list':['seed','agent']}}

@dataclass(frozen=True)
class Policy:
    order_qty_veto: int
    order_amount_veto: int
    payment_count_cap: int
    payment_pair_count: int
    payment_pair_amount: int
    reserve_capacity: int
    reserve_pair_qty: int
    reserve_pair_reserved: int
    close_max_qty: int
    close_pair_amount: int
    close_pair_shipped: int


def policy_for(seed:int) -> Policy:
    r=random.Random(seed * 7703 + 191)
    return Policy(
        order_qty_veto=r.randint(5,10),order_amount_veto=r.choice([40,60,80,100,120]),
        payment_count_cap=r.randint(2,5),payment_pair_count=r.randint(1,3),
        payment_pair_amount=r.choice([12,24,36,48,60]),
        reserve_capacity=r.randint(8,19),reserve_pair_qty=r.randint(4,9),
        reserve_pair_reserved=r.randint(3,10),close_max_qty=r.randint(5,11),
        close_pair_amount=r.choice([50,75,100,125]),close_pair_shipped=r.randint(6,13))


def make_manifest(policy:Policy, *, identifier:str='synthetic-erp') -> dict:
    O=lambda f:loc('orders',f,'o1')
    P=lambda f:loc('products',f,'p1')
    C=lambda f:loc('customers',f,'c1')
    I=lambda f:loc('invoices',f,'i1')
    def workflow(preconds,steps,extra=None):
        props={'id':{'type':'string'}}
        if extra:props[extra]={'type':'integer','minimum':1}
        return {'roles':['agent'],'evidence':'hypothesis',
                'input_schema':{'type':'object','properties':props,'required':list(props),'additionalProperties':False},
                'preconditions':preconds,'steps':steps,'result':{'path':'steps.0.after.id'}}
    validate=workflow([
        cmp(O('status'),'eq',lit('draft')),
        cmp(P('stock'),'ge',O('qty')),
        cmp(C('credit'),'ge',O('amount')),
        neither(both(cmp(O('qty'),'ge',lit(policy.order_qty_veto)),
                     cmp(O('amount'),'ge',lit(policy.order_amount_veto)))),
    ],[
        {'kind':'update','system':'erp','resource':'orders','id':lit('o1'),'values':{'status':lit('validated')}},
        step_adj('products','stock',{'negate':O('qty')},'p1'),
        step_adj('customers','credit',{'negate':O('amount')},'c1'),
    ])
    pay=workflow([
        cmp(I('status'),'eq',lit('issued')),
        cmp(I('balance'),'ge',inp('payment')),
        cmp(I('payment_count'),'lt',lit(policy.payment_count_cap)),
        neither(both(cmp(I('payment_count'),'ge',lit(policy.payment_pair_count)),
                     cmp(inp('payment'),'ge',lit(policy.payment_pair_amount)))),
    ],[
        step_adj('invoices','balance',{'negate':inp('payment')},'i1'),
        step_adj('invoices','payment_count',lit(1),'i1'),
        step_adj('customers','received',inp('payment'),'c1'),
    ],'payment')
    reserve=workflow([
        cmp(O('status'),'eq',lit('draft')),
        cmp(P('stock'),'ge',inp('reserve_qty')),
        cmp({'add':[P('reserved'),inp('reserve_qty')]},'le',lit(policy.reserve_capacity)),
        neither(both(cmp(inp('reserve_qty'),'ge',lit(policy.reserve_pair_qty)),
                     cmp(P('reserved'),'ge',lit(policy.reserve_pair_reserved)))),
    ],[
        step_adj('products','stock',{'negate':inp('reserve_qty')},'p1'),
        step_adj('products','reserved',inp('reserve_qty'),'p1'),
        step_adj('orders','allocated',inp('reserve_qty'),'o1'),
    ],'reserve_qty')
    close=workflow([
        cmp(O('status'),'eq',lit('validated')),
        cmp(O('shipped'),'ge',O('qty')),
        cmp(O('invoiced'),'eq',lit(True)),
        cmp(O('qty'),'le',lit(policy.close_max_qty)),
        neither(both(cmp(O('amount'),'ge',lit(policy.close_pair_amount)),
                     cmp(O('shipped'),'ge',lit(policy.close_pair_shipped)))),
    ],[
        {'kind':'update','system':'erp','resource':'orders','id':lit('o1'),'values':{'status':lit('closed')}},
        step_adj('customers','closes',lit(1),'c1'),
    ])
    return {'version':'0.1','organization':identifier,'reviewed':False,
      'systems':{'erp':{
        'resources':{
            'customers':res({'credit':'integer','received':'integer','closes':'integer'}),
            'products':res({'stock':'integer','reserved':'integer'}),
            'orders':res({'customer_id':'string','product_id':'string','qty':'integer','amount':'integer',
                          'status':'string','shipped':'integer','invoiced':'boolean','allocated':'integer'}),
            'invoices':res({'customer_id':'string','balance':'integer','payment_count':'integer','status':'string'}),
        },
        'workflows':{'validate_order':validate,'pay_invoice':pay,'reserve_stock':reserve,'close_order':close},
      }},'integrations':[],'scenarios':[]}


def sample_input(rng:random.Random, op:str, mode:str='mixed') -> dict:
    """Candidate interventions use only public field ranges, never policy values."""
    if mode not in ('mixed','stress'):raise ValueError(mode)
    # Interleave plausible nominal configurations, single-failure probes, and
    # independent draws; stress uses extremes / changed feature correlations.
    state_mode=rng.randrange(8)
    qty=rng.randint(1,12) if mode=='mixed' else rng.choice([1,2,3,4,5,6,8,10,12,13,16])
    amt=rng.choice([0,10,20,30,40,50,60,75,90,110,130,150,180]) if mode=='mixed' else rng.choice([0,35,45,65,85,95,115,145,170,190])
    stock=qty+rng.randint(0,12) if state_mode<5 else rng.randint(0,12)
    credit=amt+rng.randint(0,150) if state_mode<5 else rng.randint(0,190)
    status=('draft' if state_mode<5 else rng.choice(['draft','validated','closed']))
    shipped=(qty+rng.randint(0,8)) if state_mode<5 else rng.randint(0,18)
    pay=rng.randint(1,70) if mode=='mixed' else rng.choice([1,10,20,30,40,50,70,85])
    balance=pay+rng.randint(0,80) if state_mode<5 else rng.randint(0,100)
    invstatus=('issued' if state_mode<5 else rng.choice(['issued','draft','paid']))
    reserve_qty=rng.randint(1,12) if mode=='mixed' else rng.choice([1,2,4,6,8,10,14])
    reserved=rng.randint(0,9 if mode=='mixed' else 15)
    rec={
      'customer':{'credit':credit,'received':rng.randint(0,100),'closes':rng.randint(0,5)},
      'product':{'stock':stock,'reserved':reserved},
      'order':{'customer_id':'c1','product_id':'p1','qty':qty,'amount':amt,'status':status,
               'shipped':shipped,'invoiced':bool(state_mode<5 or rng.random()<0.6),'allocated':rng.randint(0,8)},
      'invoice':{'customer_id':'c1','balance':balance,'payment_count':rng.randint(0,5), 'status':invstatus}}
    if op=='close_order':
        rec['order']['status']='validated' if state_mode<5 else rng.choice(['draft','validated','closed'])
    if op=='pay_invoice':
        rec['invoice']['status']='issued' if state_mode<5 else invstatus
    return {'operation':op,'state':rec,'action':{'id': {'validate_order':'o1','close_order':'o1','reserve_stock':'p1','pay_invoice':'i1'}[op],
         **({'payment':pay} if op=='pay_invoice' else {'reserve_qty':reserve_qty} if op=='reserve_stock' else {})}}


class Oracle:
    """One reusable engine per policy, reset between all episodes."""
    def __init__(self, seed:int, *, identity:str|None=None):
        self.policy=policy_for(seed)
        self.spec=make_manifest(self.policy,identifier=identity or f'synthetic-erp-{seed}')
        self.manifest_sha256=digest(self.spec)
        self.engine=SimulationEngine(self.spec,allow_draft=True)
        self.queries=0

    def close(self):self.engine.close()

    def execute(self, sample:dict)->dict:
        self.queries+=1
        self.engine.reset()
        state=sample['state']
        records={'customers':('c1',state['customer']),'products':('p1',state['product']),
                 'orders':('o1',state['order']),'invoices':('i1',state['invoice'])}
        # Seed directly through simulator's explicit role, not via SQL writes.
        for resource,(record_id,record) in records.items():
            data={'id':record_id, **record}
            self.engine.call(f'erp.{resource}.create',data,role='seed',actor='scenario_setup')
        before=self._state()
        failure=None
        try:
            self.engine.call('erp.workflow.'+sample['operation'],sample['action'],role='agent')
            decision='ACCEPT';status=200
        except SimError as exc:
            decision='REJECT';status=exc.status;failure=exc.code
            if status!=422:raise RuntimeError(f'Unexpected simulator error: {exc.code}: {exc}') from exc
        after=self._state()
        if decision=='REJECT' and before!=after:
            raise AssertionError('Rejected operation modified simulator state')
        changed={f'{r}.{field}':new_value for r,record in after.items() for field,new_value in record.items()
                 if before[r][field]!=new_value}
        return {'decision':decision,'http_status':status,'before':before,'after':after,
                'changed':changed,'error_code':failure, 'provenance':'SYNTHETIC_ORGSIM_ORACLE'}

    def _state(self)->dict:
        return {singular:dict(self.engine.get('erp',resource,ident) or {})
          for singular,resource,ident in [('customer','customers','c1'),('product','products','p1'),
                                          ('order','orders','o1'),('invoice','invoices','i1')]}
