"""Black-box, intervention-trained executable policy memory for Model Refiner.

This module never reads OrgSim manifests, private Policy fields, or threshold
settings. It consumes a limited query function and separately labelled examples.

Scope: public workflow guards plus a monotone upper veto/sum-bound grammar.  It is
NOT a generic ERP program synthesizer. The intervention catalog names *public*
state/action fields and their ranges; hidden constants must be learned by querying.
Unrepresentable policies fail the audit and return ABSTAIN, not guessed ACCEPT.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from copy import deepcopy
import hashlib
import json
import random
from typing import Callable

from .oracle import canonical, digest
from .learning import predict_generic

VERSION='simlearn-structured-policy-v1'


def neutral(op:str)->dict:
    """Public-schema neutral starting state; no private rules or policy access."""
    ids={'validate_order':'o1','close_order':'o1','pay_invoice':'i1','reserve_stock':'p1'}
    if op not in ids:raise ValueError(f'Unknown workflow: {op}')
    return {'operation':op,
      'state':{
        'customer':{'credit':1000,'received':0,'closes':0},
        'product':{'stock':1000,'reserved':0},
        'order':{'customer_id':'c1','product_id':'p1','qty':1,'amount':0,'status':'draft',
                 'shipped':30,'invoiced':True,'allocated':0},
        'invoice':{'customer_id':'c1','balance':1000,'payment_count':0,'status':'issued'}},
      'action':{'id':ids[op],**({'payment':1} if op=='pay_invoice' else
                              {'reserve_qty':1} if op=='reserve_stock' else {})}}


def counterfactual(op:str,**kwargs)->dict:
    s=neutral(op);r=s['state'];a=s['action']
    if op=='validate_order':
        r['order']['qty']=kwargs.get('qty',1)
        r['order']['amount']=kwargs.get('amount',0)
    elif op=='pay_invoice':
        r['invoice']['payment_count']=kwargs.get('count',0)
        a['payment']=kwargs.get('payment',1)
    elif op=='reserve_stock':
        r['product']['reserved']=kwargs.get('reserved',0)
        a['reserve_qty']=kwargs.get('qty',1)
    elif op=='close_order':
        r['order']['status']='validated'
        r['order']['qty']=kwargs.get('qty',1)
        r['order']['amount']=kwargs.get('amount',0)
        r['order']['shipped']=kwargs.get('shipped',30)
        if r['order']['shipped']<r['order']['qty']:
            raise ValueError('Counterfactual must satisfy public shipping guard')
    else:raise ValueError(op)
    assert predict_generic({'sample':s})=='ACCEPT'
    return s


def value(sample:dict,expression:str)->int:
    if expression=='product.reserved+action.reserve_qty':
        return sample['state']['product']['reserved']+sample['action']['reserve_qty']
    top,field=expression.split('.')
    obj=sample['action'] if top=='action' else sample['state'][top]
    v=obj[field]
    if type(v) is not int:raise TypeError(f'Not integer: {expression}')
    return v


@dataclass
class Program:
    operation:str
    guards:list[dict]=field(default_factory=list)
    audit_ready:bool=False
    audit_total:int=0
    audit_mismatches:int=0
    queries:int=0
    evidence_hash:str=''
    scope:str='PUBLIC_GUARD_PLUS_MONOTONE_UPPER_VETO'
    def to_dict(self)->dict:
        return {'version':VERSION,'operation':self.operation,'public_guards':'public_preconditions',
                'guards':deepcopy(self.guards),'audit_ready':self.audit_ready,
                'audit_total':self.audit_total,'audit_mismatches':self.audit_mismatches,
                'queries':self.queries,'evidence_hash':self.evidence_hash,
                'provenance':'SYNTHETIC_ORGSIM_ORACLE_OBSERVATIONS_ONLY',
                'scope':self.scope,'real_ERP_behavior_verified':False}
    @classmethod
    def from_dict(cls,x:dict):
        expected={'version','operation','public_guards','guards','audit_ready','audit_total',
                  'audit_mismatches','queries','evidence_hash','provenance','scope','real_ERP_behavior_verified'}
        if set(x)!=expected or x['version']!=VERSION or x['public_guards']!='public_preconditions' or x['real_ERP_behavior_verified'] is not False:
            raise ValueError('Invalid structured policy artifact')
        if type(x['audit_ready']) is not bool or type(x['guards']) is not list:raise ValueError('Invalid policy')
        if x['scope']!='PUBLIC_GUARD_PLUS_MONOTONE_UPPER_VETO':raise ValueError('Unknown policy scope')
        allowed={'upper','sum_upper','pair_upper'}
        feature_allow={'order.qty','order.amount','order.shipped','invoice.payment_count',
                       'action.payment','action.reserve_qty','product.reserved',
                       'product.reserved+action.reserve_qty'}
        for rule in x['guards']:
            if not isinstance(rule,dict) or rule.get('kind') not in allowed:
                raise ValueError('Unknown guard kind')
            kind=rule['kind']
            expected={'kind','expression','max'} if kind in ('upper','sum_upper') else {'kind','a','b','ge_a','ge_b'}
            if set(rule)!=expected:raise ValueError('Malformed guard')
            names=[rule['expression']] if kind in ('upper','sum_upper') else [rule['a'],rule['b']]
            if any(name not in feature_allow for name in names):raise ValueError('Unknown feature')
            keys=['max'] if kind in ('upper','sum_upper') else ['ge_a','ge_b']
            if any(type(rule[k]) is not int or abs(rule[k])>10_000 for k in keys):
                raise ValueError('Invalid threshold')
        return cls(operation=x['operation'],guards=deepcopy(x['guards']),audit_ready=x['audit_ready'],
                   audit_total=x['audit_total'],audit_mismatches=x['audit_mismatches'],
                   queries=x['queries'],evidence_hash=x['evidence_hash'])
    def classify(self,sample:dict,*,fail_closed:bool=True)->str:
        if sample.get('operation')!=self.operation:raise ValueError('Operation mismatch')
        # Trusted public violations can be denied even if hidden-rule audit fails.
        if predict_generic({'sample':sample})=='REJECT':return 'REJECT'
        # An unaudited learned policy cannot authorize a synthetic transaction.
        if not self.audit_ready:return 'ABSTAIN' if fail_closed else 'REJECT'
        for rule in self.guards:
            k=rule['kind']
            if k in ('upper','sum_upper'):
                if value(sample,rule['expression'])>rule['max']:return 'REJECT'
            elif k=='pair_upper':
                if (value(sample,rule['a'])>=rule['ge_a'] and
                    value(sample,rule['b'])>=rule['ge_b']):return 'REJECT'
            else:raise ValueError(k)
        return 'ACCEPT'


class Membership:
    """Limited, memoized black-box query interface; tracks all queried examples."""
    def __init__(self, query:Callable[[dict],dict],op:str,max_queries:int=160):
        if not 20<=max_queries<=1000:raise ValueError('Probe budget outside safe bounds')
        self.query=query;self.op=op;self.limit=max_queries
        self.cache={};self.rows=[]
    def ask(self,sample:dict)->bool:
        if sample['operation']!=self.op:raise ValueError('Operation mismatch')
        if predict_generic({'sample':sample})!='ACCEPT':raise ValueError('Not a public-admissible probe')
        key=digest(sample)
        if key in self.cache:return self.cache[key]
        if len(self.cache)>=self.limit:raise RuntimeError('Probe budget exhausted; fail closed')
        outcome=self.query(deepcopy(sample))
        if outcome.get('provenance')!='SYNTHETIC_ORGSIM_ORACLE':
            raise ValueError('Refusing to train on unverified provenance')
        decision=outcome['decision']
        if decision not in ('ACCEPT','REJECT'):raise ValueError('Invalid oracle response')
        self.cache[key]=(decision=='ACCEPT')
        self.rows.append({**outcome,'sample':deepcopy(sample),'operation':self.op,'id':key,'group':'probe-'+key})
        return self.cache[key]
    def rejects(self,**axis):return not self.ask(counterfactual(self.op,**axis))


def first_rejection(check:Callable[[int],bool],lo:int,hi:int)->int|None:
    """Monotone upper-veto binary search. Independent audit tests its assumption."""
    if not check(hi):return None
    if check(lo):return lo
    while lo+1<hi:
        mid=(lo+hi)//2
        if check(mid):hi=mid
        else:lo=mid
    return hi


def _extract(m:Membership)->list[dict]:
    op=m.op
    if op=='validate_order':
        q=first_rejection(lambda x:m.rejects(qty=x,amount=200),1,24)
        a=first_rejection(lambda x:m.rejects(qty=24,amount=x),0,250)
        if q is None or a is None:raise RuntimeError('Unexpectedly no upper-veto witness')
        return [{'kind':'pair_upper','a':'order.qty','ge_a':q,'b':'order.amount','ge_b':a}]
    if op=='pay_invoice':
        cap=first_rejection(lambda n:m.rejects(count=n,payment=1),0,10)
        if cap is None:raise RuntimeError('Payment guard not identified')
        guards=[{'kind':'upper','expression':'invoice.payment_count','max':cap-1}]
        pair_n=first_rejection(lambda n:m.rejects(count=n,payment=200),0,cap-1) if cap>0 else None
        if pair_n is not None:
            amt=first_rejection(lambda x:m.rejects(count=pair_n,payment=x),1,250)
            if amt is not None:guards.append({'kind':'pair_upper','a':'invoice.payment_count','ge_a':pair_n,
                                               'b':'action.payment','ge_b':amt})
        return guards
    if op=='reserve_stock':
        cap=first_rejection(lambda x:m.rejects(qty=x,reserved=0),1,40)
        if cap is None:raise RuntimeError('Reserve-capacity guard not identified')
        capacity=cap-1
        guards=[{'kind':'sum_upper','expression':'product.reserved+action.reserve_qty','max':capacity}]
        onset=[]
        for q in range(1,min(24,capacity)+1):
            maxr=capacity-q
            if maxr<0:continue
            first=first_rejection(lambda r:m.rejects(qty=q,reserved=r),0,maxr)
            if first is not None:onset.append((q,first))
        if onset:
            guards.append({'kind':'pair_upper','a':'action.reserve_qty','ge_a':onset[0][0],
                           'b':'product.reserved','ge_b':min(v for _,v in onset)})
        return guards
    if op=='close_order':
        cap=first_rejection(lambda x:m.rejects(qty=x,amount=0,shipped=30),1,24)
        if cap is None:raise RuntimeError('Close-capacity guard not identified')
        guards=[{'kind':'upper','expression':'order.qty','max':cap-1}]
        a=first_rejection(lambda x:m.rejects(qty=1,amount=x,shipped=30),0,250)
        shipped=first_rejection(lambda x:m.rejects(qty=1,amount=250,shipped=x),1,35)
        if a is not None and shipped is not None:
            guards.append({'kind':'pair_upper','a':'order.amount','ge_a':a,
                           'b':'order.shipped','ge_b':shipped})
        return guards
    raise ValueError(op)


def _audit_samples(op:str,guards:list[dict],count:int,seed:int):
    rng=random.Random(seed*31337+len(guards)*71)
    # Random and rule-boundary counterfactuals; no disclosed holdout examples.
    for _ in range(count):
        if op=='validate_order':
            axis={'qty':rng.randint(1,20),'amount':rng.randint(0,200)}
        elif op=='pay_invoice':
            axis={'count':rng.randint(0,8),'payment':rng.randint(1,200)}
        elif op=='reserve_stock':
            axis={'qty':rng.randint(1,22),'reserved':rng.randint(0,22)}
        else:
            qty=rng.randint(1,20)
            axis={'qty':qty,'amount':rng.randint(0,200),'shipped':rng.randint(qty,35)}
        # Half of the checks target learned rule transition surfaces.
        if rng.randrange(2)==0:
            for g in guards:
                if g['kind']=='pair_upper':
                    a=g['ge_a']+rng.choice([-1,0,1]);b=g['ge_b']+rng.choice([-1,0,1])
                    if op=='validate_order':axis={'qty':max(1,a),'amount':max(0,b)}
                    elif op=='pay_invoice':axis={'count':max(0,a),'payment':max(1,b)}
                    elif op=='reserve_stock':axis={'qty':max(1,a),'reserved':max(0,b)}
                    elif op=='close_order':
                        axis={'qty':1,'amount':max(0,a),'shipped':max(1,b)}
                    break
        yield counterfactual(op,**axis)


def _monotonicity_checks(op:str):
    """Deliberately look for holes that a monotone rule grammar cannot express.

    These are development-only membership queries, never sealed evaluation data.
    """
    if op=='validate_order':
        for q in range(1,25):yield counterfactual(op,qty=q,amount=200)
    elif op=='pay_invoice':
        for count in range(0,8):
            for p in (1,2,4,8,16,32,64,128):
                yield counterfactual(op,count=count,payment=p)
    elif op=='reserve_stock':
        for q in range(1,22):
            yield counterfactual(op,qty=q,reserved=1)
    elif op=='close_order':
        for shipped in range(1,24):yield counterfactual(op,qty=1,amount=0,shipped=shipped)


def fit_structured(query:Callable[[dict],dict],op:str,*,seed:int=1,
                   max_queries:int=160,audit_count:int=48,validation:list[dict]|None=None)->tuple[Program,list[dict]]:
    """Train and audit a per-operation program from black-box outcomes only.

    ``query`` receives a synthetic state/action and returns an OrgSim outcome;
    there is no hidden-policy access. Validation is optional pre-heldout data;
    tests are *never* passed here.
    """
    membership=Membership(query,op,max_queries=max_queries)
    try:
        guards=_extract(membership)
        provisional=Program(op,guards,audit_ready=True)
        matches=total=0
        for sample in list(_monotonicity_checks(op))+list(_audit_samples(op,guards,audit_count,seed)):
            observed=membership.ask(sample)
            predicted=provisional.classify(sample)=='ACCEPT'
            total+=1;matches+=predicted==observed
        # Validate on already-existing validation outcomes; does not spend oracle budget.
        for row in validation or []:
            if row['operation']!=op:continue
            total+=1
            matches+=provisional.classify(row['sample'])==row['decision']
        ready=(matches==total and total>0)
    except (RuntimeError,ValueError,TypeError,KeyError):
        guards=[];total=matches=0;ready=False
    evidence=[{'input_sha256':digest(x['sample']),'decision':x['decision']} for x in membership.rows]
    program=Program(op,guards,audit_ready=ready,audit_total=total,
                    audit_mismatches=total-matches,queries=len(membership.rows),
                    evidence_hash=digest(evidence))
    return program,membership.rows


class ModelRefinerPolicyAdapter:
    """External verifiable policy memory attached alongside a Model Refiner artifact.

    With model_proposal, the frozen LLM is advisory only when policy is unknown;
    it cannot authorize a mutation by overriding a deny or abstention.
    This does NOT perform provider inference or modify the native SuiteArtifact.
    """
    def __init__(self,policies:dict[str,Program],*,tenant_fingerprint:str|None=None):
        self.policies=dict(policies)
        self.tenant_fingerprint=tenant_fingerprint
    def decide(self,sample:dict,model_proposal:str|None=None,*,tenant_fingerprint:str|None=None)->dict:
        op=sample['operation']
        policy=self.policies.get(op)
        matched=(self.tenant_fingerprint is None or self.tenant_fingerprint==tenant_fingerprint)
        decision=(policy.classify(sample) if policy else 'ABSTAIN') if matched else 'ABSTAIN'
        if model_proposal is not None and model_proposal not in ('ACCEPT','REJECT','ABSTAIN'):
            raise ValueError('Invalid model proposal')
        return {'decision':decision,'model_proposal':model_proposal,
                'model_override_authorized':False,'executable':decision=='ACCEPT' and policy.audit_ready,
                'source':'external_structured_policy_memory',
                'tenant_binding_matched':matched,
                'requires_independent_real_ERP_validation':True}
    def frozen_payload(self)->dict:
        p={k:v.to_dict() for k,v in sorted(self.policies.items())}
        return {'format':'model-refiner-external-program-memory-v1',
                'tenant_fingerprint':self.tenant_fingerprint,'policies':p,
                'policies_sha256':digest(p),'native_model_weights_modified':False,
                'permitted_execution_target':'synthetic_only','real_ERP_write_authorized':False}
