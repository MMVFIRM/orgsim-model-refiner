"""Gate 2A: budget-matched black-box environment discovery, completely offline.

This is a controlled experiment on *algorithmic exploration strategies*, not
proof of LLM learning or Dolibarr parity. Every arm has identical public
contracts, candidate states, example budget, induction algorithm and holdout
states. The hidden synthetic oracle is instantiated only by the evaluator;
learner functions receive (public contract, observed trajectories) only.

To avoid misleading claims, the compact hypothesis class, evaluation protocol,
and failure cases are documented in docs/GATE2A_METHOD.md.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import argparse
import json
import math
from pathlib import Path
import random
from statistics import mean
from typing import Any

from .exploration import EvidenceLedger, state_signatures, structural_delta


@dataclass(frozen=True)
class Operation:
    name: str
    fields: dict[str, tuple]
    derived: dict[str, tuple]
    outcome_fields: tuple[str, ...]
    description: str


def public_operations() -> list[Operation]:
    """Typed tool contracts, public input ranges, and arithmetic affordances.

    The *selected thresholds, state guards, and actual effects* are not here.
    These are general-purpose operation templates. Numeric arithmetic features
    are explicitly available to both arms, much as Python is available in ARC.
    """
    return [
        Operation('orders.validate', {
            'status': ('draft', 'confirmed', 'closed'), 'qty': tuple(range(1, 7)),
            'stock': tuple(range(0, 14)), 'amount': tuple(range(10, 111, 10)),
            'credit': tuple(range(0, 151, 10)), 'record_id': ('A01','A02','A03','A04'),
        }, {'stock_minus_qty': ('stock', 'qty'),
            'credit_minus_amount': ('credit', 'amount')}, ('status','stock'),
            'Validate a sales order; may require an acceptable lifecycle state, available stock, and financial capacity.'),
        Operation('invoices.validate', {
            'status': ('draft', 'open', 'cancelled'), 'line_count': tuple(range(0, 6)),
            'customer_active': (False, True), 'posted': (False,),
            'record_id': ('I01','I02','I03'),
        }, {}, ('status','posted'),
            'Validate an invoice, subject to state and prerequisites.'),
        Operation('invoices.payment', {
            'status': ('draft','open','paid'), 'due': tuple(range(0, 151, 10)),
            'payment': tuple(range(0, 151, 10)), 'customer_match': (False,True),
            'payment_count': tuple(range(0,6)), 'record_id': ('P01','P02','P03'),
        }, {'due_minus_payment': ('due','payment')}, ('due','payment_count'),
            'Register payment against an invoice, subject to state, amount, and identity checks.'),
        Operation('orders.close', {
            'status': ('draft','confirmed','closed'), 'ordered_qty': tuple(range(1,8)),
            'shipped_qty': tuple(range(0,11)), 'disputed': (False, True),
            'record_id': ('C01','C02','C03'),
        }, {'shipped_minus_ordered': ('shipped_qty','ordered_qty')}, ('status',),
            'Close an order when its lifecycle and fulfillment permits.'),
        Operation('inventory.reserve', {
            'product_active': (False,True), 'stock': tuple(range(0,16)),
            'allocated': tuple(range(0,9)), 'qty': tuple(range(1,8)),
            'record_id': ('S01','S02','S03'),
        }, {'stock_minus_allocated_minus_qty': ('stock','allocated','qty')}, ('allocated',),
            'Reserve available inventory for an active product.'),
    ]


def features_for(op: Operation, state: dict) -> dict:
    f = {k: v for k,v in state.items() if k != 'record_id'}
    for name, src in op.derived.items():
        f[name] = f[src[0]] - sum(f[k] for k in src[1:])
    return f


class _SyntheticOracle:
    """Private synthetic oracle; never handed to baseline or active policy."""
    def __init__(self, seed: int):
        r=random.Random(seed * 7411 + 937)
        self.private_rules={
            'orders.validate': (r.choice([0,1,2]),r.choice([0,10,20,30])),
            'invoices.validate': r.choice([1,2,3]),
            'invoices.payment': r.choice([0,10]),
            'orders.close': r.choice([0,1,2]),
            'inventory.reserve': r.choice([0,1,2]),
        }
        self.calls = Counter()

    def query(self, op: Operation, before: dict, *, phase: str='discovery') -> tuple[int,dict,dict]:
        if phase != 'discovery' and phase != 'evaluation':
            raise ValueError('Invalid phase')
        self.calls[(phase,op.name)] += 1
        s=before.copy(); f=features_for(op,s)
        if op.name == 'orders.validate':
            stock_pad, credit_pad = self.private_rules[op.name]
            valid=(s['status']=='draft' and f['stock_minus_qty']>=stock_pad
                   and f['credit_minus_amount']>=credit_pad)
        elif op.name == 'invoices.validate':
            valid=(s['status']=='draft' and s['line_count']>=self.private_rules[op.name]
                   and s['customer_active'] is True)
        elif op.name == 'invoices.payment':
            valid=(s['status']=='open' and s['payment']>self.private_rules[op.name]
                   and f['due_minus_payment']>=0 and s['customer_match'] is True)
        elif op.name == 'orders.close':
            valid=(s['status']=='confirmed' and f['shipped_minus_ordered']>=self.private_rules[op.name]
                   and s['disputed'] is False)
        elif op.name == 'inventory.reserve':
            valid=(s['product_active'] is True
                   and f['stock_minus_allocated_minus_qty']>=self.private_rules[op.name])
        else: raise ValueError('Unknown tool')
        if not valid:
            # Deliberately uniform rejection class, to focus on transition learning.
            return 422, s, {'ok':False,'error':'POLICY_DENIED'}
        if op.name == 'orders.validate':
            s['status']='confirmed';s['stock']-=s['qty']
        elif op.name == 'invoices.validate':
            s['status']='open';s['posted']=True
        elif op.name == 'invoices.payment':
            s['due']-=s['payment'];s['payment_count']+=1
        elif op.name == 'orders.close':
            s['status']='closed'
        elif op.name == 'inventory.reserve':
            s['allocated']+=s['qty']
        return 200,s,{'ok':True,'record':s.copy()}


def sample_state(op: Operation, r: random.Random, *, boundary_weight: float=0.55) -> dict:
    s={k:r.choice(v) for k,v in op.fields.items()}
    # IDs vary independently of business state; the learner is barred from
    # learning record_id as a policy feature. This also permits repeated
    # equivalent-state experiments without exact duplicate requests.
    s['record_id']=f'REC-{r.randrange(1,10**9):09d}'
    if r.random()<boundary_weight:
        if op.name=='orders.validate':
            if r.random()<.70: s['status']='draft'
            s['stock']=max(0,min(13,s['qty']+r.choice([-3,-2,-1,0,1,2,3,4])))
            s['credit']=max(0,min(150, s['amount']+r.choice([-30,-20,-10,0,10,20,30,40])))
        elif op.name=='invoices.validate':
            if r.random()<.70: s['status']='draft'
            s['line_count']=r.choice([0,1,2,3,4])
        elif op.name=='invoices.payment':
            if r.random()<.70: s['status']='open'
            s['payment']=max(0,min(150,s['due']+r.choice([-30,-20,-10,0,10,20,30])))
        elif op.name=='orders.close':
            if r.random()<.70: s['status']='confirmed'
            s['shipped_qty']=max(0,min(10,s['ordered_qty']+r.choice([-3,-2,-1,0,1,2,3,4])))
        elif op.name=='inventory.reserve':
            if r.random()<.70: s['product_active']=True
            s['stock']=max(0,min(15,s['allocated']+s['qty']+r.choice([-3,-2,-1,0,1,2,3,4])))
    return s


def reference_state(op: Operation) -> dict:
    """Public, documented demonstration input, not generated from hidden rules.

    Both arms must spend one of their queries to establish its actual outcome.
    This tests learning *after a valid operation is observed*, not cold start.
    """
    examples = {
        'orders.validate': {'status':'draft','qty':1,'stock':13,'amount':10,'credit':150},
        'invoices.validate': {'status':'draft','line_count':5,'customer_active':True,'posted':False},
        'invoices.payment': {'status':'open','due':150,'payment':20,'customer_match':True,'payment_count':0},
        'orders.close': {'status':'confirmed','ordered_qty':1,'shipped_qty':10,'disputed':False},
        'inventory.reserve': {'product_active':True,'stock':15,'allocated':0,'qty':1},
    }
    return {'record_id':'REFERENCE-SYNTHETIC',**examples[op.name]}


def one_factor_candidates(op: Operation, anchor: dict) -> list[dict]:
    """Counterfactual intervention menu, available identically to both arms."""
    candidates=[]
    for field,values in op.fields.items():
        if field=='record_id' or field in {'posted','payment_count'}: continue
        for value in values:
            if value == anchor[field]:continue
            s=anchor.copy();s[field]=value
            s['record_id']=f'OFAT-{field}-{str(value)}'
            candidates.append(s)
    return candidates


def _make_pool(op: Operation, seed: int, *, n: int=240, boundary_weight: float=.55) -> list[dict]:
    rng=random.Random(seed)
    pool=[]; seen=set()
    while len(pool)<n:
        s=sample_state(op,rng,boundary_weight=boundary_weight)
        sig=state_signatures(s)['exact_sha256']
        if sig not in seen:
            seen.add(sig);pool.append(s)
    return pool


@dataclass(frozen=True)
class Atom:
    field: str
    relation: str
    value: Any

    def test(self, f: dict) -> bool:
        v=f.get(self.field)
        if self.relation=='eq':return type(v)==type(self.value) and v==self.value
        if self.relation=='ge':return isinstance(v,(int,float)) and not isinstance(v,bool) and v>=self.value
        if self.relation=='le':return isinstance(v,(int,float)) and not isinstance(v,bool) and v<=self.value
        raise ValueError('Unknown relation')

    def as_dict(self) -> dict:
        return {'field':self.field,'op':self.relation,'value':self.value}


def candidate_atoms(op: Operation) -> list[Atom]:
    atoms=[]
    for name, values in op.fields.items():
        if name in {'record_id','posted','payment_count'}: continue
        if isinstance(values[0],bool) or isinstance(values[0],str):
            atoms.extend(Atom(name,'eq',v) for v in values)
        else:
            vals=sorted(set(values))
            # Public grammar: bounded coarse thresholds and meaningful endpoints.
            # This is deliberately a limited non-LLM hypothesis class.
            if len(vals)>10:
                vals=sorted(set(vals[::2]+vals[:1]+vals[-1:]))
            for v in vals:
                atoms.append(Atom(name,'ge',v)); atoms.append(Atom(name,'le',v))
    for name,sources in op.derived.items():
        vals= [-3,-2,-1,0,1,2,3,4] if 'amount' not in name else [-30,-20,-10,0,10,20,30,40]
        for val in vals:
            atoms.append(Atom(name,'ge',val));atoms.append(Atom(name,'le',val))
    return atoms


def _signature(atom: Atom) -> tuple:
    return (atom.field,atom.relation,str(atom.value))


def _state_key(state: dict) -> tuple:
    # Public candidate inputs are scalars in deterministic insertion order.
    return tuple(state.items())


def _candidate_literals(op: Operation, observations: list[dict]) -> list[Atom]:
    positives=[x['features'] for x in observations if x['accepted']]
    if not positives:return []
    return [a for a in candidate_atoms(op) if all(a.test(p) for p in positives)]


def _coverage(a: Atom, negatives: list[dict]) -> set[int]:
    return {i for i,f in enumerate(negatives) if not a.test(f)}


def _choose_rule(op: Operation, observations: list[dict], *, rng: random.Random|None=None,
                 max_literals: int=4) -> list[Atom]|None:
    positive=[x['features'] for x in observations if x['accepted']]
    negative=[x['features'] for x in observations if not x['accepted']]
    if not positive or not negative:return None
    allowed=_candidate_literals(op,observations)
    uncovered=set(range(len(negative)))
    chosen=[]
    # Greedy negative-covering conjunction, using a deterministic generality
    # prior. All consistent candidate literals are provisional, not verified.
    while uncovered and len(chosen)<max_literals:
        choices=[]
        for atom in allowed:
            if atom in chosen: continue
            covered=_coverage(atom,negative) & uncovered
            if not covered: continue
            # A coarse prior penalizes single-example coincidences; random
            # committees sample alternate consistent covers for query selection.
            empirical_accept = sum(atom.test(x['features']) for x in observations)/len(observations)
            width_prior = empirical_accept
            choices.append((len(covered),width_prior,atom))
        if not choices: return None
        choices.sort(key=lambda z:(z[0],z[1],_signature(z[2])),reverse=True)
        top_gain=choices[0][0]
        top=[x for x in choices if x[0]==top_gain]
        if rng:
            # randomized committee visits alternate explanations
            selected=rng.choice(top[:min(12,len(top))])[2]
        else:
            # choose the most permissive among current training candidates
            selected=max(top,key=lambda x:(x[1],-len(x[2].field),_signature(x[2])))[2]
        chosen.append(selected)
        uncovered-= _coverage(selected,negative)
    if uncovered:return None
    return chosen


def _expr_candidates(field: str, observed: list[dict], op: Operation) -> list[dict]:
    before=observed[0]['before'];after=observed[0]['after']
    if field not in before or field not in after:return []
    # Expressions: copy, arithmetic +/- another input field, or constant.
    result=[{'kind':'copy','from':field}]
    for src,vals in op.fields.items():
        if src==field or isinstance(before[field],(str,bool)) or isinstance(vals[0],(str,bool)):continue
        result.append({'kind':'add','from':field,'source':src})
        result.append({'kind':'subtract','from':field,'source':src})
    if type(before[field]) in {int,float}:
        for offset in (-2,-1,1,2):
            result.append({'kind':'offset','from':field,'delta':offset})
    result.append({'kind':'literal','value':after[field]})
    return result


def _compute(expr: dict, state: dict) -> Any:
    k=expr['kind']
    if k=='copy':return state[expr['from']]
    if k=='literal':return expr['value']
    if k=='add':return state[expr['from']]+state[expr['source']]
    if k=='subtract':return state[expr['from']]-state[expr['source']]
    if k=='offset':return state[expr['from']]+expr['delta']
    raise ValueError('Unknown effect')


def induce_effects(op: Operation, observations: list[dict]) -> dict:
    successes=[o for o in observations if o['accepted']]
    if not successes:return {}
    out={}
    for field in op.outcome_fields:
        cands=_expr_candidates(field,successes,op)
        matches=[e for e in cands if all(type(_compute(e,o['before']))==type(o['after'][field])
                    and _compute(e,o['before'])==o['after'][field] for o in successes)]
        # For single points, nontrivial arithmetic inference must remain
        # abstained: one outcome cannot establish a numeric update law.
        if len(successes)<2 and type(successes[0]['after'][field]) in {int,float}:
            if successes[0]['before'][field]!=successes[0]['after'][field]: continue
        if matches:
            # prefer copy, then arithmetic with independent features, then
            # constant. For categorical state change, literal is supported.
            matches.sort(key=lambda e: {'copy':0,'subtract':1,'add':2,'offset':3,'literal':4}[e['kind']])
            out[field]=matches[0]
    return out


def fit_model(op: Operation, observations: list[dict]) -> dict:
    r=_choose_rule(op,observations)
    return {'rule':[a.as_dict() for a in r] if r is not None else None,
            'effects':induce_effects(op,observations),
            'evidence':'synthetic_discovery_provisional',
            'training_examples':len(observations),
            'positive_examples':sum(o['accepted'] for o in observations)}


def predict_model(op: Operation, model: dict, state: dict) -> dict:
    if model.get('rule') is None:
        return {'decision':'abstain','after':None}
    f=features_for(op,state)
    accepted=all(Atom(z['field'],z['op'],z['value']).test(f) for z in model['rule'])
    if not accepted:return {'decision':'reject','after':state.copy()}
    after=state.copy()
    missing=[]
    for field in op.outcome_fields:
        expr=model['effects'].get(field)
        if expr is None:missing.append(field)
        else:after[field]=_compute(expr,state)
    return {'decision':'accept','after':after,'unknown_effects':missing}


def _committee(op: Operation, obs: list[dict], seed: int, n: int=16) -> list[list[Atom]]:
    r=random.Random(seed)
    rules=[]
    for _ in range(n):
        c=_choose_rule(op,obs,rng=r)
        if c is not None:rules.append(c)
    return rules


def _rank_active(op: Operation, pool: list[dict], seen: set[str], obs: list[dict],
                 ledger: EvidenceLedger, seed: int, *, budget:int) -> dict:
    committee=_committee(op,obs,seed)
    anchor=reference_state(op)
    pos=sum(x['accepted'] for x in obs)
    used_pairs={(x['features'].get(k),k) for x in obs for k in op.derived}
    options=[]
    for i,s in enumerate(pool):
        sign=_state_key(s)
        if sign in seen:continue
        f=features_for(op,s)
        if committee:
            accepts=sum(all(a.test(f) for a in rule) for rule in committee)
            p=accepts/len(committee)
            entropy = 4*p*(1-p)
        else:entropy=0.0;p=.5
        # Boundary verification: a differential feature near zero is an
        # efficient intervention candidate, not evidence of the true policy.
        near=sum(1/(1+abs(f[k])/(10 if 'amount' in k else 1)) for k in op.derived)
        nov=sum((f.get(k),k) not in used_pairs for k in op.derived)
        # Query novel, contested states; maintain some success exposure.
        balance=(p if pos<3 else 0)
        # New states are verified when probed, rather than running a
        # redundant SQLite decision-gate query for every candidate.
        verifying=.04
        num_changed=sum(s.get(k)!=anchor.get(k) for k in anchor if k!='record_id')
        causal_intervention=1.15 if num_changed==1 and len(obs)<=budget//2 else 0.0
        score=1.7*entropy+.23*near+.06*nov+.22*balance+verifying+causal_intervention
        options.append((score,-i,s))
    if not options:raise ValueError('Exhausted discovery pool')
    options.sort(reverse=True,key=lambda x:(x[0],x[1]))
    return options[0][2]


def run_arm(op: Operation, oracle: _SyntheticOracle, pool: list[dict],
            *, arm: str, budget: int, seed: int) -> dict:
    if arm not in {'uniform','franzen'}:raise ValueError('Unrecognized exploration arm')
    if budget<2 or budget>=len(pool):raise ValueError('Budget too small or large')
    r=random.Random(seed*1001 + (11 if arm=='uniform' else 37))
    anchor=reference_state(op)
    pool=pool+one_factor_candidates(op,anchor)
    shuffled=pool[:];r.shuffle(shuffled)
    seen=set();observations=[]; ledger=EvidenceLedger()
    try:
        for t in range(budget):
            if t==0:
                state=anchor  # identical public positive-anchor probe, budgeted
            elif arm=='uniform' or (arm=='franzen' and t>=(budget//2) and t%4 in {0,1}):
                # Diversity recovery: after initial causal probing, spend
                # roughly half of late-stage calls checking broad alternatives.
                state=next(s for s in shuffled if _state_key(s) not in seen)
            else:
                state=_rank_active(op,pool,seen,observations,ledger,seed*111 + t,budget=budget)
            sig=_state_key(state); seen.add(sig)
            gate=ledger.decision_gate(op.name,state,planned_mutations=2)
            code, after, response=oracle.query(op,state)
            record=ledger.record(operation=op.name, before=state, after=after,
                                 args={'state':state},status=code,response=response,
                                 role='test_agent', source='synthetic_fixture')
            observations.append({'features':features_for(op,state),'before':state,
                                 'after':after,'accepted':code==200,
                                 'state_changed':record['state_changed'],
                                 'state_sig':record['state_signature']['generalized_sha256']})
        contradictions=0
        # Confirm observed equivalent state outcomes for identical action.
        for o in observations:
            probe=ledger.stats(op.name,o['before'])
            contradictions += int(probe['conflict'])
        return {'arm':arm,'model':fit_model(op,observations),
                'observations':observations,
                'examples':len(observations),'positives':sum(o['accepted'] for o in observations),
                'contradictions':contradictions,
                'unique_exact_states':len(seen),'queries':budget}
    finally:ledger.close()


def _scores(op:Operation, model:dict, holdouts:list[tuple[dict,int,dict]]) -> dict:
    total=len(holdouts); correct=fp=fn=tp=tn=abstain=0
    after_case=after_correct=after_fields=after_fields_correct=0
    pos=sum(code==200 for _,code,_ in holdouts)
    for state,code,truth_after in holdouts:
        got=predict_model(op,model,state)
        if got['decision']=='abstain':abstain+=1;continue
        actual=code==200;accepted=got['decision']=='accept'
        correct+=int(accepted==actual)
        tp+=int(accepted and actual);tn+=int(not accepted and not actual)
        fp+=int(accepted and not actual);fn+=int(not accepted and actual)
        if actual and accepted:
            after_case+=1
            exact=True
            for field in op.outcome_fields:
                if field in got.get('unknown_effects',[]):
                    exact=False;continue
                after_fields+=1
                c=got['after'][field]==truth_after[field] and type(got['after'][field])==type(truth_after[field])
                after_fields_correct+=int(c)
                exact &= c
            after_correct+=int(exact)
    pp=pos;nn=total-pos
    return {'cases':total,'positive_truth':pos,'outcome_accuracy':round(correct/total,4),
            'balanced_accuracy':round((tp/max(1,pp)+tn/max(1,nn))/2,4),
            'false_accept_rate':round(fp/max(1,nn),4),
            'false_reject_rate':round(fn/max(1,pp),4),
            'abstentions':abstain,'coverage':round((total-abstain)/total,4),
            'effect_full_case_accuracy':round(after_correct/after_case,4) if after_case else None,
            'effect_field_accuracy':round(after_fields_correct/after_fields,4) if after_fields else None,
            'effect_field_coverage':round(after_fields/(after_case*len(op.outcome_fields)),4) if after_case else None,
            'true_pos':tp,'true_neg':tn,'false_pos':fp,'false_neg':fn}


def run_gate(*, seeds: int=12, budget: int=20, holdout_size:int=160,
             outdir:str|Path='build/gate2a', freeze_seed:int=20261010) -> dict:
    if not 2<=seeds<=40 or not 4<=budget<=100 or not 60<=holdout_size<=1000:
        raise ValueError('Seeds 2..40, budget 4..100, holdout 60..1000')
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    operations=public_operations()
    records=[]
    for si in range(seeds):
        oracle=_SyntheticOracle(freeze_seed+si)
        for oi,op in enumerate(operations):
            pool=_make_pool(op, seed=freeze_seed+si*1009+oi*89,n=max(240,budget+10))
            # Evaluation states are disjoint from discovery pool by independently
            # seeded generation and exact-state exclusion.
            disjoint={state_signatures(s)['exact_sha256'] for s in pool}
            evaluation=_make_pool(op, seed=freeze_seed*3+si*7717+oi*101+17,
                                  n=holdout_size+80,boundary_weight=.50)
            evaluation=[s for s in evaluation if state_signatures(s)['exact_sha256'] not in disjoint][:holdout_size]
            if len(evaluation)<holdout_size:raise AssertionError('Holdout pool overlap')
            # Evaluate oracle exactly once on common holdout set. No outcomes
            # enter discovery memory, model fit, query selection or file bundle.
            withheld=[]
            for s in evaluation:
                code,after,_ = oracle.query(op,s,phase='evaluation')
                withheld.append((s,code,after))
            for arm in ['uniform','franzen']:
                learned=run_arm(op,oracle,pool,arm=arm,budget=budget,seed=freeze_seed+si*101+oi)
                metrics=_scores(op,learned['model'],withheld)
                records.append({'seed':si,'operation':op.name,'arm':arm,
                                'queries':learned['queries'],'discovery_positives':learned['positives'],
                                'contradictions':learned['contradictions'],**metrics})
    comparisons=[]
    for op in operations:
        pair=[]
        for row in records:
            if row['operation']==op.name:pair.append(row)
        for arm in ['uniform','franzen']:
            rows=[x for x in pair if x['arm']==arm]
            comparisons.append({'operation':op.name,'arm':arm,'n_seeds':len(rows),
                'balanced_accuracy_mean':round(mean(r['balanced_accuracy'] for r in rows),4),
                'outcome_accuracy_mean':round(mean(r['outcome_accuracy'] for r in rows),4),
                'false_accept_rate_mean':round(mean(r['false_accept_rate'] for r in rows),4),
                'coverage_mean':round(mean(r['coverage'] for r in rows),4),
                'discovery_positives_mean':round(mean(r['discovery_positives'] for r in rows),3),
                'effects_full_case_mean':round(mean(r['effect_full_case_accuracy'] for r in rows if r['effect_full_case_accuracy'] is not None),4)
                      if any(r['effect_full_case_accuracy'] is not None for r in rows) else None})
    seed_diff=[]
    for si in range(seeds):
        a=[x for x in records if x['seed']==si and x['arm']=='franzen']
        b=[x for x in records if x['seed']==si and x['arm']=='uniform']
        seed_diff.append(mean(x['balanced_accuracy'] for x in a)-mean(x['balanced_accuracy'] for x in b))
    # Cluster bootstrap by seed (rather than treating correlated examples as independent)
    rng=random.Random(82412); bootstrap=[]
    for _ in range(1000):
        draws=[seed_diff[rng.randrange(seeds)] for _ in range(seeds)]
        bootstrap.append(mean(draws))
    bootstrap.sort()
    delta=mean(seed_diff)
    public={op.name:{'description':op.description,'fields':{k:list(v) for k,v in op.fields.items()},
                        'derived_features':{k:list(v) for k,v in op.derived.items()}}
            for op in operations}
    doc={'format':'orgsim-gate2a-v1','experiment':'SYNTHETIC_ORACLE; not Dolibarr or LLM test',
         'seeds':seeds,'freeze_seed':freeze_seed,'budget_per_action_per_arm':budget,'holdout_cases_per_action_seed':holdout_size,
         'operations':len(operations),'total_discovery_queries_per_arm':seeds*len(operations)*budget,
         'methods':{'uniform':'One common, budgeted public example + uniformly randomized candidate-state exploration + identical sparse conjunctive learner',
                    'franzen':'One common, budgeted public example + one-factor counterfactual probing + committee disagreement + transition memory + boundary guidance, with late-stage diversity recovery + identical sparse conjunctive learner'},
         'reporting':'Effect accuracy only among true accepted predicted accepts; coverage reported separately',
         'overall':{
             'uniform_balanced_accuracy':round(mean(x['balanced_accuracy'] for x in records if x['arm']=='uniform'),4),
             'franzen_balanced_accuracy':round(mean(x['balanced_accuracy'] for x in records if x['arm']=='franzen'),4),
             'difference':round(delta,4),'cluster_bootstrap_95_ci':[round(bootstrap[25],4),round(bootstrap[975],4)],
             'positive_seed_fraction':round(sum(d>0 for d in seed_diff)/len(seed_diff),4)
         },
         'by_operation':comparisons,
         'caveats':['No general-purpose LLM is called; both arms use a bounded symbolic rule learner.',
                    'Oracle guard family is expressible by known sparse-conjunction hypothesis grammar.',
                    'Both arms begin with the same budgeted public example that happens to be valid; cold start is not tested.',
                    'Action effects can be partially unidentified when only a few positive examples occur.',
                    'A failed or negative result does not mean Franzen fails at ARC or Dolibarr.',
                    'Heldout cases are generated separately and never stored in learning memory.',
                    'Execution against actual Dolibarr has not occurred.'],
         'live_calls':0,'public_contracts_sha256':sha256(json.dumps(public,sort_keys=True).encode()).hexdigest()}
    # Strict evaluation isolation: only aggregate rows and public data written.
    (outdir/'report.json').write_text(json.dumps(doc,indent=2)+'\n')
    (outdir/'per_seed_metrics.json').write_text(json.dumps(records,indent=2)+'\n')
    (outdir/'public_contracts.json').write_text(json.dumps(public,indent=2)+'\n')
    return doc


def main(argv:list[str]|None=None)->int:
    p=argparse.ArgumentParser(description='Offline black-box exploration comparison; NO Dolibarr traffic')
    p.add_argument('--seeds',type=int,default=12)
    p.add_argument('--budget',type=int,default=20)
    p.add_argument('--holdout',type=int,default=160)
    p.add_argument('--out',default='build/gate2a')
    p.add_argument('--freeze-seed',type=int,default=20261010,help='Base seed for hidden oracle family')
    args=p.parse_args(argv)
    report=run_gate(seeds=args.seeds,budget=args.budget,holdout_size=args.holdout,outdir=args.out,freeze_seed=args.freeze_seed)
    print(json.dumps({'overall':report['overall'],'outdir':args.out,'live_calls':0},indent=2))
    return 0

if __name__=='__main__': raise SystemExit(main())
