"""Offline interaction-aware program synthesis under a scoped guard grammar.

Oracle access is ONLY via the caller's observation function. The learner sees
public numeric domains and witnessed (before,after,accepted) records, never
hidden thresholds or evaluation outcomes. The grammar is intentionally narrow:
a monotone base conjunction plus at most one upper-bound unary or two-field
veto. Non-representable rules must be reported as unresolved, not guessed.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import random
from math import ceil
from typing import Any, Callable
from .gate2a_lab import Operation, reference_state, features_for, _compute
from .gate2c_counterexamples import (_applies, _fingerprint, _observation,
    fit_audited_model, run_cegis_arm)

@dataclass(frozen=True)
class Veto:
    clauses:tuple[tuple[str,float],...]

    def prohibits(self,state:dict)->bool:
        return all(state[k]>=value for k,value in self.clauses)

    def as_json(self):
        return [{'field':k,'op':'ge','value':v} for k,v in self.clauses]


def _numeric(op:Operation)->dict[str,tuple]:
    return {k:tuple(sorted(v)) for k,v in op.fields.items() if
            k!='record_id' and len(v)>1 and type(v[0]) in (int,float)}


def _veto_candidates(op:Operation,observations:list[dict],base:dict)->list[Veto]:
    """Enumerate *every* observed-consistent monotone, high-value veto.

    The single-veto maximum is a hypothesis-class restriction, not evidence
    that other programs cannot exist. Zero false accepts is not certified by it.
    """
    permitted=[o for o in observations if _applies(op,base['rule'],o['before'])]
    positives=[o['before'] for o in permitted if o['accepted']]
    negatives=[o['before'] for o in permitted if not o['accepted']]
    fields=_numeric(op)
    if not negatives:return []
    vetted=[]
    keys=list(fields)
    for i,a in enumerate(keys):
        # Reduce thresholds to the lowest negative witness coordinate; a
        # threshold above that would fail to explain that counterexample.
        for v in fields[a]:
            veto=Veto(((a,v),))
            if all(veto.prohibits(n) for n in negatives) and not any(veto.prohibits(p) for p in positives):
                vetted.append(veto)
        for b in keys[i+1:]:
            for va in fields[a]:
                for vb in fields[b]:
                    veto=Veto(((a,va),(b,vb)))
                    if all(veto.prohibits(n) for n in negatives) and not any(veto.prohibits(p) for p in positives):
                        vetted.append(veto)
    return vetted


def _probe_pool(op:Operation,base:dict,seen:set[str],seed:int)->list[dict]:
    """Budget-free public candidates; no calls to the hidden oracle."""
    rand=random.Random(seed)
    numeric=_numeric(op)
    anchor=reference_state(op)
    keys=list(numeric)
    high=anchor.copy()
    # Supplying variables are high by default; favor states where baseline
    # financial/inventory prerequisites hold, not artificially denied cases.
    refs={x for src in op.derived.values() for x in src[1:]}
    for derived,src in op.derived.items():
        high[src[0]]=max(numeric[src[0]])
    def candidate(d:dict,tag:str):
        s=high.copy();s.update(d);s['record_id']='INVESTIGATE-'+tag
        if not _applies(op,base['rule'],s):return None
        if _fingerprint(op,s) in seen:return None
        return s
    results=[];registered=set()
    def add(x):
        if x is not None:
            k=_fingerprint(op,x)
            if k not in registered:
                registered.add(k);results.append(x)
    def grid_values(vals):
        return [vals[min(len(vals)-1,ceil(q*(len(vals)-1)))] for q in (.35,.55,.7,.85,1)]
    # Each raw risk input by itself at varied high levels (including
    # payment_count and fields also used in arithmetic margins).
    for k,vs in numeric.items():
        for v in (vs if len(vs)<=16 else grid_values(vs)):
            add(candidate({k:v},f'{k}-{v}'))
    # Joint features: full upper corner and boundary-ish quantile pairs.
    for i,a in enumerate(keys):
        for b in keys[i+1:]:
            for av in grid_values(numeric[a]):
                for bv in grid_values(numeric[b]):
                    add(candidate({a:av,b:bv},f'{a}-{av}-{b}-{bv}'))
    # Perturb 1-3 randomly chosen fields around the valid anchor. Other
    # inputs remain in contract-defined domains; oracle never ranks these.
    for step in range(1000):
        changes={}
        for k in rand.sample(keys,k=min(len(keys),rand.choice([1,2,2,3]))):
            changes[k]=rand.choice(numeric[k])
        add(candidate(changes,f'random-{step}'))
    rand.shuffle(results)
    return results


def _rank_candidates(op:Operation,base:dict,options:list[Veto],pool:list[dict],
                     observations:list[dict],seed:int)->list[dict]:
    """Prioritize disagreement between *consistent* programs, never truth."""
    if not options:
        # All known base-positive samples agree. Search high-risk regions
        # rather than declaring absence of veto from lack of counterexamples.
        fields=_numeric(op)
        def score(s):
            return sum((fields[k].index(s[k])/(len(vs)-1)) for k,vs in fields.items())
        return sorted(pool,key=lambda s:(-score(s),_fingerprint(op,s)))
    if len(options)==1:
        # Certify the single candidate by probing around its corners.
        candidate=options[0]
        def score(s):
            return sum(min(abs(s[k]-v),10) for k,v in candidate.clauses)
        return sorted(pool,key=lambda s:(score(s),_fingerprint(op,s)))
    # Downsample version space only for ranking cost; keep all for prediction.
    subset=options[::max(1,len(options)//200)][:200]
    ranked=[]
    for s in pool:
        frac=sum(v.prohibits(s) for v in subset)/len(subset)
        # Disagreement is maximal near 50/50. Favor even partial disagreement
        # over redundant safe-baseline samples.
        uncertain=frac*(1-frac)
        ranked.append((uncertain,s))
    ranked.sort(key=lambda e:(-e[0],_fingerprint(op,e[1])))
    return [s for _,s in ranked]


def run_interaction_arm(op:Operation,oracle,*,budget:int=64,seed:int=2026)->dict:
    """32 controlled baseline interventions then <=budget-32 active probes."""
    if not 32<=budget<=96:raise ValueError('Budget must be 32..96')
    baseline=run_cegis_arm(op,oracle,budget=32,seed=seed)
    observations=list(baseline['observations'])
    seen={_fingerprint(op,r['before']) for r in observations}
    if len(seen)!=32:raise AssertionError('Duplicate baseline discovery state')
    proposals=_probe_pool(op,baseline['model'],seen,seed+31019)
    while len(observations)<budget:
        base=fit_audited_model(op,observations)
        options=_veto_candidates(op,observations,base)
        ranked=_rank_candidates(op,base,options,
                                [s for s in proposals if _fingerprint(op,s) not in seen],
                                observations,seed)
        if not ranked:
            break  # finite public positive region exhausted, no padding queries
        state=ranked[0]
        seen.add(_fingerprint(op,state))
        observations.append(_observation(op,oracle,state))
    return {'arm':'interaction_audited','observations':observations,
            'queries':len(observations),'model':fit_interaction_model(op,observations),
            'positive_examples':sum(r['accepted'] for r in observations)}



def _base_region_exhausted(op:Operation,base:dict,observations:list[dict])->bool:
    """Exact, finite PUBLIC input enumeration, never querying the oracle."""
    fields=[(k,v) for k,v in op.fields.items() if k!='record_id']
    count=1
    for _,v in fields:
        count*=len(v)
        if count>100_000:return False
    witnessed={_fingerprint(op,r['before']) for r in observations}
    anchor=reference_state(op)
    for vals in product(*(choices for _,choices in fields)):
        s={**anchor,**dict(zip((k for k,_ in fields),vals))}
        if _applies(op,base['rule'],s) and _fingerprint(op,s) not in witnessed:
            return False
    return True


def _high_corner_witness(op:Operation,observations:list[dict])->bool:
    """A successful global high corner excludes upward-closed vetoes if legal."""
    num=_numeric(op)
    if not num:return False
    return any(r['accepted'] and all(r['before'][k]==max(v) for k,v in num.items())
               for r in observations)

def fit_interaction_model(op:Operation,observations:list[dict])->dict:
    base=fit_audited_model(op,observations)
    bad=[r for r in observations if _applies(op,base['rule'],r['before']) and not r['accepted']]
    options=_veto_candidates(op,observations,base)
    audited=all(x['status'] in ('enumerated','adjacent_boundary','all_values_observed')
                for x in base['audit']['critical_axes'])
    effects_ready=not base['audit']['unidentified_effects']
    # A pure conjunction is allowed only if active counterexample discovery
    # was attempted AFTER the baseline audit, not merely because it looked fit.
    base_ready=audited and effects_ready and (len(observations)>=48 or _base_region_exhausted(op,base,observations) or _high_corner_witness(op,observations))
    if bad and not options:base_ready=False
    # Version-space ambiguity is exposed as partial abstention, rather than
    # quietly selecting the most convenient veto for the current evaluation.
    return {'base':base,'veto_options':options,
            'candidate_count':len(options),
            'has_witnessed_counterexample':bool(bad),
            'audited':base_ready,'training_observations':len(observations),
            'reviewed':False,'grammar':'base_conjunction_plus_optional_single_upper_veto',
            'assumption':'At most one unary/two-raw-feature nondecreasing high-value veto. No guarantee for arbitrary ERP policies.'}


def predict_interaction(op:Operation,model:dict,before:dict)->dict:
    if not model['audited']:
        return {'decision':'abstain','after':None,'reason':'audit_incomplete_or_grammar_conflict'}
    base=model['base']
    if not _applies(op,base['rule'],before):
        return {'decision':'reject','after':before.copy()}
    options=model['veto_options']
    if model['has_witnessed_counterexample']:
        if not options:
            return {'decision':'abstain','after':None,'reason':'no_consistent_policy_program'}
        results=[v.prohibits(before) for v in options]
        if any(results) and not all(results):
            return {'decision':'abstain','after':None,'reason':'consistent_policies_disagree'}
        if all(results):return {'decision':'reject','after':before.copy()}
    after=before.copy()
    for k,formula in base['effects'].items():after[k]=_compute(formula,before)
    return {'decision':'accept','after':after}


def score_interaction(op:Operation,model:dict,truth:list[tuple[dict,int,dict]])->dict:
    n=len(truth);tp=tn=fp=fn=abst=effect_ok=0
    for before,status,after in truth:
        pred=predict_interaction(op,model,before)
        if pred['decision']=='abstain':abst+=1;continue
        positive=status==200
        if pred['decision']=='accept':
            if positive:
                tp+=1
                effect_ok+=int(all(pred['after'][k]==after[k] and
                         type(pred['after'][k]) is type(after[k]) for k in op.outcome_fields))
            else:fp+=1
        else:
            if positive:fn+=1
            else:tn+=1
    pos=sum(status==200 for _,status,_ in truth)
    return {'cases':n,'answered':n-abst,'coverage':round((n-abst)/n,5),
            'false_accepts':fp,'false_rejects':fn,'true_accepts':tp,'true_rejects':tn,
            'abstentions':abst,'positive_truth':pos,
            'positive_recall':round(tp/max(1,pos),5),
            'effect_exact':round(effect_ok/tp,5) if tp else None,
            'grammar_candidates':model['candidate_count'],
            'witnessed_veto':model['has_witnessed_counterexample'],
            'audited':model['audited']}
