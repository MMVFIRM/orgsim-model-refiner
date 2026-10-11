"""Gate 2C: counterexample-guided, risk-aware black-box reconstruction.

The learner knows only *public* input ranges, arithmetic affordances and declared
critical axes. Hidden oracle thresholds, behavior, and holdout outcomes are
available to the evaluator only. An ambiguous policy cannot authorize an
execution; it either rejects via a witnessed contrast or abstains.

This is NOT a general guarantee of real enterprise semantics. The explicit risk
axes and monotonic model family are a declared assumption, and independent
counterfactual testing remains necessary before any real-system mutation.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from statistics import mean
from typing import Any
import argparse
import json
import math
import random

from .gate2a_lab import (
    Operation, _SyntheticOracle, public_operations, features_for,
    reference_state, _make_pool, _scores, run_arm, induce_effects, _compute,
)


@dataclass(frozen=True)
class CriticalAxis:
    name: str
    kind: str                    # finite enum or ordinal monotonic lower bound
    source: str | None = None   # field changed to isolate a derived feature
    values: tuple = ()          # possible one-factor feature values
    states: tuple = ()          # matching serialized input states, never oracle outcomes


def _nonoutput_fields(op: Operation) -> dict[str, tuple]:
    return {k:v for k,v in op.fields.items() if k != 'record_id' and k not in {'posted', 'payment_count'}}


def _candidate_axes(op: Operation) -> list[CriticalAxis]:
    """Deterministic, contract-only candidate risk axes.

    This small synthetic benchmark publicly declares all categorical fields,
    relational arithmetic features, and non-derived numeric fields as critical.
    `payment` is also an independent amount axis: whether the amount is positive
    is not implied by whether it is <= the total due.

    An arbitrary production API needs human/model-audited axis discovery; this
    function must not be generalized to claim unknown features were tested.
    """
    anchor=reference_state(op)
    src_fields=set(x for refs in op.derived.values() for x in refs)
    axes=[]
    fields=_nonoutput_fields(op)
    for key,choices in fields.items():
        if type(choices[0]) in {bool,str}:
            vars=[]
            for v in choices:
                s={**anchor,key:v};s['record_id']=f'AXIS-{key}-{v}'
                vars.append(s)
            axes.append(CriticalAxis(key,'enum',key,tuple(choices),tuple(vars)))
    for feat,refs in op.derived.items():
        primary=refs[0]
        # Change one upstream value at a time; all other inputs stay at the
        # public success anchor. No private threshold knowledge is used.
        vals={}
        for v in op.fields[primary]:
            s={**anchor,primary:v}
            f=features_for(op,s)[feat]
            s['record_id']=f'AXIS-{feat}-{str(f)}'
            vals[f]=s
        values=tuple(sorted(vals))
        axes.append(CriticalAxis(feat,'ordinal',primary,values,tuple(vals[v] for v in values)))
    for key,choices in fields.items():
        if type(choices[0]) not in {int,float}:
            continue
        if key not in src_fields or key in {'payment'}:
            vals={}
            for value in choices:
                s={**anchor,key:value};s['record_id']=f'AXIS-{key}-{str(value)}'
                vals[value]=s
            axes.append(CriticalAxis(key,'ordinal',key,tuple(sorted(vals)),tuple(vals[v] for v in sorted(vals))))
    return axes


def _fingerprint(op: Operation, state: dict) -> str:
    """Deduplicate business states, excluding irrelevant record identifiers."""
    return sha256(json.dumps(features_for(op,state),sort_keys=True).encode()).hexdigest()


def _observation(op:Operation, oracle:_SyntheticOracle, state:dict)->dict:
    code,after,_=oracle.query(op,state,phase='discovery')
    return {'before':state.copy(),'after':after,'accepted':code==200,
            'features':features_for(op,state),'status':code}


def _axis_observed(axis:CriticalAxis, op:Operation, samples:list[dict]) -> dict[Any,bool]:
    """Use only observations matching the anchor on every OTHER input field."""
    anchor=reference_state(op)
    outcomes={}
    for row in samples:
        state=row['before']
        if any(state.get(k)!=anchor[k] for k in anchor if k!='record_id' and k!=axis.source):
            continue
        value=row['features'].get(axis.name)
        if value in axis.values:
            if value in outcomes and outcomes[value]!=row['accepted']:
                raise ValueError('Contradictory deterministic observation')
            outcomes[value]=row['accepted']
    return outcomes


def _axis_need(axis:CriticalAxis, op:Operation, observations:list[dict]) -> tuple[dict|None,dict]:
    """Return the most useful unobserved intervention and public audit status."""
    witnessed=_axis_observed(axis,op,observations)
    samples={v:s for v,s in zip(axis.values,axis.states)}
    if axis.kind=='enum':
        unknown=[v for v in axis.values if v not in witnessed]
        if unknown:
            return samples[unknown[0]],{'status':'unverified_alternatives','tested':len(witnessed),'total':len(axis.values)}
        return None,{'status':'enumerated','tested':len(witnessed),'total':len(axis.values)}
    # Numeric monotone rule: establish a witnessed rejecting lower endpoint,
    # then binary-search to an adjacent success/rejection boundary.
    if not witnessed:
        return samples[axis.values[0]],{'status':'no_observations'}
    if axis.values[0] not in witnessed:
        return samples[axis.values[0]],{'status':'need_lower_endpoint'}
    if axis.values[-1] not in witnessed:
        return samples[axis.values[-1]],{'status':'need_upper_endpoint'}
    # If no negative exists anywhere, monotonicity is a *hypothesis*; require
    # every input value to have been checked before treating it as unrestricted.
    positives=sorted(v for v,result in witnessed.items() if result)
    negatives=sorted(v for v,result in witnessed.items() if not result)
    if not positives:
        # In these benchmark tasks the public anchor is positive; no positive
        # would signal an observation-protocol failure.
        return None,{'status':'no_positive'}
    if positives and negatives and max(negatives)>=min(positives):
        return None,{'status':'nonmonotonic_or_conflict'}
    if not negatives:
        unknown=[v for v in axis.values if v not in witnessed]
        if unknown:
            return samples[unknown[len(unknown)//2]],{'status':'no_negative_boundary'}
        return None,{'status':'all_values_observed'}
    rejected=max(negatives); accepted=min(positives)
    lo=axis.values.index(rejected);hi=axis.values.index(accepted)
    if hi-lo==1:
        return None,{'status':'adjacent_boundary','lower_reject':rejected,'lowest_accept':accepted}
    idx=(lo+hi)//2
    return samples[axis.values[idx]],{'status':'boundary_search','interval_width':hi-lo}


def _directed_interaction_cases(op:Operation,axes:list[CriticalAxis],observations:list[dict])->list[dict]:
    """Stress one pair of independently discovered guard dimensions.

    Never use the oracle to choose interactions. Attempt combinations close to
    witnessed successful thresholds, checking effects and conjunction assumption.
    """
    anchor=reference_state(op)
    targets=[]
    for axis in axes:
        seen=_axis_observed(axis,op,observations)
        successful=[v for v,succeeded in seen.items() if succeeded]
        if not successful:continue
        v=min(successful) if axis.kind=='ordinal' else successful[-1]
        candidate=dict(zip(axis.values,axis.states))[v]
        targets.append((axis,candidate))
    out=[]
    for i,(a,sa) in enumerate(targets):
        for b,sb in targets[i+1:]:
            # If the two axes manipulate the same source field, do not pretend
            # that two independent changes can be applied simultaneously.
            if a.source==b.source:continue
            s=anchor.copy()
            s[a.source]=sa[a.source];s[b.source]=sb[b.source]
            s['record_id']=f'PAIR-{a.name}-{b.name}'
            if _fingerprint(op,s)==_fingerprint(op,anchor):continue
            out.append(s)
    return out


def _risk_stress_candidates(op: Operation, observations:list[dict]) -> list[dict]:
    """Deliberate multi-feature stress to challenge the conjunction assumption.

    These are generated from public domains and *observed* axis evidence,
    never from hidden oracle rules. For dependent numeric variables, choose
    the maximum feasible supplying resource before stressing demand variables.
    """
    anchor=reference_state(op)
    axes=_candidate_axes(op)
    observed_axes={a.name:_axis_observed(a,op,observations) for a in axes}
    if any(_axis_need(a,op,observations)[0] is not None for a in axes):
        return []
    used_axis_raw={a.name for a in axes if a.kind=='ordinal' and a.source==a.name}
    # Some action parameters can be guard conditions independently of the
    # declared arithmetic margins (qty, payment_count, customer-specific etc).
    # Stress them jointly rather than assuming each single-factor rule suffices.
    extra=[k for k,v in op.fields.items() if k!='record_id' and
           type(v[0]) in {int,float} and len(v)>1 and k not in used_axis_raw]
    if not extra:return []
    max_state=anchor.copy()
    for a in axes:
        if a.kind=='ordinal' and a.source in op.fields:
            max_state[a.source]=max(op.fields[a.source])
    for field in extra:
        max_state[field]=max(op.fields[field])
    # Repair relational margins by maximizing available supplying quantities,
    # still staying within public ranges.
    for a in axes:
        if a.kind=='ordinal' and a.name in op.derived:
            max_state[a.source]=max(op.fields[a.source])
    candidates=[max_state]
    # Mid-to-high joint interventions expose conditional interactions hidden by
    # purely maximal probes. Pick values by public domain quantiles, not by
    # knowledge of the evaluator's private rules.
    for quant in (0.55,0.7):
        mid=anchor.copy()
        for axis in axes:
            if axis.kind=='ordinal' and axis.name in op.derived:
                mid[axis.source]=max(op.fields[axis.source])
        for field in extra:
            options=sorted(op.fields[field])
            mid[field]=options[min(len(options)-1,math.ceil(quant*(len(options)-1)))]
        candidates.append(mid)
    for field in extra:
        s=anchor.copy();s[field]=max(op.fields[field])
        # For a changed demand amount, give maximum known supply to avoid an
        # obvious margin failure being mistaken for an interaction.
        for a in axes:
            if a.kind=='ordinal' and a.name in op.derived:
                s[a.source]=max(op.fields[a.source])
        candidates.append(s)
    # Add pairwise high/low demand interactions under otherwise generous
    # resource conditions; a single shared extreme can miss conditional guards.
    for i,f in enumerate(extra):
        for g in extra[i+1:]:
            s=anchor.copy()
            for a in axes:
                if a.kind=='ordinal' and a.name in op.derived:
                    s[a.source]=max(op.fields[a.source])
            s[f]=max(op.fields[f]);s[g]=max(op.fields[g])
            candidates.append(s)
            mid=anchor.copy()
            for axis in axes:
                if axis.kind=='ordinal' and axis.name in op.derived:
                    mid[axis.source]=max(op.fields[axis.source])
            for item in (f,g):
                options=sorted(op.fields[item])
                mid[item]=options[min(len(options)-1,math.ceil(.6*(len(options)-1)))]
            candidates.append(mid)
    dedup=set();out=[]
    for i,s in enumerate(candidates):
        s['record_id']=f'STRESS-{i}'
        fp=_fingerprint(op,s)
        if fp in dedup or fp==_fingerprint(op,anchor):continue
        dedup.add(fp);out.append(s)
    return out


def run_cegis_arm(op:Operation,oracle:_SyntheticOracle,*,budget:int,seed:int)->dict:
    """A fixed budget including common positive anchor; no heldouts are touched."""
    if budget<2:raise ValueError('Need at least two observations')
    rng=random.Random(seed)
    axes=_candidate_axes(op)
    observations=[];seen=set();interactions=0;stress_probes=0
    anchor=reference_state(op)
    for step in range(budget):
        proposed=None
        if step==0:
            proposed=anchor
        else:
            # First finish each axis; this prevents greedily sampling near one
            # interesting boundary while omitting a whole lifecycle dimension.
            unfinished=[]
            for axis in axes:
                trial,audit=_axis_need(axis,op,observations)
                if trial is not None and _fingerprint(op,trial) not in seen:
                    unfinished.append((axis.name,trial))
            if unfinished:
                # Round robin by observation debt rather than favoring first
                # axis in the contract at every step.
                unfinished.sort(key=lambda x: sum(
                    row['before'].get(k)!=anchor.get(k) for row in observations
                    for k in [next(a.source for a in axes if a.name==x[0])]))
                proposed=unfinished[0][1]
            else:
                stress=[s for s in _risk_stress_candidates(op,observations)
                        if _fingerprint(op,s) not in seen]
                if stress:
                    proposed=stress[0];stress_probes+=1
                else:
                    pair=[s for s in _directed_interaction_cases(op,axes,observations)
                          if _fingerprint(op,s) not in seen]
                    if pair:
                        proposed=pair[0];interactions+=1
                    else:
                        # Broader random exploration to falsify interactions and
                        # model misspecification; no oracle calls to rank states.
                        for _ in range(100):
                            candidate={k:rng.choice(v) for k,v in op.fields.items()}
                            candidate['record_id']=f'RANDOM-{step}-{rng.randrange(10**9)}'
                            if _fingerprint(op,candidate) not in seen:
                                proposed=candidate;break
                        if proposed is None:
                            raise RuntimeError('Exhausted synthetic discovery states')
        seen.add(_fingerprint(op,proposed))
        observations.append(_observation(op,oracle,proposed))
    model=fit_audited_model(op,observations)
    return {'arm':'cegis','model':model,'observations':observations,'queries':len(observations),
            'positive_examples':sum(x['accepted'] for x in observations),
            'pairwise_probes':interactions,'stress_probes':stress_probes,'axes':len(axes)}


def fit_audited_model(op:Operation, observations:list[dict])->dict:
    """Compile only witnessed, independently varied guard dimensions.

    Other features are *not assumed safe*. If a critical axis is unknown, the
    shield never authorizes an accept, although it may reject based on a known
    invalid dimension. Unverified models stay reviewed:false regardless of pass.
    """
    axes=_candidate_axes(op)
    rule=[];coverage=[];ready=True
    for axis in axes:
        observed=_axis_observed(axis,op,observations)
        _,need=_axis_need(axis,op,observations)
        audited=need['status'] in {'enumerated','adjacent_boundary','all_values_observed'}
        if axis.kind=='enum':
            accepted=[v for v in axis.values if observed.get(v)]
            rejected=[v for v in axis.values if v in observed and not observed[v]]
            if not audited or not accepted:
                ready=False
            # An acceptance region can be a finite explicit enumerated set.
            # If all enum values are allowed, the dimension is irrelevant.
            if accepted and rejected:
                rule.append({'field':axis.name,'op':'in','value':accepted})
        else:
            pos=sorted(v for v,result in observed.items() if result)
            neg=sorted(v for v,result in observed.items() if not result)
            if not audited or not pos:
                ready=False
            if neg and pos and max(neg)<min(pos):
                rule.append({'field':axis.name,'op':'ge','value':min(pos)})
            elif need['status']!='all_values_observed':
                ready=False
        coverage.append({'axis':axis.name,'kind':axis.kind,'status':need['status'],
                         'observed_values':len(observed),'domain_values':len(axis.values),
                         'known_positive':sum(observed.values()),
                         'known_negative':sum(not v for v in observed.values())})
    # Every accept must be supported by a positive outcome and complete effects.
    effects=induce_effects(op,observations)
    missing=[f for f in op.outcome_fields if f not in effects]
    if missing:ready=False
    # Some pairwise counterfactual must be explicitly tested after the axes
    # are identified; otherwise higher-order interactions remain untested.
    pairs=_directed_interaction_cases(op,axes,observations)
    observed_keys={_fingerprint(op,x['before']) for x in observations}
    pair_count=sum(_fingerprint(op,s) in observed_keys for s in pairs)
    stress=_risk_stress_candidates(op,observations)
    stress_count=sum(_fingerprint(op,s) in observed_keys for s in stress)
    if len(axes)>1 and pair_count==0 and stress_count==0:ready=False
    if stress and stress_count==0:ready=False
    # Reject the entire audit if a learned conjunction disagrees with ANY
    # discovered case (including arbitrary joint/interaction probes).
    predicted_mismatch=0
    for row in observations:
        cond=_applies(op,rule,row['before'])
        if cond!=row['accepted']:
            predicted_mismatch+=1
    if predicted_mismatch:ready=False
    # Explicit post-development certification floor: short-budget fits can
    # look complete but miss interactions. This is a safety threshold, NOT
    # a proof or a claim of optimal sample complexity.
    minimum_observations=20
    if len(observations)<minimum_observations:ready=False
    return {'rule':rule,'effects':effects,'audit':{'ready_for_offline_execution':ready,
            'critical_axes':coverage,'interaction_cases_observed':pair_count,'stress_cases_observed':stress_count,
            'training_conflicts':predicted_mismatch,'unidentified_effects':missing,
            'minimum_observations_for_execution':minimum_observations,
            'assumptions':['finite enumerated categorical domains',
                           'monotone numeric lower-bound guards',
                           'known critical axes and derived features',
                           'pairwise counterexamples do not prove absence of higher-order interactions']},
            'reviewed':False,'evidence':'synthetic_discovery_provisional',
            'training_examples':len(observations)}


def _applies(op:Operation,rule:list[dict],state:dict)->bool:
    feats=features_for(op,state)
    for condition in rule:
        value=feats[condition['field']]
        if condition['op']=='in':
            if value not in condition['value']:return False
        elif condition['op']=='ge':
            if value<condition['value']:return False
        elif condition['op']=='gt':
            if value<=condition['value']:return False
        elif condition['op']=='le':
            if value>condition['value']:return False
        elif condition['op']=='lt':
            if value>=condition['value']:return False
        elif condition['op']=='eq':
            if type(value)!=type(condition['value']) or value!=condition['value']:return False
        elif condition['op']=='ne':
            if type(value)==type(condition['value']) and value==condition['value']:return False
        else:raise ValueError('Unsupported constraint atom')
    return True


def predict_audited(op:Operation, model:dict, state:dict)->dict:
    if not model['audit']['ready_for_offline_execution']:
        return {'decision':'abstain','after':None,
                'reason':'critical_precondition_or_transition_unverified'}
    if not _applies(op,model['rule'],state):
        return {'decision':'reject','after':state.copy()}
    after=state.copy()
    for field,expr in model['effects'].items():
        after[field]=_compute(expr,state)
    return {'decision':'accept','after':after,'unknown_effects':[]}


def selective_scores(op:Operation, model:dict, withheld:list[tuple[dict,int,dict]])->dict:
    tp=tn=fp=fn=abstain=effects_ok=0
    for state,status,truth_after in withheld:
        prediction=predict_audited(op,model,state)
        if prediction['decision']=='abstain':abstain+=1;continue
        positive=status==200;allowed=prediction['decision']=='accept'
        tp+=int(positive and allowed);tn+=int(not positive and not allowed)
        fp+=int(not positive and allowed);fn+=int(positive and not allowed)
        if positive and allowed:
            effects_ok+=int(all(prediction['after'][f]==truth_after[f] and
                                type(prediction['after'][f])==type(truth_after[f])
                                for f in op.outcome_fields))
    n=len(withheld); total_positive=sum(z[1]==200 for z in withheld)
    total_negative=n-total_positive
    return {'cases':n,'positive_truth':total_positive,'negative_truth':total_negative,
            'abstained_positive':total_positive-tp-fn,'abstained_negative':total_negative-tn-fp,
            'coverage':round((n-abstain)/n,5),
            'positive_accept_recall':round(tp/max(1,total_positive),5),
            'negative_reject_recall':round(tn/max(1,total_negative),5),
            'false_accepts':fp,'false_rejects':fn,'abstentions':abstain,
            'unsafe_false_accept_rate':round(fp/max(1,total_negative),5),
            'accepted_true_positives':tp,'true_negatives':tn,
            'effect_exact_on_true_accepts':round(effects_ok/max(1,tp),5) if tp else None,
            'audit_ready':model['audit']['ready_for_offline_execution']}


def run_gate2c(*,seeds:int=24,budgets:tuple[int,...]=(8,20,32),holdout_size:int=200,
               freeze_seed:int=20270111,outdir:str|Path='build/gate2c',
               oracle_family:str='simple')->dict:
    if oracle_family not in {'simple','interaction'}:
        raise ValueError('Unknown oracle family')
    if not 2<=seeds<=40 or not budgets or min(budgets)<4 or max(budgets)>100 or not 80<=holdout_size<=1000:
        raise ValueError('Invalid benchmark parameters')
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    records=[];operations=public_operations()
    for si in range(seeds):
        oracle=(_SyntheticOracle(freeze_seed+si) if oracle_family=='simple'
                else _InteractionOracle(freeze_seed+si))
        for oi,op in enumerate(operations):
            # Explore first. Do not generate or score held-outs until all arm
            # observation signatures are frozen, so exact business states
            # cannot accidentally appear in both discovery and evaluation.
            learned={}
            discovery_states=set()
            for budget in budgets:
                run=run_cegis_arm(op,oracle,budget=budget,seed=freeze_seed+si*991+oi*31)
                learned[budget]=run
                discovery_states.update(_fingerprint(op,o['before']) for o in run['observations'])
            hold_pool=_make_pool(op,freeze_seed*2+si*1607+oi*17,n=holdout_size+700)
            unseen=[st for st in hold_pool if _fingerprint(op,st) not in discovery_states]
            # Some finite synthetic operations have fewer unique business
            # states than the evaluation sample size (e.g. 3 x 6 x 2).
            # Prefer unseen combinations, then separately seeded fresh-ID
            # cases. Report novelty instead of pretending every case is new.
            evaluation=unseen[:holdout_size]
            if len(evaluation)<holdout_size:
                selected_exact={json.dumps(s,sort_keys=True) for s in evaluation}
                for st in hold_pool:
                    key=json.dumps(st,sort_keys=True)
                    if key not in selected_exact:
                        evaluation.append(st);selected_exact.add(key)
                    if len(evaluation)>=holdout_size:break
            if len(evaluation)<holdout_size:raise AssertionError('Insufficient exact-state holdouts')
            novel_cases=sum(_fingerprint(op,st) not in discovery_states for st in evaluation)
            truth=[(st,*oracle.query(op,st,phase='evaluation')[:2]) for st in evaluation]
            for budget in budgets:
                discovery=learned[budget]
                score=selective_scores(op,discovery['model'],truth)
                records.append({'seed':si,'operation':op.name,'budget':budget,
                                'queries':discovery['queries'],'positive_discovery':discovery['positive_examples'],
                                'pairwise_probes':discovery['pairwise_probes'],
                                'stress_probes':discovery['stress_probes'],
                                'audited_axes':discovery['axes'],'novel_business_states':novel_cases,**score})
    overall=[]
    for budget in budgets:
        batch=[r for r in records if r['budget']==budget]
        total_neg=sum(r['true_negatives']+r['false_accepts'] for r in batch)
        overall.append({'budget':budget,'operation_seed_pairs':len(batch),
                        'cases':sum(r['cases'] for r in batch),
                        'total_false_accepts':sum(r['false_accepts'] for r in batch),
                        'false_rejects':sum(r['false_rejects'] for r in batch),
                        'mean_coverage':round(mean(r['coverage'] for r in batch),4),
                        'aggregate_positive_accept_recall':round(sum(r['accepted_true_positives'] for r in batch)/max(1,sum(r['positive_truth'] for r in batch)),4),
                        'aggregate_negative_reject_recall':round(sum(r['true_negatives'] for r in batch)/max(1,sum(r['negative_truth'] for r in batch)),4),
                        'truth_positives':sum(r['positive_truth'] for r in batch),
                        'truth_negatives':sum(r['negative_truth'] for r in batch),
                        'business_novelty_fraction':round(sum(r['novel_business_states'] for r in batch)/max(1,sum(r['cases'] for r in batch)),4),
                        'audits_ready':sum(r['audit_ready'] for r in batch),
                        'mean_effect_exact':round(mean(r['effect_exact_on_true_accepts'] for r in batch
                                   if r['effect_exact_on_true_accepts'] is not None),4)
                                   if any(r['effect_exact_on_true_accepts'] is not None for r in batch) else None})
    # Safety is necessary, but refusing every action must not be called a pass.
    large=[r for r in overall if r['budget']==max(budgets)][0]
    safety_gate= ('PASS_OFFLINE_SYNTHETIC_ONLY' if large['total_false_accepts']==0
                  and large['mean_coverage']>=.8 and large['aggregate_positive_accept_recall']>=.8
                  and large['audits_ready']>=math.ceil(len(operations)*seeds*.8)
                  and large['mean_effect_exact']==1.0 else 'FAIL')
    report={'format':'orgsim-gate2c-offline-v1','evidence':'SYNTHETIC_ORACLE','oracle_family':oracle_family,
            'not_real_erp_fidelity':True,'not_model_in_loop':True,
            'seeds':seeds,'freeze_seed':freeze_seed,'budgets':list(budgets),
            'heldout_per_operation_seed':holdout_size,'operation_families':len(operations),
            'overall':overall,'safety_gate':safety_gate,'live_dolibarr_calls':0,
            'caveats':['Oracle known to evaluator but its private thresholds are hidden from explorer.',
                       'Public candidate risk axes are hand-defined contract-level affordances; not autonomous axis discovery.',
                       'Numerical rule class assumes monotone lower bounds; arbitrary nonlinear policies remain out of scope.',
                       'Holdouts use fresh exact states; some share business-feature combinations because the finite synthetic domains are small. Novelty fraction is reported.',
                       'Zero observed false accepts is not a formal guarantee on unseen software.',
                       'No API-backed language model was available in this environment.']}
    (outdir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (outdir/'per_seed_metrics.json').write_text(json.dumps(records,indent=2)+'\n')
    return report


def main(argv:list[str]|None=None)->int:
    parser=argparse.ArgumentParser(description='Offline counterexample-guided reconstruction, zero live network I/O')
    parser.add_argument('--seeds',type=int,default=24)
    parser.add_argument('--budgets',type=int,nargs='+',default=[8,20,32])
    parser.add_argument('--holdout',type=int,default=200)
    parser.add_argument('--freeze-seed',type=int,default=20270111)
    parser.add_argument('--out',default='build/gate2c')
    parser.add_argument('--oracle-family',choices=['simple','interaction'],default='simple')
    a=parser.parse_args(argv)
    report=run_gate2c(seeds=a.seeds,budgets=tuple(a.budgets),holdout_size=a.holdout,
                      freeze_seed=a.freeze_seed,outdir=a.out,oracle_family=a.oracle_family)
    print(json.dumps({'overall':report['overall'],'safety_gate':report['safety_gate'],
                      'live_dolibarr_calls':0},indent=2))
    return 0 if report['safety_gate']=='PASS_OFFLINE_SYNTHETIC_ONLY' else 2

class _InteractionOracle(_SyntheticOracle):
    """Second evaluator-only family: hidden mixed-feature and upper guards.

    This intentionally violates the simpler monotone-conjunction contract
    assumptions. The learner sees identical public operations, never these
    private disallow conditions. The public success anchor remains legal.
    """
    def query(self, op:Operation, before:dict, *,phase:str='discovery'):
        status,after,response=super().query(op,before,phase=phase)
        if status!=200:return status,after,response
        s=before
        if op.name=='orders.validate':
            extra_bad=s['qty']>=4 and s['amount']>=70
        elif op.name=='invoices.payment':
            extra_bad=s['payment_count']>=3
        elif op.name=='orders.close':
            extra_bad=s['ordered_qty']>=5
        elif op.name=='inventory.reserve':
            extra_bad=s['qty']>=5 and s['allocated']>=4
        else:
            extra_bad=False
        if extra_bad:return 422,s.copy(),{'ok':False,'error':'POLICY_DENIED'}
        return status,after,response

if __name__=='__main__':raise SystemExit(main())
