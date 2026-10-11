"""Small, auditable decision-boundary guard learner for discovery observations.

Like the ARC Classifier Factory, this proposes a rule from observations, not a
truth. It cannot read private datasets or infer unseen ERP policy with certainty.
Only caller-approved feature names are eligible; raw examples are not persisted.
"""
from __future__ import annotations

from collections import Counter
import re
from typing import Any

_ALLOWED = re.compile(r'^(record|input)\.[a-z][a-z0-9_.]*$', re.I)


def _compare(value: Any, op: str, threshold: Any) -> bool:
    if op == 'eq':
        return value == threshold
    if op == 'ne':
        return value != threshold
    if type(value) not in {int, float} or type(threshold) not in {int, float}:
        return False
    if op == 'ge': return value >= threshold
    if op == 'gt': return value > threshold
    if op == 'le': return value <= threshold
    if op == 'lt': return value < threshold
    return False


def infer_guard(samples: list[dict], *, allowed_features: list[str], minimum_per_class: int = 2,
                max_candidates: int = 500) -> dict:
    """Select a single predicate that separates observed success from rejection.

    Returns no executable rule if the evidence is single-class or ambiguous.
    The candidate is never labeled verified, even for perfect training accuracy.
    """
    if not allowed_features or len(allowed_features) > 30 or any(not _ALLOWED.fullmatch(f) for f in allowed_features):
        raise ValueError('Explicit 1..30 record/input feature paths required')
    if len(samples) > 2000:
        raise ValueError('Too many samples for in-memory induction')
    labels = [s.get('outcome') for s in samples]
    if any(v not in {'success', 'rejected'} for v in labels):
        raise ValueError('Outcomes must be success or rejected')
    counts = Counter(labels)
    if min(counts.get('success', 0), counts.get('rejected', 0)) < minimum_per_class:
        return {'status':'insufficient_contrast', 'hypothesis':None, 'observations':len(samples),
                'note':'Both success and rejection are required to learn preconditions'}
    candidates = []
    for field in allowed_features:
        values = [s.get('features', {}).get(field) for s in samples]
        distinct = {repr((type(x).__name__, x)): x for x in values if type(x) in {int,float,str,bool}}
        if not distinct:
            continue
        if len(distinct) <= 25:
            for threshold in distinct.values():
                for op in ['eq','ne']:
                    candidates.append((field, op, threshold))
        numeric = sorted({x for x in values if type(x) in {int,float}})
        if len(numeric) >= 2:
            for left, right in zip(numeric,numeric[1:]):
                threshold = (left+right)/2
                candidates.extend((field, op, threshold) for op in ['ge','gt','le','lt'])
    if len(candidates) > max_candidates:
        return {'status':'candidate_budget_exceeded', 'hypothesis':None,
                'candidate_count':len(candidates), 'budget':max_candidates}
    best = None
    for field, op, threshold in candidates:
        pred = [_compare(s.get('features', {}).get(field), op, threshold) for s in samples]
        tp = sum(p and y=='success' for p,y in zip(pred,labels))
        fp = sum(p and y=='rejected' for p,y in zip(pred,labels))
        tn = sum(not p and y=='rejected' for p,y in zip(pred,labels))
        fn = sum(not p and y=='success' for p,y in zip(pred,labels))
        # Prefer strong rejection fidelity, not a rule that accepts everything.
        balanced = (tp/counts['success']+tn/counts['rejected'])/2
        precision = tp/max(1,tp+fp)
        score = (round(balanced,6), round(precision,6), -fp, -fn)
        if best is None or score > best[0]:
            best=(score,(field,op,threshold,tp,fp,tn,fn))
    if best is None:
        return {'status':'no_usable_predicate','hypothesis':None}
    field, op, threshold, tp, fp, tn, fn = best[1]
    accuracy = (tp+tn)/len(samples)
    baseline = max(counts.values())/len(samples)
    if accuracy <= baseline:
        return {'status':'no_better_than_majority', 'hypothesis':None,
                'baseline_accuracy':baseline, 'best_accuracy':accuracy}
    return {'status':'provisional_guard', 'hypothesis':{
        'compare':{'left':{'path':field},'op':op,'right':{'literal':threshold}}},
        'evidence':'discovery_only_unverified', 'training_examples':len(samples),
        'success_cases':counts['success'], 'rejection_cases':counts['rejected'],
        'training_accuracy':accuracy, 'balanced_accuracy':best[0][0],
        'false_accepts':fp, 'false_rejects':fn,
        'limitations':['Single-predicate model','Not tested on hidden rules, permissions or side effects',
                       'Must pass independent validation before policy promotion']}
