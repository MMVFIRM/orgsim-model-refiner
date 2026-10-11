"""Offline supervised comparator and effect induction.

The tabular classifiers below are conventional scikit-learn models, NOT GPT or
Model Refiner's Artificial Weights. Model Refiner's real training path is in
simlearn.refiner_runner, using exactly the same simulation JSONL dataset.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import json
from typing import Any

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier

from model_refiner.suite.dataset import Record
from model_refiner.suite.runtime import Memory
from .oracle import OPS, canonical

FIELDS=(
    'customer.credit','customer.received','customer.closes',
    'product.stock','product.reserved',
    'order.qty','order.amount','order.shipped','order.allocated',
    'invoice.balance','invoice.payment_count',
)


def flatten(s:dict)->dict:
    return {f'{kind}.{k}':v for kind,record in s['state'].items() for k,v in record.items()}


def feature(sample:dict)->list[float]:
    raw=flatten(sample)
    numeric=[float(raw.get(key,0)) for key in FIELDS]
    op=sample['operation'];arg=sample['action']
    order=sample['state']['order'];invoice=sample['state']['invoice'];product=sample['state']['product']
    qty=order['qty'];reserve_qty=arg.get('reserve_qty',0);payment=arg.get('payment',0)
    return numeric+[
        float(order['status']==s) for s in ('draft','validated','closed')
    ]+[
        float(invoice['status']==s) for s in ('issued','draft','paid')
    ]+[
        float(order['invoiced']),float(payment),float(reserve_qty),
        float(product['stock']-qty),float(sample['state']['customer']['credit']-order['amount']),
        float(order['shipped']-qty),float(invoice['balance']-payment),
        float(product['stock']-reserve_qty),float(product['reserved']+reserve_qty),
    ]


def features(rows:list[dict])->np.ndarray:
    return np.array([feature(r['sample']) for r in rows],dtype=np.float64)


def labels(rows:list[dict])->np.ndarray:
    return np.array([int(r['decision']=='ACCEPT') for r in rows],dtype=np.int8)


def fit_models(train:list[dict],validation:list[dict],*,seed:int):
    """All hyperparameters and risk thresholds are selected on validation only."""
    tr_by={op:[r for r in train if r['operation']==op] for op in OPS}
    va_by={op:[r for r in validation if r['operation']==op] for op in OPS}
    trained={};choices={}
    configs=[
        ('tree4',lambda:DecisionTreeClassifier(max_depth=4,min_samples_leaf=2,random_state=seed)),
        ('tree6',lambda:DecisionTreeClassifier(max_depth=6,min_samples_leaf=2,random_state=seed)),
        ('tree9',lambda:DecisionTreeClassifier(max_depth=9,min_samples_leaf=2,random_state=seed)),
        ('extra256',lambda:ExtraTreesClassifier(n_estimators=256,max_features=1.0,min_samples_leaf=1,n_jobs=1,random_state=seed)),
        ('extra128_leaf2',lambda:ExtraTreesClassifier(n_estimators=128,max_features=1.0,min_samples_leaf=2,n_jobs=1,random_state=seed)),
        ('forest160',lambda:RandomForestClassifier(n_estimators=160,max_features=1.0,min_samples_leaf=2,n_jobs=1,random_state=seed)),
        ('hgb100',lambda:HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,
                                                         learning_rate=.08,min_samples_leaf=4,random_state=seed)),
        ('hgb200',lambda:HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=15,
                                                         learning_rate=.06,min_samples_leaf=4,random_state=seed)),
    ]
    for oi,op in enumerate(OPS):
        X=features(tr_by[op]);Y=labels(tr_by[op]);V=features(va_by[op]);truth=labels(va_by[op])
        if len(set(Y))!=2:raise ValueError('Training split lost a class')
        results=[]
        for label,builder in configs:
            model=builder();model.fit(X,Y)
            prob=model.predict_proba(V)[:,list(model.classes_).index(1)]
            # Optimize decision accuracy, use false-accepts and calibration error
            # as tie breakers. No test information is used.
            for threshold in (0.35,0.4,0.45,0.5,0.55,0.6,0.65):
                pred=prob>=threshold
                accuracy=float(np.mean(pred==truth))
                false_accept=int(np.sum(pred & (truth==0)))
                balanced=.5*(float(np.mean(pred[truth==1]))+float(np.mean(~pred[truth==0])))
                results.append((accuracy,balanced,-false_accept,-abs(.5-threshold),
                                label,threshold,model))
        best=max(results,key=lambda t:t[:4])
        trained[op]={'model':best[-1],'threshold':best[-2]}
        choices[op]={'model':best[-3],'threshold':best[-2],
                     'validation_accuracy':round(best[0],6),
                     'validation_balanced_accuracy':round(best[1],6),
                     'validation_false_accepts':-best[2]}
    return trained,choices


def predict_tabular(models:dict, rows:list[dict])->list[str]:
    out=[]
    for row in rows:
        op=row['operation'];info=models[op]
        prob=info['model'].predict_proba(np.array([feature(row['sample'])],dtype=np.float64))[0]
        idx=list(info['model'].classes_).index(1)
        out.append('ACCEPT' if prob[idx]>=info['threshold'] else 'REJECT')
    return out


def predict_probability(models:dict, rows:list[dict])->list[float]:
    out=[]
    for r in rows:
        info=models[r['operation']];cl=info['model'].classes_
        probs=info['model'].predict_proba(np.array([feature(r['sample'])]))[0]
        out.append(float(probs[list(cl).index(1)]))
    return out


def predict_generic(row:dict)->str:
    """A strong public-precondition baseline with no private veto knowledge."""
    s=row['sample'];r=s['state'];a=s['action'];op=s['operation']
    o=r['order'];p=r['product'];c=r['customer'];i=r['invoice']
    allow={
      'validate_order':o['status']=='draft' and p['stock']>=o['qty'] and c['credit']>=o['amount'],
      'pay_invoice':i['status']=='issued' and i['balance']>=a.get('payment',0),
      'reserve_stock':o['status']=='draft' and p['stock']>=a.get('reserve_qty',0),
      'close_order':o['status']=='validated' and o['shipped']>=o['qty'] and o['invoiced'],
    }[op]
    return 'ACCEPT' if allow else 'REJECT'


def predict_retrieval(train:list[dict],eval_rows:list[dict],*,k:int=5)->list[str]:
    """Uses the ACTUAL Model Refiner 0.5rc1 TF-IDF retrieval implementation."""
    recs=[Record(id=r['id'],input=canonical(r['sample']),target=r['decision'],group=r['group'],slice=r['operation']) for r in train]
    mem={op:Memory([r for r in recs if r.slice==op]) for op in OPS}
    pred=[]
    for row in eval_rows:
        near=mem[row['operation']].retrieve(canonical(row['sample']),k)
        accepted=sum(x.target=='ACCEPT' for x in near)
        pred.append('ACCEPT' if accepted>len(near)/2 else 'REJECT')
    return pred


@dataclass
class Formula:
    path:str|None=None
    sign:int=1
    const:int|None=None

    def value(self,sample:dict):
        if self.const is not None:return self.const
        keys=self.path.split('.')
        root=sample['action'] if keys[0]=='action' else sample['state'][keys[0]]
        for k in keys[1:]:root=root[k]
        return self.sign*root

    def to_dict(self):return {'path':self.path,'sign':self.sign,'const':self.const}


def _effect_formulas(op:str)->list[Formula]:
    """Small generic affine basis; no encoded business-policy thresholds."""
    paths=['action.payment','action.reserve_qty',
           'order.qty','order.amount','order.shipped','order.allocated',
           'invoice.balance','invoice.payment_count',
           'product.stock','product.reserved',
           'customer.credit','customer.received','customer.closes']
    paths=[p for p in paths if not (p=='action.payment' and op!='pay_invoice')
                             and not (p=='action.reserve_qty' and op!='reserve_stock')]
    return [Formula(const=v) for v in (0,1,-1,2,-2)]+[Formula(path=p,sign=sgn) for p in paths for sgn in (1,-1)]


class EffectLearner:
    """Infer scalar state effects from *executed* trajectories only.

    Expressions are required to explain every successful training observation;
    uncertain expressions are not elevated to executable status. Holds out
    validation to audit the fit and never sees test/stress outcomes during fit.
    """
    def __init__(self):self.models={};self.report={}

    def fit(self,train:list[dict],validation:list[dict]):
        for op in OPS:
            positives=[r for r in train if r['operation']==op and r['decision']=='ACCEPT']
            valid=[r for r in validation if r['operation']==op and r['decision']=='ACCEPT']
            if not positives:raise ValueError('No positive effect witnesses')
            keys=sorted({key for r in positives for key in r['changed']})
            model={};rejected=[]
            for key in keys:
                entity,field=key.split('.')
                previous=[r['sample']['state'][entity][field] for r in positives]
                changed=[r['after'][entity][field] for r in positives]
                if all(type(v) is str for v in changed):
                    unique=set(changed)
                    if len(unique)==1:model[key]={'literal':next(iter(unique))}
                    else:rejected.append(key)
                    continue
                if not all(type(v) is int for v in changed):
                    rejected.append(key);continue
                actual_deltas=[v-a for a,v in zip(previous,changed)]
                for formula in _effect_formulas(op):
                    try:
                        if all(formula.value(r['sample'])==delta for r,delta in zip(positives,actual_deltas)):
                            model[key]={'delta':formula.to_dict()};break
                    except (KeyError,TypeError):continue
                else:rejected.append(key)
            self.models[op]=model
            correct=sum(self.predict_accept(r['sample'],op)==r['changed'] for r in valid)
            self.report[op]={'identified_effect_fields':sorted(model),'unidentified_effect_fields':rejected,
                             'witnessed_accepts':len(positives),'validation_accepts':len(valid),
                             'validation_effect_exact':(correct/len(valid) if valid else None),
                             'audited':not rejected and bool(valid) and correct==len(valid)}
        return self

    def predict_accept(self,sample:dict,op:str)->dict:
        effect={}
        for key,expr in self.models.get(op,{}).items():
            entity,field=key.split('.')
            before=sample['state'][entity][field]
            if 'literal' in expr:after=expr['literal']
            else:
                formula=Formula(**expr['delta']);after=before+formula.value(sample)
            if after!=before:effect[key]=after
        return effect

    def predict(self,sample:dict,decision:str)->dict:
        if decision=='REJECT':return {}
        return self.predict_accept(sample,sample['operation'])
