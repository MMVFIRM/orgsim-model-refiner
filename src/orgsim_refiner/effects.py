"""Scoped, empirically checked state-effect expressions; not LLM predictions."""
from __future__ import annotations
from dataclasses import dataclass
from . import dsl
from .types import Scope,Observation
from .util import flatten,canonical,digest,loads

def _inputs(row):
    return {**{'before.'+k:v for k,v in flatten(row.before or {}).items()},
            **{'action.'+k:v for k,v in flatten(row.action or {}).items()}}

def _equal(a,b):
    return type(a)==type(b) and a==b

class EffectModel:
    def __init__(self,scope:Scope):self.scope=scope;self.programs={};self.ready=False;self.certificate={}
    def fit(self,training:list[Observation],validation:list[Observation]):
        self.programs={};self.ready=False
        for r in training+validation:
            if r.scope!=self.scope or not r.usable():raise ValueError('Effect evidence scope/provenance mismatch')
        if any(r.split not in ('train','active','audit') for r in training) or any(r.split!='validation' for r in validation):raise ValueError('Effect evidence split violation')
        if {r.sample_hash for r in training}&{r.sample_hash for r in validation}:raise ValueError('Effects validation overlap')
        tr=[r for r in training if r.label=='APPROVE' and r.before is not None and r.after is not None]
        va=[r for r in validation if r.label=='APPROVE' and r.before is not None and r.after is not None]
        if len(tr)<8 or len(va)<4:
            self.certificate={'status':'insufficient_observed_transitions'};return self
        keys=set(flatten(tr[0].after));inputkeys=set(_inputs(tr[0]))
        if any(set(flatten(r.after))!=keys or set(_inputs(r))!=inputkeys for r in tr+va):
            self.certificate={'status':'state_schema_changed'};return self
        for output in sorted(keys):
            self.programs[output]=self._solve(output,tr,va,inputkeys)
        self.ready=all(self.programs.values())
        self.certificate={'status':'empirically_checked' if self.ready else 'unsupported_effect',
            'training_accepts':len(tr),'validation_accepts':len(va),'output_fields':sorted(keys),
            'training_sha256':digest([r.to_dict() for r in tr]),'validation_sha256':digest([r.to_dict() for r in va]),
            'claim':'All declared observed fields only; unobserved side effects are not modeled.'}
        return self
    @staticmethod
    def _numeric_inputs(rows,inputkeys):
        return [k for k in sorted(inputkeys) if all(type(_inputs(r)[k]) in (int,float) and type(_inputs(r)[k])!=bool for r in rows)]

    def _candidates(self,output,rows,inputkeys):
        """Constants, copies, and before.out +/- sums of up to three numeric action fields."""
        from itertools import combinations
        first=flatten(rows[0].after)[output]
        out=[dsl.constant(first)]+[dsl.field(k) for k in sorted(inputkeys)]
        before_name='before.'+output
        nums=self._numeric_inputs(rows,inputkeys)
        if type(first) in (int,float) and before_name in nums:
            delta=first-_inputs(rows[0])[before_name]
            out.append(dsl.binary('add',dsl.field(before_name),dsl.constant(delta)))
            action_nums=[k for k in nums if k.startswith('action.')]
            for size in (1,2,3):
                for subset in combinations(action_nums,size):
                    total=dsl.field(subset[0])
                    for k in subset[1:]:total=dsl.binary('add',total,dsl.field(k))
                    out.append(dsl.binary('sub',dsl.field(before_name),total))
                    out.append(dsl.binary('add',dsl.field(before_name),total))
        return out

    def _fits(self,expr,output,rows):
        return all(_equal(dsl.evaluate(expr,_inputs(r)),flatten(r.after)[output]) for r in rows)

    def _solve(self,output,tr,va,inputkeys):
        """Global expression if one exists; otherwise a single-field conditional
        (e.g. stock unchanged for services, decremented for goods)."""
        both=tr+va
        direct=[e for e in self._candidates(output,tr,inputkeys) if self._fits(e,output,tr) and self._fits(e,output,va)]
        if direct:return list({digest(x):x for x in direct}.values())
        for key in sorted(inputkeys):
            vals=[_inputs(r)[key] for r in both]
            if not all(type(v) in (str,bool) for v in vals):continue
            groups={}
            for r in both:groups.setdefault(_inputs(r)[key],[]).append(r)
            if len(groups)<2 or len(groups)>4:continue
            chosen={}
            for v,rows_v in groups.items():
                if len(rows_v)<2:break
                ok=[e for e in self._candidates(output,rows_v,inputkeys) if self._fits(e,output,rows_v)]
                if not ok:break
                chosen[v]=min(ok,key=lambda e:len(canonical(e)))
            else:
                ordered=sorted(chosen)
                expr=chosen[ordered[-1]]
                for v in reversed(ordered[:-1]):
                    expr={'op':'if','condition':{'op':'in','arg':dsl.field(key),'values':[v]},'then':chosen[v],'else':expr}
                if self._fits(expr,output,tr) and self._fits(expr,output,va):return [expr]
        return []
    def predict(self,before,action,scope:Scope):
        if scope!=self.scope or not self.ready:return None
        inp={**{'before.'+k:v for k,v in flatten(before).items()},**{'action.'+k:v for k,v in flatten(action).items()}}
        result={}
        for key,exprs in self.programs.items():
            vals=[dsl.evaluate(e,inp) for e in exprs]
            if not vals or any(v is None for v in vals) or any(not _equal(v,vals[0]) for v in vals):return None
            result[key]=vals[0]
        return result
    def to_dict(self):return {'scope':self.scope.to_dict(),'programs':self.programs,'ready':self.ready,'certificate':self.certificate}
    @classmethod
    def from_dict(cls,p):
        m=cls(Scope(**p['scope']));m.programs=loads(canonical(p['programs']));m.ready=bool(p['ready']);m.certificate=p['certificate']
        if len(m.programs)>256 or any(not isinstance(k,str) or '__' in k for k in m.programs):raise ValueError('Invalid effect outputs')
        for exprs in m.programs.values():
            for expr in exprs:
                # Enumerate names structurally; external state schema is checked at predict.
                names=set()
                def visit(n):
                    if isinstance(n,dict):
                        if n.get('op')=='field':names.add(n['name'])
                        for v in n.values():visit(v)
                    elif isinstance(n,list):
                        for v in n:visit(v)
                visit(expr)
                if any(not name.startswith(('before.','action.')) or '__' in name for name in names):raise ValueError('Invalid effect field')
                dsl.validate(expr,names,expected='any')
        return m
