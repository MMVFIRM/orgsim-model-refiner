from __future__ import annotations
from dataclasses import dataclass, asdict
import math, random, re
from typing import Any
from .util import canonical, digest

DECISIONS=('APPROVE','REJECT','UNKNOWN')
TRUSTED=('SIMULATOR_OBSERVED','REAL_OPERATOR_CORROBORATED','REAL_API_VERIFIED','DISCLOSED_SYNTHETIC_ARCHIVE')
SPLITS=('train','active','audit','validation','holdout','stress')

@dataclass(frozen=True)
class Scope:
    tenant: str
    operation: str
    configuration: str
    def __post_init__(self):
        if any(not isinstance(x,str) or not x.strip() or len(x)>200 for x in (self.tenant,self.operation,self.configuration)):
            raise ValueError('Explicit tenant, operation and configuration required')
    def to_dict(self): return asdict(self)
    @property
    def key(self): return digest(self.to_dict())

@dataclass(frozen=True)
class Field:
    name: str
    kind: str
    low: int=0
    high: int=1
    values: tuple=()
    def __post_init__(self):
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_.]{0,95}',self.name) or '__' in self.name:
            raise ValueError('Unsafe field name')
        if self.kind not in ('int','enum','bool'): raise ValueError('Use integer units, bool, or an explicit enum')
        if self.kind=='int' and (type(self.low)!=int or type(self.high)!=int or not 0<=self.high-self.low<=10000 or abs(self.low)>10**12 or abs(self.high)>10**12):
            raise ValueError('Integer grid must have 1..10001 values')
        if self.kind=='enum':
            if not 1<=len(self.values)<=64 or any(type(x)!=str or not x or len(x)>100 for x in self.values):
                raise ValueError('Enums require bounded string values')
            if len(set(self.values))!=len(self.values): raise ValueError('Duplicate enum value')
    @property
    def domain(self):
        return list(range(self.low,self.high+1)) if self.kind=='int' else ([False,True] if self.kind=='bool' else list(self.values))
    def accepts(self,x):
        if self.kind=='int': return type(x)==int and self.low<=x<=self.high
        if self.kind=='bool': return type(x)==bool
        return type(x)==str and x in self.values
    def to_dict(self):
        d={'name':self.name,'kind':self.kind}
        if self.kind=='int': d.update(low=self.low,high=self.high)
        if self.kind=='enum': d['values']=list(self.values)
        return d
    @classmethod
    def from_dict(cls,d):
        if set(d)-{'name','kind','low','high','values'}: raise ValueError('Unknown field setting')
        d=dict(d);d['values']=tuple(d.get('values',()))
        return cls(**d)

@dataclass(frozen=True)
class Schema:
    fields: tuple[Field,...]
    def __post_init__(self):
        if not 1<=len(self.fields)<=12: raise ValueError('Schema supports 1..12 named fields')
        if len({f.name for f in self.fields})!=len(self.fields): raise ValueError('Duplicate field')
    def validate(self, sample:dict):
        if not isinstance(sample,dict) or set(sample)!={f.name for f in self.fields}:
            raise ValueError('Missing or unexpected feature; no silent defaults')
        if any(not f.accepts(sample[f.name]) for f in self.fields): raise ValueError('Feature outside declared domain/type')
        return sample
    def sample(self,rng: random.Random): return {f.name:rng.choice(f.domain) for f in self.fields}
    def encode(self,x):
        self.validate(x)
        return [x[f.name] if f.kind=='int' else f.domain.index(x[f.name]) for f in self.fields]
    def to_dict(self): return {'fields':[f.to_dict() for f in self.fields]}
    @classmethod
    def from_dict(cls,d):
        if set(d)!={'fields'}: raise ValueError('Invalid schema')
        return cls(tuple(Field.from_dict(x) for x in d['fields']))
    @property
    def key(self): return digest(self.to_dict())

@dataclass(frozen=True)
class Observation:
    scope: Scope
    sample: dict
    label: str
    split: str='train'
    provenance: str='SIMULATOR_OBSERVED'
    source_id: str=''
    before: dict | None=None
    after: dict | None=None
    action: dict | None=None
    evidence: dict | None=None
    def __post_init__(self):
        if self.label not in DECISIONS or self.split not in SPLITS: raise ValueError('Invalid label/split')
        canonical(self.sample)
    @property
    def sample_hash(self): return digest(self.sample)
    def to_dict(self):
        d=asdict(self);return d
    @classmethod
    def from_dict(cls,d):
        d=dict(d);d['scope']=Scope(**d['scope']);return cls(**d)
    def usable(self):
        return self.label in DECISIONS[:2] and self.provenance in TRUSTED
