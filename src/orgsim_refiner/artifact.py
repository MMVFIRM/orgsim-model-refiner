"""Versioned executable policy with explicit domain, evidence and uncertainty."""
from __future__ import annotations
from dataclasses import dataclass, field as datafield
from bisect import bisect_right
from itertools import product
import math
from .types import Scope,Schema
from .util import canonical, loads, digest, read_json,write_json
from . import dsl

@dataclass
class Partition:
    schema: Schema
    cuts: dict[str,list[int]]=datafield(default_factory=dict)
    def __post_init__(self):
        known={f.name for f in self.schema.fields if f.kind=='int'}
        if set(self.cuts)-known:raise ValueError('Invalid partition field')
        for f in self.schema.fields:
            if f.kind=='int':
                c=self.cuts.get(f.name,[])
                if c!=sorted(set(c)) or any(type(x)!=int or not f.low<x<=f.high for x in c):raise ValueError('Invalid cut')
                self.cuts[f.name]=list(c)
    def axes(self):
        return [range(len(self.cuts[f.name])+1) if f.kind=='int' else range(len(f.domain)) for f in self.schema.fields]
    @property
    def size(self):return math.prod(len(a) for a in self.axes())
    def key(self,sample):
        self.schema.validate(sample)
        return tuple(bisect_right(self.cuts[f.name],sample[f.name]) if f.kind=='int' else f.domain.index(sample[f.name]) for f in self.schema.fields)
    def cells(self):return product(*self.axes())
    def representative(self,key,rng=None):
        sample={}
        for f,k in zip(self.schema.fields,key):
            if f.kind=='int':
                edges=[f.low]+self.cuts[f.name]+[f.high+1]
                lo,hi=edges[k],edges[k+1]-1
                sample[f.name]=rng.randint(lo,hi) if rng else (lo+hi)//2
            else:sample[f.name]=f.domain[k]
        return sample
    def add_cuts(self,name,values):
        self.cuts[name]=sorted(set(self.cuts[name])|set(values))
    @staticmethod
    def text_key(key):return ','.join(str(int(x)) for x in key)

class PolicyArtifact:
    FORMAT='orgsim-refiner-active-policy-v1'
    def __init__(self,payload:dict):
        p=loads(canonical(payload))
        expected={'format','scope','schema','cuts','table','ast','evidence','calibration','ready','reason','assumptions','artifact_sha256'}
        if set(p)!=expected or p['format']!=self.FORMAT:raise ValueError('Policy artifact schema mismatch')
        if digest({k:v for k,v in p.items() if k!='artifact_sha256'})!=p['artifact_sha256']:raise ValueError('Policy hash mismatch')
        self.scope=Scope(**p['scope']);self.schema=Schema.from_dict(p['schema'])
        self._partition=Partition(self.schema,p['cuts'])
        if self._partition.size>65536:raise ValueError('Partition too large')
        keys={Partition.text_key(k) for k in self._partition.cells()}
        if set(p['table'])-keys or any(v not in ('APPROVE','REJECT') for v in p['table'].values()):raise ValueError('Invalid partition table')
        if type(p['ready'])!=bool:raise ValueError('Invalid ready flag')
        if p['ast'] is not None:
            dsl.validate(p['ast'],{f.name for f in self.schema.fields})
            for k in self._partition.cells():
                known=p['table'].get(Partition.text_key(k))
                if known and dsl.label(p['ast'],self._partition.representative(k))!=known:
                    raise ValueError('Compiled AST does not reproduce region table')
        self._data=p
    @classmethod
    def create(cls,**payload):
        p={'format':cls.FORMAT,**payload};return cls({**p,'artifact_sha256':digest(p)})
    @property
    def data(self):return loads(canonical(self._data))
    @property
    def partition(self):return Partition(self.schema,loads(canonical(self._data['cuts'])))
    @property
    def fingerprint(self):return self._data['artifact_sha256']
    @property
    def ready(self):return self._data['ready']
    def to_dict(self):return loads(canonical(self._data))
    def save(self,path):write_json(path,self._data)
    @classmethod
    def load(cls,path):return cls(read_json(path))
    def predict(self,sample,scope:Scope,*,advisory=False):
        out={'decision':'UNKNOWN','artifact_sha256':self.fingerprint,'erp_authorization':False}
        if scope!=self.scope:return {**out,'reason':'scope_mismatch'}
        try:key=self._partition.key(sample)
        except ValueError:return {**out,'reason':'outside_declared_schema'}
        if not self.ready and not advisory:return {**out,'reason':'policy_not_calibrated'}
        if not self._data['table'] and self._data['ast'] is not None:
            y=dsl.label(self._data['ast'],sample)
            if y=='UNKNOWN':return {**out,'reason':'ast_undefined'}
            return {**out,'decision':y,'reason':'executable_ast','region':[],
                    'assumption':'Executable expression policy; empirically audited at its boundaries, not proved.',
                    'evidence_status':'calibrated' if self.ready else 'unpromoted_advisory'}
        y=self._data['table'].get(Partition.text_key(key))
        if y is None:return {**out,'reason':'unobserved_region'}
        return {**out,'decision':y,'reason':'empirical_region_interpolation','region':list(key),
                'assumption':'Label is constant within the learned region; empirically checked, not proved.',
                'evidence_status':'calibrated' if self.ready else 'unpromoted_advisory'}
    def with_calibration(self,calibration,ready,reason):
        p=self.to_dict();p.pop('artifact_sha256');p.pop('format')
        p.update(calibration=calibration,ready=ready,reason=reason)
        return PolicyArtifact.create(**p)
