"""Hard scope filtering BEFORE distance/ranking; never global fallback."""
from __future__ import annotations
from .types import Schema,Scope,Observation
from .util import digest

def distance(schema:Schema,a:dict,b:dict):
    terms=[]
    for f in schema.fields:
        x,y=a[f.name],b[f.name]
        terms.append(abs(x-y)/max(1,f.high-f.low) if f.kind=='int' else float(x!=y))
    return sum(terms)/len(terms)

class TenantMemory:
    def __init__(self,schema:Schema,rows:list[Observation]):
        self.schema=schema;self.rows=list(rows)
        for r in rows:
            schema.validate(r.sample)
            if r.split not in ('train','active','audit') or not r.usable():raise ValueError('Retrieval memory must be observed training evidence')
    def retrieve(self,scope:Scope,sample:dict,k:int=8):
        self.schema.validate(sample)
        if type(k)!=int or not 1<=k<=10000:raise ValueError('Bad retrieval count')
        eligible=[r for r in self.rows if r.scope==scope]
        # IDs and tenant identifiers are not vectorized. Scope is an access key.
        return sorted(eligible,key=lambda r:(distance(self.schema,sample,r.sample),r.sample_hash))[:k]
    def vote(self,scope:Scope,sample:dict,k:int=5):
        rows=self.retrieve(scope,sample,k)
        if not rows:return 'UNKNOWN'
        a=sum(r.label=='APPROVE' for r in rows);b=len(rows)-a
        return 'APPROVE' if a>b else ('REJECT' if b>a else 'UNKNOWN')
