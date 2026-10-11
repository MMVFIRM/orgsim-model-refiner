"""Executable candidate bank and exact replay of all permitted evidence.

This is a sampled candidate bank, not enumeration of all possible programs.
Agreement is not a proof that the true rule belongs to the sampled bank.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from . import dsl
from .types import Schema, Observation
from .util import digest

def export_tree(tree, schema:Schema):
    t=tree.tree_
    def node(i):
        if t.children_left[i]==t.children_right[i]:
            cls=int(tree.classes_[int(np.argmax(t.value[i][0]))])
            return dsl.constant(bool(cls))
        j=int(t.feature[i]);f=schema.fields[j];cut=float(t.threshold[i])
        cond=(dsl.binary('le',dsl.field(f.name),dsl.constant(cut)) if f.kind=='int' else
              {'op':'in','arg':dsl.field(f.name),'values':[v for k,v in enumerate(f.domain) if k<=cut]})
        return {'op':'if','condition':cond,'then':node(int(t.children_left[i])),'else':node(int(t.children_right[i]))}
    ast=node(0);dsl.validate(ast,{f.name for f in schema.fields});return ast

def replay(ast,schema,rows):
    dsl.validate(ast,{f.name for f in schema.fields})
    return [r.sample_hash for r in rows if dsl.label(ast,r.sample)!=r.label]

class CandidateBank:
    def __init__(self,schema:Schema,*,seed:int=0,count:int=12):
        self.schema=schema;self.seed=seed;self.count=count;self.asts=[];self.models=[];self.rejections=[]
    def fit(self,rows:list[Observation],extra_candidates=()):
        self.asts=[];self.models=[];self.rejections=[]
        if not rows:return self
        X=np.asarray([self.schema.encode(r.sample) for r in rows],dtype=float)
        y=np.asarray([r.label=='APPROVE' for r in rows],dtype=int)
        seen={}
        for r in rows:
            if r.sample_hash in seen and seen[r.sample_hash]!=r.label:raise ValueError('Conflicting evidence')
            seen[r.sample_hash]=r.label
        unique=set()
        for i in range(self.count):
            model=DecisionTreeClassifier(criterion='entropy' if i%2 else 'gini',
                splitter='best' if i<self.count//2 else 'random',random_state=self.seed+i,
                max_features=None if i<2 else 'sqrt')
            model.fit(X,y)
            if not np.array_equal(model.predict(X),y):continue
            ast=export_tree(model,self.schema);h=digest(ast)
            if h not in unique:
                unique.add(h);self.models.append(model);self.asts.append(ast)
        for ast in extra_candidates:
            try:
                errors=replay(ast,self.schema,rows)
                if errors:self.rejections.append({'reason':'contradicts_evidence','count':len(errors),'example_sha256':errors[:8]});continue
                h=digest(ast)
                if h not in unique:unique.add(h);self.asts.append(ast)
            except (ValueError,KeyError,TypeError):self.rejections.append({'reason':'invalid_ast'})
        return self
    def probabilities(self,samples):
        if not samples:return np.array([])
        if not self.asts:return np.full(len(samples),.5)
        X=np.asarray([self.schema.encode(x) for x in samples],dtype=float)
        votes=[m.predict(X) for m in self.models]
        for ast in self.asts[len(self.models):]:votes.append(np.array([dsl.evaluate(ast,x) for x in samples],dtype=float))
        return np.mean(votes,axis=0)
    def predict(self,sample,*,consensus=True):
        p=float(self.probabilities([sample])[0])
        if consensus and 0<p<1:return 'UNKNOWN'
        return 'APPROVE' if p>.5 else ('REJECT' if p<.5 else 'UNKNOWN')
    def smallest(self):
        if not self.asts:return None
        return min(self.asts,key=lambda a:(len(str(a)),digest(a)))

def compile_partition(partition,table,seed=0):
    """Compress cell labels without moving their learned raw-space boundaries.

Training an ordinary tree on cell midpoints would put cuts BETWEEN midpoints,
not at the observed transition. Translate index splits through the partition.
"""
    from .artifact import Partition
    keys=list(partition.cells())
    X=np.asarray(keys,dtype=float);y=np.array([table[Partition.text_key(k)]=='APPROVE' for k in keys],dtype=int)
    options=[]
    for criterion in ('gini','entropy'):
        model=DecisionTreeClassifier(criterion=criterion,random_state=seed).fit(X,y);t=model.tree_
        def node(i):
            if t.children_left[i]==t.children_right[i]:return dsl.constant(bool(model.classes_[int(np.argmax(t.value[i][0]))]))
            j=int(t.feature[i]);f=partition.schema.fields[j];cut=int(np.floor(t.threshold[i]))
            if f.kind=='int':
                # Index 0 covers [low, cuts[0]-1], etc.
                boundary=partition.cuts[f.name][cut]
                cond=dsl.binary('lt',dsl.field(f.name),dsl.constant(boundary))
            else:cond={'op':'in','arg':dsl.field(f.name),'values':f.domain[:cut+1]}
            return {'op':'if','condition':cond,'then':node(int(t.children_left[i])),'else':node(int(t.children_right[i]))}
        ast=node(0);dsl.validate(ast,{f.name for f in partition.schema.fields});options.append(ast)
    return min(options,key=lambda x:len(str(x)))
