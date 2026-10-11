"""Tenant/configuration isolation, phase separation and append-only receipts."""
from __future__ import annotations
import json, sqlite3, threading
from pathlib import Path
from typing import Callable
from .types import Scope, Schema, Observation, TRUSTED
from .util import canonical, digest, loads

class EvidenceError(ValueError): pass
class QueryBudgetExceeded(EvidenceError): pass

class EvidenceLedger:
    """Local integrity chain, NOT an authenticated transaction log.

SQLite stores canonical payloads rather than platform-dependent text hashes.
A single IMMEDIATE transaction serializes each append across processes.
"""
    def __init__(self,path:Path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        with self._connect() as con:
            con.execute('CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL, previous TEXT NOT NULL, hash TEXT NOT NULL UNIQUE)')
            con.execute("CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events BEGIN SELECT RAISE(ABORT, 'append-only evidence'); END")
            con.execute("CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events BEGIN SELECT RAISE(ABORT, 'append-only evidence'); END")
    def _connect(self):return sqlite3.connect(self.path,timeout=30)
    def append(self,kind:str,payload:dict):
        encoded=canonical(payload)
        with self._connect() as con:
            con.execute('BEGIN IMMEDIATE')
            last=con.execute('SELECT seq, hash FROM events ORDER BY seq DESC LIMIT 1').fetchone()
            seq=last[0]+1 if last else 1;prev=last[1] if last else '0'*64
            h=digest({'seq':seq,'kind':kind,'payload':loads(encoded),'previous':prev})
            con.execute('INSERT INTO events VALUES (?,?,?,?,?)',(seq,kind,encoded,prev,h))
        return h
    def events(self):
        with self._connect() as con:
            rows=con.execute('SELECT seq,kind,payload,previous,hash FROM events ORDER BY seq').fetchall()
        return [{'seq':s,'kind':k,'payload':loads(p),'previous':pr,'hash':h} for s,k,p,pr,h in rows]
    def verify(self):
        prev='0'*64;rows=self.events()
        for i,e in enumerate(rows,1):
            if e['seq']!=i or e['previous']!=prev or digest({k:e[k] for k in ('seq','kind','payload','previous')})!=e['hash']:
                raise EvidenceError('Evidence chain mismatch')
            prev=e['hash']
        return {'events':len(rows),'chain_sha256':prev,'externally_timestamped':False}

class EvidenceSet:
    def __init__(self,schema:Schema,scope:Scope):
        self.schema=schema;self.scope=scope;self.rows=[];self._seen={};self.conflicted=False
    def add(self,row:Observation):
        if row.scope!=self.scope:raise EvidenceError('Cross-tenant/operation/configuration evidence rejected')
        self.schema.validate(row.sample)
        if row.split not in ('train','active','audit'):raise EvidenceError('Evaluation labels cannot enter learning evidence')
        if not row.usable():raise EvidenceError('Unknown, uncorroborated and model-generated labels are not training truth')
        old=self._seen.get(row.sample_hash)
        if old and old.label!=row.label:
            self.conflicted=True
            raise EvidenceError('Contradictory observed labels; scope quarantined')
        if not old:
            # Detach from mutable caller data.
            row=Observation.from_dict(loads(canonical(row.to_dict())))
            self.rows.append(row);self._seen[row.sample_hash]=row
        return row
    def get(self,sample):return self._seen.get(digest(sample))
    @property
    def fingerprint(self):return digest([r.to_dict() for r in sorted(self.rows,key=lambda x:x.sample_hash)])

class TrainingOracle:
    """Explicit training-only callback; no provider/HTTP/shell dispatch.

Caller supplies a verified simulator callback or a local pending-request inbox.
Heldout identities are excluded before invoking that callback. The callback must
return the exact requested sample and scope, not a proposed label from a model.
"""
    def __init__(self,schema:Schema,scope:Scope,callback:Callable[[dict],Observation],*,
                 budget:int,forbidden_hashes:set[str]|None=None,ledger:EvidenceLedger|None=None):
        if type(budget)!=int or not 0<=budget<=100000:raise ValueError('Query budget 0..100000')
        self.schema=schema;self.scope=scope;self.callback=callback;self.budget=budget
        self.forbidden=set(forbidden_hashes or ());self.ledger=ledger;self.calls=0
        self.closed=False;self.observations=[];self.cache={};self.lock=threading.Lock()
    def query(self,sample,*,purpose='active',force_repeat=False):
        self.schema.validate(sample);h=digest(sample)
        if purpose not in ('active','audit'):raise EvidenceError('Oracle usable for training/audit only')
        with self.lock:
            if self.closed:raise EvidenceError('Oracle closed after policy freeze')
            if h in self.forbidden:raise EvidenceError('Reserved evaluation input cannot be queried')
            if h in self.cache and not force_repeat:return self.cache[h]
            if self.calls>=self.budget:raise QueryBudgetExceeded('Hard oracle request budget exhausted')
            self.calls+=1
            request={'scope':self.scope.to_dict(),'sample_sha256':h,'purpose':purpose,'attempt':self.calls}
            if self.ledger:self.ledger.append('oracle_request',request)
            try:
                row=self.callback(loads(canonical(sample)))
                if not isinstance(row,Observation) or row.scope!=self.scope or row.sample_hash!=h:
                    raise EvidenceError('Oracle response scope/sample mismatch')
                if not row.usable() or row.provenance=='DISCLOSED_SYNTHETIC_ARCHIVE':
                    raise EvidenceError('Callback did not supply an observed outcome')
                row=Observation.from_dict({**row.to_dict(),'split':purpose})
                if h in self.cache and self.cache[h].label!=row.label:
                    raise EvidenceError('Repeated query changed label; oracle is not stationary/deterministic')
                self.cache[h]=row;self.observations.append(row)
                if self.ledger:self.ledger.append('oracle_response',{**request,'observation':row.to_dict()})
                return row
            except Exception as e:
                if self.ledger:self.ledger.append('oracle_failure',{**request,'error_type':type(e).__name__})
                raise
    def query_many(self,samples,purposes,*,workers=1):
        """Concurrent training queries. Returns (rows, budget_exhausted).
        Same checks as query(); results are returned in input order. The caller
        supplies a thread-safe callback when workers>1."""
        from concurrent.futures import ThreadPoolExecutor
        todo=[]
        with self.lock:
            if self.closed:raise EvidenceError('Oracle closed after policy freeze')
            exhausted=False
            for x,purpose in zip(samples,purposes):
                self.schema.validate(x);h=digest(x)
                if purpose not in ('active','audit'):raise EvidenceError('Oracle usable for training/audit only')
                if h in self.forbidden:raise EvidenceError('Reserved evaluation input cannot be queried')
                if h in self.cache:continue
                if self.calls>=self.budget:exhausted=True;break
                self.calls+=1
                req={'scope':self.scope.to_dict(),'sample_sha256':h,'purpose':purpose,'attempt':self.calls}
                if self.ledger:self.ledger.append('oracle_request',req)
                todo.append((x,h,purpose,req))
        def run(item):
            x,h,purpose,req=item
            row=self.callback(loads(canonical(x)))
            if not isinstance(row,Observation) or row.scope!=self.scope or row.sample_hash!=h:
                raise EvidenceError('Oracle response scope/sample mismatch')
            if not row.usable() or row.provenance=='DISCLOSED_SYNTHETIC_ARCHIVE':
                raise EvidenceError('Callback did not supply an observed outcome')
            return Observation.from_dict({**row.to_dict(),'split':purpose}),req
        if workers>1 and len(todo)>1:
            with ThreadPoolExecutor(max_workers=workers) as ex:results=list(ex.map(run,todo))
        else:results=[run(t) for t in todo]
        with self.lock:
            for row,req in results:
                self.cache[row.sample_hash]=row;self.observations.append(row)
                if self.ledger:self.ledger.append('oracle_response',{**req,'observation':row.to_dict()})
        return [self.cache[digest(x)] for x in samples if digest(x) in self.cache],exhausted
    def close(self):
        self.closed=True
        if self.ledger:self.ledger.append('oracle_closed',{'calls':self.calls,'scope':self.scope.to_dict()})
