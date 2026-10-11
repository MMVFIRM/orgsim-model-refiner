"""Retry/circuit-breaker wrapper with durable per-attempt reservations.

A reservation is recorded BEFORE every upstream attempt. Interrupted attempts
remain charged to the local budget. This bounds attempts, not provider dollars.
Results cached within a run allow resumable fits; cache is never global across
customers or independent experiments.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import asdict
import random
import sqlite3
import threading
import time
from pathlib import Path

from ..backend import BackendError, ModelBackend, _retryable_exception
from ..types import ModelResult
from .common import canonical, digest, integer, nonempty


class BudgetExceeded(RuntimeError): pass
class CircuitOpen(RuntimeError): pass
class Cancelled(RuntimeError): pass


class AttemptLedger:
    def __init__(self,path: str | Path,run_id: str,max_attempts: int,max_reserved_tokens: int):
        self.path=str(path); self.run_id=nonempty(run_id,"run_id",256)
        self.max_attempts=integer(max_attempts,"max_attempts",1,1_000_000)
        self.max_reserved_tokens=integer(max_reserved_tokens,"max_reserved_tokens",1000,10**12)
        Path(self.path).parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("CREATE TABLE IF NOT EXISTS suite_budgets(run_id TEXT PRIMARY KEY,attempts INTEGER NOT NULL DEFAULT 0,reserved_tokens INTEGER NOT NULL DEFAULT 0, max_attempts INTEGER NOT NULL,max_tokens INTEGER NOT NULL)")
            con.execute("CREATE TABLE IF NOT EXISTS suite_cache(run_id TEXT NOT NULL,key TEXT NOT NULL,value TEXT NOT NULL,PRIMARY KEY(run_id,key))")
            con.execute("INSERT OR IGNORE INTO suite_budgets(run_id,max_attempts,max_tokens) VALUES(?,?,?)",(self.run_id,max_attempts,max_reserved_tokens))
            row=con.execute("SELECT max_attempts,max_tokens FROM suite_budgets WHERE run_id=?",(run_id,)).fetchone()
            if tuple(row)!=(max_attempts,max_reserved_tokens):
                raise ValueError("Existing run budget cannot be changed")

    @contextmanager
    def connect(self):
        con=sqlite3.connect(self.path,timeout=30)
        con.execute("PRAGMA busy_timeout=30000")
        try:
            yield con
            con.commit()
        except BaseException:
            con.rollback()
            raise
        finally:
            con.close()

    def reserve(self,tokens:int):
        integer(tokens,"reserved tokens",1,10**9)
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            cursor=con.execute("UPDATE suite_budgets SET attempts=attempts+1,reserved_tokens=reserved_tokens+? WHERE run_id=? AND attempts<max_attempts AND reserved_tokens+?<=max_tokens",(tokens,self.run_id,tokens))
            if cursor.rowcount!=1:
                raise BudgetExceeded("Run attempt/token-reservation budget exhausted")

    def snapshot(self):
        with self.connect() as con:
            row=con.execute("SELECT attempts,reserved_tokens,max_attempts,max_tokens FROM suite_budgets WHERE run_id=?",(self.run_id,)).fetchone()
        return dict(zip(("upstream_attempts","reserved_tokens","max_attempts","max_reserved_tokens"),row))

    def get(self,key):
        with self.connect() as con:
            row=con.execute("SELECT value FROM suite_cache WHERE run_id=? AND key=?",(self.run_id,key)).fetchone()
        if row:
            from .common import strict_json
            data=strict_json(row[0]);data["metadata"]={**data.get("metadata",{}),"cache_hit":True}
            return ModelResult(**data)
        return None

    def put(self,key,result):
        with self.connect() as con:
            con.execute("INSERT OR IGNORE INTO suite_cache(run_id,key,value) VALUES(?,?,?)",(self.run_id,key,canonical(asdict(result))))


class GuardedBackend(ModelBackend):
    def __init__(self,inner:ModelBackend,ledger:AttemptLedger,*,retries=2,max_prompt_chars=100_000,
                 max_output_tokens=512,cache=True,cancel_check=None,sleep=time.sleep,
                 circuit_threshold=5,circuit_cooldown=30.0):
        self.inner=inner;self.ledger=ledger;self.retries=retries
        self.max_prompt_chars=max_prompt_chars;self.max_output_tokens=max_output_tokens
        self.cache=cache;self.cancel_check=cancel_check;self.sleep=sleep
        self.circuit_threshold=circuit_threshold;self.circuit_cooldown=circuit_cooldown
        self.failures=0;self.open_until=0.0
        self.lock=threading.Lock();self.hits=0

    def close(self):
        closer=getattr(self.inner,"close",None)
        if callable(closer):closer()
        else:
            client=getattr(self.inner,"client",None)
            if client is not None and callable(getattr(client,"close",None)):client.close()

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.close()
        return False

    def cache_identity(self):
        return self.inner.cache_identity()

    def with_output_cap(self,tokens):
        """A view sharing this backend's ledger, cache and circuit breaker but capping output at ``tokens``."""
        return _CapView(self,self.inner.with_output_cap(tokens),int(tokens))

    def generate(self,*,model,messages):
        return self._run(self.inner,self.max_output_tokens,model,messages)

    def _run(self,inner,max_output_tokens,model,messages):
        if self.cancel_check and self.cancel_check():
            raise Cancelled("Job cancelled or lease lost")
        if sum(len(x["content"]) for x in messages)>self.max_prompt_chars:
            raise ValueError("Rendered prompt exceeds limit")
        key=digest({"backend":inner.cache_identity(),"model":model,"messages":messages})
        if self.cache:
            hit=self.ledger.get(key)
            if hit is not None:
                self.hits+=1
                return hit
        # UTF-8 bytes plus framing and configured output cap is a conservative
        # accounting reservation, not a provider-tokenizer guarantee.
        tokens=len(canonical(messages).encode("utf-8"))+128+64*len(messages)+max_output_tokens
        for attempt in range(self.retries+1):
            if self.cancel_check and self.cancel_check():
                raise Cancelled("Job cancelled or lease lost")
            with self.lock:
                if time.monotonic()<self.open_until:
                    raise CircuitOpen("Provider circuit breaker is open")
            self.ledger.reserve(tokens)
            try:
                result=inner.generate(model=model,messages=messages)
                if not isinstance(result,ModelResult) or not isinstance(result.text,str):
                    raise BackendError("Malformed provider result",retryable=False)
                if len(result.text)>1_000_000:
                    raise BackendError("Provider output exceeds hard response limit",retryable=False)
                with self.lock:
                    self.failures=0;self.open_until=0
                if self.cache:
                    self.ledger.put(key,result)
                return result
            except Exception as exc:
                if not _retryable_exception(exc):
                    raise
                with self.lock:
                    self.failures+=1
                    if self.failures>=self.circuit_threshold:
                        self.open_until=time.monotonic()+self.circuit_cooldown
                if attempt==self.retries:
                    raise
                self.sleep(min(8.0,0.5*2**attempt)*(0.75+random.random()*0.5))
        raise AssertionError("unreachable")


class _CapView(ModelBackend):
    def __init__(self,parent,inner,tokens):
        self.parent=parent;self.inner=inner;self.tokens=tokens

    def cache_identity(self):
        return self.inner.cache_identity()

    def with_output_cap(self,tokens):
        return self.parent.with_output_cap(tokens)

    def generate(self,*,model,messages):
        return self.parent._run(self.inner,self.tokens,model,messages)
