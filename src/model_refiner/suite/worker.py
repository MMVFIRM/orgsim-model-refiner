"""Fenced, resumable jobs with per-job durable upstream budgets."""
from __future__ import annotations
from dataclasses import asdict
import logging
import threading

from .common import digest,normalized_input
from .config import TrainConfig,ReleasePolicy
from .dataset import Record,Preference
from .evaluation import evaluate_artifact
from .guarded import AttemptLedger,GuardedBackend
from .runtime import SuiteRuntime
from .training import train

LOG=logging.getLogger(__name__)


def validate_holdout(artifact,records):
    if not records:raise ValueError("Empty holdout")
    seen=set();ids=set()
    data=artifact.to_dict()
    provenance=data["provenance"].get("dataset",{}).get("splits",{})
    protected={x for split in ("train","router","validation") for x in provenance.get(split,{}).get("input_hashes",[])}
    protected_groups={x for split in ("train","router","validation") for x in provenance.get(split,{}).get("group_hashes",[])}
    protected_groups.update(digest(row.get("group") or row["id"]) for row in data["memory"])
    for row in records:
        h=digest(normalized_input(row.input))
        if row.id in ids or h in seen:raise ValueError("Duplicate holdout record")
        if h in protected or digest(row.group or row.id) in protected_groups:raise ValueError("Holdout overlaps development data")
        if row.split not in {None,"test"}:raise ValueError("Evaluation records must be test-only")
        seen.add(h);ids.add(row.id)


class SuiteWorker:
    def __init__(self,store,backend_factory,provider,policy:ReleasePolicy,*,lease_seconds=120,poll_seconds=.2):
        self.store=store;self.backend_factory=backend_factory;self.provider=provider;self.policy=policy
        self.lease_seconds=lease_seconds;self.poll_seconds=poll_seconds
        self.stop_event=threading.Event();self.thread=None

    def start(self):
        if self.thread and self.thread.is_alive():return
        self.stop_event.clear()
        self.thread=threading.Thread(target=self.loop,daemon=True,name="refiner-suite-worker")
        self.thread.start()

    def stop(self,timeout=5):
        self.stop_event.set()
        if self.thread:self.thread.join(timeout)

    def loop(self):
        while not self.stop_event.is_set():
            try:
                job=self.store.claim(self.lease_seconds)
                if job:self.execute(job)
                else:self.stop_event.wait(self.poll_seconds)
            except Exception as exc:
                LOG.error("suite_worker_loop_error class=%s",type(exc).__name__)
                self.stop_event.wait(self.poll_seconds)

    def execute(self,job):
        stop_heartbeat=threading.Event();lost=threading.Event()
        def heartbeat():
            while not stop_heartbeat.wait(max(.05,self.lease_seconds/3)):
                try:
                    if not self.store.heartbeat(job["id"],job["owner"],self.lease_seconds):
                        lost.set();return
                except Exception:
                    lost.set();return
        thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
        def cancelled():
            return self.stop_event.is_set() or lost.is_set() or not self.store.alive(job["id"],job["owner"])
        def progress(event):
            if not self.store.heartbeat(job["id"],job["owner"],self.lease_seconds,event):lost.set()
        raw=None
        try:
            request=job["request"]
            config=TrainConfig.from_dict(request["config"])
            raw=self.backend_factory(config)
            ledger=AttemptLedger(self.store.path,"job:"+job["id"],config.max_attempts,config.max_reserved_tokens)
            backend=GuardedBackend(raw,ledger,retries=config.retries,max_prompt_chars=config.max_prompt_chars,
                                   max_output_tokens=config.max_output_tokens,cancel_check=cancelled)
            rows=[Record.from_dict(x,i) for i,x in enumerate(request["records"])]
            if job["kind"]=="fit":
                preferences=[Preference.from_dict(x) for x in request.get("preferences",[])]
                result=train(backend,self.provider,rows,config,preferences=preferences,policy=self.policy,progress=progress,run_id=job["id"])
                artifact,report=result.artifact,result.report
            else:
                _,artifact=self.store.load(job["tenant"],job["name"],request["version"])
                validate_holdout(artifact,rows)
                runtime=SuiteRuntime(backend,artifact,provider=self.provider)
                report=evaluate_artifact(runtime,rows,policy=self.policy,progress=progress)
                artifact=None
            report["budget"]=ledger.snapshot()
            self.store.finish(job,artifact,report)
        except Exception as exc:
            # API/SDK error messages can contain credentials or customer data.
            # Only a class-level code reaches job status and default logs.
            LOG.error("suite_job_failed job_id=%s class=%s",job["id"],type(exc).__name__)
            self.store.fail(job,type(exc).__name__)
        finally:
            stop_heartbeat.set();thread.join(1)
            client=getattr(raw,"client",None)
            if client and hasattr(client,"close"):client.close()
