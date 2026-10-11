"""Authenticated, tenant-scoped v2 API and local operator console.

This is a single-host service, not a substitute for TLS, shared gateway rate
limits, managed secrets, encrypted storage, or deployment-specific security review.
"""
from __future__ import annotations
from contextlib import asynccontextmanager
from dataclasses import asdict,dataclass
import hashlib
import hmac
from importlib.resources import files as resources
import re
import threading
import time
import uuid
from typing import Any

from .artifact import SuiteArtifact
from .common import canonical,digest,integer,keys,nonempty,normalized_input,strict_json
from .config import TrainConfig,ReleasePolicy
from .dataset import Dataset,Record,Preference
from .guarded import AttemptLedger,GuardedBackend
from .runtime import SuiteRuntime
from .store import SuiteStore,Conflict,NotFound
from .training import plan_fit
from .worker import SuiteWorker,validate_holdout

import logging
LOG=logging.getLogger("model_refiner.suite")


@dataclass(frozen=True)
class Principal:
    tenant:str
    role:str
    name:str="operator"


@dataclass(frozen=True)
class ServiceLimits:
    max_body_bytes:int=4_000_000
    max_records:int=2000
    max_attempts:int=5000
    max_reserved_tokens:int=50_000_000
    max_output_tokens:int=4096
    max_concurrency:int=8
    requests_per_minute:int=120
    max_active_jobs:int=8
    queue_wait_ms:int=2000        # how long a prediction may wait for a free slot before it is shed with 429

    def __post_init__(self):
        for key,value in asdict(self).items():
            integer(value,key,1,10**12)


class BodyLimit:
    def __init__(self,app,max_bytes):self.app=app;self.max_bytes=max_bytes
    async def __call__(self,scope,receive,send):
        if scope["type"]!="http":return await self.app(scope,receive,send)
        async def reject(code,message):
            await send({"type":"http.response.start","status":code,"headers":[(b"content-type",b"application/json")]})
            await send({"type":"http.response.body","body":canonical({"detail":message}).encode()})
        headers=dict(scope.get("headers",[]));length=headers.get(b"content-length")
        if length:
            try:
                if int(length)<0:return await reject(400,"invalid content length")
                if int(length)>self.max_bytes:return await reject(413,"request too large")
            except ValueError:return await reject(400,"invalid content length")
        chunks=[];total=0
        while True:
            message=await receive()
            if message["type"]=="http.disconnect":return
            chunk=message.get("body",b"");total+=len(chunk)
            if total>self.max_bytes:return await reject(413,"request too large")
            chunks.append(chunk)
            if not message.get("more_body",False):break
        body=b"".join(chunks)
        if body and b"application/json" in headers.get(b"content-type",b""):
            try:strict_json(body.decode("utf-8"))
            except (ValueError,UnicodeError,RecursionError):return await reject(400,"invalid, duplicate-key, or non-finite JSON")
        consumed=False
        async def replay():
            nonlocal consumed
            if not consumed:
                consumed=True
                return {"type":"http.request","body":body,"more_body":False}
            return await receive()
        await self.app(scope,replay,send)


PERMISSIONS={"viewer":{"read"},"predictor":{"read","predict","feedback"},
             "trainer":{"read","predict","feedback","train"},"admin":{"read","predict","feedback","train","review","deploy"}}


def create_suite_app(*,store_path,backend_factory,provider,auth_keys:dict[str,Principal],allowed_models:set[str],
                     policy:ReleasePolicy|None=None,limits:ServiceLimits|None=None,start_worker=True,lease_seconds=120):
    from fastapi import FastAPI,Depends,Header,HTTPException,Body,Path
    from fastapi.responses import JSONResponse,Response,HTMLResponse
    if provider not in {"openai","openai-compatible","anthropic","toy"}:
        raise ValueError("Unsupported serving provider; shell command backends are forbidden")
    if not auth_keys or not allowed_models:raise ValueError("Authentication and a nonempty model allowlist are required")
    for secret,principal in auth_keys.items():
        if not isinstance(secret,str) or len(secret)<24:raise ValueError("API keys must have at least 24 characters")
        if principal.role not in PERMISSIONS:raise ValueError("Unknown role")
        nonempty(principal.tenant,"tenant",128);nonempty(principal.name,"principal name",128)
    key_hashes=[(hashlib.sha256(k.encode()).digest(),v) for k,v in auth_keys.items()]
    policy=policy or ReleasePolicy();limits=limits or ServiceLimits();store=SuiteStore(store_path)
    worker=SuiteWorker(store,backend_factory,provider,policy,lease_seconds=lease_seconds)
    semaphore=threading.BoundedSemaphore(limits.max_concurrency)
    rate={};rate_lock=threading.Lock();metrics_lock=threading.Lock()
    counts={"predictions":0,"errors":0,"shed":0,"inflight":0,"rate_limited":0}
    @asynccontextmanager
    async def lifespan(app):
        if start_worker:worker.start()
        try:yield
        finally:
            if start_worker:worker.stop()
    app=FastAPI(title="Model Refiner Suite",version="0.5.0rc1",lifespan=lifespan,docs_url=None,redoc_url=None)
    app.add_middleware(BodyLimit,max_bytes=limits.max_body_bytes)
    app.state.store=store;app.state.worker=worker
    def authenticate(authorization:str|None=Header(None)):
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401,"Bearer API key required")
        candidate=hashlib.sha256(authorization[7:].encode()).digest();principal=None
        for expected,value in key_hashes:
            if hmac.compare_digest(candidate,expected):principal=value
        if principal is None:raise HTTPException(401,"Invalid credentials")
        with rate_lock:
            minute=int(time.time()//60);key=(principal.tenant,principal.name)
            current=rate.get(key,(minute,0))
            n=current[1]+1 if current[0]==minute else 1
            rate[key]=(minute,n)
            if n>limits.requests_per_minute:
                with metrics_lock:counts["rate_limited"]+=1
                raise HTTPException(429,"Per-process rate limit exceeded",headers={"Retry-After":"60"})
        return principal
    def require(permission):
        def check(principal=Depends(authenticate)):
            if permission not in PERMISSIONS[principal.role]:raise HTTPException(403,"Insufficient role")
            return principal
        return check
    def safe_name(name):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}",name):raise ValueError("Name must use 1-80 letters, digits, '.', '_' or '-'")
        return name
    def config_policy(config):
        models=[config.model,config.optimizer_model,config.verifier_model,config.judge_model]
        if any(m and m not in allowed_models for m in models):raise HTTPException(403,"Model is not allowlisted")
        if config.max_attempts>limits.max_attempts or config.max_reserved_tokens>limits.max_reserved_tokens or config.max_output_tokens>limits.max_output_tokens or config.max_compile_tokens>limits.max_output_tokens:
            raise HTTPException(413,"Training/prediction budget exceeds operator limits")
        return config
    def artifact_policy(artifact):
        data=artifact.to_dict()
        if data["provider"]!=provider:raise HTTPException(409,"Artifact provider mismatch")
        for model in (data["model"],data["verifier_model"]):
            if model and model not in allowed_models:raise HTTPException(403,"Artifact model is not allowlisted")
        if data["runtime"]["max_output_tokens"]>limits.max_output_tokens:raise HTTPException(413,"Artifact output cap exceeds service limit")
    def normalize_fit(body):
        keys(body,{"config","records","preferences"},"fit")
        config=config_policy(TrainConfig.from_dict(body["config"]))
        if not isinstance(body["records"],list) or len(body["records"])>limits.max_records:raise ValueError("Too many/invalid records")
        rows=[Record.from_dict(x,i) for i,x in enumerate(body["records"])]
        if not isinstance(body.get("preferences",[]),list):raise ValueError("preferences must be a list")
        prefs=[Preference.from_dict(x) for x in body.get("preferences",[])]
        data=Dataset.prepare(rows,config.task,config.seed);data.audit_preferences(prefs)
        if prefs and not config.optimizer_model:raise ValueError("Preferences require optimizer_model")
        plan=plan_fit(data,config,prefs)
        if plan["provider_attempt_upper_bound"]>config.max_attempts:raise HTTPException(413,"Preflight provider-attempt ceiling exceeds requested budget")
        return {"config":config.to_dict(),"records":[asdict(x) for x in rows],"preferences":[asdict(x) for x in prefs]},plan
    @app.exception_handler(Conflict)
    async def conflict_handler(request,exc):return JSONResponse(status_code=409,content={"detail":str(exc)})
    @app.exception_handler(NotFound)
    async def missing_handler(request,exc):return JSONResponse(status_code=404,content={"detail":"Resource not found"})
    @app.exception_handler(OverflowError)
    async def overflow_handler(request,exc):
        # e.g. a version number beyond SQLite's 64-bit integers: a client error, not a server fault
        return JSONResponse(status_code=422,content={"detail":"Numeric value out of range"})
    @app.exception_handler(ValueError)
    async def validation_handler(request,exc):return JSONResponse(status_code=422,content={"detail":str(exc)[:500]})
    @app.exception_handler(KeyError)
    async def missing_field_handler(request,exc):return JSONResponse(status_code=422,content={"detail":"Missing required request field"})
    @app.get("/healthz")
    def health():return {"status":"ok"}
    @app.get("/readyz")
    def ready():
        with store.connect() as con:con.execute("SELECT 1").fetchone()
        return {"status":"ready","provider_checked":False}
    @app.get("/v2/info")
    def info(p=Depends(require("read"))):
        return {"provider":provider,"allowed_models":sorted(allowed_models),"tenant":p.tenant,"role":p.role,
                "limits":asdict(limits),"release_policy":asdict(policy)}
    @app.post("/v2/plan")
    def plan(body:dict=Body(...),p=Depends(require("train"))):return normalize_fit(body)[1]
    @app.get("/v2/refiners")
    def list_refiners(p=Depends(require("read"))):return {"versions":store.versions(p.tenant)}
    @app.get("/v2/refiners/{name}/versions")
    def versions(name:str,p=Depends(require("read"))):return {"versions":store.versions(p.tenant,safe_name(name))}
    @app.post("/v2/refiners/{name}/fits")
    def fit(name:str,body:dict=Body(...),idempotency_key:str=Header(...,alias="Idempotency-Key"),p=Depends(require("train"))):
        request,_=normalize_fit(body)
        return JSONResponse(status_code=202,content=store.submit(p.tenant,safe_name(name),"fit",request,idempotency_key,actor=p.name,max_active=limits.max_active_jobs))
    @app.get("/v2/jobs/{job_id}")
    def job(job_id:str,p=Depends(require("read"))):return store.job(p.tenant,job_id)
    @app.delete("/v2/jobs/{job_id}")
    def cancel(job_id:str,p=Depends(require("train"))):
        store.cancel(p.tenant,job_id,p.name);return {"status":"cancelled"}
    @app.post("/v2/refiners/{name}/import")
    def import_artifact(name:str,body:dict=Body(...),p=Depends(require("train"))):
        keys(body,{"artifact"},"import");artifact=SuiteArtifact(body["artifact"]);artifact_policy(artifact)
        version=store.add(p.tenant,safe_name(name),artifact,p.name)
        return {"version":version,"artifact_id":artifact.artifact_id,"status":"staged","trusted_evaluation":False}
    @app.get("/v2/refiners/{name}/versions/{version}/artifact")
    def export_artifact(name:str,version:int=Path(...,ge=1,le=2**31-1),p=Depends(require("train"))):return store.load(p.tenant,safe_name(name),version)[1].to_dict()
    @app.get("/v2/refiners/{name}/versions/{version}/report")
    def report(name:str,version:int=Path(...,ge=1,le=2**31-1),p=Depends(require("train"))):return store.latest_report(p.tenant,safe_name(name),version)[1]
    @app.post("/v2/refiners/{name}/versions/{version}/evaluate")
    def evaluate(name:str,version:int,body:dict=Body(...),idempotency_key:str=Header(...,alias="Idempotency-Key"),p=Depends(require("train"))):
        keys(body,{"records","max_attempts"},"evaluation")
        _,artifact=store.load(p.tenant,safe_name(name),version);artifact_policy(artifact)
        if not isinstance(body["records"],list) or len(body["records"])>limits.max_records:raise ValueError("Invalid evaluation dataset size")
        rows=[Record.from_dict(x,i) for i,x in enumerate(body["records"])];validate_holdout(artifact,rows)
        cfg=dict(artifact.to_dict()["provenance"]["config"])
        cfg["max_attempts"]=body.get("max_attempts",min(cfg["max_attempts"],limits.max_attempts))
        config=config_policy(TrainConfig.from_dict(cfg))
        request={"version":version,"config":config.to_dict(),"records":[asdict(x) for x in rows]}
        return JSONResponse(status_code=202,content=store.submit(p.tenant,name,"evaluate",request,idempotency_key,actor=p.name,max_active=limits.max_active_jobs))
    @app.post("/v2/refiners/{name}/promote")
    def promote(name:str,body:dict=Body(...),p=Depends(require("deploy"))):
        keys(body,{"version","expected_current"},"promotion")
        version=integer(body["version"],"version",1,10**9)
        expected=body.get("expected_current")
        if expected is not None:integer(expected,"expected_current",1,10**9)
        _,artifact=store.load(p.tenant,safe_name(name),version);artifact_policy(artifact)
        # Check current operator policy too: an older lenient gate is not enough.
        from .evaluation import release_gate
        _,report=store.latest_report(p.tenant,name,version)
        if not release_gate(report,policy)["passed"]:raise Conflict("Artifact fails current operator release policy")
        return store.promote(p.tenant,name,version,actor=p.name,expected_current=expected)

    def do_predict(name,text,version,p):
        safe_name(name);nonempty(text,"input")
        if version is not None:integer(version,"version",1,10**9)
        version,artifact=store.load(p.tenant,name,version);artifact_policy(artifact)
        # Absorb short bursts: wait briefly for a slot instead of failing instantly; shed only when still saturated.
        if not semaphore.acquire(timeout=limits.queue_wait_ms/1000):
            with metrics_lock:counts["shed"]+=1
            raise HTTPException(429,"Inference concurrency limit reached",headers={"Retry-After":"1"})
        with metrics_lock:counts["inflight"]+=1
        raw=None
        try:
            cfg=dict(artifact.to_dict()["provenance"]["config"])
            # Artifacts carry generation parameters; clients cannot change them.
            config=TrainConfig.from_dict(cfg)
            raw=backend_factory(config)
            request_id=str(uuid.uuid4())
            cap=(1+config.repair_attempts+bool(config.verifier_model))*(config.retries+1)
            ledger=AttemptLedger(store.path,"predict:"+request_id,min(limits.max_attempts,cap),limits.max_reserved_tokens)
            backend=GuardedBackend(raw,ledger,retries=config.retries,cache=False,max_prompt_chars=config.max_prompt_chars,max_output_tokens=config.max_output_tokens)
            runtime=SuiteRuntime(backend,artifact,provider=provider)
            result=runtime.predict(text)
            telemetry={key:value for key,value in result.to_dict().items() if key not in {"text","errors","retrieved_ids","request_ids"}}
            telemetry["output_hash"]=digest(normalized_input(result.text))
            pid=store.prediction(p.tenant,name,version,artifact.artifact_id,digest(normalized_input(text)),telemetry)
            with metrics_lock:counts["predictions"]+=1
            return {"prediction_id":pid,"name":name,"version":version,"artifact_id":artifact.artifact_id,
                    "status":result.status,"output":result.text if result.status=="accepted" else None,
                    "recipe_id":result.recipe_id,"route":result.route,"usage":{"input_tokens":result.input_tokens,"output_tokens":result.output_tokens},
                    "logical_calls":result.logical_calls,"budget":ledger.snapshot(),"latency_ms":result.latency_ms,
                    "provider_request_ids":result.request_ids}
        except (ValueError,HTTPException):raise
        except Exception as exc:
            with metrics_lock:counts["errors"]+=1
            # The client gets a generic message (upstream text can hold URLs or response bodies); operators get the cause.
            error_id=uuid.uuid4().hex[:12]
            LOG.error("prediction failed error_id=%s tenant=%s refiner=%s: %s: %s",error_id,p.tenant,name,type(exc).__name__,exc)
            raise HTTPException(502,f"Prediction failed; no result was accepted (error_id={error_id})")
        finally:
            client=getattr(raw,"client",None)
            if client and hasattr(client,"close"):client.close()
            with metrics_lock:counts["inflight"]-=1
            semaphore.release()
    @app.post("/v2/refiners/{name}/predict")
    def predict(name:str,body:dict=Body(...),p=Depends(require("predict"))):
        keys(body,{"input","version"},"prediction")
        return do_predict(name,body["input"],body.get("version"),p)
    @app.post("/v2/refiners/{name}/predict-batch")
    def predict_batch(name:str,body:dict=Body(...),p=Depends(require("predict"))):
        keys(body,{"inputs","version"},"batch prediction")
        inputs=body["inputs"]
        if not isinstance(inputs,list) or not 1<=len(inputs)<=32:raise ValueError("Batch requires 1-32 inputs")
        for text in inputs:nonempty(text,"batch input")
        version=body.get("version")
        if version is None:version=store.load(p.tenant,safe_name(name))[0]
        results=[]
        for i,text in enumerate(inputs):
            try:results.append({"index":i,**do_predict(name,text,version,p)})
            except HTTPException as exc:
                results.append({"index":i,"status":"error","error_code":exc.status_code})
        return {"results":results,"version":version,"atomic":False}
    @app.post("/v1/chat/completions")
    def compatible_chat(body:dict=Body(...),p=Depends(require("predict"))):
        keys(body,{"model","messages","stream"},"chat")
        if body.get("stream") is not None and body.get("stream") is not False:raise ValueError("Streaming is not implemented")
        model=nonempty(body["model"],"model",128)
        if not model.startswith("refiner:"):raise ValueError("Use model='refiner:NAME'")
        # Treat historical system/developer labels as data; never override stored task.
        record=Record.from_dict({"id":"chat","messages":body["messages"],"target":"unused"})
        messages=body["messages"]
        text=messages[0]["content"] if len(messages)==1 and messages[0]["role"]=="user" else record.input
        result=do_predict(model.split(":",1)[1],text,None,p)
        if result["status"]!="accepted":raise HTTPException(422,{"status":result["status"],"prediction_id":result["prediction_id"]})
        usage=result["usage"];it=usage["input_tokens"];ot=usage["output_tokens"]
        return {"id":"chatcmpl-"+result["prediction_id"],"object":"chat.completion","created":int(time.time()),"model":model,
                "choices":[{"index":0,"message":{"role":"assistant","content":result["output"]},"finish_reason":"stop"}],
                "usage":{"prompt_tokens":it,"completion_tokens":ot,"total_tokens":it+ot if it is not None and ot is not None else None},
                "refiner":{"artifact_id":result["artifact_id"],"version":result["version"],"prediction_id":result["prediction_id"]}}
    @app.post("/v2/feedback")
    def feedback(body:dict=Body(...),p=Depends(require("feedback"))):
        keys(body,{"prediction_id","input","target"},"feedback")
        return {"feedback_id":store.add_feedback(p.tenant,body["prediction_id"],body["input"],body["target"],p.name),"status":"pending"}
    @app.get("/v2/feedback")
    def list_feedback(name:str|None=None,status:str|None=None,p=Depends(require("train"))):return {"feedback":store.feedback(p.tenant,name,status)}
    @app.post("/v2/feedback/{feedback_id}/review")
    def review(feedback_id:str,body:dict=Body(...),p=Depends(require("review"))):
        keys(body,{"approve"},"review")
        if type(body["approve"]) is not bool:raise ValueError("approve must be boolean")
        store.review_feedback(p.tenant,feedback_id,body["approve"],p.name)
        return {"status":"approved" if body["approve"] else "rejected"}
    @app.get("/v2/audit")
    def audit(p=Depends(require("read"))):return {"events":store.events(p.tenant)}
    @app.get("/v2/refiners/{name}/monitor")
    def monitor(name:str,p=Depends(require("read"))):
        with store.connect() as con:
            rows=con.execute("SELECT telemetry FROM suite_predictions WHERE tenant=? AND name=? ORDER BY created_at DESC LIMIT 1000",(p.tenant,safe_name(name))).fetchall()
        entries=[strict_json(x[0]) for x in rows];n=len(entries)
        return {"sample_size":n,"accepted_rate":sum(x["status"]=="accepted" for x in entries)/n if n else None,
                "low_support_rate":sum(bool(x.get("route",{}).get("fallback")) for x in entries if x.get("route"))/sum(bool(x.get("route")) for x in entries) if any(x.get("route") for x in entries) else None,
                "note":"Recent operational proxies; not a measured accuracy estimate or a formal drift test."}
    @app.get("/metrics")
    def metrics(p=Depends(require("read"))):
        # Do not expose other tenants' activity via shared global counters.
        with store.connect() as con:
            predictions=con.execute("SELECT COUNT(*) FROM suite_predictions WHERE tenant=?",(p.tenant,)).fetchone()[0]
            jobs=con.execute("SELECT status,COUNT(*) FROM suite_jobs WHERE tenant=? GROUP BY status",(p.tenant,)).fetchall()
        with metrics_lock:snap=dict(counts)
        lines=[f"model_refiner_suite_predictions_total {predictions}",
               f"model_refiner_suite_prediction_errors_total {snap['errors']}",
               f"model_refiner_suite_shed_total {snap['shed']}",
               f"model_refiner_suite_rate_limited_total {snap['rate_limited']}",
               f"model_refiner_suite_inflight {snap['inflight']}"]
        lines.extend(f'model_refiner_suite_jobs{{status="{x[0]}"}} {x[1]}' for x in jobs)
        return Response("\n".join(lines)+"\n",media_type="text/plain")
    @app.get("/console",response_class=HTMLResponse)
    def console():
        return HTMLResponse(resources("model_refiner.suite").joinpath("static/index.html").read_text(encoding="utf-8"),headers={
            "Content-Security-Policy":"default-src 'self'; script-src 'self'; style-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'",
            "Cache-Control":"no-store","X-Content-Type-Options":"nosniff"})
    @app.get("/console/{asset}")
    def asset(asset:str):
        if asset not in {"app.js","style.css"}:raise HTTPException(404)
        media="text/javascript" if asset.endswith("js") else "text/css"
        return Response(resources("model_refiner.suite").joinpath("static/"+asset).read_text(encoding="utf-8"),media_type=media,headers={"Cache-Control":"no-store","X-Content-Type-Options":"nosniff"})
    return app
