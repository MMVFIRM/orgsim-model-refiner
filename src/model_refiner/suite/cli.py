"""CLI for the black-box frozen-model suite."""
from __future__ import annotations
import argparse
from dataclasses import asdict,replace
import json
import os
from pathlib import Path
import sys
import uuid
from urllib.parse import urlparse

from ..backend import make_backend
from .artifact import SuiteArtifact
from .common import atomic_json,canonical,digest,strict_json,normalized_input
from .config import TrainConfig,ReleasePolicy
from .dataset import Dataset,Record,Preference,load_records,load_jsonl
from .evaluation import evaluate_artifact,release_gate
from .guarded import AttemptLedger,GuardedBackend
from .runtime import SuiteRuntime
from .store import SuiteStore
from .training import train,plan_fit
from .toy import SuiteToyBackend
from .worker import validate_holdout


def read_json(path):
    p=Path(path)
    if p.stat().st_size>20_000_000:raise ValueError("JSON file exceeds size limit")
    return strict_json(p.read_text(encoding="utf-8"))


def raw_backend(provider,config,base_url=None,key_env=None):
    if provider=="toy":return SuiteToyBackend()
    if provider not in {"openai","openai-compatible","anthropic"}:raise ValueError("Unsupported suite provider")
    if provider=="openai":
        if base_url:raise ValueError("Use openai-compatible for custom URLs; native OpenAI is provider-bound")
        key=os.environ.get(key_env or "OPENAI_API_KEY", "")
        if not key:raise ValueError("OpenAI API key environment variable is missing")
        from ..backend import OpenAIResponsesBackend
        return OpenAIResponsesBackend(api_key=key,max_output_tokens=config.max_output_tokens,timeout=config.timeout_seconds)
    if base_url:
        parsed=urlparse(base_url)
        if parsed.username or parsed.password or parsed.query or parsed.fragment:raise ValueError("Base URLs cannot embed credentials/query/fragment")
        local=parsed.hostname in {"127.0.0.1","localhost","::1"}
        if parsed.scheme!="https" and not (parsed.scheme=="http" and local):raise ValueError("Use HTTPS except on literal loopback hosts")
    return make_backend(provider,base_url=base_url,key_env=key_env,max_tokens=config.max_output_tokens,timeout=config.timeout_seconds)


def guarded(a,config,workspace,run_id,cache=True):
    raw=raw_backend(a.provider,config,getattr(a,"base_url",None),getattr(a,"key_env",None))
    ledger=AttemptLedger(Path(workspace)/"ledger.sqlite3",run_id,config.max_attempts,config.max_reserved_tokens)
    backend=GuardedBackend(raw,ledger,retries=config.retries,max_prompt_chars=config.max_prompt_chars,
                           max_output_tokens=config.max_output_tokens,cache=cache)
    return backend,ledger


def _train(a):
    config=TrainConfig.from_dict(read_json(a.config));records=load_records(a.data)
    preferences=[Preference.from_dict(x) for x in load_jsonl(a.preferences)] if a.preferences else []
    data=Dataset.prepare(records,config.task,config.seed);data.audit_preferences(preferences)
    plan=plan_fit(data,config,preferences)
    if a.dry_run:
        print(canonical({"plan":plan,"manifest":data.manifest()}))
        ceiling=plan["provider_attempt_upper_bound"]
        if ceiling>config.max_attempts:
            print(f"dry-run FAILED: worst-case provider attempts {ceiling} exceed max_attempts={config.max_attempts}; "
                  f"set max_attempts to at least {ceiling} (a cap, not an expected spend)",file=sys.stderr)
            return 2
        return 0
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    identity=digest({"config":config.to_dict(),"provider":a.provider,"records":[asdict(x) for x in records],
                     "preferences":[asdict(x) for x in preferences],"base_url":a.base_url})
    manifest_path=out/"run.json"
    if manifest_path.exists():
        if not a.resume:raise ValueError("Output workspace already contains a run; use --resume or a new directory")
        manifest=read_json(manifest_path)
        if manifest["identity"]!=identity:raise ValueError("Resume config/data/provider differs from original run")
        run_id=manifest["run_id"]
    else:
        run_id=str(uuid.uuid4());atomic_json(manifest_path,{"run_id":run_id,"identity":identity,"plan":plan},overwrite=False)
    policy=ReleasePolicy.from_dict(read_json(a.policy)) if a.policy else ReleasePolicy()
    backend,ledger=guarded(a,config,out,run_id)
    with backend:
        result=train(backend,a.provider,records,config,preferences=preferences,policy=policy,run_id=run_id,
                     progress=lambda event:print(canonical(event),file=sys.stderr,flush=True))
    result.report["budget"]=ledger.snapshot()
    result.artifact.save(out/"artifact.json");atomic_json(out/"report.json",result.report)
    atomic_json(out/"manifest.json",result.report["dataset_manifest"])
    (out/"model-card.md").write_text(model_card(result.artifact,result.report),encoding="utf-8")
    print(json.dumps({"artifact":str(out/"artifact.json"),"artifact_id":result.artifact.artifact_id,
                      "scores":{k:v["score"] for k,v in result.report["arms"].items()},"gate":result.report["gate"],
                      "budget":ledger.snapshot()},indent=2))
    return 0


def model_card(artifact,report):
    data=artifact.to_dict()
    scores="\n".join(f"| {name} | {arm['score']:.4f} | {arm['coverage']:.4f} |" for name,arm in report["arms"].items())
    return (f"# Model Refiner model card\n\nArtifact: `{artifact.artifact_id}`\n\nProvider / model: {data['provider']} / {data['model']}\n\n"
            f"Metric: {report['task_metric']}\n\nProvider weights modified: **No**\n\nSelected path: {report['selected_mode']}\n\n"
            "| Arm | Score | Accepted coverage |\n|---|---:|---:|\n"+scores+
            f"\n\nQuality gate: **{'PASS' if report['gate']['passed'] else 'FAIL'}**\n\nFailed checks: {', '.join(report['gate']['failed']) or 'none'}.\n\n"
            "## Limitations\n\nA toy-provider result verifies machinery, not LLM improvement. External weights are recipe-selection weights, "
            "not changes to the provider's neural network. Lexical scores do not establish factuality or semantic quality. "
            "A quality gate does not certify security, live-provider compatibility, throughput, or independent replicated efficacy. "
            "Artifacts, reports and local caches may contain sensitive training data.\n")


def _predict(a):
    artifact=SuiteArtifact.load(a.artifact);data=artifact.to_dict();a.provider=a.provider or data["provider"]
    config=TrainConfig.from_dict(data["provenance"]["config"])
    backend,ledger=guarded(a,config,a.workspace,"predict:"+str(uuid.uuid4()),cache=False)
    with backend:
        result=SuiteRuntime(backend,artifact,provider=a.provider).predict(a.input)
    print(json.dumps({"artifact_id":artifact.artifact_id,**result.to_dict(),"budget":ledger.snapshot()},indent=2))
    return 0 if result.status=="accepted" else 2


def _evaluate(a):
    artifact=SuiteArtifact.load(a.artifact);data=artifact.to_dict();a.provider=a.provider or data["provider"]
    rows=load_records(a.data);validate_holdout(artifact,rows)
    config=TrainConfig.from_dict(data["provenance"]["config"])
    if a.max_attempts:
        cfg=config.to_dict();cfg["max_attempts"]=a.max_attempts;config=TrainConfig.from_dict(cfg)
    backend,ledger=guarded(a,config,a.workspace,"eval:"+str(uuid.uuid4()))
    with backend:
        report=evaluate_artifact(SuiteRuntime(backend,artifact,provider=a.provider),rows,
                                 policy=ReleasePolicy.from_dict(read_json(a.policy)) if a.policy else ReleasePolicy())
    report["budget"]=ledger.snapshot();atomic_json(a.out,report)
    print(json.dumps(report["gate"],indent=2));return 0


def _serve(a):
    from .api import Principal,ServiceLimits,create_suite_app
    import uvicorn
    keys={}
    if a.auth_file:
        for item in read_json(a.auth_file):
            token=os.environ.get(item["token_env"],"")
            if not token:raise ValueError(f"Required token environment variable is missing: {item['token_env']}")
            if token in keys:raise ValueError("Duplicate API token in auth config")
            keys[token]=Principal(item["tenant"],item["role"],item.get("name",item["token_env"]))
    else:
        token=os.environ.get("MODEL_REFINER_SERVICE_KEY","")
        if not token:raise ValueError("Set MODEL_REFINER_SERVICE_KEY or --auth-file")
        keys[token]=Principal(a.tenant,"admin")
    models={x.strip() for x in a.allowed_models.split(",") if x.strip()}
    policy=ReleasePolicy.from_dict(read_json(a.policy)) if a.policy else ReleasePolicy()
    limits=ServiceLimits(max_attempts=a.max_attempts,max_concurrency=a.max_concurrency,requests_per_minute=a.requests_per_minute,queue_wait_ms=a.queue_wait_ms)
    app=create_suite_app(store_path=a.store,backend_factory=lambda c:raw_backend(a.provider,c,a.base_url,a.key_env),
                         provider=a.provider,auth_keys=keys,allowed_models=models,policy=policy,limits=limits,start_worker=not a.no_worker,lease_seconds=a.lease_seconds)
    uvicorn.run(app,host=a.host,port=a.port,log_level="info",access_log=False)
    return 0


def _split(a):
    config=TrainConfig.from_dict(read_json(a.config))
    data=Dataset.prepare(load_records(a.data),config.task,config.seed)
    path=Path(a.out)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8",newline="\n") as stream:
        for split in ("train","router","validation","test"):
            for row in getattr(data,split):
                stream.write(canonical(asdict(replace(row,split=split)))+"\n")
    atomic_json(str(path)+".manifest.json",data.manifest())
    print(canonical({"output":str(path),"manifest":data.manifest()}))
    return 0


def _revision(a):
    rows=load_records(a.data)
    if not all(x.split for x in rows):raise ValueError("Revision requires explicit splits; materialize the original split manifest first")
    if a.store:
        approved=SuiteStore(a.store).feedback(a.tenant,a.name,"approved")
    else:
        approved=load_jsonl(a.feedback)
    seen={normalized_input(x.input) for x in rows};ids={x.id for x in rows};new=[]
    for row in approved:
        if row.get("status")!="approved":raise ValueError("Only explicitly approved feedback can enter a revision")
        ex=Record.from_dict({"id":"feedback-"+row["id"],"input":row["input"],"target":row["target"],"split":"train",
                             "metadata":{"source":"reviewed_feedback","feedback_id":row["id"]}})
        if ex.id in ids or normalized_input(ex.input) in seen:raise ValueError("Feedback overlaps the original dataset; do not contaminate frozen holdouts")
        seen.add(normalized_input(ex.input));ids.add(ex.id);new.append(ex)
    path=Path(a.out)
    if path.exists():raise FileExistsError("Revision destination already exists")
    with path.open("x",encoding="utf-8") as f:
        for row in rows+new:f.write(canonical(asdict(row))+"\n")
    print(canonical({"output":str(path),"new_train_rows":len(new),"test_and_validation_unchanged":True}));return 0


def main(argv=None):
    p=argparse.ArgumentParser(prog="model-refiner suite",description="Adapt frozen black-box models without touching provider weights.")
    sub=p.add_subparsers(dest="command",required=True)
    def provider(parser,default="openai"):
        parser.add_argument("--provider",choices=("openai","openai-compatible","anthropic","toy"),default=default)
        parser.add_argument("--base-url");parser.add_argument("--key-env")
    t=sub.add_parser("train");t.add_argument("--config",required=True);t.add_argument("--data",required=True)
    t.add_argument("--preferences");t.add_argument("--policy");t.add_argument("--out",default="./runs/refiner")
    t.add_argument("--dry-run",action="store_true");t.add_argument("--resume",action="store_true");provider(t);t.set_defaults(fn=_train)
    t=sub.add_parser("predict");t.add_argument("--artifact",required=True);t.add_argument("--input",required=True)
    t.add_argument("--workspace",default="./runs/predict");provider(t,None);t.set_defaults(fn=_predict)
    t=sub.add_parser("evaluate");t.add_argument("--artifact",required=True);t.add_argument("--data",required=True)
    t.add_argument("--out",default="evaluation.json");t.add_argument("--workspace",default="./runs/evaluate")
    t.add_argument("--policy");t.add_argument("--max-attempts",type=int);provider(t,None);t.set_defaults(fn=_evaluate)
    t=sub.add_parser("serve");t.add_argument("--store",default="./data/suite.sqlite3");t.add_argument("--host",default="127.0.0.1")
    t.add_argument("--port",type=int,default=8080);t.add_argument("--auth-file");t.add_argument("--tenant",default="default")
    t.add_argument("--allowed-models",default=os.environ.get("MODEL_REFINER_ALLOWED_MODELS",""));t.add_argument("--policy")
    t.add_argument("--queue-wait-ms",type=int,default=2000,help="max time a prediction waits for a free concurrency slot before a 429");t.add_argument("--lease-seconds",type=float,default=120.0,help="fit/evaluate job lease; a crashed worker's job is reclaimed after it expires");t.add_argument("--no-worker",action="store_true");t.add_argument("--max-attempts",type=int,default=5000)
    t.add_argument("--max-concurrency",type=int,default=8);t.add_argument("--requests-per-minute",type=int,default=120)
    provider(t);t.set_defaults(fn=_serve)
    t=sub.add_parser("backup");t.add_argument("--store",required=True);t.add_argument("--out",required=True)
    t.set_defaults(fn=lambda a:print(SuiteStore(a.store).backup(a.out)))
    t=sub.add_parser("integrity");t.add_argument("--store",required=True)
    t.set_defaults(fn=lambda a:print(SuiteStore(a.store).integrity()))
    t=sub.add_parser("gate");t.add_argument("--report",required=True);t.add_argument("--policy")
    def gate(a):
        result=release_gate(read_json(a.report),ReleasePolicy.from_dict(read_json(a.policy)) if a.policy else ReleasePolicy())
        print(json.dumps(result,indent=2));return 0 if result["passed"] else 2
    t.set_defaults(fn=gate)
    t=sub.add_parser("split");t.add_argument("--config",required=True);t.add_argument("--data",required=True)
    t.add_argument("--out",required=True);t.set_defaults(fn=_split)
    t=sub.add_parser("prepare-revision");t.add_argument("--data",required=True);t.add_argument("--out",required=True)
    group=t.add_mutually_exclusive_group(required=True);group.add_argument("--store");group.add_argument("--feedback")
    t.add_argument("--tenant",default="default");t.add_argument("--name",default="support");t.set_defaults(fn=_revision)
    a=p.parse_args(argv)
    try:return a.fn(a) or 0
    except (ValueError,KeyError,FileExistsError,FileNotFoundError) as exc:
        p.error(str(exc))
        return 2


if __name__=="__main__":raise SystemExit(main())
