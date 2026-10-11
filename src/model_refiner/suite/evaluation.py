"""Held-out evaluation, paired cluster intervals, and fail-closed release decisions."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import asdict
import math
import random
import statistics

from .common import digest, normalized_input
from .config import ReleasePolicy
from .dataset import Record
from .runtime import SuiteRuntime
from .tasks import TaskSpec
from .judging import Scorer


def summarize(rows:list[dict],task:TaskSpec) -> dict:
    if not rows: raise ValueError("Cannot evaluate an empty arm")
    accepted=sum(x["status"]=="accepted" for x in rows)
    scores=[x["score"] for x in rows]
    slices={}
    for name in sorted({x["slice"] for x in rows}):
        subset=[x for x in rows if x["slice"]==name]
        slices[name]={"count":len(subset),"score":statistics.mean(x["score"] for x in subset),
                      "coverage":sum(x["status"]=="accepted" for x in subset)/len(subset)}
    def tokens(key):
        values=[x[key] for x in rows]
        return sum(values) if all(type(x) is int for x in values) else None
    result={"count":len(rows),"score":statistics.mean(scores),"coverage":accepted/len(rows),
            "accepted_only_score":sum(scores)/accepted if accepted else None,
            "invalid_rate":sum(x["status"]=="invalid" for x in rows)/len(rows),
            "input_tokens":tokens("input_tokens"),"output_tokens":tokens("output_tokens"),
            "logical_calls":sum(x["logical_calls"] for x in rows),"cache_hits":sum(x["cache_hits"] for x in rows),
            "p50_latency_ms":statistics.median(x["latency_ms"] for x in rows),
            "p95_latency_ms":sorted(x["latency_ms"] for x in rows)[math.ceil(.95*len(rows))-1],
            "slices":slices}
    if task.kind=="classification":
        confusion={label:{} for label in task.labels}
        for row in rows:
            pred=row["text"].strip() if row["status"]=="accepted" else "<"+row["status"]+">"
            confusion.setdefault(row["target"],{})[pred]=confusion.get(row["target"],{}).get(pred,0)+1
        metrics={}
        for label in task.labels:
            tp=sum(x["target"]==label and x["text"].strip()==label and x["status"]=="accepted" for x in rows)
            fp=sum(x["target"]!=label and x["text"].strip()==label and x["status"]=="accepted" for x in rows)
            fn=sum(x["target"]==label for x in rows)-tp
            p=tp/(tp+fp) if tp+fp else 0.0;r=tp/(tp+fn) if tp+fn else 0.0
            metrics[label]={"precision":p,"recall":r,"f1":2*p*r/(p+r) if p+r else 0.0,"support":tp+fn}
        result.update(confusion_matrix=confusion,per_label=metrics,
                      macro_f1=statistics.mean(x["f1"] for x in metrics.values()))
    return result


def paired_interval(left:list[dict],right:list[dict],*,samples:int=2000,seed:int=42) -> dict:
    if [x["id"] for x in left]!=[x["id"] for x in right]:
        raise ValueError("Paired arms have mismatched IDs/order")
    groups=defaultdict(list)
    for a,b in zip(left,right):
        if a["group"]!=b["group"]: raise ValueError("Mismatched paired groups")
        groups[a["group"]].append(b["score"]-a["score"])
    clusters=list(groups.values());rng=random.Random(seed)
    bootstrap=[]
    for _ in range(samples):
        sample=[clusters[rng.randrange(len(clusters))] for _ in clusters]
        bootstrap.append(sum(sum(x) for x in sample)/sum(len(x) for x in sample))
    bootstrap.sort()
    delta=statistics.mean(b["score"]-a["score"] for a,b in zip(left,right))
    return {"delta":delta,"ci95":[bootstrap[int(.025*(samples-1))],bootstrap[int(.975*(samples-1))]],
            "groups":len(groups),"method":"paired group bootstrap percentile; descriptive, not proof of causality",
            "right_wins":sum(b["score"]>a["score"] for a,b in zip(left,right)),
            "left_wins":sum(a["score"]>b["score"] for a,b in zip(left,right))}


def release_gate(report:dict,policy:ReleasePolicy) -> dict:
    arms=report["arms"];selected=arms["selected"];static=arms["static"];baseline=arms["baseline"]
    checks={
        "enough_test_examples":selected["count"]>=policy.min_test_examples,
        "enough_independent_groups":report["test_groups"]>=policy.min_test_groups,
        "minimum_score":selected["score"]>=policy.min_score,
        "baseline_gain":selected["score"]-baseline["score"]>=policy.min_baseline_gain-1e-12,
        "static_noninferiority_ci":report["comparisons"]["static_vs_selected"]["ci95"][0]>=-policy.static_noninferiority-1e-12,
        # A deployed recipe that is credibly worse than plain retrieval costs more for less: refuse to release it.
        "retrieval_noninferiority_ci":report["comparisons"]["retrieval_vs_selected"]["ci95"][0]>=-policy.retrieval_noninferiority-1e-12,
        "coverage":selected["coverage"]>=policy.min_coverage,
        "invalid_rate":selected["invalid_rate"]<=policy.max_invalid_rate,
        "selection_before_test":report.get("selection_before_test") is True,
        "live_provider":not policy.require_live_provider or report.get("provider") not in {"toy","suite-toy"},
    }
    regressions=[];small_slices=[]
    for name,s in selected["slices"].items():
        if s["count"]<policy.min_slice_examples:
            small_slices.append(name);continue
        if s["score"]<static["slices"][name]["score"]-policy.max_slice_regression-1e-12:
            regressions.append(name)
    checks["slice_regressions"]=not regressions
    return {"passed":all(checks.values()),"checks":checks,"failed":[k for k,v in checks.items() if not v],
            "regressing_slices":regressions,"underpowered_slices":small_slices,"policy":asdict(policy),
            "scope":"quality gate only; not security certification, SLA proof, or real-model efficacy evidence"}


def evaluate_artifact(runtime:SuiteRuntime,test:list[Record],*,policy:ReleasePolicy,baseline_id="baseline",retrieval_id:str|None=None,
                      progress=None) -> dict:
    if not test: raise ValueError("Empty test set")
    task=runtime.predictor.task
    config=runtime.data["provenance"].get("config",{})
    scorer=Scorer(task,runtime.predictor.backend,config.get("judge_model"),config.get("judge_rubric"))
    # Test targets are accessed here, after artifact construction and selection.
    recipes=runtime.recipes
    if baseline_id not in recipes: raise ValueError("Artifact has no baseline recipe")
    retrieval_id=retrieval_id or runtime.data["provenance"].get("retrieval_recipe")
    if retrieval_id not in recipes: raise ValueError("Artifact has no recorded retrieval comparator")
    outputs={key:[] for key in ("baseline","retrieval","static")}
    if runtime.weights:outputs["weights"] = []
    for i,ex in enumerate(test):
        if task.validate_output(ex.target):raise ValueError("Invalid evaluation target")
        per_case={}
        def run_recipe(rid):
            if rid not in per_case:
                per_case[rid]=runtime.predict(ex.input,mode="recipe",recipe_id=rid)
            return per_case[rid]
        predictions={"baseline":run_recipe(baseline_id),"retrieval":run_recipe(retrieval_id),
                     "static":run_recipe(runtime.data["default_recipe"])}
        if runtime.weights:
            # Explicitly run the weights path including its verifier, not just a label lookup.
            predictions["weights"]=runtime.predict(ex.input,mode="weights")
        for key,p in predictions.items():
            score=scorer.score(ex.input,p.text,ex.target) if p.status=="accepted" else 0.0
            outputs[key].append({"id":ex.id,"group":ex.group or ex.id,"slice":ex.slice,"target":ex.target,
                                 **p.to_dict(),"score":score})
        if progress and (i%10==0 or i==len(test)-1):progress({"phase":"test","completed":i+1,"total":len(test)})
    selected_key="weights" if runtime.data["use_weights"] else "static"
    outputs["selected"]=outputs[selected_key]
    comparisons={}
    for left,right in (("baseline","selected"),("retrieval","selected"),("static","selected")):
        comparisons[left+"_vs_"+right]=paired_interval(outputs[left],outputs[right],samples=policy.bootstrap_samples)
    if "weights" in outputs:
        comparisons["static_vs_weights"]=paired_interval(outputs["static"],outputs["weights"],samples=policy.bootstrap_samples)
    report={"schema_version":1,"artifact_id":runtime.artifact.artifact_id,"provider":runtime.data["provider"],
            "model":runtime.data["model"],"task_metric":task.metric,"selected_mode":selected_key,
            "selection_before_test":True,"test_groups":len({x.group or x.id for x in test}),
            "test_manifest":{"sha256":digest([asdict(x) for x in sorted(test,key=lambda x:x.id)]),
                             "count":len(test),"input_hashes":sorted(digest(normalized_input(x.input)) for x in test),
                             "group_hashes":sorted({digest(x.group or x.id) for x in test})},
            "arms":{key:summarize(rows,task) for key,rows in outputs.items()},"comparisons":comparisons,
            "predictions":outputs,
            "caveats":["Text token-F1 is lexical overlap, not a semantic-quality or factuality score.",
                       "Resampled intervals do not account for repeated test-set reuse or provider nondeterminism.",
                       "Cache-hit latency is not representative production latency."]}
    report["grading"]=scorer.usage()
    report["gate"]=release_gate(report,policy)
    return report
