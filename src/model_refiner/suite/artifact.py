"""Version-2 deployment artifacts. Evaluation receipts live separately to avoid circular hashes."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import atomic_json, canonical, digest, integer, keys, nonempty, strict_json
from .dataset import Record
from .config import TrainConfig
from .tasks import TaskSpec
from .weights import ArtificialWeights


@dataclass(frozen=True)
class Recipe:
    id: str
    rules: str = ""
    k: int = 0
    retrieval: str = "tfidf"
    source: str = "baseline"

    def __post_init__(self):
        nonempty(self.id,"recipe id",128)
        if not isinstance(self.rules,str) or len(self.rules)>40_000:
            raise ValueError("Invalid recipe rules")
        integer(self.k,"recipe k",0,32)
        if self.retrieval not in {"tfidf","mmr"}:
            raise ValueError("Unknown retrieval algorithm")
        if self.source not in {"baseline","retrieval","supervised","preference","hybrid"}:
            raise ValueError("Unknown recipe source")

    @classmethod
    def from_dict(cls,obj):
        keys(obj,set(cls.__dataclass_fields__),"recipe")
        return cls(**obj)


class SuiteArtifact:
    """Defensive-copy JSON object: callers cannot mutate the live artifact payload."""
    SCHEMA=2

    def __init__(self,payload:dict):
        data=strict_json(canonical(payload))
        keys(data,{"schema_version","artifact_id","created_at","provider","model","verifier_model","backend_identity",
                   "task","recipes","default_recipe","use_weights","weights","memory","runtime","provenance"},"artifact")
        if data.get("schema_version")!=self.SCHEMA:
            raise ValueError("Unsupported suite artifact schema")
        logical={k:v for k,v in data.items() if k!="artifact_id"}
        if data.get("artifact_id")!=digest(logical):
            raise ValueError("Artifact integrity check failed")
        nonempty(data.get("provider"),"provider",64);nonempty(data.get("model"),"model",256)
        if data.get("verifier_model") is not None:
            nonempty(data["verifier_model"],"verifier_model",256)
        TaskSpec.from_dict(data["task"])
        if not isinstance(data["recipes"],list) or not 1<=len(data["recipes"])<=64:
            raise ValueError("Artifact needs 1-64 recipes")
        recipes=[Recipe.from_dict(x) for x in data["recipes"]]
        ids=[x.id for x in recipes]
        if len(set(ids))!=len(ids) or data["default_recipe"] not in ids:
            raise ValueError("Invalid default/duplicate recipe")
        if type(data["use_weights"]) is not bool:
            raise ValueError("use_weights must be boolean")
        if data["weights"] is not None:
            weights=ArtificialWeights.from_dict(data["weights"])
            if set(weights.recipe_ids)!=set(ids):
                raise ValueError("Weights refer to a different recipe bank")
        elif data["use_weights"]:
            raise ValueError("Artifact selects absent weights")
        if not isinstance(data["memory"],list) or len(data["memory"])>10_000:
            raise ValueError("Oversized memory")
        memory=[Record.from_dict(x,i) for i,x in enumerate(data["memory"])]
        if len({x.id for x in memory})!=len(memory) or any(x.split not in {None,"train"} for x in memory):
            raise ValueError("Artifact memory must contain unique train-only rows")
        if any(x.k for x in recipes) and not memory:
            raise ValueError("Retrieval recipe requires memory")
        runtime=data["runtime"]
        keys(runtime,{"max_prompt_chars","repair_attempts","verification_margin","max_output_tokens","timeout_seconds"},"runtime")
        from .common import number
        integer(runtime["max_prompt_chars"],"max_prompt_chars",1024,500_000)
        integer(runtime["repair_attempts"],"repair_attempts",0,2)
        integer(runtime["max_output_tokens"],"max_output_tokens",16,32_000)
        number(runtime["verification_margin"],"verification_margin",0,1)
        number(runtime["timeout_seconds"],"timeout_seconds",1,300)
        if not isinstance(data["backend_identity"],dict) or not isinstance(data["provenance"],dict):
            raise ValueError("Invalid artifact provenance")
        config=TrainConfig.from_dict(data["provenance"].get("config"))
        if config.model!=data["model"] or config.verifier_model!=data["verifier_model"] or canonical(config.task.to_dict())!=canonical(data["task"]):
            raise ValueError("Artifact config/model/task mismatch")
        for name,value in runtime.items():
            if getattr(config,name)!=value:
                raise ValueError("Artifact runtime/config mismatch")
        lookup={r.id:r for r in recipes}
        baseline=lookup.get("baseline")
        if baseline is None or baseline.rules or baseline.k or baseline.source!="baseline":
            raise ValueError("Artifact must preserve a genuine baseline comparator")
        rid=data["provenance"].get("retrieval_recipe")
        retrieval=lookup.get(rid) if isinstance(rid,str) else None
        if retrieval is None or retrieval.rules or retrieval.k<1 or retrieval.source!="retrieval":
            raise ValueError("Artifact needs a genuine retrieval-only comparator")
        if len(canonical(data))>20_000_000:
            raise ValueError("Artifact exceeds size limit")
        self._json=canonical(data)
        self.artifact_id=data["artifact_id"]

    @classmethod
    def create(cls,**payload):
        logical={"schema_version":cls.SCHEMA,"created_at":datetime.now(timezone.utc).isoformat(),**payload}
        return cls({**logical,"artifact_id":digest(logical)})

    def to_dict(self):
        return strict_json(self._json)

    def save(self,path):
        atomic_json(path,self.to_dict())

    @classmethod
    def load(cls,path):
        p=Path(path)
        if p.stat().st_size>20_000_000:
            raise ValueError("Artifact exceeds size limit")
        return cls(strict_json(p.read_text(encoding="utf-8")))
