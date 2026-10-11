from __future__ import annotations
from dataclasses import asdict, dataclass, field
from .common import integer, keys, nonempty, number
from .tasks import TaskSpec


@dataclass(frozen=True)
class TrainConfig:
    model: str
    task: TaskSpec
    optimizer_model: str | None = None
    verifier_model: str | None = None
    judge_model: str | None = None
    judge_rubric: str | None = None
    seed: int = 42
    k_values: tuple[int, ...] = (2, 4)
    rounds: int = 2
    max_train_errors: int = 24
    max_train_scan: int = 64
    max_rules_chars: int = 6000
    max_prompt_chars: int = 100_000
    max_attempts: int = 5000
    max_output_tokens: int = 2048
    max_compile_tokens: int = 4096
    max_reserved_tokens: int = 50_000_000
    retries: int = 2
    timeout_seconds: float = 60.0
    repair_attempts: int = 1
    enable_weights: bool = True
    preflight_probe: bool = True
    balanced_error_sampling: bool = True
    rule_coverage_check: bool = True
    weight_dimension: int = 512
    weight_epochs: int = 40
    route_min_similarity: float = 0.1
    route_min_support: int = 2
    verification_margin: float = 0.1
    token_penalty: float = 0.0

    def __post_init__(self):
        nonempty(self.model, "model", 256)
        for key in ("optimizer_model", "verifier_model", "judge_model"):
            if getattr(self, key) is not None:
                nonempty(getattr(self, key), key, 256)
        if not isinstance(self.task, TaskSpec):
            raise ValueError("task must be a TaskSpec")
        if self.task.metric == "rubric":
            nonempty(self.judge_model, "judge_model", 256)
            nonempty(self.judge_rubric, "judge_rubric", 10_000)
        elif self.judge_model is not None or self.judge_rubric is not None:
            raise ValueError("Judge settings require the rubric task metric")
        if not isinstance(self.k_values, (list, tuple)) or not 1 <= len(self.k_values) <= 6:
            raise ValueError("k_values must have 1-6 entries")
        for k in self.k_values:
            integer(k, "retrieval k", 1, 32)
        limits = {"seed": (0, 2**32-1), "rounds": (0, 5), "max_train_errors": (1, 100),
                  "max_train_scan": (1, 500), "max_rules_chars": (128, 20_000),
                  "max_prompt_chars": (1024, 500_000), "max_attempts": (1, 1_000_000),
                  "max_output_tokens": (16, 32_000), "max_compile_tokens": (16, 32_000), "max_reserved_tokens": (1000, 10**12),
                  "retries": (0, 5), "repair_attempts": (0, 2), "weight_dimension": (32, 2048),
                  "weight_epochs": (1, 200), "route_min_support": (1, 100)}
        for name, (lo, hi) in limits.items():
            integer(getattr(self, name), name, lo, hi)
        number(self.timeout_seconds, "timeout_seconds", 1, 300)
        number(self.route_min_similarity, "route_min_similarity", 0, 1)
        number(self.verification_margin, "verification_margin", 0, 1)
        number(self.token_penalty, "token_penalty", 0, 1)
        if type(self.enable_weights) is not bool:
            raise ValueError("enable_weights must be boolean")
        for flag in ("preflight_probe", "balanced_error_sampling", "rule_coverage_check"):
            if type(getattr(self, flag)) is not bool:
                raise ValueError(f"{flag} must be boolean")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, obj):
        keys(obj, set(cls.__dataclass_fields__), "training config")
        data = dict(obj)
        nonempty(data.get("model"), "model", 256)
        if not isinstance(data.get("k_values", (2,4)), (tuple,list)):
            raise ValueError("k_values must be a list")
        data["task"] = TaskSpec.from_dict(data.get("task"))
        data["k_values"] = tuple(data.get("k_values", (2, 4)))
        return cls(**data)


@dataclass(frozen=True)
class ReleasePolicy:
    min_test_examples: int = 100
    min_test_groups: int = 30
    min_score: float = 0.80
    min_baseline_gain: float = 0.0
    static_noninferiority: float = 0.02
    retrieval_noninferiority: float = 0.02
    min_coverage: float = 0.99
    max_invalid_rate: float = 0.01
    max_slice_regression: float = 0.05
    min_slice_examples: int = 10
    bootstrap_samples: int = 2000
    require_live_provider: bool = True

    def __post_init__(self):
        for name, lo, hi in (("min_test_examples",1,100_000),("min_test_groups",1,100_000),
                             ("min_slice_examples",1,100_000),("bootstrap_samples",100,10_000)):
            integer(getattr(self,name),name,lo,hi)
        for name in ("min_score", "min_coverage", "max_invalid_rate", "max_slice_regression", "static_noninferiority", "retrieval_noninferiority"):
            number(getattr(self,name),name,0,1)
        number(self.min_baseline_gain,"min_baseline_gain",-1,1)
        if type(self.require_live_provider) is not bool:
            raise ValueError("require_live_provider must be boolean")

    @classmethod
    def from_dict(cls, obj):
        keys(obj,set(cls.__dataclass_fields__),"release policy")
        return cls(**obj)
