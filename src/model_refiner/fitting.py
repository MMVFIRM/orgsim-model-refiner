"""Fit a deployable artifact while preserving held-out evaluation semantics."""
from __future__ import annotations

from dataclasses import dataclass

from .artifact import RefinerArtifact
from .data import split_examples
from .engine import ExperimentConfig, ExperimentReport, RefinerExperiment
from .types import Example


@dataclass(frozen=True)
class FitResult:
    report: ExperimentReport
    artifact: RefinerArtifact
    estimated_call_upper_bound: int


def estimate_call_upper_bound(examples: list[Example], config: ExperimentConfig) -> int:
    """Conservative upper bound on provider calls for one experiment.

    Rule discovery may inspect every training example and then make one compiler
    call.  Each candidate k is evaluated on validation for retrieval and refined
    arms, then all three arms are evaluated on test.
    """
    train, val, test = split_examples(examples, seed=config.seed)
    ks = sorted(set(k for k in config.k_values if k > 0)) or [1]
    compile_calls = (len(train) + 1) if config.optimizer_model else 0
    validation_calls = 2 * len(ks) * len(val)
    test_calls = 3 * len(test)
    return compile_calls + validation_calls + test_calls


def fit_artifact(
    *,
    backend,
    provider: str,
    examples: list[Example],
    config: ExperimentConfig,
    max_calls: int | None = 5000,
) -> FitResult:
    estimate = estimate_call_upper_bound(examples, config)
    if max_calls is not None and estimate > max_calls:
        raise ValueError(
            f"Refusing fit: worst-case provider calls ({estimate}) exceed max_calls={max_calls}. "
            "Raise the limit only after reviewing expected cost."
        )
    report = RefinerExperiment(backend, config).run(examples)
    train, _, _ = split_examples(examples, seed=config.seed)
    metrics = {arm.name: arm.accuracy for arm in report.arms}
    comparisons = {
        f"{c.left}_vs_{c.right}": {
            "delta": c.right_minus_left_accuracy,
            "p_value": c.exact_p_value,
            "right_wins": c.right_wins,
            "left_wins": c.left_wins,
            "ties": c.ties,
        }
        for c in report.comparisons
    }
    artifact = RefinerArtifact.create(
        provider=provider,
        model=config.model,
        optimizer_model=config.optimizer_model,
        base_instruction=config.base_instruction,
        compiled_rules=report.compiled_rules,
        retrieval_k=report.selected_refined_k,
        examples=train,
        metadata={
            "seed": report.seed,
            "train_count": report.train_count,
            "val_count": report.val_count,
            "test_count": report.test_count,
            "evaluated_arm": "refined",
            "accuracy": metrics,
            "comparisons": comparisons,
            "estimated_call_upper_bound": estimate,
        },
    )
    return FitResult(report=report, artifact=artifact, estimated_call_upper_bound=estimate)
