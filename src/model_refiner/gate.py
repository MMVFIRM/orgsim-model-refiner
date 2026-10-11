"""Gate 1: the predeclared success criteria from the README, applied mechanically to finished reports."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .engine import ExperimentReport

MIN_TEST = 100
MIN_VS_BASELINE = 0.08
MIN_VS_RETRIEVAL = 0.03
MIN_REPEATS = 3


@dataclass
class Criterion:
    name: str
    passed: bool
    detail: str


@dataclass
class GateResult:
    passed: bool
    criteria: list[Criterion]

    def to_dict(self) -> dict:
        return asdict(self)


def _arm(report: ExperimentReport, name: str):
    return next(a for a in report.arms if a.name == name)


def token_overhead(report: ExperimentReport) -> float | None:
    """Refined input tokens relative to baseline input tokens, or None if the provider did not report usage."""
    b, r = _arm(report, "baseline").total_input_tokens, _arm(report, "refined").total_input_tokens
    return None if not b or r is None else r / b


def evaluate_run(report: ExperimentReport, *, min_test: int = MIN_TEST, min_vs_baseline: float = MIN_VS_BASELINE,
                 min_vs_retrieval: float = MIN_VS_RETRIEVAL) -> GateResult:
    base, ret, ref = _arm(report, "baseline"), _arm(report, "retrieval"), _arm(report, "refined")
    d_base, d_ret = ref.accuracy - base.accuracy, ref.accuracy - ret.accuracy
    oh = token_overhead(report)
    crit = [
        Criterion("held_out_size", report.test_count >= min_test, f"{report.test_count} test examples (need >= {min_test})"),
        Criterion("beats_baseline", d_base >= min_vs_baseline - 1e-12,
                  f"refined - baseline = {d_base:+.1%} (need >= {min_vs_baseline:+.0%})"),
        Criterion("beats_retrieval", d_ret >= min_vs_retrieval - 1e-12,
                  f"refined - retrieval = {d_ret:+.1%} (need >= {min_vs_retrieval:+.0%})"),
        Criterion("token_overhead_reported", oh is not None,
                  "refined/baseline input tokens = " + (f"{oh:.2f}x" if oh is not None else "not reported by the provider")),
    ]
    return GateResult(all(c.passed for c in crit), crit)


def evaluate_repeats(reports: list[ExperimentReport], *, min_repeats: int = MIN_REPEATS, **kw) -> GateResult:
    """The improvement must repeat: every run passes, across at least `min_repeats` seeds (or tasks)."""
    runs = [evaluate_run(r, **kw) for r in reports]
    crit = [Criterion("enough_repeats", len(reports) >= min_repeats, f"{len(reports)} runs (need >= {min_repeats})")]
    crit += [Criterion(f"run_{r.seed}", g.passed, "; ".join(c.detail for c in g.criteria if not c.passed) or "all criteria met")
             for r, g in zip(reports, runs)]
    return GateResult(all(c.passed for c in crit), crit)


def format_gate(result: GateResult) -> str:
    lines = [f"Gate 1: {'PASS' if result.passed else 'FAIL'}"]
    lines += [f"  [{'x' if c.passed else ' '}] {c.name}: {c.detail}" for c in result.criteria]
    return "\n".join(lines)
