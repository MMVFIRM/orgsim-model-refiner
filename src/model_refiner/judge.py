"""Bridge to Artificial Weights (`aw`): use any model-refiner backend as the LLM judge of `aw judge`.

Requires the `aw` extra (`pip install 'model-refiner[aw]'`). The judge protocol, integrity rules and report are
documented in docs/JUDGE_PROTOCOL.md.
"""
from __future__ import annotations

from pathlib import Path

from .backend import CachedBackend, ModelBackend, RetryingBackend, make_backend


def aw_available() -> bool:
    try:
        import aw.judge  # noqa: F401
        return True
    except Exception:
        return False


def RefinerJudgeBackend(inner: ModelBackend, model: str, name: str = "refiner"):
    """Wrap a ModelBackend as an `aw.judge.Backend` (one user turn in, reply text out)."""
    from aw.judge import Backend

    class _Judge(Backend):
        def __init__(self):
            self.name, self.model = name, model

        def ask(self, prompt: str) -> str:
            return inner.generate(model=model, messages=[{"role": "user", "content": prompt}]).text

    return _Judge()


def ask(work: str | Path, provider: str, model: str, *, base_url: str | None = None, key_env: str | None = None,
        command: str | None = None, workers: int = 4, retries: int = 3, backoff: float = 2.0, timeout: float = 120.0,
        limit: int | None = None, cache: str | Path | None = None):
    """Ask a model-refiner backend every query in `work/queries.jsonl`, writing `work/answers.jsonl` (resumable)."""
    from aw.judge import read_jsonl, run_judge

    work = Path(work)
    queries = read_jsonl(work / "queries.jsonl")
    if not queries:
        raise SystemExit("No queries found: run `aw judge prepare --work ...` first")
    inner: ModelBackend = RetryingBackend(make_backend(provider, base_url=base_url, key_env=key_env,
                                                       command=command, max_tokens=8, timeout=timeout), retries=retries, backoff=backoff)
    if cache:
        inner = CachedBackend(inner, cache)
    backend = RefinerJudgeBackend(inner, model, name=f"refiner:{provider}")
    return run_judge(queries, backend, work / "answers.jsonl", workers=workers, retries=1, limit=limit)
