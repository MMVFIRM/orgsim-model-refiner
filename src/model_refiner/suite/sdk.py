"""Guarded Python entry point for local experiments.

The lower-level `train` function accepts arbitrary test backends. Application
code should use `fit` so every provider attempt is durably budgeted. Providers
are configured by the operator; no key or URL is serialized into artifacts.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import uuid

from ..backend import ModelBackend
from .common import atomic_json, digest, strict_json
from .config import TrainConfig, ReleasePolicy
from .dataset import Record, Preference
from .guarded import AttemptLedger, GuardedBackend
from .training import TrainResult, train


def fit(backend: ModelBackend, provider: str, records: list[Record], config: TrainConfig,
        *, workspace: str | Path, preferences: list[Preference] | None = None,
        policy: ReleasePolicy | None = None, resume: bool = False,
        progress=None) -> TrainResult:
    """Fit, freeze, and evaluate an adapter. Never promotes a production version.

    The caller owns the supplied backend's lifecycle. Resume reuses responses
    only within this experiment; it is not an independent replication.
    """
    destination = Path(workspace)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_file = destination / "sdk-run.json"
    preferences = preferences or []
    identity = digest({"provider": provider, "backend": backend.cache_identity(),
                       "config": config.to_dict(), "records": [asdict(x) for x in records],
                       "preferences": [asdict(x) for x in preferences]})
    if manifest_file.exists():
        if not resume:
            raise ValueError("Workspace already has a run; use resume=True or a fresh workspace")
        manifest = strict_json(manifest_file.read_text(encoding="utf-8"))
        if manifest["identity"] != identity:
            raise ValueError("Cannot resume with different configuration, backend, or data")
        run_id = manifest["run_id"]
    else:
        run_id = str(uuid.uuid4())
        atomic_json(manifest_file, {"identity": identity, "run_id": run_id}, overwrite=False)
    ledger = AttemptLedger(destination / "ledger.sqlite3", run_id, config.max_attempts,
                           config.max_reserved_tokens)
    guarded = GuardedBackend(backend, ledger, retries=config.retries,
                             max_prompt_chars=config.max_prompt_chars,
                             max_output_tokens=config.max_output_tokens)
    result = train(guarded, provider, records, config, preferences=preferences,
                   policy=policy, progress=progress, run_id=run_id)
    result.report["budget"] = ledger.snapshot()
    result.artifact.save(destination / "artifact.json")
    atomic_json(destination / "report.json", result.report)
    return result
