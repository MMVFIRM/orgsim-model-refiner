"""Production HTTP serving and fitting surface for Model Refiner.

Provider transports, base URLs, credentials and shell commands are process-level
configuration. API clients may choose only allowlisted model names; they cannot
turn the service into a generic SSRF or shell-execution proxy.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict
import hmac
import threading
from typing import Any

from .artifact import RefinerArtifact
from .backend import ModelBackend
from .engine import ExperimentConfig
from .fitting import estimate_call_upper_bound
from .jobs import FitJob, FitJobStore, FitWorker, IdempotencyConflict
from .runtime import RefinerRuntime
from .store import ArtifactStore
from .types import Example


def create_app(
    *,
    store_path: str,
    backend: ModelBackend,
    provider: str,
    api_key: str | None,
    fit_backend: ModelBackend | None = None,
    allowed_models: set[str] | None = None,
    max_input_chars: int = 100_000,
    max_prompt_chars: int = 500_000,
    max_concurrency: int = 8,
    max_artifact_examples: int = 10_000,
    max_artifact_chars: int = 5_000_000,
    max_fit_examples: int = 5_000,
    max_fit_chars: int = 20_000_000,
    max_fit_calls: int = 5_000,
    fit_lease_seconds: float = 300.0,
    fit_retry_multiplier: int = 1,
    allow_provider_mismatch: bool = False,
):
    try:
        from fastapi import Body, Depends, FastAPI, Header, HTTPException, Response
        from fastapi.responses import JSONResponse
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Serving requires: pip install 'model-refiner[serve]'") from exc

    if max_input_chars < 1 or max_prompt_chars < 1 or max_concurrency < 1:
        raise ValueError("input, prompt and concurrency limits must be positive")
    if (max_artifact_examples < 1 or max_artifact_chars < 1 or max_fit_examples < 3 or
            max_fit_chars < 1 or max_fit_calls < 1 or fit_retry_multiplier < 1):
        raise ValueError("artifact/fit limits are invalid")

    store = ArtifactStore(store_path)
    jobs = FitJobStore(store_path)
    worker = FitWorker(
        jobs=jobs,
        artifacts=store,
        backend=fit_backend or backend,
        provider=provider,
        lease_seconds=fit_lease_seconds,
    )
    allow = set(allowed_models or ())
    semaphore = threading.BoundedSemaphore(max_concurrency)
    runtime_cache: dict[tuple[str, int, str], RefinerRuntime] = {}
    runtime_lock = threading.Lock()
    metrics_lock = threading.Lock()
    counters = {
        "predict_requests": 0,
        "predict_errors": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "imports": 0,
        "promotions": 0,
        "fit_submissions": 0,
    }

    @asynccontextmanager
    async def lifespan(app):
        worker.start()
        try:
            yield
        finally:
            worker.stop()

    app = FastAPI(title="Model Refiner", version="0.3.0-rc1", lifespan=lifespan)

    def auth(
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> None:
        if api_key is None:
            return
        supplied = x_api_key
        if authorization and authorization.lower().startswith("bearer "):
            supplied = authorization[7:].strip()
        if not supplied or not hmac.compare_digest(supplied, api_key):
            raise HTTPException(status_code=401, detail="invalid API key")

    def model_policy(model: str) -> None:
        if allow and model not in allow:
            raise HTTPException(status_code=403, detail=f"model {model!r} is not allowlisted")

    def artifact_policy(artifact: RefinerArtifact) -> None:
        if artifact.provider != provider and not allow_provider_mismatch:
            raise HTTPException(
                status_code=409,
                detail=f"artifact provider {artifact.provider!r} does not match serving provider {provider!r}",
            )
        model_policy(artifact.model)
        if len(artifact.examples) > max_artifact_examples:
            raise HTTPException(status_code=413, detail=f"artifact exceeds {max_artifact_examples} memory examples")
        if artifact.retrieval_k > 128:
            raise HTTPException(status_code=413, detail="artifact retrieval_k exceeds service limit 128")
        parts = [artifact.base_instruction, artifact.compiled_rules]
        parts.extend(x.input for x in artifact.examples)
        parts.extend(x.target for x in artifact.examples)
        if any(len(x) > max_input_chars for x in parts):
            raise HTTPException(status_code=413, detail="artifact contains an oversized instruction/example")
        if sum(len(x) for x in parts) > max_artifact_chars:
            raise HTTPException(status_code=413, detail=f"artifact exceeds {max_artifact_chars} total characters")

    def load_runtime(name: str, version: int | None):
        try:
            stored, artifact = store.load(name, version)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        artifact_policy(artifact)
        key = (name, stored.version, artifact.artifact_id)
        with runtime_lock:
            rt = runtime_cache.get(key)
            if rt is None:
                rt = RefinerRuntime(
                    backend,
                    artifact,
                    runtime_provider=provider,
                    allow_provider_mismatch=allow_provider_mismatch,
                    max_prompt_chars=max_prompt_chars,
                )
                runtime_cache[key] = rt
        return stored, artifact, rt

    @staticmethod
    def job_view(job: FitJob) -> dict[str, Any]:
        # Do not echo the training dataset back in job/status responses.
        return {
            "job_id": job.job_id,
            "name": job.name,
            "status": job.status,
            "artifact_version": job.artifact_version,
            "artifact_id": job.artifact_id,
            "result": job.result,
            "error": job.error,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
        }

    def normalize_fit(body: dict[str, Any]) -> tuple[dict[str, Any], int, int]:
        model = body.get("model")
        task = body.get("task")
        rows = body.get("examples")
        if not isinstance(model, str) or not model.strip():
            raise HTTPException(status_code=422, detail="model must be a non-empty string")
        model = model.strip()
        model_policy(model)
        optimizer_model = body.get("optimizer_model", model)
        if isinstance(optimizer_model, str) and optimizer_model.lower() in {"none", "off", "false"}:
            optimizer_model = None
        if optimizer_model is not None:
            if not isinstance(optimizer_model, str) or not optimizer_model.strip():
                raise HTTPException(status_code=422, detail="optimizer_model must be a model name or null")
            optimizer_model = optimizer_model.strip()
            model_policy(optimizer_model)
        if not isinstance(task, str) or not task.strip():
            raise HTTPException(status_code=422, detail="task must be a non-empty string")
        if len(task) > max_input_chars:
            raise HTTPException(status_code=413, detail="task instruction is too large")
        if not isinstance(rows, list) or len(rows) < 3:
            raise HTTPException(status_code=422, detail="examples must contain at least 3 rows")
        if len(rows) > max_fit_examples:
            raise HTTPException(status_code=413, detail=f"fit exceeds {max_fit_examples} examples")

        examples: list[Example] = []
        normalized_rows = []
        seen_ids: set[str] = set()
        total_fit_chars = len(task)
        for i, row in enumerate(rows):
            if not isinstance(row, dict) or "input" not in row or "target" not in row:
                raise HTTPException(status_code=422, detail=f"example {i} needs input and target")
            inp, target = row["input"], row["target"]
            if not isinstance(inp, str) or not isinstance(target, str) or not inp.strip() or not target.strip():
                raise HTTPException(status_code=422, detail=f"example {i} input/target must be non-empty strings")
            if len(inp) > max_input_chars or len(target) > max_input_chars:
                raise HTTPException(status_code=413, detail=f"example {i} is too large")
            total_fit_chars += len(inp) + len(target)
            if total_fit_chars > max_fit_chars:
                raise HTTPException(status_code=413, detail=f"fit exceeds {max_fit_chars} total characters")
            identifier = str(row.get("id", i + 1))
            if not identifier or identifier in seen_ids:
                raise HTTPException(status_code=422, detail=f"example {i} has a duplicate/empty id")
            seen_ids.add(identifier)
            split = row.get("split")
            if split is not None and not isinstance(split, str):
                raise HTTPException(status_code=422, detail=f"example {i} split must be a string")
            metadata = row.get("metadata", {})
            if not isinstance(metadata, dict):
                raise HTTPException(status_code=422, detail=f"example {i} metadata must be an object")
            nr = {
                "id": identifier,
                "input": inp,
                "target": target,
                "split": split,
                "metadata": dict(metadata),
            }
            normalized_rows.append(nr)
            examples.append(Example(**nr))

        raw_k = body.get("k_values", [1, 2, 4])
        if isinstance(raw_k, str):
            try:
                raw_k = [int(x) for x in raw_k.split(",") if x.strip()]
            except ValueError as exc:
                raise HTTPException(status_code=422, detail="k_values must contain integers") from exc
        if not isinstance(raw_k, list) or not raw_k:
            raise HTTPException(status_code=422, detail="k_values must be a non-empty list")
        try:
            k_values = sorted(set(int(x) for x in raw_k if int(x) > 0))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="k_values must contain positive integers") from exc
        if not k_values or max(k_values) > 128:
            raise HTTPException(status_code=422, detail="k_values must be between 1 and 128")
        try:
            seed = int(body.get("seed", 42))
            max_rule_errors = int(body.get("max_rule_errors", 24))
            max_rules = int(body.get("max_rules", 12))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="seed/rule limits must be integers") from exc
        if not (1 <= max_rule_errors <= 1000 and 1 <= max_rules <= 100):
            raise HTTPException(status_code=422, detail="rule limits are out of range")

        config = ExperimentConfig(
            model=model, optimizer_model=optimizer_model, base_instruction=task,
            k_values=tuple(k_values), seed=seed, max_rule_errors=max_rule_errors, max_rules=max_rules,
        )
        try:
            estimate = estimate_call_upper_bound(examples, config)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        attempt_ceiling = estimate * fit_retry_multiplier
        if attempt_ceiling > max_fit_calls:
            raise HTTPException(
                status_code=422,
                detail=(f"fit needs {estimate} logical generations and up to {attempt_ceiling} upstream provider calls "
                        f"with retries, exceeding server cap {max_fit_calls}"),
            )
        return {
            "model": model,
            "optimizer_model": optimizer_model,
            "task": task,
            "k_values": k_values,
            "seed": seed,
            "max_rule_errors": max_rule_errors,
            "max_rules": max_rules,
            "max_prompt_chars": max_prompt_chars,
            "max_calls": max_fit_calls,
            "examples": normalized_rows,
        }, estimate, attempt_ceiling

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz():
        try:
            ok = store.ping()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"store unavailable: {exc}") from exc
        if not ok:
            raise HTTPException(status_code=503, detail="store unavailable")
        return {"status": "ready", "provider": provider}

    @app.get("/v1/refiners/{name}/versions", dependencies=[Depends(auth)])
    def versions(name: str):
        return {"name": name, "versions": [asdict(v) for v in store.list_versions(name)]}

    @app.get("/v1/refiners/{name}/promotions", dependencies=[Depends(auth)])
    def promotions(name: str):
        return {"name": name, "promotions": store.promotion_history(name)}

    @app.post("/v1/refiners/{name}/versions/import", dependencies=[Depends(auth)])
    def import_artifact(name: str, body: dict[str, Any] = Body(...)):
        try:
            artifact = RefinerArtifact.from_dict(dict(body.get("artifact", {})))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        artifact_policy(artifact)
        try:
            promote_flag = bool(body.get("promote", False))
            stored = store.add(name, artifact, promote=promote_flag)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        with metrics_lock:
            counters["imports"] += 1
            if promote_flag:
                counters["promotions"] += 1
        return asdict(stored)

    @app.post("/v1/refiners/{name}/versions/{version}/promote", dependencies=[Depends(auth)])
    def promote(name: str, version: int):
        try:
            stored, artifact = store.load(name, version)
            artifact_policy(artifact)
            stored = store.promote(name, version)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        with metrics_lock:
            counters["promotions"] += 1
        return asdict(stored)

    @app.post("/v1/refiners/{name}/fits", dependencies=[Depends(auth)])
    def submit_fit(
        name: str,
        body: dict[str, Any] = Body(...),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ):
        request, estimate, attempt_ceiling = normalize_fit(body)
        try:
            job = jobs.submit(name, request, idempotency_key=idempotency_key)
        except IdempotencyConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        with metrics_lock:
            counters["fit_submissions"] += 1
        payload = job_view(job)
        payload["estimated_logical_calls"] = estimate
        payload["estimated_provider_attempt_ceiling"] = attempt_ceiling
        return JSONResponse(status_code=202, content=payload)

    @app.get("/v1/fit-jobs/{job_id}", dependencies=[Depends(auth)])
    def get_fit_job(job_id: str):
        try:
            return job_view(jobs.get(job_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/v1/refiners/{name}/predict", dependencies=[Depends(auth)])
    def predict(name: str, body: dict[str, Any] = Body(...)):
        user_input = body.get("input")
        if not isinstance(user_input, str) or not user_input:
            raise HTTPException(status_code=422, detail="input must be a non-empty string")
        version = body.get("version")
        if version is not None and (not isinstance(version, int) or version < 1):
            raise HTTPException(status_code=422, detail="version must be a positive integer")
        if len(user_input) > max_input_chars:
            raise HTTPException(status_code=413, detail=f"input exceeds {max_input_chars} characters")
        stored, artifact, rt = load_runtime(name, version)
        acquired = semaphore.acquire(timeout=0.01)
        if not acquired:
            raise HTTPException(status_code=429, detail="server concurrency limit reached")
        try:
            with metrics_lock:
                counters["predict_requests"] += 1
            result = rt.predict(user_input)
            with metrics_lock:
                counters["input_tokens"] += int(result.input_tokens or 0)
                counters["output_tokens"] += int(result.output_tokens or 0)
            return {
                "name": name,
                "version": stored.version,
                "status": stored.status,
                "artifact_id": artifact.artifact_id,
                "model": result.model,
                "output": result.text,
                "usage": {"input_tokens": result.input_tokens, "output_tokens": result.output_tokens},
                "retrieved_example_ids": list(result.retrieved_example_ids),
                "provider_metadata": result.provider_metadata,
            }
        except HTTPException:
            raise
        except Exception as exc:
            with metrics_lock:
                counters["predict_errors"] += 1
            raise HTTPException(status_code=502, detail=f"upstream generation failed: {exc}") from exc
        finally:
            semaphore.release()

    @app.get("/metrics", dependencies=[Depends(auth)])
    def metrics():
        with metrics_lock:
            values = dict(counters)
        values.update({f"fit_jobs_{k}": v for k, v in jobs.status_counts().items()})
        lines = [f"model_refiner_{k} {v}" for k, v in sorted(values.items())]
        return Response("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")

    return app
