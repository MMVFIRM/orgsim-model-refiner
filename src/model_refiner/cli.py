from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from .artifact import RefinerArtifact
from .backend import PROVIDERS, CachedBackend, RetryingBackend, SQLiteCachedBackend, make_backend
from .data import load_jsonl
from .engine import ExperimentConfig, RefinerExperiment
from .fitting import estimate_call_upper_bound, fit_artifact
from .gate import evaluate_repeats, evaluate_run, format_gate
from .runtime import RefinerRuntime
from .store import ArtifactStore

SUBCOMMANDS = {"run", "fit", "predict", "store-import", "store-promote", "serve", "judge-ask", "aw"}


def _add_provider_args(p: argparse.ArgumentParser, *, default: str | None = "openai") -> None:
    p.add_argument("--provider", choices=PROVIDERS, default=default,
                   help="openai = Responses API (needs the openai extra); openai-compatible/anthropic use plain HTTP; "
                        "command pipes prompts to a shell command; toy is an offline plumbing check")
    p.add_argument("--base-url", default=None)
    p.add_argument("--key-env", default=None, help="environment variable holding the API key")
    p.add_argument("--command", default=None, help="shell command for --provider command (prompt on stdin)")


def _backend(a: argparse.Namespace, *, cached: bool = True):
    raw = make_backend(
        a.provider,
        base_url=getattr(a, "base_url", None),
        key_env=getattr(a, "key_env", None),
        command=getattr(a, "command", None),
        max_tokens=getattr(a, "max_tokens", 1024),
        timeout=getattr(a, "timeout", 120.0),
    )
    b = RetryingBackend(raw, retries=getattr(a, "retries", 3), backoff=getattr(a, "backoff", 2.0))
    if cached:
        return CachedBackend(b, getattr(a, "cache", ".model_refiner_cache.jsonl"))
    return b


def _optimizer_name(a: argparse.Namespace) -> str | None:
    value = getattr(a, "optimizer_model", "")
    return None if value.lower() in {"none", "off", "false"} else (value or a.model)


def _config(a: argparse.Namespace, seed: int | None = None) -> ExperimentConfig:
    return ExperimentConfig(
        model=a.model,
        optimizer_model=_optimizer_name(a),
        base_instruction=a.task,
        k_values=tuple(int(x) for x in a.k.split(",") if x.strip()),
        seed=a.seed if seed is None else seed,
        max_rule_errors=getattr(a, "max_rule_errors", 24),
        max_rules=getattr(a, "max_rules", 12),
        max_prompt_chars=getattr(a, "max_prompt_chars", 500_000),
    )


def _cmd_run(a: argparse.Namespace) -> int:
    examples = load_jsonl(a.data)
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()] if a.seeds else [a.seed]
    if len(seeds) > 1 and any(e.split for e in examples):
        raise SystemExit("--seeds only varies random splits; this dataset has explicit splits, so every seed would be identical")
    backend = _backend(a, cached=True)
    reports = [RefinerExperiment(backend, _config(a, seed)).run(examples) for seed in seeds]

    out = Path(a.out)
    if len(reports) == 1:
        RefinerExperiment.save_report(reports[0], out)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps([r.to_dict() for r in reports], indent=2, ensure_ascii=False), encoding="utf-8")
    summary = [{"seed": r.seed, "results": {x.name: round(x.accuracy, 4) for x in r.arms},
                "selected_retrieval_k": r.selected_retrieval_k, "selected_refined_k": r.selected_refined_k,
                "test_count": r.test_count} for r in reports]
    print(json.dumps({"report": str(out.resolve()), "compiled_rules": reports[-1].compiled_rules,
                      "runs": summary, "cache": {"hits": backend.hits, "misses": backend.misses}}, indent=2))
    if a.gate:
        gate = evaluate_run(reports[0]) if len(reports) == 1 else evaluate_repeats(reports)
        print(format_gate(gate))
        if len(reports) == 1:
            print("  (a single run cannot satisfy the repeat criterion: use --seeds or run unrelated tasks)")
        return 0 if gate.passed else 2
    return 0


def _cmd_fit(a: argparse.Namespace) -> int:
    examples = load_jsonl(a.data)
    config = _config(a)
    estimate = estimate_call_upper_bound(examples, config)
    attempt_ceiling = estimate * (a.retries + 1)
    if attempt_ceiling > a.max_calls:
        raise SystemExit(
            f"Refusing fit: {estimate} logical generations with up to {a.retries} retries imply "
            f"a worst-case {attempt_ceiling} upstream attempts, exceeding --max-calls {a.max_calls}. "
            "Raise the cap only after reviewing expected spend."
        )
    backend = _backend(a, cached=True)
    result = fit_artifact(
        backend=backend,
        provider=a.provider,
        examples=examples,
        config=config,
        max_calls=a.max_calls,
    )
    RefinerExperiment.save_report(result.report, a.report)
    result.artifact.save(a.artifact)
    summary = {
        "artifact": str(Path(a.artifact).resolve()),
        "artifact_id": result.artifact.artifact_id,
        "report": str(Path(a.report).resolve()),
        "provider": result.artifact.provider,
        "model": result.artifact.model,
        "retrieval_k": result.artifact.retrieval_k,
        "estimated_logical_calls": result.estimated_call_upper_bound,
        "estimated_provider_attempt_ceiling": attempt_ceiling,
        "results": {x.name: round(x.accuracy, 4) for x in result.report.arms},
        "cache": {"hits": backend.hits, "misses": backend.misses},
    }
    print(json.dumps(summary, indent=2))
    if a.gate:
        gate = evaluate_run(result.report)
        print(format_gate(gate))
        print("  (single-fit gate cannot satisfy the independent-repeat criterion by itself)")
        return 0 if gate.passed else 2
    return 0


def _cmd_predict(a: argparse.Namespace) -> int:
    artifact = RefinerArtifact.load(a.artifact)
    if a.provider is None:
        a.provider = artifact.provider
    backend = _backend(a, cached=bool(a.cache)) if a.cache else _backend(a, cached=False)
    runtime = RefinerRuntime(
        backend,
        artifact,
        runtime_provider=a.provider,
        allow_provider_mismatch=a.allow_provider_mismatch,
    )
    if a.input is not None:
        user_input = a.input
    elif a.input_file:
        user_input = Path(a.input_file).read_text(encoding="utf-8")
    else:
        user_input = sys.stdin.read()
    if not user_input:
        raise SystemExit("Prediction input is empty")
    r = runtime.predict(user_input)
    print(json.dumps({
        "artifact_id": r.artifact_id,
        "model": r.model,
        "output": r.text,
        "usage": {"input_tokens": r.input_tokens, "output_tokens": r.output_tokens},
        "retrieved_example_ids": list(r.retrieved_example_ids),
    }, ensure_ascii=False, indent=2))
    return 0


def _cmd_store_import(a: argparse.Namespace) -> int:
    store = ArtifactStore(a.store)
    artifact = RefinerArtifact.load(a.artifact)
    row = store.add(a.name, artifact, promote=a.promote)
    print(json.dumps(row.__dict__, indent=2))
    return 0


def _cmd_store_promote(a: argparse.Namespace) -> int:
    store = ArtifactStore(a.store)
    try:
        row = store.promote(a.name, a.version)
    except KeyError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(row.__dict__, indent=2))
    return 0


def _cmd_serve(a: argparse.Namespace) -> int:
    try:
        import uvicorn
    except Exception as exc:
        raise SystemExit("Serving requires: pip install 'model-refiner[serve]'") from exc
    key = os.environ.get(a.service_key_env)
    if not key and not a.allow_unauthenticated:
        raise SystemExit(
            f"Set {a.service_key_env} or pass --allow-unauthenticated explicitly. "
            "The server refuses to start unauthenticated by default."
        )
    allowed_raw = a.allowed_models or os.environ.get("MODEL_REFINER_ALLOWED_MODELS", "")
    if not allowed_raw and not a.allow_any_model:
        raise SystemExit("Set --allowed-models / MODEL_REFINER_ALLOWED_MODELS or pass --allow-any-model explicitly. The server refuses an unbounded model surface by default.")
    from .service import create_app
    backend = _backend(a, cached=False)
    fit_backend = _backend(a, cached=False)
    if a.fit_cache:
        fit_backend = SQLiteCachedBackend(fit_backend, a.fit_cache)
    allowed = None if a.allow_any_model else {x.strip() for x in allowed_raw.split(",") if x.strip()}
    app = create_app(
        store_path=a.store,
        backend=backend,
        fit_backend=fit_backend,
        provider=a.provider,
        api_key=key,
        allowed_models=allowed,
        max_input_chars=a.max_input_chars,
        max_prompt_chars=a.max_prompt_chars,
        max_concurrency=a.max_concurrency,
        max_artifact_examples=a.max_artifact_examples,
        max_artifact_chars=a.max_artifact_chars,
        max_fit_examples=a.max_fit_examples,
        max_fit_chars=a.max_fit_chars,
        max_fit_calls=a.max_fit_calls,
        fit_lease_seconds=a.fit_lease_seconds,
        fit_retry_multiplier=a.retries + 1,
        allow_provider_mismatch=a.allow_provider_mismatch,
    )
    uvicorn.run(app, host=a.host, port=a.port, log_level=a.log_level)
    return 0


def _cmd_judge_ask(a: argparse.Namespace) -> int:
    from . import judge
    if not judge.aw_available():
        raise SystemExit("The aw package needs its extra dependencies: pip install 'model-refiner[aw]'")
    st = judge.ask(a.work, a.provider, a.model, base_url=a.base_url, key_env=a.key_env, command=a.command,
                   workers=a.workers, retries=a.retries, backoff=a.backoff, timeout=a.timeout, limit=a.limit, cache=a.cache)
    print(f"done: asked {st.asked}, reused {st.cached}, unparseable {st.unparseable}, failed {st.failed}")
    return 0 if st.failed == 0 else 1


def _add_experiment_args(r: argparse.ArgumentParser) -> None:
    r.add_argument("--data", required=True, help="JSONL with id,input,target and optional split")
    r.add_argument("--model", required=True, help="the frozen model being refined")
    r.add_argument("--optimizer-model", default="", help="model that compiles rules (default: --model); 'none' disables rules")
    r.add_argument("--task", required=True, help="base developer instruction")
    r.add_argument("--k", default="1,2,4", help="comma-separated retrieval depths to choose from on validation")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--max-rule-errors", type=int, default=24)
    r.add_argument("--max-rules", type=int, default=12)
    r.add_argument("--max-prompt-chars", type=int, default=500_000, help="hard cap on rendered prompt size")
    r.add_argument("--max-tokens", type=int, default=1024, help="output cap per provider call")
    r.add_argument("--retries", type=int, default=3)
    r.add_argument("--backoff", type=float, default=2.0, help="initial retry delay in seconds (doubles each attempt)")
    r.add_argument("--timeout", type=float, default=120.0, help="per-provider-call timeout in seconds")
    r.add_argument("--cache", default=".model_refiner_cache.jsonl")
    _add_provider_args(r)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="model-refiner", description="Refine and deploy frozen language models externally")
    p.add_argument("--version", action="version", version=f"model-refiner {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a baseline / retrieval / learned-refinement experiment")
    _add_experiment_args(r)
    r.add_argument("--seeds", default=None, help="comma-separated seeds for repeated random splits, e.g. 1,2,3")
    r.add_argument("--out", default="model_refiner_report.json")
    r.add_argument("--gate", action="store_true", help="apply Gate 1 criteria; exit code 2 if they fail")
    r.set_defaults(fn=_cmd_run)

    f = sub.add_parser("fit", help="evaluate and emit an immutable deployable refiner artifact")
    _add_experiment_args(f)
    f.add_argument("--artifact", default="model_refiner_artifact.json")
    f.add_argument("--report", default="model_refiner_report.json")
    f.add_argument("--max-calls", type=int, default=5000, help="hard cap on worst-case upstream attempts including configured retries")
    f.add_argument("--gate", action="store_true", help="also print Gate 1 status")
    f.set_defaults(fn=_cmd_fit)

    pred = sub.add_parser("predict", help="run one immutable artifact")
    pred.add_argument("--artifact", required=True)
    g = pred.add_mutually_exclusive_group()
    g.add_argument("--input", default=None)
    g.add_argument("--input-file", default=None)
    pred.add_argument("--max-tokens", type=int, default=1024)
    pred.add_argument("--retries", type=int, default=3)
    pred.add_argument("--backoff", type=float, default=2.0)
    pred.add_argument("--timeout", type=float, default=120.0)
    pred.add_argument("--cache", default=None, help="optional request cache; disabled by default for deployment inference")
    pred.add_argument("--allow-provider-mismatch", action="store_true")
    _add_provider_args(pred, default=None)
    pred.set_defaults(fn=_cmd_predict)

    si = sub.add_parser("store-import", help="import a deployable artifact into a versioned SQLite store")
    si.add_argument("--store", required=True)
    si.add_argument("--name", required=True)
    si.add_argument("--artifact", required=True)
    si.add_argument("--promote", action="store_true", help="make this version production immediately")
    si.set_defaults(fn=_cmd_store_import)

    sp = sub.add_parser("store-promote", help="atomically promote or roll back to a stored version")
    sp.add_argument("--store", required=True)
    sp.add_argument("--name", required=True)
    sp.add_argument("--version", required=True, type=int)
    sp.set_defaults(fn=_cmd_store_promote)

    s = sub.add_parser("serve", help="serve versioned artifacts over an authenticated HTTP API")
    s.add_argument("--store", default="./data/model_refiner.sqlite3")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--service-key-env", default="MODEL_REFINER_SERVICE_KEY")
    s.add_argument("--allow-unauthenticated", action="store_true", help="explicitly disable API-key auth (development only)")
    s.add_argument("--allowed-models", default="", help="comma-separated exact model allowlist")
    s.add_argument("--allow-any-model", action="store_true", help="explicitly disable the model allowlist (not recommended)")
    s.add_argument("--max-input-chars", type=int, default=100_000)
    s.add_argument("--max-prompt-chars", type=int, default=500_000)
    s.add_argument("--max-concurrency", type=int, default=8)
    s.add_argument("--max-artifact-examples", type=int, default=10_000)
    s.add_argument("--max-artifact-chars", type=int, default=5_000_000)
    s.add_argument("--max-fit-examples", type=int, default=5_000)
    s.add_argument("--max-fit-chars", type=int, default=20_000_000)
    s.add_argument("--max-fit-calls", type=int, default=5_000)
    s.add_argument("--fit-lease-seconds", type=float, default=300.0)
    s.add_argument("--fit-cache", default="./data/model_refiner_fit_cache.sqlite3", help="SQLite cache for fit/eval calls; set empty to disable")
    s.add_argument("--max-tokens", type=int, default=1024)
    s.add_argument("--retries", type=int, default=3)
    s.add_argument("--backoff", type=float, default=2.0)
    s.add_argument("--timeout", type=float, default=120.0)
    s.add_argument("--log-level", default="info", choices=("critical", "error", "warning", "info", "debug", "trace"))
    s.add_argument("--allow-provider-mismatch", action="store_true", help="serve artifacts fitted with another provider (experimental)")
    _add_provider_args(s)
    s.set_defaults(fn=_cmd_serve)

    j = sub.add_parser("judge-ask", help="answer an `aw judge` queries.jsonl with any model-refiner provider")
    j.add_argument("--work", required=True, help="the --work directory used by `aw judge prepare`")
    j.add_argument("--model", required=True)
    j.add_argument("--workers", type=int, default=4)
    j.add_argument("--retries", type=int, default=3)
    j.add_argument("--backoff", type=float, default=2.0)
    j.add_argument("--timeout", type=float, default=120.0)
    j.add_argument("--limit", type=int, default=None, help="ask at most this many new items (cheap trial)")
    j.add_argument("--cache", default=None, help="optional request cache file")
    _add_provider_args(j)
    j.set_defaults(fn=_cmd_judge_ask)

    sub.add_parser("suite", help="frozen-model adaptation suite: train, evaluate, serve, review feedback")
    sub.add_parser("aw", help="Artificial Weights CLI (same as the `aw` command); everything after `aw` is passed through")
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "suite":
        from .suite.cli import main as suite_main
        return suite_main(argv[1:])
    if argv and argv[0] == "aw":
        try:
            from aw.cli import main as aw_main
            return aw_main(argv[1:]) or 0
        except ModuleNotFoundError as exc:
            if exc.name not in {"numpy", "scipy", "sklearn", "torch", "pandas", "tabulate", "pyarrow", "PIL"}:
                raise
            raise SystemExit(f"Artificial Weights needs its extra dependencies (missing {exc.name}): pip install 'model-refiner[aw]'")
    if argv and argv[0].startswith("-") and argv[0] not in {"-h", "--help", "--version"}:
        argv.insert(0, "run")
    args = _build_parser().parse_args(argv)
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
