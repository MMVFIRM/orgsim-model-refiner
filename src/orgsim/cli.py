"""orgsim CLI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml
from .compiler import compile_spec, merge_manifests
from .engine import SimulationEngine, SimError
from .openapi import import_openapi
from .scenarios import generate_scenarios, write_jsonl
from .spec import load_spec, SpecError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="orgsim", description="Offline enterprise simulation compiler + runner")
    commands = parser.add_subparsers(dest="command", required=True)

    imp = commands.add_parser("import-openapi", help="Extract a draft resource catalog from OpenAPI 3.x")
    imp.add_argument("files", nargs="+")
    imp.add_argument("--system", required=True)
    imp.add_argument("--organization", default="Unreviewed organization")
    imp.add_argument("--base-path", default="", help="Optional prefix to strip, e.g. /api/v2")
    imp.add_argument("--out", required=True)

    merge = commands.add_parser("merge", help="Combine multiple system contracts into a draft organization")
    merge.add_argument("manifests", nargs="+")
    merge.add_argument("--organization", required=True)
    merge.add_argument("--out", required=True)

    comp = commands.add_parser("compile", help="Compile manifest into simulator artifacts")
    comp.add_argument("manifest")
    comp.add_argument("--dest", required=True)
    comp.add_argument("--allow-draft", action="store_true")

    serve = commands.add_parser("serve", help="Run a locally bound simulation server")
    serve.add_argument("manifest")
    serve.add_argument("--db", default=":memory:")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--allow-draft", action="store_true")

    invoke = commands.add_parser("call", help="Execute a tool against a local SQLite simulation")
    invoke.add_argument("manifest")
    invoke.add_argument("tool")
    invoke.add_argument("--db", default="runtime.sqlite")
    invoke.add_argument("--args", default="{}", help="JSON argument object")
    invoke.add_argument("--role", default="agent")
    invoke.add_argument("--actor", default="cli")

    for name, help_text in [("inspect", "Print the tool registry"),
                             ("validate", "Validate stored state"),
                             ("reset", "Reset local simulated state"),
                             ("snapshot", "Save a portable state snapshot")]:
        cmd = commands.add_parser(name, help=help_text)
        cmd.add_argument("manifest")
        cmd.add_argument("--db", default="runtime.sqlite")
        if name == "snapshot":
            cmd.add_argument("--out", required=True)

    tick = commands.add_parser("tick", help="Advance simulated clock and process integration events")
    tick.add_argument("manifest")
    tick.add_argument("--db", default="runtime.sqlite")
    tick.add_argument("--steps", type=int, default=1)

    scenarios = commands.add_parser("scenarios", help="Generate seeded training/evaluation scenarios")
    scenarios.add_argument("manifest")
    scenarios.add_argument("--count", type=int, default=3)
    scenarios.add_argument("--seed", type=int, default=42)
    scenarios.add_argument("--out", required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "import-openapi":
            draft = import_openapi(args.files, system=args.system, organization=args.organization, base_path=args.base_path)
            Path(args.out).write_text(yaml.safe_dump(draft, sort_keys=False), encoding="utf-8")
            print(json.dumps({"draft": args.out, "resources": list(draft["systems"][args.system]["resources"]),
                              "review_notes": len(draft["review_notes"])}))
            return 0
        if args.command == "merge":
            merged = merge_manifests([load_spec(p) for p in args.manifests], organization=args.organization)
            Path(args.out).write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")
            print(json.dumps({"draft": args.out, "systems": list(merged["systems"])}))
            return 0
        spec = load_spec(args.manifest)
        if args.command == "compile":
            print(json.dumps(compile_spec(spec, args.dest, allow_draft=args.allow_draft), indent=2))
            return 0
        if args.command == "scenarios":
            instances = generate_scenarios(spec, count_per_template=args.count, seed=args.seed)
            write_jsonl(args.out, instances)
            print(json.dumps({"generated": len(instances), "path": args.out}))
            return 0
        if args.command == "serve":
            import uvicorn
            from .api import create_app
            uvicorn.run(create_app(args.manifest, args.db, allow_draft=args.allow_draft), host="127.0.0.1", port=args.port)
            return 0
        engine = SimulationEngine(spec, getattr(args, "db", ":memory:"))
        if args.command == "call":
            result = engine.call(args.tool, json.loads(args.args), role=args.role, actor=args.actor)
        elif args.command == "inspect":
            result = engine.registry
        elif args.command == "validate":
            result = engine.validate()
        elif args.command == "reset":
            engine.reset()
            result = {"ok": True}
        elif args.command == "tick":
            result = engine.advance(args.steps)
        elif args.command == "snapshot":
            Path(args.out).write_text(json.dumps(engine.snapshot(), indent=2) + "\n", encoding="utf-8")
            result = {"saved": args.out}
        else:
            parser.error("Command not implemented")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        engine.close()
        return 0
    except (SpecError, SimError, ValueError, OSError) as exc:
        if isinstance(exc, SimError):
            print(json.dumps(exc.as_dict()), file=sys.stderr)
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
