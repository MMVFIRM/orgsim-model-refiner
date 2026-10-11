"""Compile a reviewed organization manifest to portable simulator artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from .spec import validate_spec, build_registry


def compile_spec(spec: dict, dest: str | Path, *, allow_draft: bool = False) -> dict:
    validate_spec(spec, allow_draft=allow_draft)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    registry = build_registry(spec)
    artifacts = {
        "manifest.json": spec,
        "tool_registry.json": registry,
        "environment_graph.json": {"organization": spec["organization"], "nodes": [
            f"{sys}.{resource}" for sys, val in spec["systems"].items() for resource in val["resources"]],
            "edges": registry["dependency_edges"]},
        "build_report.json": {
            "organization": spec["organization"], "reviewed": bool(spec.get("reviewed")),
            "assurance": "policy_reviewed" if spec.get("reviewed") else "contract_only_unreviewed",
            "tool_count": len(registry["tools"]), "resource_count": sum(len(s["resources"]) for s in spec["systems"].values()),
            "integration_count": len(spec.get("integrations", [])),
            "workflow_count": sum(len(s.get("workflows", {})) for s in spec["systems"].values()),
            "lookup_count": sum(len(s.get("lookups", {})) for s in spec["systems"].values()),
            "notes": spec.get("review_notes", []),
            "limits": ["Only configured policy invariants are enforced", "Tool roles are untrusted caller-supplied lab identities",
                       "Unmodeled custom operations remain unsupported until a reviewed declarative action is supplied", "No live system I/O or organization data cloning"]},
    }
    for filename, content in artifacts.items():
        (dest / filename).write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (dest / "run_server.py").write_text('''"""Run only on a local developer machine: python run_server.py"""\nfrom pathlib import Path\nimport uvicorn\nfrom orgsim.api import create_app\napp = create_app(Path(__file__).parent / "manifest.json", Path(__file__).parent / "runtime.sqlite")\nif __name__ == "__main__":\n    uvicorn.run(app, host="127.0.0.1", port=8765)\n''', encoding="utf-8")
    return artifacts["build_report.json"]


def merge_manifests(manifests: list[dict], *, organization: str) -> dict:
    """Merge independently discovered system drafts without losing review provenance.

    Deliberately does not synthesize cross-system policies: that step needs review.
    """
    if not manifests:
        raise ValueError("At least one manifest is required")
    import copy
    merged = {"version": "0.1", "organization": organization, "reviewed": False,
              "systems": {}, "integrations": [], "scenarios": [], "review_notes": []}
    for entry in manifests:
        if entry.get("version") != "0.1":
            raise ValueError("Cannot merge unsupported manifest version")
        for system, conf in entry.get("systems", {}).items():
            if system in merged["systems"]:
                raise ValueError(f"Duplicate system ID: {system}. Merge resource definitions manually.")
            merged["systems"][system] = copy.deepcopy(conf)
        merged["integrations"].extend(copy.deepcopy(entry.get("integrations", [])))
        merged["scenarios"].extend(copy.deepcopy(entry.get("scenarios", [])))
        merged["review_notes"].extend(copy.deepcopy(entry.get("review_notes", [])))
    merged["review_notes"].append("REVIEW REQUIRED: Model identity mapping, cross-system policy invariants and event integrations; nothing was inferred.")
    return merged
