"""Seeded scenario templates. No dependence on an LLM or production records."""
from __future__ import annotations

import copy
import json
import random
import re
from typing import Any

from .engine import SimulationEngine, SimError

_PATTERN = re.compile(r"\$\{([a-zA-Z][a-zA-Z_0-9]*)\}")


def _expand(value: Any, variables: dict) -> Any:
    if isinstance(value, str):
        match = _PATTERN.fullmatch(value)
        if match:
            if match.group(1) not in variables:
                raise ValueError(f"Unrecognized scenario variable {match.group(1)}")
            return copy.deepcopy(variables[match.group(1)])
        def substitute(m):
            if m.group(1) not in variables:
                raise ValueError(f"Unrecognized scenario variable {m.group(1)}")
            return str(variables[m.group(1)])
        return _PATTERN.sub(substitute, value)
    if isinstance(value, dict):
        return {k: _expand(v, variables) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v, variables) for v in value]
    return value


def instantiate_scenario(template: dict, *, index: int = 0, seed: int = 1) -> dict:
    rng = random.Random(f"{seed}:{template['id']}:{index}")
    variables = {"case": f"{index:04d}"}
    for name, conf in template.get("variables", {}).items():
        if "int" in conf:
            lo, hi = conf["int"]
            variables[name] = rng.randint(lo, hi)
        elif "choice" in conf:
            variables[name] = rng.choice(conf["choice"])
        elif "literal" in conf:
            variables[name] = copy.deepcopy(conf["literal"])
        else:
            raise ValueError(f"Unknown scenario variable generator: {name}")
    for name, expr in template.get("derived", {}).items():
        if not isinstance(expr, dict) or len(expr) != 1:
            raise ValueError(f"Invalid derived variable: {name}")
        operation, operands = next(iter(expr.items()))
        values = _expand(operands, variables)
        if not isinstance(values, list) or len(values) < 2 or not all(type(x) in (int, float) for x in values):
            raise ValueError(f"Derived variable {name} needs numeric operands")
        if operation == "mul":
            result_value = 1
            for val in values:
                result_value *= val
        elif operation == "add":
            result_value = sum(values)
        elif operation == "sub":
            result_value = values[0] - sum(values[1:])
        else:
            raise ValueError(f"Unsupported derived operation: {operation}")
        variables[name] = result_value
    result = _expand(template, variables)
    result.pop("variables", None)
    result.pop("derived", None)
    result["instance_id"] = f"{template['id']}:{index:04d}:seed{seed}"
    result["variables"] = variables
    return result


def generate_scenarios(spec: dict, count_per_template: int = 1, seed: int = 1) -> list[dict]:
    if count_per_template < 1:
        raise ValueError("count_per_template must be positive")
    return [instantiate_scenario(template, index=i, seed=seed)
            for template in spec.get("scenarios", []) for i in range(count_per_template)]


def prepare_scenario(engine: SimulationEngine, scenario: dict) -> dict:
    """Reset a simulation to a known state then execute deterministic setup calls.

    Agent should receive only the task and tool registry, not seed_calls/checks.
    This function is for the evaluator, not a tool available to the agent.
    """
    engine.reset()
    seeded = 0
    for invocation in scenario.get("seed_calls", []):
        engine.call(invocation["tool"], invocation.get("args", {}),
                    role=invocation.get("role", "admin"), actor="scenario_setup")
        seeded += 1
    for ticks in scenario.get("seed_advance", []):
        engine.advance(ticks)
    return {"instance_id": scenario["instance_id"], "task": scenario["task"],
            "agent_role": scenario.get("agent_role", "agent"),
            "seeded_calls": seeded, "tick": engine.tick}


def score_scenario(engine: SimulationEngine, scenario: dict) -> dict:
    checks = engine.evaluate(scenario["checks"])
    validity = engine.validate()
    return {"instance_id": scenario["instance_id"], "passed": checks["passed"] and validity["valid"],
            "checks": checks["checks"], "validation": validity,
            "audit_events": len(engine.audit()), "pending_events": sum(e["state"] == "pending" for e in engine.events())}


def write_jsonl(path: str, items: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n")


def export_episode(engine: SimulationEngine, scenario: dict) -> dict:
    """Export one scored, reviewable training episode, including failed tool calls."""
    from hashlib import sha256
    report = score_scenario(engine, scenario)
    state = engine.snapshot()
    return {
        "format": "orgsim-episode-v1", "scenario_id": scenario["instance_id"],
        "task": scenario["task"], "role": scenario.get("agent_role", "agent"),
        "steps": engine.tool_trace(exclude_seed=True),
        "score": report, "final_state_sha256": sha256(json.dumps(state["entities"], sort_keys=True).encode()).hexdigest(),
        "simulator_reviewed": engine.spec.get("reviewed", False),
    }
