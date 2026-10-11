"""Offline deterministic smoke run, producing scored trajectories."""
from __future__ import annotations

import json
from pathlib import Path

from orgsim import SimulationEngine, load_spec, SimError
from orgsim.scenarios import generate_scenarios, prepare_scenario, export_episode

BASE = Path(__file__).resolve().parent
SPEC = load_spec(BASE / "demo_org.yaml")


def main():
    sim = SimulationEngine(SPEC)
    cases = generate_scenarios(SPEC, count_per_template=6, seed=42)
    episodes = []
    for case in cases:
        prepared = prepare_scenario(sim, case)
        quote = sim.get("crm", "quotes", "Q-" + case["variables"]["case"])
        caseid = case["variables"]["case"]
        payload = {"id": f"ORD-{caseid}", "account_id": f"A-{caseid}", "quote_id": f"Q-{caseid}",
                   "sku": f"SKU-{caseid}", "quantity": quote["quantity"],
                   "total_cents": quote["quantity"] * quote["unit_price_cents"], "status": "confirmed"}
        if case["id"] == "credit_exception":
            # A rejected attempt is still captured in the training trace.
            try:
                sim.call("erp.orders.create", payload, role="agent", actor="demo_agent")
            except SimError as exc:
                assert exc.code == "POLICY_DENIED"
            payload["approval_id"] = f"APP-{caseid}"
        sim.call("erp.orders.create", payload, role="agent", actor="demo_agent")
        sim.advance(1)
        assert sim.get("crm", "order_mirror", payload["id"])
        assert sim.get("lake", "order_facts", payload["id"]) is None
        sim.advance(1)
        episodes.append(export_episode(sim, case))
        print(f"{case['instance_id']}: {'PASS' if episodes[-1]['score']['passed'] else 'FAIL'} | "
              f"agent calls={len(episodes[-1]['steps'])}")
    sim.close()
    output = BASE / "demo_episodes.jsonl"
    with output.open("w", encoding="utf-8") as f:
        for episode in episodes:
            f.write(json.dumps(episode, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"episodes": len(episodes), "passed": sum(e["score"]["passed"] for e in episodes),
                      "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
