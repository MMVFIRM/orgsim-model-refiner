from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from orgsim.compiler import compile_spec
from orgsim.engine import SimulationEngine, SimError
from orgsim.openapi import import_openapi
from orgsim.scenarios import generate_scenarios, prepare_scenario, score_scenario, export_episode
from orgsim.api import create_app
from orgsim.spec import load_spec, validate_spec, SpecError

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
SPEC = load_spec(EXAMPLES / "demo_org.yaml")


@pytest.fixture
def sim():
    eng = SimulationEngine(SPEC)
    yield eng
    eng.close()


def setup_simple(sim, *, limit=50000, inventory=10, status="accepted"):
    sim.call("crm.accounts.create", {"id": "A1", "name": "Test Customer", "credit_limit_cents": limit}, role="admin")
    sim.call("erp.inventory.create", {"id": "SKU1", "description": "Item", "available": inventory, "unit_price_cents": 9000}, role="admin")
    sim.call("crm.quotes.create", {"id": "Q1", "account_id": "A1", "sku": "SKU1", "quantity": 3,
                                   "unit_price_cents": 9000, "status": status}, role="admin")


def order(**overrides):
    return {**{"id": "ORD1", "account_id": "A1", "quote_id": "Q1", "sku": "SKU1",
               "quantity": 3, "total_cents": 27000, "status": "confirmed"}, **overrides}


def test_compile_registry_and_deps(tmp_path):
    report = compile_spec(SPEC, tmp_path)
    assert report["tool_count"] == 23
    assert report["resource_count"] == 7
    assert (tmp_path / "run_server.py").is_file()
    graph = json.loads((tmp_path / "environment_graph.json").read_text())
    assert any(e["kind"] == "event" and e["to"] == "lake.order_facts" for e in graph["edges"])
    assert any(e["kind"] == "foreign_key" and e["from"] == "erp.orders" for e in graph["edges"])


def test_schema_validation(sim):
    with pytest.raises(SimError, match="'name' is a required property") as exc:
        sim.call("crm.accounts.create", {"id": "A1", "credit_limit_cents": 10}, role="admin")
    assert exc.value.code == "SCHEMA_VIOLATION"
    with pytest.raises(SimError, match="Additional properties"):
        sim.call("crm.accounts.create", {"id": "A1", "name": "Me", "credit_limit_cents": 10, "secret": "x"}, role="admin")
    assert not sim.list_records("crm", "accounts")


def test_permission_denied_is_logged(sim):
    with pytest.raises(SimError) as e:
        sim.call("erp.inventory.create", {"id": "X", "description": "X", "available": 1, "unit_price_cents": 10}, role="agent")
    assert e.value.code == "FORBIDDEN"
    assert sim.tool_trace()[-1]["ok"] is False


def test_fkey_and_uniqueness(sim):
    with pytest.raises(SimError) as exc:
        sim.call("crm.quotes.create", {"id": "Q1", "account_id": "MISSING", "sku": "S", "quantity": 1,
                                       "unit_price_cents": 100, "status": "accepted"}, role="admin")
    assert exc.value.code == "FOREIGN_KEY"
    sim.call("crm.accounts.create", {"id": "A1", "name": "Acme", "credit_limit_cents": 10000}, role="admin")
    with pytest.raises(SimError) as exc:
        sim.call("crm.accounts.create", {"id": "A2", "name": "Acme", "credit_limit_cents": 3000}, role="admin")
    assert exc.value.code == "UNIQUE_VIOLATION"


def test_successful_order_reserves_inventory(sim):
    setup_simple(sim)
    response = sim.call("erp.orders.create", order(), role="agent")
    assert response["result"]["total_cents"] == 27000
    assert sim.get("erp", "inventory", "SKU1")["available"] == 7
    assert len(sim.events()) == 2
    assert sim.validate()["valid"]


def test_rejected_credit_no_partial_side_effects(sim):
    setup_simple(sim, limit=10000)
    initial = sim.snapshot()
    with pytest.raises(SimError) as exc:
        sim.call("erp.orders.create", order(), role="agent", actor="test_agent")
    assert exc.value.code == "POLICY_DENIED"
    assert sim.get("erp", "orders", "ORD1") is None
    assert sim.get("erp", "inventory", "SKU1")["available"] == 10
    assert sim.events() == []
    assert sim.snapshot()["entities"] == initial["entities"]
    assert any(not step["ok"] and step["output"]["error"]["code"] == "POLICY_DENIED"
               for step in sim.tool_trace())


def test_approved_credit_exception(sim):
    setup_simple(sim, limit=10000)
    sim.call("approvals.requests.create", {"id": "APP1", "account_id": "A1", "quote_id": "Q1",
                                           "amount_cents": 27000, "status": "pending"}, role="sales")
    with pytest.raises(SimError) as e:
        sim.call("erp.orders.create", order(approval_id="APP1"), role="agent")
    assert e.value.code == "POLICY_DENIED"
    with pytest.raises(SimError) as e:
        sim.call("approvals.requests.update", {"id": "APP1", "patch": {"status": "approved"}}, role="agent")
    assert e.value.code == "FORBIDDEN"
    sim.call("approvals.requests.update", {"id": "APP1", "patch": {"status": "approved"}}, role="finance")
    sim.call("erp.orders.create", order(approval_id="APP1"), role="agent")
    assert sim.get("erp", "inventory", "SKU1")["available"] == 7


def test_approval_wrong_quote_fails(sim):
    setup_simple(sim, limit=10000)
    sim.call("crm.quotes.create", {"id": "Q2", "account_id": "A1", "sku": "SKU1", "quantity": 3,
                                   "unit_price_cents": 9000, "status": "accepted"}, role="admin")
    sim.call("approvals.requests.create", {"id": "APP1", "account_id": "A1", "quote_id": "Q2",
                                           "amount_cents": 27000, "status": "approved"}, role="admin")
    with pytest.raises(SimError) as e:
        sim.call("erp.orders.create", order(approval_id="APP1"), role="agent")
    assert e.value.code == "POLICY_DENIED"


@pytest.mark.parametrize("change", [{"quantity": 4}, {"total_cents": 28000}, {"sku": "NOPE"},
                                     {"account_id": "NOPE"}, {"status": "draft"}])
def test_bad_order_is_rejected_and_atomic(sim, change):
    setup_simple(sim)
    with pytest.raises(SimError):
        sim.call("erp.orders.create", order(**change), role="agent")
    assert sim.get("erp", "inventory", "SKU1")["available"] == 10
    assert sim.get("erp", "orders", "ORD1") is None


def test_stock_unavailable_denied(sim):
    setup_simple(sim, inventory=2)
    with pytest.raises(SimError) as exc:
        sim.call("erp.orders.create", order(), role="agent")
    assert exc.value.code == "POLICY_DENIED"
    assert sim.get("erp", "inventory", "SKU1")["available"] == 2


def test_async_projection_delivery(sim):
    setup_simple(sim)
    sim.call("erp.orders.create", order(), role="agent")
    assert sim.get("crm", "order_mirror", "ORD1") is None
    assert sim.get("lake", "order_facts", "ORD1") is None
    first = sim.advance(1)
    assert first["delivered"] == 1
    assert sim.get("crm", "order_mirror", "ORD1") is not None
    assert sim.get("lake", "order_facts", "ORD1") is None
    second = sim.advance(1)
    assert second["delivered"] == 1
    assert sim.get("lake", "order_facts", "ORD1")["total_cents"] == 27000
    assert sim.advance(10)["delivered"] == 0
    assert all(ev["state"] == "delivered" for ev in sim.events())


def test_failed_integration_retried_and_dead_letter():
    spec = copy.deepcopy(SPEC)
    spec["integrations"][1]["mapping"]["status"] = {"literal": "INVALID"}
    eng = SimulationEngine(spec)
    try:
        setup_simple(eng)
        eng.call("erp.orders.create", order(), role="agent")
        eng.advance(2)
        assert eng.get("crm", "order_mirror", "ORD1") is not None
        assert eng.get("lake", "order_facts", "ORD1") is None
        eng.advance(1)
        eng.advance(1)
        assert any(e["state"] == "dead_letter" and e["attempts"] == 3 for e in eng.events())
        assert eng.validate()["valid"]
    finally:
        eng.close()


def test_update_id_immutable(sim):
    setup_simple(sim)
    with pytest.raises(SimError) as exc:
        sim.call("crm.accounts.update", {"id": "A1", "patch": {"id": "A2"}}, role="admin")
    assert exc.value.code == "INVALID_ARGUMENTS"
    assert sim.get("crm", "accounts", "A1")


def test_disallowed_delete(sim):
    setup_simple(sim)
    with pytest.raises(SimError) as exc:
        sim.call("crm.accounts.delete", {"id": "A1"}, role="admin")
    assert exc.value.code == "FORBIDDEN"


def test_autoids_rollback_sequence(sim):
    with pytest.raises(SimError):
        sim.call("crm.accounts.create", {"name": "", "credit_limit_cents": 200}, role="admin")
    res = sim.call("crm.accounts.create", {"name": "Test", "credit_limit_cents": 200}, role="admin")
    assert res["result"]["id"] == "crm_accounts_000001"


def test_restore_snapshot_and_replay(sim):
    setup_simple(sim)
    snapshot = sim.snapshot()
    sim.call("erp.orders.create", order(), role="agent")
    sim.advance(5)
    assert sim.get("lake", "order_facts", "ORD1")
    sim.restore(snapshot)
    assert sim.get("erp", "orders", "ORD1") is None
    assert sim.events() == []
    sim.call("erp.orders.create", order(), role="agent")
    sim.advance(5)
    assert sim.validate()["valid"]
    assert sim.get("lake", "order_facts", "ORD1")["total_cents"] == 27000


def test_reset_and_fresh_instance(sim):
    setup_simple(sim)
    sim.reset()
    assert sim.tick == 0
    assert sim.validate()["records"] == 0
    assert len(sim.tool_trace()) == 0


def test_unsupported_expressions_cannot_compile():
    spec = copy.deepcopy(SPEC)
    spec["systems"]["erp"]["resources"]["orders"]["guards"]["create"] = [{"python": "__import__('os').system('echo hi')"}]
    with pytest.raises(SpecError, match="unsupported condition"):
        validate_spec(spec)


def test_unreviewed_manifest_requires_explicit_override(tmp_path):
    spec = copy.deepcopy(SPEC)
    spec["reviewed"] = False
    with pytest.raises(SpecError):
        compile_spec(spec, tmp_path / "reject")
    report = compile_spec(spec, tmp_path / "draft", allow_draft=True)
    assert report["assurance"] == "contract_only_unreviewed"


def test_openapi_import_is_not_falsely_approved(tmp_path):
    draft = import_openapi([EXAMPLES / "crm_openapi.yaml"], system="crm", organization="TestCo")
    assert draft["reviewed"] is False
    assert "accounts" in draft["systems"]["crm"]["resources"]
    assert any("/approve" in note for note in draft["review_notes"])
    with pytest.raises(SpecError):
        compile_spec(draft, tmp_path)
    compile_spec(draft, tmp_path, allow_draft=True)


def test_scenario_generation_deterministic_and_valid(sim):
    a = generate_scenarios(SPEC, count_per_template=5, seed=7)
    b = generate_scenarios(SPEC, count_per_template=5, seed=7)
    assert a == b
    assert len(a) == 10
    one = a[0]
    info = prepare_scenario(sim, one)
    assert info["seeded_calls"] == 3
    quote = sim.get("crm", "quotes", "Q-0000")
    assert one["variables"]["total"] == quote["quantity"] * quote["unit_price_cents"]
    assert one["variables"]["expected_remaining"] == 20 - quote["quantity"]
    assert not score_scenario(sim, one)["passed"]
    sim.call("erp.orders.create", {"id": "ORD-0000", "account_id": "A-0000", "quote_id": "Q-0000",
                                   "sku": "SKU-0000", "quantity": quote["quantity"],
                                   "total_cents": one["variables"]["total"], "status": "confirmed"}, role="agent")
    sim.advance(2)
    report = score_scenario(sim, one)
    assert report["passed"], report
    episode = export_episode(sim, one)
    assert episode["score"]["passed"]
    assert episode["steps"][0]["tool"] == "erp.orders.create"
    assert all(c["passed"] for c in episode["score"]["checks"])


def test_credit_scenario_rejected_then_completed(sim):
    scenario = generate_scenarios(SPEC, count_per_template=1, seed=1)[1]
    prepare_scenario(sim, scenario)
    with pytest.raises(SimError) as exc:
        sim.call("erp.orders.create", {"id": "ORD-0000", "account_id": "A-0000", "quote_id": "Q-0000",
                                        "sku": "SKU-0000", "quantity": 3, "total_cents": 27000,
                                        "status": "confirmed"}, role="agent")
    assert exc.value.code == "POLICY_DENIED"
    sim.call("erp.orders.create", {"id": "ORD-0000", "account_id": "A-0000", "quote_id": "Q-0000",
                                    "sku": "SKU-0000", "quantity": 3, "total_cents": 27000,
                                    "status": "confirmed", "approval_id": "APP-0000"}, role="agent")
    sim.advance(2)
    assert score_scenario(sim, scenario)["passed"]
    assert any(not s["ok"] for s in export_episode(sim, scenario)["steps"])


def test_sqlite_persists_across_restarts(tmp_path):
    path = tmp_path / "sim.sqlite"
    e1 = SimulationEngine(SPEC, path)
    setup_simple(e1)
    e1.close()
    e2 = SimulationEngine(SPEC, path)
    try:
        assert e2.get("crm", "accounts", "A1")["name"] == "Test Customer"
        assert e2.validate()["valid"]
    finally:
        e2.close()


def test_http_api_registry_and_error_codes(tmp_path, monkeypatch):
    app = create_app(EXAMPLES / "demo_org.yaml", tmp_path / "app.sqlite")
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert len(client.get("/v1/registry").json()["tools"]) == 23
    assert client.post("/v1/admin/reset").status_code == 403
    monkeypatch.setenv("ORGSIM_ADMIN_TOKEN", "dev-token")
    assert client.get("/v1/admin/validate", headers={"x-orgsim-admin-token": "dev-token"}).status_code == 200
    token = client.post("/v1/admin/session", json={"role": "agent", "actor": "eval-agent"},
                        headers={"x-orgsim-admin-token": "dev-token"}).json()["session"]
    assert client.post("/v1/call", json={"tool": "erp.orders.create", "session": "invalid", "args": order()}).status_code == 401
    response = client.post("/v1/call", json={"tool": "erp.orders.create", "session": token, "args": order()})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "FOREIGN_KEY"
    assert client.post("/v1/clock/advance", json={"steps": 1}).status_code == 403


def test_evaluation_is_readonly(sim):
    setup_simple(sim)
    before = sim.snapshot()
    result = sim.evaluate([{"compare": {"left": {"count": {"system": "crm", "resource": "accounts"}},
                                         "op": "eq", "right": {"literal": 1}}}])
    assert result["passed"]
    assert sim.snapshot() == before


def test_failed_policy_tries_dont_leave_events(sim):
    setup_simple(sim, status="draft")
    with pytest.raises(SimError):
        sim.call("erp.orders.create", order(), role="agent")
    assert not sim.events()
    assert len(sim.audit()) == 3


def test_invalid_ref_compiler_fails():
    bad = copy.deepcopy(SPEC)
    bad["systems"]["erp"]["resources"]["orders"]["foreign_keys"][0]["resource"] = "oops"
    with pytest.raises(SpecError):
        validate_spec(bad)


def test_manifest_db_mismatch_rejected(tmp_path):
    path = tmp_path / "persist.sqlite"
    eng = SimulationEngine(SPEC, path)
    eng.close()
    altered = copy.deepcopy(SPEC)
    altered["organization"] += " change"
    with pytest.raises(SimError) as exc:
        SimulationEngine(altered, path)
    assert exc.value.code == "MANIFEST_MISMATCH"


def test_merge_imported_organization_drafts():
    from orgsim.compiler import merge_manifests
    crm = import_openapi([EXAMPLES / "crm_openapi.yaml"], system="crm")
    erp = import_openapi([EXAMPLES / "erp_openapi.yaml"], system="erp")
    combined = merge_manifests([crm, erp], organization="New Co")
    assert set(combined["systems"]) == {"crm", "erp"}
    assert combined["reviewed"] is False
    assert any("cross-system" in note for note in combined["review_notes"])
    with pytest.raises(ValueError, match="Duplicate system"):
        merge_manifests([crm, crm], organization="X")


def test_concurrent_competing_orders_cannot_oversell(sim):
    from concurrent.futures import ThreadPoolExecutor
    setup_simple(sim, inventory=5)
    sim.call("crm.quotes.create", {"id": "Q2", "account_id": "A1", "sku": "SKU1", "quantity": 3,
                                   "unit_price_cents": 9000, "status": "accepted"}, role="admin")

    def place(suffix, quote):
        try:
            sim.call("erp.orders.create", order(id=suffix, quote_id=quote), role="agent")
            return "ok"
        except SimError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda args: place(*args), [("ORD1", "Q1"), ("ORD2", "Q2")]))
    assert sorted(results) == ["POLICY_DENIED", "ok"]
    assert sim.get("erp", "inventory", "SKU1")["available"] == 2
    assert len(sim.list_records("erp", "orders")) == 1
    assert sim.validate()["valid"]


def test_openapi_common_prefix_import():
    from orgsim.openapi import import_openapi
    from tempfile import TemporaryDirectory
    import yaml
    with TemporaryDirectory() as temp:
        source = load_spec(EXAMPLES / "crm_openapi.yaml")
        source["paths"] = {"/api/v2" + path: op for path, op in source["paths"].items()}
        target = Path(temp) / "prefixed.yaml"
        target.write_text(yaml.safe_dump(source), encoding="utf-8")
        draft = import_openapi([target], system="crm")
        assert "accounts" in draft["systems"]["crm"]["resources"]
        assert "api" not in draft["systems"]["crm"]["resources"]
        explicit = import_openapi([target], system="crm", base_path="/api/v2")
        assert "accounts" in explicit["systems"]["crm"]["resources"]
