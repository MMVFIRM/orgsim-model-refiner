"""Local training/test API. Not authenticated production software."""
from __future__ import annotations

import os
import secrets
from pathlib import Path

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field

from .engine import SimulationEngine, SimError
from .spec import load_spec


class CallPayload(BaseModel):
    tool: str
    args: dict = Field(default_factory=dict)
    session: str


class SessionPayload(BaseModel):
    role: str
    actor: str = "training_agent"


class TickPayload(BaseModel):
    steps: int = 1


class EvalPayload(BaseModel):
    checks: list[dict]


def create_app(manifest: str | Path, db_path: str | Path = ":memory:", *, allow_draft: bool = False) -> FastAPI:
    engine = SimulationEngine(load_spec(manifest), db_path, allow_draft=allow_draft)
    app = FastAPI(title="OrgSim Environment", version="0.2.0",
                  description="Experimental isolated simulation. Role assertions must not be trusted on a network.")
    app.state.engine = engine
    app.state.sessions = {}

    @app.exception_handler(SimError)
    async def sim_error_handler(_request, exc: SimError):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=exc.status, content=exc.as_dict())

    def admin_guard(token: str | None):
        expected = os.getenv("ORGSIM_ADMIN_TOKEN")
        # Administrative API is disabled unless explicitly configured.
        if not expected or token != expected:
            raise HTTPException(status_code=403, detail="Admin endpoint disabled or invalid token")

    @app.get("/health")
    def health():
        return {"ok": True, "organization": engine.spec["organization"], "tick": engine.tick,
                "reviewed": engine.spec.get("reviewed", False)}

    @app.get("/v1/registry")
    def registry():
        return engine.registry

    @app.post("/v1/admin/session")
    def create_session(req: SessionPayload, x_orgsim_admin_token: str | None = Header(default=None)):
        admin_guard(x_orgsim_admin_token)
        allowed_roles = {role for tool in engine.registry["tools"] for role in tool["roles"]}
        if req.role not in allowed_roles:
            raise HTTPException(status_code=422, detail="Unknown simulation role")
        token = secrets.token_urlsafe(32)
        app.state.sessions[token] = {"role": req.role, "actor": req.actor}
        return {"session": token, "role": req.role, "actor": req.actor}

    @app.post("/v1/call")
    def call(req: CallPayload):
        identity = app.state.sessions.get(req.session)
        if identity is None:
            raise HTTPException(status_code=401, detail="Invalid simulation session")
        return engine.call(req.tool, req.args, **identity)

    @app.post("/v1/clock/advance")
    def advance(req: TickPayload, x_orgsim_admin_token: str | None = Header(default=None)):
        admin_guard(x_orgsim_admin_token)
        # Caller can progress *simulated* time only, never real time.
        return engine.advance(req.steps)

    @app.post("/v1/evaluate")
    def evaluate(req: EvalPayload, x_orgsim_admin_token: str | None = Header(default=None)):
        admin_guard(x_orgsim_admin_token)
        return engine.evaluate(req.checks)

    @app.get("/v1/admin/validate")
    def validate(x_orgsim_admin_token: str | None = Header(default=None)):
        admin_guard(x_orgsim_admin_token)
        return engine.validate()

    @app.get("/v1/admin/events")
    def events(x_orgsim_admin_token: str | None = Header(default=None)):
        admin_guard(x_orgsim_admin_token)
        return engine.events()

    @app.get("/v1/admin/snapshot")
    def snapshot(x_orgsim_admin_token: str | None = Header(default=None)):
        admin_guard(x_orgsim_admin_token)
        return engine.snapshot()

    @app.post("/v1/admin/reset")
    def reset(x_orgsim_admin_token: str | None = Header(default=None)):
        admin_guard(x_orgsim_admin_token)
        engine.reset()
        app.state.sessions.clear()
        return {"ok": True, "tick": engine.tick}

    return app
