"""Durable fit-job queue with SQLite leases and idempotency."""
from __future__ import annotations
from contextlib import contextmanager

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time
import uuid

from .data import split_examples
from .engine import ExperimentConfig
from .fitting import fit_artifact
from .store import ArtifactStore
from .types import Example


class IdempotencyConflict(ValueError):
    pass


@dataclass(frozen=True)
class FitJob:
    job_id: str
    name: str
    status: str
    request_hash: str
    idempotency_key: str | None
    request: dict
    artifact_version: int | None
    artifact_id: str | None
    result: dict | None
    error: str | None
    created_at: str
    updated_at: str
    lease_owner: str | None
    lease_until: float | None


class FitJobStore:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self):
        con = sqlite3.connect(self.path, timeout=30.0)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA busy_timeout=30000")
        try:
            yield con
            con.commit()
        except BaseException:
            con.rollback()
            raise
        finally:
            con.close()

    def _init_db(self):
        with self._connect() as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS fit_jobs (
                    job_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('queued','running','succeeded','failed')),
                    request_hash TEXT NOT NULL,
                    idempotency_key TEXT,
                    request_json TEXT NOT NULL,
                    artifact_version INTEGER,
                    artifact_id TEXT,
                    result_json TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    lease_owner TEXT,
                    lease_until REAL
                )
                """
            )
            con.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_fit_jobs_idempotency "
                "ON fit_jobs(name,idempotency_key) WHERE idempotency_key IS NOT NULL"
            )
            con.execute("CREATE INDEX IF NOT EXISTS ix_fit_jobs_status_created ON fit_jobs(status,created_at)")

    @staticmethod
    def _canonical(obj: dict) -> str:
        return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def _hash(cls, obj: dict) -> str:
        return hashlib.sha256(cls._canonical(obj).encode("utf-8")).hexdigest()

    @staticmethod
    def _job(row: sqlite3.Row) -> FitJob:
        return FitJob(
            job_id=row["job_id"], name=row["name"], status=row["status"],
            request_hash=row["request_hash"], idempotency_key=row["idempotency_key"],
            request=json.loads(row["request_json"]),
            artifact_version=row["artifact_version"], artifact_id=row["artifact_id"],
            result=json.loads(row["result_json"]) if row["result_json"] else None,
            error=row["error"], created_at=row["created_at"], updated_at=row["updated_at"],
            lease_owner=row["lease_owner"], lease_until=row["lease_until"],
        )

    def submit(self, name: str, request: dict, idempotency_key: str | None = None) -> FitJob:
        clean_name = name.strip()
        if not clean_name or len(clean_name) > 128:
            raise ValueError("Refiner name must be 1-128 characters")
        if idempotency_key is not None:
            idempotency_key = idempotency_key.strip()
            if not idempotency_key or len(idempotency_key) > 128:
                raise ValueError("Idempotency-Key must be 1-128 characters")
        request_hash = self._hash(request)
        request_json = self._canonical(request)
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            if idempotency_key is not None:
                row = con.execute(
                    "SELECT * FROM fit_jobs WHERE name=? AND idempotency_key=?",
                    (clean_name, idempotency_key),
                ).fetchone()
                if row is not None:
                    job = self._job(row)
                    if job.request_hash != request_hash:
                        raise IdempotencyConflict("Idempotency-Key was already used with a different fit request")
                    return job
            job_id = str(uuid.uuid4())
            con.execute(
                "INSERT INTO fit_jobs(job_id,name,status,request_hash,idempotency_key,request_json,created_at,updated_at) "
                "VALUES(?,?,'queued',?,?,?,?,?)",
                (job_id, clean_name, request_hash, idempotency_key, request_json, now, now),
            )
            row = con.execute("SELECT * FROM fit_jobs WHERE job_id=?", (job_id,)).fetchone()
            return self._job(row)

    def get(self, job_id: str) -> FitJob:
        with self._connect() as con:
            row = con.execute("SELECT * FROM fit_jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(f"Unknown fit job {job_id!r}")
            return self._job(row)

    def status_counts(self) -> dict[str, int]:
        counts = {"queued": 0, "running": 0, "succeeded": 0, "failed": 0}
        with self._connect() as con:
            for row in con.execute("SELECT status,COUNT(*) AS n FROM fit_jobs GROUP BY status").fetchall():
                counts[row["status"]] = int(row["n"])
        return counts

    def claim(self, worker_id: str, lease_seconds: float = 300.0) -> FitJob | None:
        now_epoch = time.time()
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            # Jobs abandoned by a dead worker become eligible only after their lease expires.
            con.execute(
                "UPDATE fit_jobs SET status='queued', lease_owner=NULL, lease_until=NULL, updated_at=? "
                "WHERE status='running' AND lease_until IS NOT NULL AND lease_until < ?",
                (now_iso, now_epoch),
            )
            row = con.execute(
                "SELECT * FROM fit_jobs WHERE status='queued' ORDER BY created_at, job_id LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            lease_until = now_epoch + lease_seconds
            con.execute(
                "UPDATE fit_jobs SET status='running', lease_owner=?, lease_until=?, updated_at=? "
                "WHERE job_id=? AND status='queued'",
                (worker_id, lease_until, now_iso, row["job_id"]),
            )
            row = con.execute("SELECT * FROM fit_jobs WHERE job_id=?", (row["job_id"],)).fetchone()
            return self._job(row)

    def heartbeat(self, job_id: str, worker_id: str, lease_seconds: float) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as con:
            cur = con.execute(
                "UPDATE fit_jobs SET lease_until=?, updated_at=? WHERE job_id=? AND status='running' AND lease_owner=?",
                (time.time() + lease_seconds, now, job_id, worker_id),
            )
            return cur.rowcount == 1

    def succeed(self, job_id: str, worker_id: str, *, artifact_version: int, artifact_id: str, result: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as con:
            cur = con.execute(
                "UPDATE fit_jobs SET status='succeeded', artifact_version=?, artifact_id=?, result_json=?, error=NULL, "
                "updated_at=?, lease_owner=NULL, lease_until=NULL WHERE job_id=? AND status='running' AND lease_owner=?",
                (artifact_version, artifact_id, self._canonical(result), now, job_id, worker_id),
            )
            if cur.rowcount != 1:
                raise RuntimeError("Fit job lease was lost before completion")

    def fail(self, job_id: str, worker_id: str, error: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as con:
            con.execute(
                "UPDATE fit_jobs SET status='failed', error=?, updated_at=?, lease_owner=NULL, lease_until=NULL "
                "WHERE job_id=? AND status='running' AND lease_owner=?",
                (error[:4000], now, job_id, worker_id),
            )


class FitWorker:
    """One durable worker. Multiple service processes may run workers against the same DB safely."""

    def __init__(
        self,
        *,
        jobs: FitJobStore,
        artifacts: ArtifactStore,
        backend,
        provider: str,
        lease_seconds: float = 300.0,
        poll_seconds: float = 0.25,
    ):
        self.jobs = jobs
        self.artifacts = artifacts
        self.backend = backend
        self.provider = provider
        self.lease_seconds = max(30.0, float(lease_seconds))
        self.poll_seconds = poll_seconds
        self.worker_id = f"worker-{uuid.uuid4()}"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name=self.worker_id, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    def _loop(self) -> None:
        while not self._stop.is_set():
            job = self.jobs.claim(self.worker_id, self.lease_seconds)
            if job is None:
                self._stop.wait(self.poll_seconds)
                continue
            self._run_job(job)

    def _run_job(self, job: FitJob) -> None:
        beat_stop = threading.Event()

        def beat():
            interval = max(5.0, self.lease_seconds / 3.0)
            while not beat_stop.wait(interval):
                if not self.jobs.heartbeat(job.job_id, self.worker_id, self.lease_seconds):
                    return

        heart = threading.Thread(target=beat, name=f"heartbeat-{job.job_id}", daemon=True)
        heart.start()
        try:
            req = job.request
            examples = [
                Example(
                    id=str(x.get("id", i + 1)), input=str(x["input"]), target=str(x["target"]),
                    split=x.get("split"), metadata=dict(x.get("metadata", {})),
                )
                for i, x in enumerate(req["examples"])
            ]
            config = ExperimentConfig(
                model=req["model"], optimizer_model=req.get("optimizer_model"),
                base_instruction=req["task"], k_values=tuple(req["k_values"]), seed=int(req["seed"]),
                max_rule_errors=int(req["max_rule_errors"]), max_rules=int(req["max_rules"]),
                max_prompt_chars=int(req["max_prompt_chars"]),
            )
            result = fit_artifact(
                backend=self.backend, provider=self.provider, examples=examples, config=config,
                max_calls=int(req["max_calls"]),
            )
            stored = self.artifacts.add(job.name, result.artifact, promote=False)
            summary = {
                "provider": result.artifact.provider,
                "model": result.artifact.model,
                "artifact_id": result.artifact.artifact_id,
                "version": stored.version,
                "status": stored.status,
                "estimated_call_upper_bound": result.estimated_call_upper_bound,
                "results": {x.name: x.accuracy for x in result.report.arms},
                "selected_refined_k": result.report.selected_refined_k,
            }
            self.jobs.succeed(
                job.job_id, self.worker_id, artifact_version=stored.version,
                artifact_id=result.artifact.artifact_id, result=summary,
            )
        except Exception as exc:
            self.jobs.fail(job.job_id, self.worker_id, f"{type(exc).__name__}: {exc}")
        finally:
            beat_stop.set()
            heart.join(timeout=1.0)
