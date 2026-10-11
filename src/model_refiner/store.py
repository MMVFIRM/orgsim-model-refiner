"""SQLite control-plane store for versioned Model Refiner artifacts."""
from __future__ import annotations
from contextlib import contextmanager

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

from .artifact import RefinerArtifact


@dataclass(frozen=True)
class StoredVersion:
    name: str
    version: int
    artifact_id: str
    provider: str
    model: str
    status: str
    created_at: str
    promoted_at: str | None


class ArtifactStore:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self):
        con = sqlite3.connect(self.path, timeout=30.0)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys=ON")
        con.execute("PRAGMA busy_timeout=30000")
        try:
            yield con
            con.commit()
        except BaseException:
            con.rollback()
            raise
        finally:
            con.close()

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                    name TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    artifact_id TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('staged','production','archived')),
                    created_at TEXT NOT NULL,
                    promoted_at TEXT,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY(name, version),
                    UNIQUE(name, artifact_id)
                )
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS ix_artifacts_name_status ON artifacts(name, status)")
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS promotion_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    promoted_at TEXT NOT NULL
                )
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS ix_promotions_name ON promotion_events(name,id)")

    @staticmethod
    def _row(row: sqlite3.Row) -> StoredVersion:
        return StoredVersion(
            name=row["name"], version=int(row["version"]), artifact_id=row["artifact_id"],
            provider=row["provider"], model=row["model"], status=row["status"],
            created_at=row["created_at"], promoted_at=row["promoted_at"],
        )

    def ping(self) -> bool:
        with self._connect() as con:
            return con.execute("SELECT 1").fetchone()[0] == 1

    def add(self, name: str, artifact: RefinerArtifact, *, promote: bool = False) -> StoredVersion:
        artifact.verify()
        clean = name.strip()
        if not clean or len(clean) > 128:
            raise ValueError("Refiner name must be 1-128 characters")
        payload = json.dumps(artifact.to_dict(), ensure_ascii=False, separators=(",", ":"))
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            existing = con.execute(
                "SELECT * FROM artifacts WHERE name=? AND artifact_id=?", (clean, artifact.artifact_id)
            ).fetchone()
            if existing:
                row = self._row(existing)
                if promote and row.status != "production":
                    self._promote_in_tx(con, clean, row.version, now)
                    existing = con.execute("SELECT * FROM artifacts WHERE name=? AND version=?", (clean, row.version)).fetchone()
                    return self._row(existing)
                return row
            next_version = int(con.execute(
                "SELECT COALESCE(MAX(version),0)+1 FROM artifacts WHERE name=?", (clean,)
            ).fetchone()[0])
            con.execute(
                "INSERT INTO artifacts(name,version,artifact_id,provider,model,status,created_at,payload_json) "
                "VALUES(?,?,?,?,?,'staged',?,?)",
                (clean, next_version, artifact.artifact_id, artifact.provider, artifact.model, now, payload),
            )
            if promote:
                self._promote_in_tx(con, clean, next_version, now)
            row = con.execute("SELECT * FROM artifacts WHERE name=? AND version=?", (clean, next_version)).fetchone()
            return self._row(row)

    def _promote_in_tx(self, con: sqlite3.Connection, name: str, version: int, now: str) -> None:
        exists = con.execute("SELECT 1 FROM artifacts WHERE name=? AND version=?", (name, version)).fetchone()
        if not exists:
            raise KeyError(f"Unknown artifact {name!r} version {version}")
        con.execute(
            "UPDATE artifacts SET status='staged' WHERE name=? AND status='production'", (name,)
        )
        con.execute(
            "UPDATE artifacts SET status='production', promoted_at=? WHERE name=? AND version=?",
            (now, name, version),
        )
        con.execute(
            "INSERT INTO promotion_events(name,version,promoted_at) VALUES(?,?,?)",
            (name, version, now),
        )

    def promote(self, name: str, version: int) -> StoredVersion:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            self._promote_in_tx(con, name, int(version), now)
            row = con.execute("SELECT * FROM artifacts WHERE name=? AND version=?", (name, int(version))).fetchone()
            return self._row(row)

    def load(self, name: str, version: int | None = None) -> tuple[StoredVersion, RefinerArtifact]:
        with self._connect() as con:
            if version is None:
                row = con.execute(
                    "SELECT * FROM artifacts WHERE name=? AND status='production' ORDER BY version DESC LIMIT 1", (name,)
                ).fetchone()
                if row is None:
                    raise KeyError(f"No production version for {name!r}")
            else:
                row = con.execute("SELECT * FROM artifacts WHERE name=? AND version=?", (name, int(version))).fetchone()
                if row is None:
                    raise KeyError(f"Unknown artifact {name!r} version {version}")
            return self._row(row), RefinerArtifact.from_dict(json.loads(row["payload_json"]))

    def promotion_history(self, name: str) -> list[dict]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT version,promoted_at FROM promotion_events WHERE name=? ORDER BY id DESC", (name,)
            ).fetchall()
            return [{"version": int(x["version"]), "promoted_at": x["promoted_at"]} for x in rows]

    def list_versions(self, name: str) -> list[StoredVersion]:
        with self._connect() as con:
            rows = con.execute("SELECT * FROM artifacts WHERE name=? ORDER BY version DESC", (name,)).fetchall()
            return [self._row(x) for x in rows]
