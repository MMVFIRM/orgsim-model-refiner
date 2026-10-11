"""Tenant-scoped SQLite registry, fenced jobs, feedback review, and audit trail.

Single-host SQLite deployment. All API writes use an authenticated tenant;
production pointers only move after a trusted, internally generated evaluation.
"""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime,timezone
from pathlib import Path
import os
import sqlite3
import time
import uuid

from .artifact import SuiteArtifact
from .common import canonical,digest,nonempty,strict_json


class Conflict(ValueError): pass
class NotFound(KeyError): pass
class LeaseLost(RuntimeError): pass


def now(): return datetime.now(timezone.utc).isoformat()


class SuiteStore:
    def __init__(self,path:str|Path):
        self.path=str(path);Path(self.path).parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.executescript('''
            CREATE TABLE IF NOT EXISTS suite_schema(version INTEGER PRIMARY KEY);
            INSERT OR IGNORE INTO suite_schema VALUES(1);
            CREATE TABLE IF NOT EXISTS suite_versions(
              tenant TEXT NOT NULL,name TEXT NOT NULL,version INTEGER NOT NULL,artifact_id TEXT NOT NULL,
              payload TEXT NOT NULL,created_at TEXT NOT NULL,
              PRIMARY KEY(tenant,name,version),UNIQUE(tenant,name,artifact_id));
            CREATE TABLE IF NOT EXISTS suite_deployments(
              tenant TEXT NOT NULL,name TEXT NOT NULL,version INTEGER NOT NULL,updated_at TEXT NOT NULL,
              PRIMARY KEY(tenant,name),FOREIGN KEY(tenant,name,version) REFERENCES suite_versions(tenant,name,version));
            CREATE TABLE IF NOT EXISTS suite_evaluations(
              id TEXT PRIMARY KEY,tenant TEXT NOT NULL,name TEXT NOT NULL,version INTEGER NOT NULL,
              artifact_id TEXT NOT NULL,report TEXT NOT NULL,report_hash TEXT NOT NULL,passed INTEGER NOT NULL,created_at TEXT NOT NULL,
              FOREIGN KEY(tenant,name,version) REFERENCES suite_versions(tenant,name,version));
            CREATE TABLE IF NOT EXISTS suite_events(
              seq INTEGER PRIMARY KEY AUTOINCREMENT,tenant TEXT NOT NULL,event TEXT NOT NULL,actor TEXT NOT NULL,
              detail TEXT NOT NULL,created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS suite_jobs(
              id TEXT PRIMARY KEY,tenant TEXT NOT NULL,name TEXT NOT NULL,kind TEXT NOT NULL,
              status TEXT NOT NULL,request TEXT NOT NULL,request_hash TEXT NOT NULL,idempotency_key TEXT NOT NULL,
              owner TEXT,lease_until REAL,attempt INTEGER NOT NULL DEFAULT 0,
              progress TEXT,result TEXT,error TEXT,created_at TEXT NOT NULL,updated_at TEXT NOT NULL,
              UNIQUE(tenant,kind,name,idempotency_key));
            CREATE INDEX IF NOT EXISTS suite_job_status ON suite_jobs(status,created_at);
            CREATE TABLE IF NOT EXISTS suite_predictions(
              id TEXT PRIMARY KEY,tenant TEXT NOT NULL,name TEXT NOT NULL,version INTEGER NOT NULL,
              artifact_id TEXT NOT NULL,input_hash TEXT NOT NULL,telemetry TEXT NOT NULL,created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS suite_feedback(
              id TEXT PRIMARY KEY,tenant TEXT NOT NULL,name TEXT NOT NULL,version INTEGER NOT NULL,
              prediction_id TEXT NOT NULL,input TEXT NOT NULL,target TEXT NOT NULL,status TEXT NOT NULL,
              actor TEXT NOT NULL,reviewer TEXT,created_at TEXT NOT NULL,reviewed_at TEXT,
              UNIQUE(tenant,prediction_id));
            ''')
            version=con.execute("SELECT MAX(version) FROM suite_schema").fetchone()[0]
            if version!=1:raise ValueError("Unsupported suite database version")

    @contextmanager
    def connect(self):
        con=sqlite3.connect(self.path,timeout=30)
        con.row_factory=sqlite3.Row
        con.execute("PRAGMA foreign_keys=ON");con.execute("PRAGMA busy_timeout=30000")
        try:
            yield con
            con.commit()
        except BaseException:
            con.rollback();raise
        finally:con.close()

    def _event(self,con,tenant,event,actor,detail):
        con.execute("INSERT INTO suite_events(tenant,event,actor,detail,created_at) VALUES(?,?,?,?,?)",
                    (tenant,event,actor,canonical(detail),now()))

    def events(self,tenant,limit=100):
        with self.connect() as con:
            rows=con.execute("SELECT seq,event,actor,detail,created_at FROM suite_events WHERE tenant=? ORDER BY seq DESC LIMIT ?",(tenant,min(1000,max(1,limit)))).fetchall()
        return [{**dict(x),"detail":strict_json(x["detail"])} for x in rows]

    def _add(self,con,tenant,name,artifact):
        nonempty(tenant,"tenant",128);nonempty(name,"name",128)
        row=con.execute("SELECT version FROM suite_versions WHERE tenant=? AND name=? AND artifact_id=?",(tenant,name,artifact.artifact_id)).fetchone()
        if row:return int(row[0])
        version=con.execute("SELECT COALESCE(MAX(version),0)+1 FROM suite_versions WHERE tenant=? AND name=?",(tenant,name)).fetchone()[0]
        con.execute("INSERT INTO suite_versions VALUES(?,?,?,?,?,?)",(tenant,name,version,artifact.artifact_id,canonical(artifact.to_dict()),now()))
        return int(version)

    def add(self,tenant,name,artifact,actor="operator"):
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            version=self._add(con,tenant,name,artifact)
            self._event(con,tenant,"artifact_staged",actor,{"name":name,"version":version,"artifact_id":artifact.artifact_id})
            return version

    def load(self,tenant,name,version=None):
        with self.connect() as con:
            if version is None:
                p=con.execute("SELECT version FROM suite_deployments WHERE tenant=? AND name=?",(tenant,name)).fetchone()
                if not p:raise NotFound("No production version")
                version=int(p[0])
            row=con.execute("SELECT payload FROM suite_versions WHERE tenant=? AND name=? AND version=?",(tenant,name,version)).fetchone()
            if not row:raise NotFound("Unknown artifact/version")
            return int(version),SuiteArtifact(strict_json(row[0]))

    def versions(self,tenant,name=None):
        with self.connect() as con:
            rows=con.execute("SELECT v.name,v.version,v.artifact_id,v.created_at,CASE WHEN p.version=v.version THEN 'production' ELSE 'staged' END AS status "  # nosec B608 - fixed SQL fragments only; every value is a bound parameter
                             "FROM suite_versions v LEFT JOIN suite_deployments p ON v.tenant=p.tenant AND v.name=p.name WHERE v.tenant=? "+
                             ("AND v.name=? " if name is not None else "")+"ORDER BY v.name,v.version DESC LIMIT 1000",(tenant,name) if name is not None else (tenant,)).fetchall()
            return [dict(x) for x in rows]

    def _evaluation(self,con,tenant,name,version,report):
        row=con.execute("SELECT artifact_id FROM suite_versions WHERE tenant=? AND name=? AND version=?",(tenant,name,version)).fetchone()
        if not row or row[0]!=report.get("artifact_id"):
            raise Conflict("Evaluation is not bound to this artifact")
        eid=str(uuid.uuid4())
        con.execute("INSERT INTO suite_evaluations VALUES(?,?,?,?,?,?,?,?,?)",(eid,tenant,name,version,row[0],canonical(report),digest(report),int(report["gate"]["passed"]),now()))
        return eid

    def record_evaluation(self,tenant,name,version,report,actor="trusted-evaluator"):
        """Trusted local capability only: no HTTP endpoint accepts arbitrary report receipts."""
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            eid=self._evaluation(con,tenant,name,version,report)
            self._event(con,tenant,"evaluation_completed",actor,{"name":name,"version":version,"evaluation_id":eid,"passed":report["gate"]["passed"]})
            return eid

    def latest_report(self,tenant,name,version):
        with self.connect() as con:
            row=con.execute("SELECT * FROM suite_evaluations WHERE tenant=? AND name=? AND version=? ORDER BY created_at DESC,rowid DESC LIMIT 1",(tenant,name,version)).fetchone()
        if not row:raise NotFound("No trusted evaluation for this version")
        report=strict_json(row["report"])
        if digest(report)!=row["report_hash"]:raise Conflict("Evaluation integrity check failed")
        return row["id"],report

    def promote(self,tenant,name,version,*,actor,expected_current:int|None=None):
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            artifact=con.execute("SELECT artifact_id,payload FROM suite_versions WHERE tenant=? AND name=? AND version=?",(tenant,name,version)).fetchone()
            if not artifact:raise NotFound("Unknown version")
            SuiteArtifact(strict_json(artifact["payload"]))
            evaluation=con.execute("SELECT * FROM suite_evaluations WHERE tenant=? AND name=? AND version=? ORDER BY created_at DESC,rowid DESC LIMIT 1",(tenant,name,version)).fetchone()
            if not evaluation:raise Conflict("No trusted evaluation; imports cannot self-certify")
            report=strict_json(evaluation["report"])
            if not evaluation["passed"] or digest(report)!=evaluation["report_hash"] or report["artifact_id"]!=artifact["artifact_id"]:
                raise Conflict("Latest evaluation failed or its integrity binding is invalid")
            current=con.execute("SELECT version FROM suite_deployments WHERE tenant=? AND name=?",(tenant,name)).fetchone()
            previous=int(current[0]) if current else None
            if expected_current!=previous:
                raise Conflict("Production pointer changed; compare-and-swap failed")
            con.execute("INSERT INTO suite_deployments VALUES(?,?,?,?) ON CONFLICT(tenant,name) DO UPDATE SET version=excluded.version,updated_at=excluded.updated_at",(tenant,name,version,now()))
            self._event(con,tenant,"promoted",actor,{"name":name,"from_version":previous,"to_version":version,"evaluation_id":evaluation["id"]})
            return {"name":name,"version":version,"previous_version":previous,"artifact_id":artifact["artifact_id"]}

    @staticmethod
    def _job(row,include_request=False):
        result=dict(row)
        if include_request:result["request"]=strict_json(result["request"])
        else:result.pop("request",None)
        for k in ("progress","result"):
            if result.get(k):result[k]=strict_json(result[k])
        return result

    def submit(self,tenant,name,kind,request,idempotency_key,*,actor="operator",max_active=8):
        nonempty(name,"name",128);nonempty(idempotency_key,"idempotency_key",128)
        if kind not in {"fit","evaluate"}:raise ValueError("Unsupported job kind")
        rhash=digest(request);stamp=now()
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            existing=con.execute("SELECT * FROM suite_jobs WHERE tenant=? AND kind=? AND name=? AND idempotency_key=?",(tenant,kind,name,idempotency_key)).fetchone()
            if existing:
                if existing["request_hash"]!=rhash:raise Conflict("Idempotency key reused with different request")
                return self._job(existing)
            count=con.execute("SELECT COUNT(*) FROM suite_jobs WHERE tenant=? AND status IN ('queued','running')",(tenant,)).fetchone()[0]
            if count>=max_active:raise Conflict("Active job quota exhausted")
            jid=str(uuid.uuid4())
            con.execute("INSERT INTO suite_jobs(id,tenant,name,kind,status,request,request_hash,idempotency_key,created_at,updated_at) VALUES(?,?,?,?,'queued',?,?,?,?,?)",(jid,tenant,name,kind,canonical(request),rhash,idempotency_key,stamp,stamp))
            self._event(con,tenant,"job_submitted",actor,{"job_id":jid,"kind":kind,"name":name})
            return self._job(con.execute("SELECT * FROM suite_jobs WHERE id=?",(jid,)).fetchone())

    def job(self,tenant,jid):
        with self.connect() as con:
            row=con.execute("SELECT * FROM suite_jobs WHERE tenant=? AND id=?",(tenant,jid)).fetchone()
            if not row:raise NotFound("Unknown job")
            return self._job(row)

    def claim(self,lease_seconds=120):
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row=con.execute("SELECT * FROM suite_jobs WHERE status='queued' OR (status='running' AND lease_until<?) ORDER BY created_at,id LIMIT 1",(time.time(),)).fetchone()
            if not row:return None
            # Unique ownership token on EVERY attempt is the fence.
            owner=str(uuid.uuid4())
            con.execute("UPDATE suite_jobs SET status='running',owner=?,lease_until=?,attempt=attempt+1,updated_at=? WHERE id=?",(owner,time.time()+lease_seconds,now(),row["id"]))
            return self._job(con.execute("SELECT * FROM suite_jobs WHERE id=?",(row["id"],)).fetchone(),True)

    def alive(self,jid,owner):
        with self.connect() as con:
            row=con.execute("SELECT 1 FROM suite_jobs WHERE id=? AND status='running' AND owner=? AND lease_until>=?",(jid,owner,time.time())).fetchone()
            return bool(row)

    def heartbeat(self,jid,owner,lease_seconds=120,progress=None):
        with self.connect() as con:
            cursor=con.execute("UPDATE suite_jobs SET lease_until=?,updated_at=?,progress=COALESCE(?,progress) WHERE id=? AND status='running' AND owner=? AND lease_until>=?",(time.time()+lease_seconds,now(),canonical(progress) if progress else None,jid,owner,time.time()))
            return cursor.rowcount==1

    def cancel(self,tenant,jid,actor):
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row=con.execute("SELECT status FROM suite_jobs WHERE tenant=? AND id=?",(tenant,jid)).fetchone()
            if not row:raise NotFound("Unknown job")
            if row[0] not in {"queued","running"}:raise Conflict("Job is already terminal")
            con.execute("UPDATE suite_jobs SET status='cancelled',owner=NULL,lease_until=NULL,updated_at=? WHERE id=?",(now(),jid))
            self._event(con,tenant,"job_cancelled",actor,{"job_id":jid})

    def finish(self,job,artifact:SuiteArtifact|None,report:dict):
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row=con.execute("SELECT * FROM suite_jobs WHERE id=? AND status='running' AND owner=? AND lease_until>=?",(job["id"],job["owner"],time.time())).fetchone()
            if not row:raise LeaseLost("Cannot publish results after cancellation or lease loss")
            version=self._add(con,job["tenant"],job["name"],artifact) if artifact else job["request"]["version"]
            eid=self._evaluation(con,job["tenant"],job["name"],version,report)
            result={"version":version,"artifact_id":report["artifact_id"],"evaluation_id":eid,"gate":report["gate"],
                    "scores":{name:arm["score"] for name,arm in report["arms"].items()}}
            con.execute("UPDATE suite_jobs SET status='succeeded',result=?,owner=NULL,lease_until=NULL,updated_at=? WHERE id=?",(canonical(result),now(),job["id"]))
            self._event(con,job["tenant"],"job_completed","worker",{"job_id":job["id"],"version":version,"gate_passed":report["gate"]["passed"]})
            return result

    def fail(self,job,error_code):
        with self.connect() as con:
            con.execute("UPDATE suite_jobs SET status='failed',error=?,owner=NULL,lease_until=NULL,updated_at=? WHERE id=? AND status='running' AND owner=? AND lease_until>=?",(error_code[:200],now(),job["id"],job["owner"],time.time()))

    def prediction(self,tenant,name,version,artifact_id,input_hash,telemetry):
        pid=str(uuid.uuid4())
        with self.connect() as con:
            con.execute("INSERT INTO suite_predictions VALUES(?,?,?,?,?,?,?,?)",(pid,tenant,name,version,artifact_id,input_hash,canonical(telemetry),now()))
        return pid

    def add_feedback(self,tenant,prediction_id,user_input,target,actor):
        from .common import normalized_input
        nonempty(user_input,"input");nonempty(target,"target")
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            prediction=con.execute("SELECT * FROM suite_predictions WHERE tenant=? AND id=?",(tenant,prediction_id)).fetchone()
            if not prediction:raise NotFound("Unknown prediction")
            if prediction["input_hash"]!=digest(normalized_input(user_input)):
                raise Conflict("Feedback input does not match prediction receipt")
            old=con.execute("SELECT id,input,target FROM suite_feedback WHERE tenant=? AND prediction_id=?",(tenant,prediction_id)).fetchone()
            if old:
                if old["input"]!=user_input or old["target"]!=target:raise Conflict("Prediction already has different feedback")
                return old["id"]
            fid=str(uuid.uuid4())
            con.execute("INSERT INTO suite_feedback VALUES(?,?,?,?,?,?,?,'pending',?,NULL,?,NULL)",(fid,tenant,prediction["name"],prediction["version"],prediction_id,user_input,target,actor,now()))
            self._event(con,tenant,"feedback_submitted",actor,{"feedback_id":fid,"prediction_id":prediction_id})
            return fid

    def feedback(self,tenant,name=None,status=None):
        query="SELECT * FROM suite_feedback WHERE tenant=?";args=[tenant]
        if name is not None:query+=" AND name=?";args.append(name)
        if status is not None:query+=" AND status=?";args.append(status)
        with self.connect() as con:
            return [dict(x) for x in con.execute(query+" ORDER BY created_at LIMIT 1000",args).fetchall()]

    def review_feedback(self,tenant,fid,approve,actor):
        from .tasks import TaskSpec
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row=con.execute("SELECT * FROM suite_feedback WHERE tenant=? AND id=?",(tenant,fid)).fetchone()
            if not row:raise NotFound("Unknown feedback")
            if row["status"]!="pending":raise Conflict("Feedback already reviewed")
            if approve:
                artifact=con.execute("SELECT payload FROM suite_versions WHERE tenant=? AND name=? AND version=?",(tenant,row["name"],row["version"])).fetchone()
                task=TaskSpec.from_dict(strict_json(artifact[0])["task"])
                if task.validate_output(row["target"]):raise ValueError("Corrected target violates task contract")
            status="approved" if approve else "rejected"
            con.execute("UPDATE suite_feedback SET status=?,reviewer=?,reviewed_at=? WHERE id=?",(status,actor,now(),fid))
            self._event(con,tenant,"feedback_reviewed",actor,{"feedback_id":fid,"status":status})

    def backup(self,destination:str|Path):
        target=Path(destination)
        if target.resolve()==Path(self.path).resolve():raise ValueError("Backup cannot overwrite the live database")
        if target.exists():raise FileExistsError("Backup destination already exists")
        target.parent.mkdir(parents=True,exist_ok=True)
        temp=target.with_name(target.name+"."+uuid.uuid4().hex+".tmp")
        try:
            with self.connect() as source:
                sink=sqlite3.connect(temp)
                try:
                    source.backup(sink)
                    if sink.execute("PRAGMA integrity_check").fetchone()[0]!="ok":raise RuntimeError("Backup integrity check failed")
                finally:sink.close()
            # Same-directory hard link is an atomic no-clobber publish.
            # Works on normal POSIX and Windows NTFS local filesystems.
            os.link(temp,target)
            temp.unlink()
        finally:
            if temp.exists():temp.unlink()
        return str(target)

    def integrity(self):
        with self.connect() as con:return con.execute("PRAGMA integrity_check").fetchone()[0]
