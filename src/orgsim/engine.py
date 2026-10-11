"""Stateful SQLite simulator. No production connectors, arbitrary code, or live system writes.

Simulated time advances only by explicit ticks. State writes, ledger adjustments,
audit records and event enqueuing are transactional.  Isolation: one SQLite
transaction + one in-process reentrant lock per simulator instance.
"""
from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import copy
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any

from jsonschema import Draft202012Validator
from .spec import validate_spec, build_registry


class SimError(Exception):
    def __init__(self, code: str, message: str, *, status: int = 422, diagnostics: dict | None = None):
        super().__init__(message)
        self.code = code
        self.status = status
        self.message = message
        # Safe for benchmark logs: field names and error categories only, never values.
        self.diagnostics = diagnostics or {}

    def as_dict(self) -> dict:
        return {"ok": False, "error": {"code": self.code, "message": self.message}}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _at_path(root: Any, dotted: str) -> Any:
    cur = root
    for part in dotted.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list) and part.isdecimal() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return None
    return cur


class SimulationEngine:
    def __init__(self, spec: dict, db_path: str | Path = ":memory:", *, allow_draft: bool = False):
        validate_spec(spec, allow_draft=allow_draft)
        self.spec = copy.deepcopy(spec)
        self.spec_hash = sha256(_json(self.spec).encode("utf-8")).hexdigest()
        self.registry = build_registry(self.spec)
        self.path = str(db_path)
        self._lock = threading.RLock()
        self.db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None, timeout=20)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA busy_timeout=20000")
        if self.path != ":memory:":
            self.db.execute("PRAGMA journal_mode=WAL")
        self._init_tables()

    def close(self) -> None:
        self.db.close()

    def _init_tables(self) -> None:
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS entities (
          system TEXT NOT NULL, resource TEXT NOT NULL, id TEXT NOT NULL,
          body TEXT NOT NULL, PRIMARY KEY(system,resource,id)
        );
        CREATE TABLE IF NOT EXISTS counters (
          system TEXT NOT NULL, resource TEXT NOT NULL, value INTEGER NOT NULL,
          PRIMARY KEY(system,resource)
        );
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT OR IGNORE INTO meta(key,value) VALUES ('tick','0');
        CREATE TABLE IF NOT EXISTS events (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT NOT NULL, integration_index INTEGER NOT NULL,
          data TEXT NOT NULL, due_tick INTEGER NOT NULL,
          state TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
          error TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_pending ON events(state,due_tick,id);
        CREATE TABLE IF NOT EXISTS tool_trace (
          id INTEGER PRIMARY KEY AUTOINCREMENT, tick INTEGER NOT NULL,
          tool TEXT NOT NULL, actor TEXT NOT NULL, role TEXT NOT NULL,
          args TEXT NOT NULL, ok INTEGER NOT NULL, output TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS audit (
          id INTEGER PRIMARY KEY AUTOINCREMENT, tick INTEGER NOT NULL,
          tool TEXT NOT NULL, actor TEXT NOT NULL, role TEXT NOT NULL,
          before_state TEXT, after_state TEXT
        );
        """)
        stored = self.db.execute("SELECT value FROM meta WHERE key='manifest_sha256'").fetchone()
        if stored is None:
            self.db.execute("INSERT INTO meta(key,value) VALUES('manifest_sha256',?)", (self.spec_hash,))
        elif stored[0] != self.spec_hash:
            self.db.close()
            raise SimError("MANIFEST_MISMATCH", "Stored simulator DB was created for a different manifest; use a new DB or explicitly migrate.")

    @contextmanager
    def _transaction(self):
        with self._lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield
                self.db.execute("COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                raise

    @property
    def tick(self) -> int:
        with self._lock:
            return int(self.db.execute("SELECT value FROM meta WHERE key='tick'").fetchone()[0])

    def _resource(self, system: str, resource: str) -> dict:
        try:
            return self.spec["systems"][system]["resources"][resource]
        except (KeyError, TypeError):
            raise SimError("UNKNOWN_RESOURCE", f"Unknown resource: {system}.{resource}", status=404)

    def _read(self, system: str, resource: str, rid: str) -> dict | None:
        row = self.db.execute("SELECT body FROM entities WHERE system=? AND resource=? AND id=?",
                              (system, resource, str(rid))).fetchone()
        return json.loads(row[0]) if row else None

    def get(self, system: str, resource: str, rid: str) -> dict | None:
        self._resource(system, resource)
        with self._lock:
            return self._read(system, resource, rid)

    def _list(self, system: str, resource: str, *, where: dict | None = None, limit: int = 100) -> list[dict]:
        records = [json.loads(row[0]) for row in self.db.execute(
            "SELECT body FROM entities WHERE system=? AND resource=? ORDER BY id", (system, resource))]
        if where:
            records = [r for r in records if all(r.get(k) == v for k, v in where.items())]
        return records[:limit]

    def list_records(self, system: str, resource: str) -> list[dict]:
        self._resource(system, resource)
        with self._lock:
            return self._list(system, resource, limit=100_000)

    def resolve(self, expr: dict, ctx: dict[str, Any]) -> Any:
        kind, value = next(iter(expr.items()))
        if kind == "literal":
            return value
        if kind == "path":
            return _at_path(ctx, value)
        if kind == "lookup":
            rid = self.resolve(value["id"], ctx)
            if rid is None:
                return None
            self._resource(value["system"], value["resource"])
            record = self._read(value["system"], value["resource"], str(rid))
            return _at_path(record, value["field"]) if record is not None and "field" in value else record
        if kind == "count":
            self._resource(value["system"], value["resource"])
            where = {k: self.resolve(v, ctx) for k, v in value.get("where", {}).items()}
            return len(self._list(value["system"], value["resource"], where=where, limit=100_000))
        if kind == "negate":
            x = self.resolve(value, ctx)
            return -x if isinstance(x, (float, int)) and not isinstance(x, bool) else None
        if kind in {"add", "mul"}:
            parts = [self.resolve(part, ctx) for part in value]
            if any(not isinstance(p, (int, float)) or isinstance(p, bool) for p in parts):
                return None
            if kind == "add":
                return sum(parts)
            out = 1
            for part in parts:
                out *= part
            return out
        raise SimError("INVALID_EXPRESSION", f"Unsupported expression: {kind}")

    def satisfies(self, condition: dict, ctx: dict[str, Any] | None = None) -> bool:
        ctx = ctx or {}
        kind, value = next(iter(condition.items()))
        if kind == "all":
            return all(self.satisfies(c, ctx) for c in value)
        if kind == "any":
            return any(self.satisfies(c, ctx) for c in value)
        if kind == "not":
            return not self.satisfies(value, ctx)
        if kind == "exists":
            return self.resolve(value, ctx) is not None
        if kind == "compare":
            left, right = self.resolve(value["left"], ctx), self.resolve(value["right"], ctx)
            op = value["op"]
            if op == "eq":
                return left == right
            if op == "ne":
                return left != right
            if left is None or right is None:
                return False
            try:
                if op == "le":
                    return left <= right
                if op == "lt":
                    return left < right
                if op == "ge":
                    return left >= right
                if op == "gt":
                    return left > right
                if op == "in":
                    return left in right
                if op == "not_in":
                    return left not in right
            except (TypeError, ValueError):
                return False
        raise SimError("INVALID_CONDITION", f"Unrecognized condition: {kind}")

    def _numeric_identity_mode(self, system: str, resource: str) -> str:
        """Identity representation inferred from declared JSON storage and POST return.

        When an API reads ``id`` as a string but POST returns a number,
        generate a decimal *string* internally, so the declared output adapter
        can convert it losslessly to an integer. This does not reconstruct real
        allocation policies or values from an actual ERP.
        """
        rc = self._resource(system, resource)
        field = rc.get('id_field', 'id')
        id_type = rc['schema'].get('properties', {}).get(field, {}).get('type')
        if id_type == 'integer':
            return 'integer'
        if rc.get('response_contract', {}).get('create', {}).get('kind') == 'number':
            return 'decimal_string'
        return 'opaque_string'

    def _next_id(self, system: str, resource: str) -> str | int:
        row = self.db.execute('SELECT value FROM counters WHERE system=? AND resource=?',
                              (system, resource)).fetchone()
        next_value = (row[0] if row else 0) + 1
        mode = self._numeric_identity_mode(system, resource)
        while True:
            if mode == 'integer':
                result = next_value
            elif mode == 'decimal_string':
                result = str(next_value)
            else:
                result = f'{system}_{resource}_{next_value:06d}'
            if self._read(system, resource, str(result)) is None:
                break
            next_value += 1
        self.db.execute('INSERT INTO counters(system,resource,value) VALUES(?,?,?) '
                        'ON CONFLICT(system,resource) DO UPDATE SET value=excluded.value',
                        (system, resource, next_value))
        return result

    def _observe_explicit_id(self, system: str, resource: str, value: Any) -> None:
        """Don't reuse integral seeded IDs when a later create auto-allocates one."""
        if self._numeric_identity_mode(system, resource) == 'opaque_string':
            return
        numeric = str(value)
        if not numeric.isdecimal() or int(numeric) > 2**63 - 1:
            return
        num = int(numeric)
        self.db.execute('INSERT INTO counters(system,resource,value) VALUES(?,?,?) '
                        'ON CONFLICT(system,resource) DO UPDATE SET value=max(value,excluded.value)',
                        (system, resource, num))

    def _validate_schema_instance(self, schema: dict, body: dict, label: str, code: str) -> None:
        errors = sorted(Draft202012Validator(schema).iter_errors(body), key=lambda e: str(e.path))
        if errors:
            err = errors[0]
            safe = {'validator': str(err.validator), 'path': '.'.join(map(str, err.path)) or '<root>'}
            if err.validator == 'required' and isinstance(err.validator_value, list) and isinstance(err.instance, dict):
                missing = [k for k in err.validator_value if k not in err.instance]
                if missing:
                    safe['missing_field'] = missing[0]
            if err.validator == 'type':
                safe['expected_type'] = err.validator_value
                value = err.instance
                safe['received_kind'] = ('null' if value is None else 'boolean' if isinstance(value, bool)
                                         else 'number' if isinstance(value, (int, float)) else
                                         'array' if isinstance(value, list) else
                                         'object' if isinstance(value, dict) else 'string' if isinstance(value, str) else 'unknown')
            raise SimError(code, f"{label} {'.'.join(map(str, err.path)) or '<root>'}: {err.message}", diagnostics=safe)

    def _validate_record(self, system: str, resource: str, body: dict, *, prior_id: str | None = None) -> None:
        rc = self._resource(system, resource)
        self._validate_schema_instance(rc['schema'], body, f'{system}.{resource}', 'SCHEMA_VIOLATION')
        for fk in rc.get("foreign_keys", []):
            field_value = body.get(fk["field"])
            if field_value is None and fk.get("nullable", False):
                continue
            if field_value is None or self._read(fk["system"], fk["resource"], str(field_value)) is None:
                raise SimError("FOREIGN_KEY", f"{system}.{resource}.{fk['field']} must reference {fk['system']}.{fk['resource']}")
        for group in rc.get("unique", []):
            key = tuple(body.get(f) for f in group)
            if any(x is None for x in key):
                continue
            for row in self._list(system, resource, limit=100_000):
                if str(row.get(rc.get("id_field", "id"))) == str(prior_id):
                    continue
                if tuple(row.get(f) for f in group) == key:
                    raise SimError("UNIQUE_VIOLATION", f"{system}.{resource}: duplicate unique fields {group}")

    def _write(self, system: str, resource: str, body: dict, *, old: dict | None = None) -> None:
        rc = self._resource(system, resource)
        rid = str(body[rc.get("id_field", "id")])
        self._validate_record(system, resource, body, prior_id=rid if old else None)
        self.db.execute("INSERT INTO entities(system,resource,id,body) VALUES(?,?,?,?) "
                        "ON CONFLICT(system,resource,id) DO UPDATE SET body=excluded.body",
                        (system, resource, rid, _json(body)))

    def _apply_effects(self, effects: list, ctx: dict[str, Any]) -> None:
        for effect in effects:
            target = effect["target"]
            rid = self.resolve(target["id"], ctx)
            if rid is None:
                raise SimError("EFFECT_TARGET", "Side effect target ID missing")
            doc = self._read(target["system"], target["resource"], str(rid))
            if doc is None:
                raise SimError("EFFECT_TARGET", f"Effect target doesn't exist: {target['system']}.{target['resource']}:{rid}")
            amount = self.resolve(effect["by"], ctx)
            field = effect["field"]
            if not isinstance(amount, (int, float)) or isinstance(amount, bool) or not isinstance(doc.get(field), (int, float)):
                raise SimError("EFFECT_TYPE", f"Cannot adjust nonnumeric value: {field}")
            original = copy.deepcopy(doc)
            doc[field] += amount
            self._write(target["system"], target["resource"], doc, old=original)

    def _emit(self, name: str, record: dict, *, before: dict | None = None) -> None:
        ctx = {"event": {"name": name, "record": record, "before": before}}
        for i, integration in enumerate(self.spec.get("integrations", [])):
            if integration["event"] != name:
                continue
            self.db.execute("INSERT INTO events(name,integration_index,data,due_tick) VALUES(?,?,?,?)",
                            (name, i, _json(ctx["event"]), self.tick + integration.get("delay_ticks", 0)))

    def _record_tool_call(self, tool: str, args: Any, role: str, actor: str, result: dict, ok: bool) -> None:
        with self._lock:
            self.db.execute(
                "INSERT INTO tool_trace(tick,tool,actor,role,args,ok,output) VALUES(?,?,?,?,?,?,?)",
                (self.tick, tool, actor, role, _json(args), int(ok), _json(result)))

    def tool_trace(self, *, exclude_seed: bool = False) -> list[dict]:
        with self._lock:
            query = "SELECT * FROM tool_trace"
            if exclude_seed:
                query += " WHERE actor != 'scenario_setup'"
            query += " ORDER BY id"
            records = [dict(row) for row in self.db.execute(query)]
            for row in records:
                row["args"] = json.loads(row["args"])
                row["output"] = json.loads(row["output"])
                row["ok"] = bool(row["ok"])
            return records

    def call(self, tool: str, args: dict | None = None, *, role: str = "agent", actor: str = "anonymous") -> dict:
        # Log rejected attempts too. Logging happens outside the tool transaction,
        # so a rollback does not erase the failed training experience.
        with self._lock:
            try:
                result = self._call_impl(tool, args, role=role, actor=actor)
            except SimError as exc:
                self._record_tool_call(tool, args, role, actor, exc.as_dict(), False)
                raise
            self._record_tool_call(tool, args, role, actor, result, True)
            return result

    def _call_impl(self, tool: str, args: dict | None = None, *, role: str = "agent", actor: str = "anonymous") -> dict:
        parts = tool.split(".")
        if len(parts) != 3:
            raise SimError("UNKNOWN_TOOL", f"Tool name must be system.resource.action: {tool}", status=404)
        system, resource, action = parts
        if resource == "workflow":
            return self._execute_workflow(tool, system, action, args, role=role, actor=actor)
        if resource == "lookup":
            return self._execute_lookup(tool, system, action, args, role=role)
        rc = self._resource(system, resource)
        if action not in {"create", "get", "list", "update", "delete"}:
            raise SimError("UNKNOWN_TOOL", f"Unknown action: {tool}", status=404)
        if role not in rc.get("allow", {}).get(action, []):
            raise SimError("FORBIDDEN", f"Role {role!r} cannot invoke {tool}", status=403)
        args = copy.deepcopy(args or {})
        if not isinstance(args, dict):
            raise SimError("INVALID_ARGUMENTS", "Tool args must be an object", status=400)
        if action in {"get", "list"}:
            with self._lock:
                if action == "get":
                    if args.get("id") is None:
                        raise SimError("INVALID_ARGUMENTS", "get requires id", status=400)
                    doc = self._read(system, resource, str(args["id"]))
                    if doc is None:
                        raise SimError("NOT_FOUND", "Record not found", status=404)
                    result = doc
                else:
                    where, limit = args.get("where", {}), args.get("limit", 100)
                    if not isinstance(where, dict) or not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 1000:
                        raise SimError("INVALID_ARGUMENTS", "list requires where object and limit 1..1000", status=400)
                    result = self._list(system, resource, where=where, limit=limit)
                result = self._project_read(rc, action, result)
                return {"ok": True, "tool": tool, "result": result, "tick": self.tick}
        with self._transaction():
            id_field = rc.get("id_field", "id")
            before = None
            if action == "create":
                body = args
                if isinstance(rc.get('create_input_schema'), dict):
                    input_schema = copy.deepcopy(rc['create_input_schema'])
                    # Server-generated IDs should not become required user inputs
                    # simply because Swagger reused a full read model for POST.
                    input_schema['required'] = [f for f in input_schema.get('required', []) if f != id_field]
                    self._validate_schema_instance(input_schema, body,
                                                   f'{system}.{resource} create request', 'REQUEST_SCHEMA_VIOLATION')
                if not body.get(id_field):
                    body[id_field] = self._next_id(system, resource)
                rid = str(body[id_field])
                if self._read(system, resource, rid) is not None:
                    raise SimError("ALREADY_EXISTS", f"Record already exists: {rid}", status=409)
                self._observe_explicit_id(system, resource, body[id_field])
            else:
                if args.get("id") is None:
                    raise SimError("INVALID_ARGUMENTS", f"{action} requires id", status=400)
                rid = str(args["id"])
                before = self._read(system, resource, rid)
                if before is None:
                    raise SimError("NOT_FOUND", f"Record not found: {rid}", status=404)
                if action == "update":
                    patch = args.get("patch")
                    if not isinstance(patch, dict) or id_field in patch:
                        raise SimError("INVALID_ARGUMENTS", "update requires patch object, immutable ID", status=400)
                    body = {**before, **patch}
                else:
                    body = before
            ctx = {"input": body if action == "create" else args.get("patch", {}) if action == "update" else args,
                   "record": body, "before": before}
            if action != "delete":
                self._validate_record(system, resource, body, prior_id=rid if before is not None else None)
            for i, cond in enumerate(rc.get("guards", {}).get(action, [])):
                if not self.satisfies(cond, ctx):
                    raise SimError("POLICY_DENIED", f"{tool}: policy check {i+1} failed")
            if action == "delete":
                # Explicit deletion can be disabled by excluding the role from allow.delete.
                # Reject deletion if other resources have live foreign-key dependents.
                for s, sc in self.spec["systems"].items():
                    for r, conf in sc["resources"].items():
                        for fk in conf.get("foreign_keys", []):
                            if fk["system"] == system and fk["resource"] == resource:
                                for item in self._list(s, r, limit=100_000):
                                    if str(item.get(fk["field"])) == rid:
                                        raise SimError("REFERENCED", f"Cannot delete {system}.{resource}:{rid}; referenced by {s}.{r}")
                self.db.execute("DELETE FROM entities WHERE system=? AND resource=? AND id=?", (system, resource, rid))
                after = None
            else:
                self._write(system, resource, body, old=before)
                self._apply_effects(rc.get("effects", {}).get(action, []), ctx)
                after = self._read(system, resource, rid)
            event_name = f"{system}.{resource}.{'created' if action == 'create' else 'updated' if action == 'update' else 'deleted'}"
            self._emit(event_name, after or before, before=before)
            self.db.execute("INSERT INTO audit(tick,tool,actor,role,before_state,after_state) VALUES(?,?,?,?,?,?)",
                            (self.tick, tool, actor, role, _json(before) if before else None,
                             _json(after) if after else None))
            return {"ok": True, "tool": tool, "result": after, "tick": self.tick}

    def _project_read(self, rc: dict, action: str, result: Any) -> Any:
        """Explicit response projection, never automatic filler of unknown values."""
        projection = rc.get('response_projection', {}).get(action)
        if not projection:
            return result
        def one(record):
            ctx = {'record': record}
            fields = {k: self.resolve(v, ctx) for k, v in projection.get('fields', {}).items()}
            return {**record, **fields} if projection.get('include_base', False) else fields
        if isinstance(result, list):
            return [one(r) for r in result]
        return one(result)

    def _execute_lookup(self, tool: str, system: str, name: str, args: dict | None, *, role: str) -> dict:
        conf = self.spec.get('systems', {}).get(system, {}).get('lookups', {}).get(name)
        if conf is None:
            raise SimError('UNKNOWN_TOOL', f'Unknown lookup: {tool}', status=404)
        if role not in conf['roles']:
            raise SimError('FORBIDDEN', f'Role {role!r} cannot invoke {tool}', status=403)
        if not isinstance(args, dict):
            raise SimError('INVALID_ARGUMENTS', 'lookup arguments must be object', status=400)
        schema = conf.get('input_schema', {'type':'object','properties':{conf['arg']:{'type':'string'}},
                                           'required':[conf['arg']]})
        self._validate_schema_instance(schema, args, tool, 'REQUEST_SCHEMA_VIOLATION')
        if conf['arg'] not in args:
            raise SimError('INVALID_ARGUMENTS', 'Missing lookup key', status=400)
        with self._lock:
            # Query only the known target resource, never dynamic SQL/field names.
            hits = [r for r in self._list(system, conf['resource'], limit=100_000)
                    if r.get(conf['field']) == args[conf['arg']]]
            if not hits:
                raise SimError('NOT_FOUND', 'Lookup target not found', status=404)
            if len(hits) > 1:
                raise SimError('AMBIGUOUS_LOOKUP', 'Alternate key is not unique', status=409)
            rc = self._resource(system, conf['resource'])
            data = self._project_read(rc, 'get', hits[0])
            return {'ok':True, 'tool':tool, 'result':data, 'tick': self.tick,
                    'evidence': conf.get('evidence','hypothesis')}

    def _execute_workflow(self, tool: str, system: str, name: str, args: dict | None, *,
                          role: str, actor: str) -> dict:
        """Execute a declarative workflow atomically; no arbitrary code or external I/O.

        Workflow rules are not automatically true of the real ERP. The response
        reports their evidence grade to prevent provisional rules being promoted.
        """
        conf = self.spec.get('systems', {}).get(system, {}).get('workflows', {}).get(name)
        if conf is None:
            raise SimError('UNKNOWN_TOOL', f'Workflow not defined: {tool}', status=404)
        if role not in conf['roles']:
            raise SimError('FORBIDDEN', f'Role {role!r} cannot invoke {tool}', status=403)
        if not isinstance(args, dict):
            raise SimError('INVALID_ARGUMENTS', 'Workflow args must be an object', status=400)
        args = copy.deepcopy(args)
        self._validate_schema_instance(conf.get('input_schema', {'type': 'object'}), args,
                                       tool + ' input', 'REQUEST_SCHEMA_VIOLATION')
        ctx = {'input': args, 'steps': [], 'before': None, 'after': None, 'record': None}
        with self._transaction():
            for idx, check in enumerate(conf.get('preconditions', [])):
                if not self.satisfies(check, ctx):
                    raise SimError('POLICY_DENIED', f'{tool}: precondition {idx+1} failed')
            for idx, step in enumerate(conf['steps']):
                sys_id, res_id, kind = step['system'], step['resource'], step['kind']
                rc = self._resource(sys_id, res_id)
                id_field = rc.get('id_field', 'id')
                rid = None
                before = None
                if kind != 'create':
                    rid = self.resolve(step['id'], ctx)
                    if rid is None:
                        raise SimError('INVALID_ARGUMENTS', f'{tool}: step {idx+1} target ID absent')
                    before = self._read(sys_id, res_id, str(rid))
                    if before is None:
                        raise SimError('NOT_FOUND', f'{tool}: step {idx+1} target missing', status=404)
                ctx['before'] = before
                ctx['record'] = before
                for check_i, check in enumerate(step.get('require', [])):
                    if not self.satisfies(check, ctx):
                        raise SimError('POLICY_DENIED', f'{tool}: step {idx+1} condition {check_i+1} failed')
                if kind == 'create':
                    body = {k: self.resolve(v, ctx) for k, v in step['values'].items()}
                    if body.get(id_field) is None:
                        body[id_field] = self._next_id(sys_id, res_id)
                    rid = body[id_field]
                    if self._read(sys_id, res_id, str(rid)) is not None:
                        raise SimError('ALREADY_EXISTS', f'{tool}: record exists', status=409)
                    self._observe_explicit_id(sys_id, res_id, rid)
                elif kind == 'update':
                    body = copy.deepcopy(before)
                    body.update({k: self.resolve(v, ctx) for k, v in step['values'].items()})
                elif kind == 'adjust':
                    amount = self.resolve(step['by'], ctx)
                    current = before.get(step['field'])
                    if (type(amount) not in {int, float} or type(current) not in {int, float}):
                        raise SimError('EFFECT_TYPE', f'{tool}: nonnumeric adjustment field')
                    body = copy.deepcopy(before)
                    body[step['field']] = current + amount
                else:
                    raise SimError('INVALID_WORKFLOW', f'{tool}: unsupported step kind')
                self._write(sys_id, res_id, body, old=before)
                after = self._read(sys_id, res_id, str(rid))
                event_name = f"{sys_id}.{res_id}.{'created' if kind == 'create' else 'updated'}"
                self._emit(event_name, after, before=before)
                self.db.execute('INSERT INTO audit(tick,tool,actor,role,before_state,after_state) VALUES(?,?,?,?,?,?)',
                                (self.tick, tool, actor, role, _json(before) if before is not None else None,
                                 _json(after)))
                ctx['record'] = after
                ctx['after'] = after
                ctx['steps'].append({'kind': kind, 'before': before, 'after': after})
            result = self.resolve(conf['result'], ctx) if 'result' in conf else ctx['after']
            return {'ok': True, 'tool': tool, 'result': result, 'tick': self.tick,
                    'evidence': conf.get('evidence', 'hypothesis'), 'steps_executed': len(ctx['steps'])}

    def _deliver(self, integration: dict, event_data: dict) -> None:
        dest = integration["target"]
        system, resource = dest["system"], dest["resource"]
        ctx = {"event": event_data}
        body = {name: self.resolve(expr, ctx) for name, expr in integration["mapping"].items()}
        rc = self._resource(system, resource)
        id_field = rc.get("id_field", "id")
        if not body.get(id_field):
            raise SimError("PROJECTION_ID", f"Integration must supply {system}.{resource}.{id_field}")
        existing = self._read(system, resource, str(body[id_field]))
        # Mirror records are service-owned projections. Do not execute their user write guards.
        # Schema, uniqueness and FK checks still apply to all projected records.
        if existing:
            body = {**existing, **body}
        self._write(system, resource, body, old=existing)

    def advance(self, steps: int = 1, *, max_events: int = 1000) -> dict:
        if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= 100_000:
            raise SimError("INVALID_TICKS", "steps must be integer 1..100000", status=400)
        with self._transaction():
            new_tick = self.tick + steps
            self.db.execute("UPDATE meta SET value=? WHERE key='tick'", (str(new_tick),))
            delivered, failed = 0, 0
            due = list(self.db.execute("SELECT * FROM events WHERE state='pending' AND due_tick<=? ORDER BY due_tick,id LIMIT ?",
                                       (new_tick, max_events)))
            for event in due:
                # Delivery is individually savepoint-isolated; a malformed mapping cannot
                # leave partial projection changes in the database.
                self.db.execute("SAVEPOINT delivery")
                try:
                    integration = self.spec["integrations"][event["integration_index"]]
                    self._deliver(integration, json.loads(event["data"]))
                    self.db.execute("RELEASE delivery")
                    self.db.execute("UPDATE events SET state='delivered',error=NULL WHERE id=?", (event["id"],))
                    delivered += 1
                except (SimError, TypeError, ValueError, KeyError) as exc:
                    self.db.execute("ROLLBACK TO delivery")
                    self.db.execute("RELEASE delivery")
                    attempts = event["attempts"] + 1
                    self.db.execute("UPDATE events SET attempts=?, state=?, error=?,due_tick=? WHERE id=?",
                                    (attempts, "dead_letter" if attempts >= 3 else "pending", str(exc), new_tick + 1, event["id"]))
                    failed += 1
            return {"tick": new_tick, "delivered": delivered, "failed": failed,
                    "remaining": self.db.execute("SELECT COUNT(*) FROM events WHERE state='pending'").fetchone()[0]}

    def events(self) -> list[dict]:
        with self._lock:
            return [dict(row) for row in self.db.execute("SELECT * FROM events ORDER BY id")]

    def audit(self) -> list[dict]:
        with self._lock:
            return [dict(row) for row in self.db.execute("SELECT * FROM audit ORDER BY id")]

    def reset(self) -> None:
        with self._transaction():
            for table in ("entities", "events", "audit", "counters", "tool_trace"):
                self.db.execute(f"DELETE FROM {table}")
            self.db.execute("UPDATE meta SET value='0' WHERE key='tick'")

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "format": "orgsim-snapshot-v1", "organization": self.spec["organization"], "manifest_sha256": self.spec_hash, "tick": self.tick,
                "entities": [dict(row) for row in self.db.execute("SELECT * FROM entities ORDER BY system,resource,id")],
                "events": [dict(row) for row in self.db.execute("SELECT * FROM events ORDER BY id")],
                "counters": [dict(row) for row in self.db.execute("SELECT * FROM counters ORDER BY system,resource")],
                "audit": [dict(row) for row in self.db.execute("SELECT * FROM audit ORDER BY id")],
                "tool_trace": [dict(row) for row in self.db.execute("SELECT * FROM tool_trace ORDER BY id")],
            }

    def restore(self, snapshot: dict) -> None:
        if (snapshot.get("format") != "orgsim-snapshot-v1" or
            snapshot.get("organization") != self.spec["organization"] or
            snapshot.get("manifest_sha256") != self.spec_hash):
            raise SimError("BAD_SNAPSHOT", "Snapshot format/organization/manifest mismatch")
        with self._transaction():
            for table in ("entities", "events", "audit", "counters", "tool_trace"):
                self.db.execute(f"DELETE FROM {table}")
            self.db.execute("UPDATE meta SET value=? WHERE key='tick'", (str(snapshot["tick"]),))
            shapes = {"entities": ("system", "resource", "id", "body"),
                      "events": ("id", "name", "integration_index", "data", "due_tick", "state", "attempts", "error"),
                      "counters": ("system", "resource", "value"),
                      "tool_trace": ("id", "tick", "tool", "actor", "role", "args", "ok", "output"),
                      "audit": ("id", "tick", "tool", "actor", "role", "before_state", "after_state")}
            for table, keys in shapes.items():
                params = ",".join("?" for _ in keys)
                cols = ",".join(keys)
                for row in snapshot.get(table, []):
                    self.db.execute(f"INSERT INTO {table}({cols}) VALUES({params})", tuple(row[k] for k in keys))
            report = self.validate()
            if not report["valid"]:
                raise SimError("BAD_SNAPSHOT", "Snapshot failed invariant validation: " + "; ".join(report["errors"][:3]))

    def validate(self) -> dict:
        errors = []
        total = 0
        with self._lock:
            for row in self.db.execute("SELECT system,resource,id,body FROM entities ORDER BY system,resource,id"):
                total += 1
                try:
                    doc = json.loads(row["body"])
                    rc = self._resource(row["system"], row["resource"])
                    if str(doc.get(rc.get("id_field", "id"))) != row["id"]:
                        raise SimError("ID_MISMATCH", "Primary key does not match body")
                    self._validate_record(row["system"], row["resource"], doc, prior_id=row["id"])
                except (SimError, ValueError, TypeError) as exc:
                    errors.append(f"{row['system']}.{row['resource']}:{row['id']}: {exc}")
            # Compare uniqueness across every configured group, not just at insertion.
            for s, sc in self.spec["systems"].items():
                for r, conf in sc["resources"].items():
                    for group in conf.get("unique", []):
                        seen = set()
                        for rec in self._list(s, r, limit=100_000):
                            key = tuple(rec.get(f) for f in group)
                            if any(v is None for v in key):
                                continue
                            if key in seen:
                                errors.append(f"{s}.{r}: duplicate uniqueness group {group}")
                            seen.add(key)
        return {"valid": not errors, "records": total, "errors": errors, "tick": self.tick,
                "assurance": "policy_reviewed" if self.spec.get("reviewed") else "contract_only_unreviewed"}

    def evaluate(self, checks: list[dict]) -> dict:
        with self._lock:
            results = [{"index": i, "passed": self.satisfies(c), "condition": c} for i, c in enumerate(checks)]
        return {"passed": all(c["passed"] for c in results), "checks": results}
