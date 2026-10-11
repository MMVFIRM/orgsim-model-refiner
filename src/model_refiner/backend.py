"""Model backends. The core package uses only the standard library; the OpenAI SDK is optional."""
from __future__ import annotations
from contextlib import contextmanager

import copy
import hashlib
import json
import os
import subprocess
import threading
import time
import sqlite3
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

from .types import ModelResult

PROVIDERS = ("openai", "openai-compatible", "anthropic", "command", "toy")


class ModelBackend(ABC):
    @abstractmethod
    def generate(self, *, model: str, messages: list[dict[str, str]]) -> ModelResult:
        raise NotImplementedError

    def cache_identity(self) -> dict[str, Any]:
        """Secret-free backend configuration that affects deterministic outputs."""
        return {"backend": self.__class__.__name__}

    def with_output_cap(self, tokens: int) -> "ModelBackend":
        """A view of this backend with a different output-token cap (used for long-form calls such as rule compilation).

        Backends without a configurable cap return themselves."""
        return self


class BackendError(RuntimeError):
    """An API call failed. `retryable` is False for client errors (bad key, bad request) that retrying cannot fix."""

    def __init__(self, message: str, retryable: bool = True, status: int | None = None):
        super().__init__(message)
        self.retryable, self.status = retryable, status


def _post_json(url: str, headers: dict[str, str], body: dict, timeout: float = 120) -> dict:
    if urllib.parse.urlsplit(url).scheme not in ("http", "https"):
        # Operators choose the base URL, but a typo or hostile config must not reach file:// or other handlers.
        raise BackendError(f"refusing non-HTTP(S) URL scheme in {url!r}", retryable=False)
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers={"content-type": "application/json", **headers}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # nosec B310 - scheme restricted to http/https above
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise BackendError(f"HTTP {exc.code} from {url}: {detail}",
                           retryable=exc.code in (408, 409, 425, 429) or exc.code >= 500, status=exc.code) from exc
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        raise BackendError(f"request to {url} failed: {exc}") from exc
    except ValueError as exc:
        raise BackendError(f"invalid JSON from {url}: {exc}") from exc


def _split_system(messages: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    """Fold developer/system messages into one system string; everything else stays a chat turn."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] in ("developer", "system"))
    turns = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] not in ("developer", "system")]
    return system, turns


class OpenAIResponsesBackend(ModelBackend):
    """OpenAI Responses API adapter. Requires the `openai` package and OPENAI_API_KEY."""

    def __init__(self, api_key: str | None = None, max_output_tokens: int | None = None, timeout: float = 120):
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - environment-specific
            raise RuntimeError("The OpenAI SDK is required. Install with: pip install 'model-refiner[openai]'") from exc
        kwargs: dict[str, Any] = {"max_retries": 0, "timeout": timeout, "base_url": "https://api.openai.com/v1"}
        if api_key:
            kwargs["api_key"] = api_key
        self.client = OpenAI(**kwargs)
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout

    def with_output_cap(self, tokens: int) -> "ModelBackend":
        clone = copy.copy(self)
        clone.max_output_tokens = int(tokens)
        return clone

    def cache_identity(self) -> dict[str, Any]:
        return {"provider": "openai", "max_output_tokens": self.max_output_tokens, "timeout": self.timeout, "base_url": "https://api.openai.com/v1", "store": False}

    def generate(self, *, model: str, messages: list[dict[str, str]]) -> ModelResult:
        kw: dict[str, Any] = {"store": False}
        if self.max_output_tokens:
            kw["max_output_tokens"] = max(16, self.max_output_tokens)   # the API rejects smaller values
        response = self.client.responses.create(model=model, input=messages, **kw)
        text = getattr(response, "output_text", None)
        if not text:
            chunks: list[str] = []
            for item in getattr(response, "output", []) or []:
                for content in getattr(item, "content", []) or []:
                    maybe = getattr(content, "text", None)
                    if maybe:
                        chunks.append(str(maybe))
            text = "\n".join(chunks)
        usage = getattr(response, "usage", None)
        return ModelResult(
            text=str(text or ""),
            input_tokens=getattr(usage, "input_tokens", None) if usage else None,
            output_tokens=getattr(usage, "output_tokens", None) if usage else None,
            metadata={
                "response_id": getattr(response, "id", None),
                "request_id": getattr(response, "_request_id", None),
                "status": getattr(response, "status", None),
                "incomplete": getattr(response, "status", None) == "incomplete",
                "refusal": any(getattr(c, "type", None) == "refusal" for item in (getattr(response, "output", None) or []) for c in (getattr(item, "content", None) or [])),
            },
        )


class OpenAIChatBackend(ModelBackend):
    """Chat Completions over plain HTTP: works with OpenAI and any OpenAI-compatible server (vLLM, Ollama, ...)."""

    def __init__(self, base_url: str = "https://api.openai.com/v1", key_env: str = "OPENAI_API_KEY",
                 max_tokens: int = 256, timeout: float = 120, temperature: float | None = None):
        self.base, self.max_tokens, self.timeout, self.temperature = base_url.rstrip("/"), max_tokens, timeout, temperature
        self.key = os.environ.get(key_env, "")
        local = any(h in self.base for h in ("localhost", "127.0.0.1", "[::1]"))
        if not self.key and not local:
            raise RuntimeError(f"Set {key_env} to use this backend")

    def with_output_cap(self, tokens: int) -> "ModelBackend":
        clone = copy.copy(self)
        clone.max_tokens = int(tokens)
        return clone

    def cache_identity(self) -> dict[str, Any]:
        return {"provider": "openai-compatible", "base_url": self.base, "max_tokens": self.max_tokens, "temperature": self.temperature}

    def generate(self, *, model: str, messages: list[dict[str, str]]) -> ModelResult:
        system, turns = _split_system(messages)
        chat = ([{"role": "system", "content": system}] if system else []) + turns
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        # api.openai.com rejects `max_tokens` for newer models; other servers still expect it.
        limit_key = "max_completion_tokens" if "api.openai.com" in self.base else "max_tokens"
        body: dict[str, Any] = {"model": model, "messages": chat, limit_key: self.max_tokens}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        out = _post_json(f"{self.base}/chat/completions", headers, body, self.timeout)
        usage = out.get("usage") or {}
        try:
            text = out["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise BackendError(f"unexpected response shape: {str(out)[:200]}") from exc
        return ModelResult(
            text=text,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            metadata={"response_id": out.get("id"),
                      "refusal": bool(out["choices"][0]["message"].get("refusal")),
                      "incomplete": out["choices"][0].get("finish_reason") == "length"},
        )


class AnthropicBackend(ModelBackend):
    """Anthropic Messages API over plain HTTP. Developer messages become the system prompt."""

    def __init__(self, base_url: str = "https://api.anthropic.com", key_env: str = "ANTHROPIC_API_KEY",
                 max_tokens: int = 256, timeout: float = 120, temperature: float | None = None):
        self.base, self.max_tokens, self.timeout, self.temperature = base_url.rstrip("/"), max_tokens, timeout, temperature
        self.key = os.environ.get(key_env)
        if not self.key:
            raise RuntimeError(f"Set {key_env} to use the anthropic backend")

    def with_output_cap(self, tokens: int) -> "ModelBackend":
        clone = copy.copy(self)
        clone.max_tokens = int(tokens)
        return clone

    def cache_identity(self) -> dict[str, Any]:
        return {"provider": "anthropic", "base_url": self.base, "max_tokens": self.max_tokens, "temperature": self.temperature}

    def generate(self, *, model: str, messages: list[dict[str, str]]) -> ModelResult:
        system, turns = _split_system(messages)
        body: dict[str, Any] = dict(model=model, max_tokens=self.max_tokens, messages=turns)
        if self.temperature is not None:      # some current models reject an explicit temperature
            body["temperature"] = self.temperature
        if system:
            body["system"] = system
        out = _post_json(f"{self.base}/v1/messages",
                         {"x-api-key": self.key, "anthropic-version": "2023-06-01"}, body, self.timeout)
        usage = out.get("usage") or {}
        return ModelResult(
            text="".join(b.get("text", "") for b in out.get("content", []) if b.get("type", "text") == "text"),
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            metadata={"response_id": out.get("id"),
                      "incomplete": out.get("stop_reason") == "max_tokens",
                      "refusal": out.get("stop_reason") == "refusal"},
        )


class CommandBackend(ModelBackend):
    """Run a shell command per request. stdin gets the system text then the user turn; stdout is the reply."""

    def __init__(self, command: str, timeout: float = 120):
        self.command, self.timeout = command, timeout

    def cache_identity(self) -> dict[str, Any]:
        # Hash the executable string so cache keys change with command configuration without persisting it in cache rows.
        return {"provider": "command", "command_sha256": hashlib.sha256(self.command.encode()).hexdigest(), "timeout": self.timeout}

    def generate(self, *, model: str, messages: list[dict[str, str]]) -> ModelResult:
        system, turns = _split_system(messages)
        prompt = (system + "\n\n" if system else "") + "\n\n".join(t["content"] for t in turns)
        # nosec B602 - the command is operator-supplied process configuration, never client input: the suite
        # and the HTTP service cannot select this provider (see tests/suite/test_production_guards.py).
        r = subprocess.run(self.command, input=prompt, capture_output=True, text=True, shell=True,  # nosec B602
                           timeout=self.timeout, encoding="utf-8")
        if r.returncode != 0:
            raise RuntimeError(f"command failed ({r.returncode}): {r.stderr[:200]}")
        return ModelResult(text=r.stdout.strip())


def _retryable_exception(exc: Exception) -> bool:
    explicit = getattr(exc, "retryable", None)
    if explicit is not None:
        return bool(explicit)
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "status", None)
    if isinstance(status, int):
        return status in (408, 409, 425, 429) or status >= 500
    # Programming/configuration errors should not be multiplied by retries.
    if isinstance(exc, (ValueError, TypeError, AssertionError, KeyError)):
        return False
    return True


class RetryingBackend(ModelBackend):
    """Retry transient failures with exponential backoff."""

    def __init__(self, inner: ModelBackend, retries: int = 3, backoff: float = 2.0,
                 sleep: Callable[[float], None] = time.sleep):
        self.inner, self.retries, self.backoff, self.sleep = inner, retries, backoff, sleep

    def cache_identity(self) -> dict[str, Any]:
        return self.inner.cache_identity()

    def generate(self, *, model: str, messages: list[dict[str, str]]) -> ModelResult:
        for attempt in range(self.retries + 1):
            try:
                return self.inner.generate(model=model, messages=messages)
            except Exception as exc:
                if attempt == self.retries or not _retryable_exception(exc):
                    raise
                self.sleep(self.backoff * (2 ** attempt))
        raise AssertionError("unreachable")


class CachedBackend(ModelBackend):
    """Deterministic append-only disk cache (JSON lines), safe to share between threads.

    A truncated final line (for example from a killed run) is ignored on load.
    """

    def __init__(self, inner: ModelBackend, path: str | Path = ".model_refiner_cache.jsonl"):
        self.inner = inner
        self.path = Path(path)
        self._lock = threading.Lock()
        self.cache: dict[str, dict[str, Any]] = {}
        self.hits = self.misses = 0
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                    self.cache[row["key"]] = row["value"]
                except (ValueError, KeyError, TypeError):
                    continue

    def _key(self, model: str, messages: list[dict[str, str]]) -> str:
        payload = json.dumps(
            {"backend": self.inner.cache_identity(), "model": model, "messages": messages},
            sort_keys=True, ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def cache_identity(self) -> dict[str, Any]:
        return self.inner.cache_identity()

    def generate(self, *, model: str, messages: list[dict[str, str]]) -> ModelResult:
        key = self._key(model, messages)
        with self._lock:
            obj = self.cache.get(key)
        if obj is not None:
            self.hits += 1
            return ModelResult(text=obj["text"], input_tokens=obj.get("input_tokens"),
                               output_tokens=obj.get("output_tokens"),
                               metadata={**obj.get("metadata", {}), "cache_hit": True})
        result = self.inner.generate(model=model, messages=messages)
        value = {"text": result.text, "input_tokens": result.input_tokens,
                 "output_tokens": result.output_tokens, "metadata": result.metadata}
        with self._lock:
            self.misses += 1
            self.cache[key] = value
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")
        return result


class SQLiteCachedBackend(ModelBackend):
    """Process-safe SQLite request cache for expensive fitting/evaluation calls.

    Prompts and replies are stored in plaintext. Use only on a protected volume.
    """

    def __init__(self, inner: ModelBackend, path: str | Path):
        self.inner = inner
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._counts_lock = threading.Lock()
        self.hits = self.misses = 0
        with self._connect() as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute(
                "CREATE TABLE IF NOT EXISTS response_cache (key TEXT PRIMARY KEY, value_json TEXT NOT NULL, created_at REAL NOT NULL)"
            )

    @contextmanager
    def _connect(self):
        con = sqlite3.connect(self.path, timeout=30.0)
        con.execute("PRAGMA busy_timeout=30000")
        try:
            yield con
            con.commit()
        except BaseException:
            con.rollback()
            raise
        finally:
            con.close()

    def cache_identity(self) -> dict[str, Any]:
        return self.inner.cache_identity()

    def _key(self, model: str, messages: list[dict[str, str]]) -> str:
        payload = json.dumps(
            {"backend": self.inner.cache_identity(), "model": model, "messages": messages},
            sort_keys=True, ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def generate(self, *, model: str, messages: list[dict[str, str]]) -> ModelResult:
        key = self._key(model, messages)
        with self._connect() as con:
            row = con.execute("SELECT value_json FROM response_cache WHERE key=?", (key,)).fetchone()
        if row is not None:
            obj = json.loads(row[0])
            with self._counts_lock:
                self.hits += 1
            return ModelResult(
                text=obj["text"], input_tokens=obj.get("input_tokens"), output_tokens=obj.get("output_tokens"),
                metadata={**obj.get("metadata", {}), "cache_hit": True},
            )
        result = self.inner.generate(model=model, messages=messages)
        value = {"text": result.text, "input_tokens": result.input_tokens,
                 "output_tokens": result.output_tokens, "metadata": result.metadata}
        with self._connect() as con:
            con.execute(
                "INSERT OR IGNORE INTO response_cache(key,value_json,created_at) VALUES(?,?,?)",
                (key, json.dumps(value, ensure_ascii=False), time.time()),
            )
        with self._counts_lock:
            self.misses += 1
        return result



def make_backend(provider: str, *, base_url: str | None = None, key_env: str | None = None,
                 command: str | None = None, max_tokens: int = 256, timeout: float = 120) -> ModelBackend:
    """Build an uncached, un-retried backend by provider name."""
    if provider == "openai":
        return OpenAIResponsesBackend(max_output_tokens=max_tokens, timeout=timeout)
    if provider == "openai-compatible":
        return OpenAIChatBackend(base_url or "https://api.openai.com/v1", key_env or "OPENAI_API_KEY", max_tokens, timeout=timeout)
    if provider == "anthropic":
        return AnthropicBackend(base_url or "https://api.anthropic.com", key_env or "ANTHROPIC_API_KEY", max_tokens, timeout=timeout)
    if provider == "command":
        if not command:
            raise ValueError("--command is required for the command provider")
        return CommandBackend(command, timeout=timeout)
    if provider == "toy":
        from .toy import ToyBackend
        return ToyBackend()
    raise ValueError(f"Unknown provider {provider!r}; choose from {', '.join(PROVIDERS)}")
