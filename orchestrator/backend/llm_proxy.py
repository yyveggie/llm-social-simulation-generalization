from __future__ import annotations

import http.client
import json
import os
import random
import threading
import time
import urllib.parse
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import HTTPException, Request
from pydantic import ValidationError

from . import settings, token_ledger
from .schemas import OpenAIChatResponse, UnifiedLLM

_LOCK = threading.Lock()
_TOKENS: dict[str, UnifiedLLM] = {}
_TOKEN_JOBS: dict[str, str] = {}
_TOKEN_CALLS: dict[str, int] = {}
_TOKEN_CREATED: dict[str, float] = {}


_JOB_CALLS_BASE: dict[str, int] = {}
_TOKEN_PREFIX = "orch-proxy-"
_ORPHAN_TOKEN_MAX_AGE = 24 * 3600.0


_TOKENS_PATH = settings.RUNS_DIR / "llm_proxy_tokens.json"
_TOKENS_SCHEMA = "llm_proxy_tokens/v2"
_TOKENS_LOADED = False
_TOKENS_LAST_SAVE = 0.0
_TOKENS_SAVE_INTERVAL = 2.0
_DEFAULT_TIMEOUT = 120
_UPSTREAM_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


_MIN_POSITIVE_TOP_P = 0.01


def _env_int(name: str, default: int, minimum: int) -> int:
    try:
        return max(minimum, int(os.environ.get(name, str(default)) or default))
    except ValueError:
        return max(minimum, default)


def _env_float(name: str, default: float, minimum: float) -> float:
    try:
        return max(minimum, float(os.environ.get(name, str(default)) or default))
    except ValueError:
        return max(minimum, default)


_UPSTREAM_MIN_INTERVAL = _env_float("ORCH_LLM_PROXY_MIN_INTERVAL", 0.05, 0.0)
_UPSTREAM_RETRIES = _env_int("ORCH_LLM_PROXY_UPSTREAM_RETRIES", 4, 1)


class _UpstreamGate:

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._max = 4
        self._sem = threading.BoundedSemaphore(self._max)

    def set_max(self, n: int) -> None:
        n = max(1, int(n))
        with self._lock:
            if n == self._max:
                return
            self._max = n
            self._sem = threading.BoundedSemaphore(n)

    def acquire(self) -> "threading.BoundedSemaphore":


        with self._lock:
            sem = self._sem
        sem.acquire()
        return sem

    @staticmethod
    def release(sem: "threading.BoundedSemaphore") -> None:
        sem.release()

    @property
    def max_concurrency(self) -> int:
        with self._lock:
            return self._max


_UPSTREAM_GATE = _UpstreamGate()


def apply_from_raw_config(raw: dict[str, Any]) -> None:
    n = settings.proxy_max_from_raw(raw)
    env_override = os.environ.get("ORCH_LLM_PROXY_MAX_UPSTREAM_CONCURRENCY")
    if env_override:
        try:
            n = max(1, int(env_override))
        except ValueError:
            pass
    _UPSTREAM_GATE.set_max(n)


def upstream_limits() -> dict[str, float | int]:
    return {
        "max_concurrency": _UPSTREAM_GATE.max_concurrency,
        "min_interval": _UPSTREAM_MIN_INTERVAL,
    }


try:
    apply_from_raw_config(settings.load_raw_config())
except Exception:
    _UPSTREAM_GATE.set_max(_env_int("ORCH_LLM_PROXY_MAX_UPSTREAM_CONCURRENCY", 4, 1))
_UPSTREAM_PACE_LOCK = threading.Lock()
_UPSTREAM_NEXT_AT = 0.0


_POOL = threading.local()


def proxy_base_url() -> str:
    return (
        str(getattr(settings, "LLM_PROXY_BASE_URL", "") or "")
        or "http://127.0.0.1:8000/api/llm-proxy/v1"
    ).rstrip("/")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _save_tokens_locked() -> None:
    data = {
        "schema": _TOKENS_SCHEMA,
        "updated_at": _now_iso(),
        "tokens": {
            token: {
                "llm": llm.model_dump(),
                "job_id": _TOKEN_JOBS.get(token),
                "calls": int(_TOKEN_CALLS.get(token, 0)),
                "created_at": float(_TOKEN_CREATED.get(token, 0.0)),
            }
            for token, llm in _TOKENS.items()
        },
        "job_calls_base": {job: int(n) for job, n in _JOB_CALLS_BASE.items() if n},
    }
    try:
        settings.RUNS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _TOKENS_PATH.with_name(_TOKENS_PATH.name + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(_TOKENS_PATH)
    except Exception:
        pass


def _maybe_save_tokens_locked() -> None:
    global _TOKENS_LAST_SAVE
    now = time.monotonic()
    if now - _TOKENS_LAST_SAVE < _TOKENS_SAVE_INTERVAL:
        return
    _TOKENS_LAST_SAVE = now
    _save_tokens_locked()


def load_tokens() -> None:
    global _TOKENS_LOADED
    with _LOCK:
        if _TOKENS_LOADED:
            return
        _TOKENS_LOADED = True
        data: Any = None
        if _TOKENS_PATH.is_file():
            try:
                data = json.loads(_TOKENS_PATH.read_text(encoding="utf-8"))
            except Exception:
                data = None
        if not isinstance(data, dict):
            return
        base = data.get("job_calls_base")
        if isinstance(base, dict):
            for job, n in base.items():
                try:
                    _JOB_CALLS_BASE.setdefault(str(job), int(n))
                except (TypeError, ValueError):
                    continue
        tokens = data.get("tokens")
        if not isinstance(tokens, dict):
            return
        for token, rec in tokens.items():
            if not (isinstance(token, str) and token.startswith(_TOKEN_PREFIX)):
                continue
            if not isinstance(rec, dict):
                continue
            try:
                llm = UnifiedLLM(**(rec.get("llm") or {}))
            except (ValidationError, TypeError):
                continue
            _TOKENS.setdefault(token, llm)
            job_id = rec.get("job_id")
            if isinstance(job_id, str) and job_id:
                _TOKEN_JOBS.setdefault(token, job_id)
            _TOKEN_CALLS.setdefault(token, int(rec.get("calls") or 0))
            try:
                _TOKEN_CREATED.setdefault(token, float(rec.get("created_at") or 0.0))
            except (TypeError, ValueError):
                _TOKEN_CREATED.setdefault(token, 0.0)


def flush_tokens() -> None:
    with _LOCK:
        _save_tokens_locked()


def register_llm(llm: UnifiedLLM) -> dict[str, str]:
    load_tokens()
    token = _TOKEN_PREFIX + uuid4().hex
    with _LOCK:
        _TOKENS[token] = llm.model_copy(deep=True)
        _TOKEN_CALLS[token] = 0
        _TOKEN_CREATED[token] = time.time()
        _save_tokens_locked()
    return {"base_url": proxy_base_url(), "api_key": token, "model": llm.model}


def _retire_token_locked(token: str) -> None:
    job_id = _TOKEN_JOBS.pop(token, None)
    calls = int(_TOKEN_CALLS.pop(token, 0) or 0)
    _TOKENS.pop(token, None)
    _TOKEN_CREATED.pop(token, None)
    if job_id and calls:
        _JOB_CALLS_BASE[job_id] = _JOB_CALLS_BASE.get(job_id, 0) + calls


def retire_job_tokens(job_id: str, *, drop_base: bool = False,
                      created_before: float | None = None) -> None:
    if not job_id:
        return
    with _LOCK:
        stale = [
            t for t, j in _TOKEN_JOBS.items()
            if j == job_id
            and (created_before is None or float(_TOKEN_CREATED.get(t, 0.0)) < created_before)
        ]
        for t in stale:
            _retire_token_locked(t)
        if drop_base:
            _JOB_CALLS_BASE.pop(job_id, None)
        if stale or drop_base:
            _save_tokens_locked()


def attach_tokens_to_job(tokens: "str | None | list[str | None]", job_id: str) -> None:
    if isinstance(tokens, str) or tokens is None:
        tokens = [tokens]
    keep = [t for t in tokens if t]
    if not keep:
        return
    with _LOCK:
        attached = {t for t in keep if t in _TOKENS}
        if not attached:
            return
        for t in attached:
            _TOKEN_JOBS[t] = job_id
        for t in [t for t, j in _TOKEN_JOBS.items() if j == job_id and t not in attached]:
            _retire_token_locked(t)
        _save_tokens_locked()


def prune_tokens(valid_job_ids: set[str], active_job_ids: set[str]) -> None:
    load_tokens()
    now = time.time()
    with _LOCK:
        changed = False
        for token in list(_TOKENS):
            job_id = _TOKEN_JOBS.get(token)
            if job_id:
                if job_id in active_job_ids:
                    continue
                _retire_token_locked(token)
                if job_id not in valid_job_ids:
                    _JOB_CALLS_BASE.pop(job_id, None)
                changed = True
            elif now - float(_TOKEN_CREATED.get(token, 0.0)) > _ORPHAN_TOKEN_MAX_AGE:
                _retire_token_locked(token)
                changed = True
        for job_id in [j for j in _JOB_CALLS_BASE if j not in valid_job_ids]:
            _JOB_CALLS_BASE.pop(job_id, None)
            changed = True
        if changed:
            _save_tokens_locked()


def calls_for_job(job_id: str) -> int:
    with _LOCK:
        live = sum(count for token, count in _TOKEN_CALLS.items() if _TOKEN_JOBS.get(token) == job_id)
        return _JOB_CALLS_BASE.get(job_id, 0) + live


def token_from_request(req: Request) -> str:
    auth = (req.headers.get("authorization") or "").strip()
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return (req.headers.get("x-api-key") or "").strip()


def chat_completions_response(token: str, body: dict[str, Any]) -> dict[str, Any]:
    llm = _llm_for_token(token)
    with _LOCK:
        _TOKEN_CALLS[token] = _TOKEN_CALLS.get(token, 0) + 1
        _maybe_save_tokens_locked()
    sem = _UPSTREAM_GATE.acquire()
    try:
        _pace_upstream_requests()
        text, raw = complete_chat(llm, body)
    finally:
        _UPSTREAM_GATE.release(sem)
    usage = _usage(raw)
    token_ledger.record(llm.model, usage)
    now = int(time.time())
    out = {
        "id": "chatcmpl-orch-" + uuid4().hex[:12],
        "object": "chat.completion",
        "created": now,
        "model": llm.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": usage,
        "orch_upstream_raw": raw,
    }
    if not text and isinstance(raw, dict):
        out["orch_upstream_error"] = json.loads(_openai_empty_content_detail(raw))
    return out


def complete_chat(llm: UnifiedLLM, body: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    kind = llm.provider_kind
    if kind == "openai_chat":
        return _complete_openai_chat(llm, body)
    if kind == "openai_responses":
        return _complete_openai_responses(llm, body)
    if kind == "anthropic":
        return _complete_anthropic(llm, body)
    if kind == "gemini":
        return _complete_gemini(llm, body)
    if kind == "huggingface":
        return _complete_huggingface(llm, body)
    if kind == "ollama":
        return _complete_ollama(llm, body)
    raise HTTPException(status_code=400, detail=f"未知 provider_kind: {kind}")


def _llm_for_token(token: str) -> UnifiedLLM:
    with _LOCK:
        llm = _TOKENS.get(token)
    if llm is None:
        raise HTTPException(status_code=401, detail="无效或过期的 LLM proxy token。请重新启动实验。")
    return llm


_THINKING_OFF_TYPE = {"thinking": {"type": "disabled"}}
_THINKING_OFF_FLAG = {"enable_thinking": False}
_THINKING_OFF_OPENAI = {"reasoning_effort": "none"}


_THINKING_OFF_DEEPSEEK = {
    "thinking": {"type": "disabled"},
    "enable_thinking": False,
    "chat_template_kwargs": {"thinking": False},
}


_THINKING_OFF_GEMINI_OPENAI = {"extra_body": {"google": {"thinking_config": {"thinking_budget": 0}}}}


def _thinking_off_overrides(llm: UnifiedLLM) -> dict[str, Any]:
    text = f"{llm.model or ''} {llm.base_url or ''}".lower()
    if "deepseek" in text:
        return _THINKING_OFF_DEEPSEEK

    if "gemini" in text and ("gemini-3" in text or "2.5" in text):
        return _THINKING_OFF_GEMINI_OPENAI
    if any(k in text for k in ("kimi", "moonshot", "glm", "zhipu", "bigmodel", "doubao", "volces")):
        return _THINKING_OFF_TYPE
    if any(k in text for k in ("qwen", "dashscope", "aliyuncs", "bailian")):
        return _THINKING_OFF_FLAG
    if any(k in text for k in ("claude", "anthropic")):
        return {}


    if "minimax" in text:
        return {}

    return _THINKING_OFF_OPENAI


def _glm_like_endpoint(llm: UnifiedLLM) -> bool:
    text = f"{llm.model or ''} {llm.base_url or ''}".lower()
    return any(key in text for key in ("glm", "zhipu", "bigmodel"))


def _positive_top_p_compat(llm: UnifiedLLM, value: Any) -> tuple[Any, dict[str, Any] | None]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value, None
    if numeric <= 0 and _glm_like_endpoint(llm):
        return _MIN_POSITIVE_TOP_P, {
            "param": "top_p",
            "original": value,
            "sent": _MIN_POSITIVE_TOP_P,
            "reason": "model_requires_positive_top_p",
        }
    return value, None


def _without_empty_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def _empty(m: Any) -> bool:
        if not isinstance(m, dict):
            return False
        content = m.get("content")
        if content is None:
            return True
        return isinstance(content, str) and not content.strip()

    cleaned = [m for m in messages if not _empty(m)]
    return cleaned or messages


def _complete_openai_chat(llm: UnifiedLLM, body: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    payload = dict(body)
    request_adjustments: list[dict[str, Any]] = []
    payload["model"] = llm.model

    payload["stream"] = False
    if payload.get("top_p") not in (None, ""):
        payload["top_p"], adjustment = _positive_top_p_compat(llm, payload["top_p"])
        if adjustment:
            request_adjustments.append(adjustment)

    if isinstance(payload.get("messages"), list):
        payload["messages"] = _without_empty_messages(payload["messages"])

    for key, value in _thinking_off_overrides(llm).items():
        payload.setdefault(key, value)
    url = (llm.base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
    data = _post_json(url, payload, _bearer_headers(llm.api_key), llm.timeout)
    if request_adjustments and isinstance(data, dict):
        data = dict(data)
        data["orch_request_adjustments"] = request_adjustments
    return _extract_openai_chat(data), data


def _complete_openai_responses(llm: UnifiedLLM, body: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    messages = _messages(body)
    instructions, input_text = _messages_to_responses_input(messages)
    request_adjustments: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "model": llm.model,
        "input": input_text,
    }
    max_tokens = body.get("max_tokens") or body.get("max_completion_tokens") or llm.max_tokens
    if max_tokens not in (None, ""):
        payload["max_output_tokens"] = int(max_tokens)
    temp = body.get("temperature", llm.temperature)
    if temp not in (None, "") and abs(float(temp) - 1.0) > 1e-9:
        payload["temperature"] = float(temp)
    top_p = body.get("top_p")
    if top_p not in (None, "") and abs(float(top_p) - 1.0) > 1e-9:
        adjusted_top_p, adjustment = _positive_top_p_compat(llm, top_p)
        payload["top_p"] = float(adjusted_top_p)
        if adjustment:
            request_adjustments.append(adjustment)
    if instructions:
        payload["instructions"] = instructions

    url = (llm.base_url or "https://api.openai.com/v1").rstrip("/") + "/responses"
    data = _post_json(url, payload, _bearer_headers(llm.api_key), llm.timeout)
    if request_adjustments and isinstance(data, dict):
        data = dict(data)
        data["orch_request_adjustments"] = request_adjustments
    return _extract_responses_text(data), data


def _complete_anthropic(llm: UnifiedLLM, body: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    messages = _messages(body)
    system, anthropic_messages = _messages_to_anthropic(messages)
    payload: dict[str, Any] = {
        "model": llm.model,
        "messages": anthropic_messages,
        "max_tokens": int(body.get("max_tokens") or body.get("max_completion_tokens") or llm.max_tokens),
        "temperature": float(body.get("temperature", llm.temperature)),
    }
    if system:
        payload["system"] = system
    if body.get("top_p") not in (None, ""):
        payload["top_p"] = float(body["top_p"])

    url = (llm.base_url or "https://api.anthropic.com/v1").rstrip("/") + "/messages"
    data = _post_json(url, payload, {
        "x-api-key": llm.api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }, llm.timeout)
    return "".join(block.get("text", "") for block in data.get("content", [])).strip(), data


def _complete_gemini(llm: UnifiedLLM, body: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    messages = _messages(body)
    system, contents = _messages_to_gemini(messages)
    payload: dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "temperature": float(body.get("temperature", llm.temperature)),
            "maxOutputTokens": int(body.get("max_tokens") or body.get("max_completion_tokens") or llm.max_tokens),
        },
    }
    if body.get("top_p") not in (None, ""):
        payload["generationConfig"]["topP"] = float(body["top_p"])


    if "flash" in (llm.model or "").lower() and "2.5" in (llm.model or ""):
        payload["generationConfig"].setdefault("thinkingConfig", {"thinkingBudget": 0})
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}

    base = (llm.base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
    model = urllib.parse.quote(llm.model, safe="")
    url = f"{base}/models/{model}:generateContent?key={urllib.parse.quote(llm.api_key, safe='')}"
    data = _post_json(url, payload, {"content-type": "application/json"}, llm.timeout)
    parts = ((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
    return "".join(part.get("text", "") for part in parts).strip(), data


def _complete_huggingface(llm: UnifiedLLM, body: dict[str, Any]) -> tuple[str, dict[str, Any] | list[Any]]:
    prompt = _messages_to_prompt(_messages(body))
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": float(body.get("temperature", llm.temperature)),
            "max_new_tokens": int(body.get("max_tokens") or body.get("max_completion_tokens") or llm.max_tokens),
        },
        "options": {"use_cache": False},
    }
    if body.get("top_p") not in (None, ""):
        payload["parameters"]["top_p"] = float(body["top_p"])
    api_url = _huggingface_url(llm)
    data = _post_json(api_url, payload, _bearer_headers(llm.api_key), llm.timeout)
    if isinstance(data, list) and data:
        text = data[0].get("generated_text", "")
    elif isinstance(data, dict):
        text = data.get("generated_text") or data.get("text") or ""
    else:
        text = ""
    if text.startswith(prompt):
        text = text[len(prompt):]
    return text.strip(), data


def _complete_ollama(llm: UnifiedLLM, body: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    payload = {
        "model": llm.model,
        "messages": _messages_to_ollama(_messages(body)),
        "stream": False,
        "options": {
            "temperature": float(body.get("temperature", llm.temperature)),
            "num_predict": int(body.get("max_tokens") or body.get("max_completion_tokens") or llm.max_tokens),
        },
    }
    if body.get("top_p") not in (None, ""):
        payload["options"]["top_p"] = float(body["top_p"])
    base = (llm.base_url or "http://localhost:11434").rstrip("/")
    url = base if base.endswith("/api/chat") else base + "/api/chat"
    data = _post_json(url, payload, {"content-type": "application/json"}, llm.timeout)
    msg = data.get("message") or {}
    return (msg.get("content") or data.get("response") or "").strip(), data


def _get_conn(scheme: str, host: str, port: int, timeout: int) -> tuple[http.client.HTTPConnection, tuple]:
    conns: dict = getattr(_POOL, "conns", None)
    if conns is None:
        conns = {}
        _POOL.conns = conns
    key = (scheme, host, port)
    conn = conns.get(key)
    if conn is None:
        cls = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
        conn = cls(host, port, timeout=timeout)
        conns[key] = conn
    else:
        conn.timeout = timeout
        if conn.sock is not None:
            try:
                conn.sock.settimeout(timeout)
            except OSError:
                pass
    return conn, key


def _drop_conn(key: tuple) -> None:
    conns: dict = getattr(_POOL, "conns", None)
    if not conns:
        return
    conn = conns.pop(key, None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass


def _pace_upstream_requests() -> None:
    global _UPSTREAM_NEXT_AT
    interval = _UPSTREAM_MIN_INTERVAL
    if interval <= 0:
        return
    with _UPSTREAM_PACE_LOCK:
        now = time.monotonic()
        wait = max(0.0, _UPSTREAM_NEXT_AT - now)
        _UPSTREAM_NEXT_AT = max(now, _UPSTREAM_NEXT_AT) + interval
    if wait > 0:
        time.sleep(wait)


def _retry_delay(status: int, retry_after: str | None, attempt: int) -> float:
    if retry_after:
        try:
            return min(60.0, max(0.0, float(retry_after.strip())))
        except ValueError:
            pass
    base = 1.5 if status == 429 else 0.8
    return min(30.0, base * (2 ** attempt)) + random.uniform(0.0, 0.4)


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int | None) -> Any:
    parts = urllib.parse.urlsplit(url)
    scheme = parts.scheme or "https"
    host = parts.hostname or ""
    port = parts.port or (443 if scheme == "https" else 80)
    path = parts.path or "/"
    if parts.query:
        path += "?" + parts.query
    body = json.dumps(payload).encode("utf-8")
    hdrs = {
        **headers,
        "Accept": "application/json",
        "Content-Length": str(len(body)),
        "Connection": "keep-alive",
    }
    to = timeout or _DEFAULT_TIMEOUT

    last_exc: Exception | None = None
    for attempt in range(_UPSTREAM_RETRIES):
        conn, key = _get_conn(scheme, host, port, to)
        try:
            conn.request("POST", path, body=body, headers=hdrs)
            resp = conn.getresponse()
            data = resp.read()
            text = data.decode("utf-8", errors="replace")
            if resp.status >= 400:
                detail = text[:1200] or f"Upstream returned HTTP {resp.status} with an empty body"
                if resp.status in _UPSTREAM_RETRYABLE_STATUS and attempt < _UPSTREAM_RETRIES - 1:
                    if resp.status >= 500:
                        _drop_conn(key)
                    time.sleep(_retry_delay(resp.status, resp.getheader("retry-after"), attempt))
                    continue
                raise HTTPException(status_code=resp.status, detail=detail)
            try:
                return json.loads(text)
            except json.JSONDecodeError as exc:
                sse_data = _openai_sse_to_chat_response(text)
                if sse_data is not None:
                    return sse_data
                preview = text[:300] if text else "<empty body>"
                if attempt < _UPSTREAM_RETRIES - 1:
                    _drop_conn(key)
                    time.sleep(_retry_delay(502, None, attempt))
                    continue
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "Upstream returned a non-JSON response; "
                        f"status={resp.status}; body={preview}; parse_error={exc}"
                    ),
                ) from exc
        except HTTPException:
            raise
        except (http.client.HTTPException, OSError, ConnectionError) as exc:
            _drop_conn(key)
            last_exc = exc
            if attempt < _UPSTREAM_RETRIES - 1:
                time.sleep(_retry_delay(502, None, attempt))
                continue
        except Exception as exc:
            _drop_conn(key)
            raise HTTPException(status_code=502, detail=str(exc)) from exc
    raise HTTPException(status_code=502, detail=str(last_exc) if last_exc else "LLM 请求失败")


def _openai_sse_to_chat_response(text: str) -> dict[str, Any] | None:
    if "data:" not in text:
        return None
    chunks: list[str] = []
    model = ""
    usage: dict[str, Any] | None = None
    saw_chunk = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(":") or line.startswith("event:"):
            continue
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            item = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        saw_chunk = True
        model = str(item.get("model") or model)
        if isinstance(item.get("usage"), dict):
            usage = item["usage"]
        for choice in item.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
            message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
            content = delta.get("content", message.get("content"))

            if content is None:
                content = delta.get("text", choice.get("text"))
            if isinstance(content, list):
                chunks.append(_content_text(content))
            elif content is not None:
                chunks.append(str(content))
    if not saw_chunk:
        return None
    return {
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "".join(chunks).strip()},
            "finish_reason": "stop",
        }],
        "model": model,
        "usage": usage or {},
    }


def _bearer_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def _huggingface_url(llm: UnifiedLLM) -> str:
    base = (llm.base_url or "").strip()
    if not base:
        return f"https://api-inference.huggingface.co/models/{llm.model}"
    if "{model}" in base:
        return base.replace("{model}", urllib.parse.quote(llm.model, safe="/"))
    if base.rstrip("/").endswith("/models"):
        return base.rstrip("/") + "/" + llm.model
    return base


def _messages(body: dict[str, Any]) -> list[dict[str, Any]]:
    messages = body.get("messages")
    if isinstance(messages, list):
        return deepcopy(messages)
    text = body.get("input") or body.get("prompt") or ""
    return [{"role": "user", "content": str(text)}]


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                chunks.append(str(item.get("text") or item.get("content") or ""))
            else:
                chunks.append(str(item))
        return "\n".join(c for c in chunks if c)
    return "" if content is None else str(content)


def _messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    lines = []
    for msg in messages:
        role = msg.get("role", "user")
        text = _content_text(msg.get("content"))
        if text:
            lines.append(f"{role}: {text}")
    return "\n".join(lines)


def _messages_to_responses_input(messages: list[dict[str, Any]]) -> tuple[str, str]:
    system = "\n".join(_content_text(m.get("content")) for m in messages if m.get("role") == "system")
    rest = [m for m in messages if m.get("role") != "system"]
    return system, _messages_to_prompt(rest)


def _messages_to_anthropic(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, str]]]:
    system_parts: list[str] = []
    out: list[dict[str, str]] = []
    for msg in messages:
        role = msg.get("role", "user")
        text = _content_text(msg.get("content"))
        if not text:
            continue
        if role == "system":
            system_parts.append(text)
        elif role == "assistant":
            out.append({"role": "assistant", "content": text})
        else:
            out.append({"role": "user", "content": text})
    if not out:
        out.append({"role": "user", "content": ""})
    return "\n".join(system_parts), out


def _messages_to_gemini(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        text = _content_text(msg.get("content"))
        if not text:
            continue
        if role == "system":
            system_parts.append(text)
        else:
            out.append({"role": "model" if role == "assistant" else "user", "parts": [{"text": text}]})
    if not out:
        out.append({"role": "user", "parts": [{"text": ""}]})
    return "\n".join(system_parts), out


def _messages_to_ollama(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    out = []
    for msg in messages:
        role = msg.get("role", "user")
        if role not in ("system", "user", "assistant"):
            role = "user"
        out.append({"role": role, "content": _content_text(msg.get("content"))})
    return out or [{"role": "user", "content": ""}]


def _extract_openai_chat(data: dict[str, Any]) -> str:
    try:
        parsed = OpenAIChatResponse.model_validate(data)
    except ValidationError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"无法解析 OpenAI Chat 响应结构：{exc.errors()}",
        ) from exc
    choice = parsed.choices[0]
    content = choice.message.content
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    return json.dumps(content, ensure_ascii=False)


def _openai_empty_content_detail(data: dict[str, Any]) -> str:
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
    reasoning = msg.get("reasoning_content", delta.get("reasoning_content"))
    if reasoning is None:
        reasoning_len = 0
    else:
        reasoning_len = len(str(reasoning))
    payload = {
        "error": "upstream returned empty assistant content",
        "finish_reason": choice.get("finish_reason"),
        "usage": data.get("usage") or {},
        "choice_keys": sorted(choice.keys()),
        "message_keys": sorted(msg.keys()),
        "content_type": type(msg.get("content")).__name__,
        "reasoning_content_present": reasoning is not None,
        "reasoning_content_length": reasoning_len,
    }
    return json.dumps(payload, ensure_ascii=False)


def _extract_responses_text(data: dict[str, Any]) -> str:
    text = data.get("output_text") or ""
    if text:
        return str(text).strip()
    chunks: list[str] = []
    for item in data.get("output", []) or []:
        for c in item.get("content", []) or []:
            if c.get("type") in ("output_text", "text") and c.get("text"):
                chunks.append(str(c["text"]))
    return "".join(chunks).strip()


def _usage(raw: Any) -> dict[str, int]:
    if isinstance(raw, dict):
        usage = raw.get("usage")
        if isinstance(usage, dict):
            prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
            completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
            total = int(usage.get("total_tokens") or 0) or (prompt + completion)
            return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}
        meta = raw.get("usageMetadata")
        if isinstance(meta, dict):
            prompt = int(meta.get("promptTokenCount") or 0)
            completion = int(meta.get("candidatesTokenCount") or 0)
            total = int(meta.get("totalTokenCount") or 0) or (prompt + completion)
            return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}
        if "prompt_eval_count" in raw or "eval_count" in raw:
            prompt = int(raw.get("prompt_eval_count") or 0)
            completion = int(raw.get("eval_count") or 0)
            return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
