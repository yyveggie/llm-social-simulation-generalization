from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import LLMSpec

logger = logging.getLogger(__name__)


_MIN_POSITIVE_TOP_P = 0.01


_THINKING_OFF_DEEPSEEK = {
    "thinking": {"type": "disabled"},
    "enable_thinking": False,
    "chat_template_kwargs": {"thinking": False},
}
_THINKING_OFF_TYPE = {"thinking": {"type": "disabled"}}
_THINKING_OFF_FLAG = {"enable_thinking": False}
_THINKING_OFF_GEMINI = {"extra_body": {"google": {"thinking_config": {"thinking_budget": 0}}}}


def _thinking_off_overrides(model: str | None, base_url: str | None) -> dict:


    if os.environ.get("LLM_THINKING", "").lower() == "on":
        return {}
    text = f"{model or ''} {base_url or ''}".lower()

    if "127.0.0.1" in text or "localhost" in text:
        return {}
    if "deepseek" in text:
        return _THINKING_OFF_DEEPSEEK
    if "gemini" in text and ("gemini-3" in text or "2.5" in text):
        return _THINKING_OFF_GEMINI
    if any(k in text for k in ("kimi", "moonshot", "glm", "zhipu", "bigmodel", "doubao", "volces")):
        return _THINKING_OFF_TYPE
    if any(k in text for k in ("qwen", "dashscope", "aliyuncs", "bailian")):
        return _THINKING_OFF_FLAG
    if any(k in text for k in ("claude", "anthropic", "minimax")):
        return {}
    return {"reasoning_effort": "none"}


@dataclass
class LLMResponse:
    text: str
    raw: dict | None = None


class LLMProviderError(RuntimeError):

    def __init__(self, message: str, raw: dict | None = None):
        super().__init__(message)
        self.raw = raw


class EmptyLLMResponseError(LLMProviderError):
    pass


_PARAM_CONSTRAINT_RE = re.compile(
    r"invalid\s+([A-Za-z_]+)\s*:\s*only\s+([0-9.]+)\s+is\s+allowed", re.IGNORECASE
)
_TOP_P_POSITIVE_RANGE_RE = re.compile(
    r"top_p[^\"'}]*between\s+0\s*\(\s*exclusive\s*\)\s+and\s+1\s*\(\s*inclusive\s*\)|"
    r"top_p[^\"'}]*(?:must\s+be\s+)?(?:greater\s+than|>)\s*0",
    re.IGNORECASE,
)


def _parse_param_constraint(text: str):
    m = _PARAM_CONSTRAINT_RE.search(text or "")
    if m:
        name = m.group(1)
        try:
            val = float(m.group(2))
        except ValueError:
            return None
        if name in ("max_tokens", "n", "top_k", "best_of"):
            val = int(val)
        return name, val, "only_allowed"
    if _TOP_P_POSITIVE_RANGE_RE.search(text or ""):
        return "top_p", _MIN_POSITIVE_TOP_P, "positive_range"
    return None


def _glm_like_endpoint(model: str | None, base_url: str | None) -> bool:
    text = f"{model or ''} {base_url or ''}".lower()
    return any(key in text for key in ("glm", "zhipu", "bigmodel"))


def _response_payload(response: httpx.Response) -> dict:
    try:
        body = response.json()
    except Exception:
        body = response.text
    return {
        "http_status": response.status_code,
        "url": str(response.request.url),
        "body": body,
    }


class LLMClient:

    def __init__(self, model_cfg: LLMSpec, request_timeout: int = 120, max_retries: int = 5):
        self.cfg = model_cfg
        self.request_timeout = request_timeout
        self.max_retries = max_retries

        self._forced_params: dict = {}
        self._sampling_warnings_seen: set[str] = set()
        self._impl = self._build()


    def _build(self):
        kind = self.cfg.kind
        if kind in ("openai_chat", "openai_responses"):
            base = (self.cfg.base_url or "https://api.openai.com/v1").rstrip("/")
            headers = {
                "Authorization": f"Bearer {self.cfg.api_key}",
                "Content-Type": "application/json",
            }
            return httpx.Client(
                base_url=base,
                headers=headers,
                timeout=self.request_timeout,
            )
        if kind == "anthropic":


            from anthropic import Anthropic
            return Anthropic(
                api_key=self.cfg.api_key,
                timeout=self.request_timeout,
                max_retries=0,
            )
        raise ValueError(f"Unknown provider kind: {kind}")


    def complete(self, prompt: str, overrides: dict | None = None) -> LLMResponse:
        return self._complete_with_retry(prompt, overrides or {})


    def _complete_with_retry(self, prompt: str, overrides: dict) -> LLMResponse:
        @retry(
            reraise=True,
            stop=stop_after_attempt(max(1, self.max_retries)),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type(_TRANSIENT_ERRORS),
        )
        def _do() -> LLMResponse:
            return self._dispatch(prompt, overrides)

        return _do()

    def _dispatch(self, prompt: str, overrides: dict) -> LLMResponse:
        kind = self.cfg.kind
        if kind == "openai_chat":
            return self._openai_chat(prompt, overrides)
        if kind == "openai_responses":
            return self._openai_responses(prompt, overrides)
        if kind == "anthropic":
            return self._anthropic(prompt, overrides)
        raise ValueError(f"Unknown provider kind: {kind}")


    def _openai_chat(self, prompt: str, overrides: dict) -> LLMResponse:
        messages = []
        if self.cfg.system:
            messages.append({"role": "system", "content": self.cfg.system})
        messages.append({"role": "user", "content": prompt})
        local_forced_params: dict = {}
        request_adjustments: list[dict] = []

        def _remember_adjustment(param: str, original, sent, reason: str) -> None:
            if original == sent:
                return
            rec = {"param": param, "original": original, "sent": sent, "reason": reason}
            if rec not in request_adjustments:
                request_adjustments.append(rec)

        def _positive_top_p_for_endpoint(value):
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return value
            if numeric <= 0 and _glm_like_endpoint(self.cfg.model, self.cfg.base_url):
                warn_key = "glm_positive_top_p"
                if warn_key not in self._sampling_warnings_seen:
                    self._sampling_warnings_seen.add(warn_key)
                    logger.warning(
                        "模型 %s / 端点不接受 top_p=0；发送时自动改用 top_p=%g 作为近似 0 的兼容值"
                        "（param_sweep 条件名仍保留 p0，请在结论中注明）。",
                        self.cfg.model, _MIN_POSITIVE_TOP_P,
                    )
                _remember_adjustment("top_p", value, _MIN_POSITIVE_TOP_P, "model_requires_positive_top_p")
                return _MIN_POSITIVE_TOP_P
            return value

        def _body() -> dict:
            b = {
                "model": self.cfg.model,
                "messages": messages,
                "temperature": overrides.get("temperature", self.cfg.temperature),
                "max_tokens": overrides.get("max_tokens", self.cfg.max_tokens),
            }
            tp = overrides.get("top_p", self.cfg.top_p)
            if tp is not None:
                b["top_p"] = _positive_top_p_for_endpoint(tp)
            b.update(_thinking_off_overrides(self.cfg.model, self.cfg.base_url))
            b.update(local_forced_params)
            b.update(self._forced_params)
            return b


        applied: set[str] = set()
        body = _body()
        r = self._impl.post("/chat/completions", json=body)
        for _ in range(6):
            if r.status_code != 400:
                break
            pc = _parse_param_constraint(r.text)
            if not pc or pc[0] in applied:
                break
            name, value, reason = pc
            applied.add(name)
            attempted = body.get(name)
            _remember_adjustment(name, attempted, value, f"provider_400_{reason}")
            if reason == "only_allowed":
                self._forced_params[name] = value
            else:
                local_forced_params[name] = value
            warn_key = f"{name}:{value}:{reason}"
            if warn_key not in self._sampling_warnings_seen:
                self._sampling_warnings_seen.add(warn_key)
                logger.warning(
                    "模型 %s 不接受请求参数 %s=%r，已自动改用兼容值 %s=%s"
                    "（模型强制；本次在该参数上偏离原文，请在结论中注明）。",
                    self.cfg.model, name, attempted, name, value,
                )
            body = _body()
            r = self._impl.post("/chat/completions", json=body)
        if r.status_code >= 400:

            if r.status_code in (408, 409, 429) or 500 <= r.status_code < 600:
                raise httpx.HTTPStatusError(
                    f"transient HTTP {r.status_code}: {r.text[:200]}",
                    request=r.request, response=r,
                )
            raw_payload = _response_payload(r)
            if request_adjustments:
                raw_payload["_client_sampling_adjustments"] = request_adjustments
            raise LLMProviderError(f"HTTP {r.status_code}: {r.text[:500]}", raw=raw_payload)
        data = r.json()
        choices = data.get("choices") or []
        if not choices:
            raise LLMProviderError("OpenAI-compatible response has no choices", raw=data)
        choice0 = choices[0]
        message = choice0.get("message") or {}
        text = (message.get("content") or "").strip()
        if not text:
            finish = choice0.get("finish_reason")
            usage = data.get("usage") or {}
            raise EmptyLLMResponseError(
                f"empty assistant content (finish_reason={finish!r}, usage={usage})",
                raw=data,
            )
        if request_adjustments:
            data = dict(data)
            data["_client_sampling_adjustments"] = request_adjustments
        return LLMResponse(text=text, raw=data)

    def _openai_responses(self, prompt: str, overrides: dict) -> LLMResponse:
        body: dict = {
            "model": self.cfg.model,
            "input": prompt,
            "max_output_tokens": overrides.get("max_tokens", self.cfg.max_tokens),
        }
        temp = overrides.get("temperature", self.cfg.temperature)
        if abs(temp - 1.0) > 1e-9:
            body["temperature"] = temp
        top_p = overrides.get("top_p", self.cfg.top_p)

        if top_p is not None and abs(top_p - 1.0) > 1e-9:
            body["top_p"] = top_p
        if self.cfg.system:
            body["instructions"] = self.cfg.system
        r = self._impl.post("/responses", json=body)
        if r.status_code >= 400:
            if r.status_code in (408, 409, 429) or 500 <= r.status_code < 600:
                raise httpx.HTTPStatusError(
                    f"transient HTTP {r.status_code}: {r.text[:200]}",
                    request=r.request, response=r,
                )
            raise LLMProviderError(f"HTTP {r.status_code}: {r.text[:500]}", raw=_response_payload(r))
        data = r.json()
        text = data.get("output_text") or ""
        if not text:
            for item in data.get("output", []) or []:
                for c in item.get("content", []) or []:
                    if c.get("type") == "output_text" and c.get("text"):
                        text += c["text"]
        text = text.strip()
        if not text:
            status = data.get("status")
            usage = data.get("usage") or {}
            raise EmptyLLMResponseError(
                f"empty Responses API output (status={status!r}, usage={usage})",
                raw=data,
            )
        return LLMResponse(text=text, raw=data)

    def _anthropic(self, prompt: str, overrides: dict) -> LLMResponse:
        kwargs = {
            "model": self.cfg.model,
            "max_tokens": overrides.get("max_tokens", self.cfg.max_tokens),
            "temperature": overrides.get("temperature", self.cfg.temperature),
            "messages": [{"role": "user", "content": prompt}],
        }
        top_p = overrides.get("top_p", self.cfg.top_p)
        if top_p is not None:
            kwargs["top_p"] = top_p
        if self.cfg.system:
            kwargs["system"] = self.cfg.system
        resp = self._impl.messages.create(**kwargs)
        parts = []
        for block in resp.content or []:
            t = getattr(block, "text", None)
            if t:
                parts.append(t)
        text = "".join(parts).strip()
        if not text:
            stop_reason = getattr(resp, "stop_reason", None)
            usage = getattr(resp, "usage", None)
            raise EmptyLLMResponseError(
                f"empty Anthropic message content (stop_reason={stop_reason!r}, usage={usage!r})"
            )
        try:
            raw = resp.model_dump()
        except Exception:
            raw = None
        return LLMResponse(text=text, raw=raw)


def _transient_errors() -> tuple[type[BaseException], ...]:
    errs: list[type[BaseException]] = [
        TimeoutError,
        ConnectionError,
        httpx.TimeoutException,
        httpx.NetworkError,
        httpx.HTTPStatusError,
    ]
    try:
        import anthropic
        errs.extend([
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
            anthropic.RateLimitError,
            anthropic.InternalServerError,
        ])
    except ImportError:
        pass
    return tuple(errs)


_TRANSIENT_ERRORS = _transient_errors()
