from __future__ import annotations

import json
import logging
import os
import re
import time

from .settings import ModelConfig

logger = logging.getLogger(__name__)


def write_sampling_meta(client: "LLMClient", run_dir: str, stage: str) -> None:
    path = os.path.join(run_dir, "sampling_meta.json")
    data: dict = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
        except (OSError, json.JSONDecodeError):
            data = {}
    data[stage] = client.effective_sampling()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

_OPENAI_STYLE = {"openai", "deepseek", "gemini", "openai_compatible"}


_PARAM_CONSTRAINT_RE = re.compile(
    r"invalid\s+([A-Za-z_]+)\s*:\s*only\s+([0-9.]+)\s+is\s+allowed", re.IGNORECASE
)


def _parse_param_constraint(text: str):
    m = _PARAM_CONSTRAINT_RE.search(text or "")
    if not m:
        return None
    name = m.group(1)
    try:
        val = float(m.group(2))
    except ValueError:
        return None
    if name in ("max_tokens", "n", "top_k", "best_of"):
        val = int(val)
    return name, val


class LLMClient:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.provider = cfg.provider
        self._forced_params: dict = {}
        self._client = self._build_client()

    def _build_client(self):
        if self.provider == "anthropic":
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError("缺少 anthropic 包, 请先 pip install anthropic") from exc
            kwargs = {"api_key": self.cfg.api_key, "timeout": self.cfg.request_timeout}
            if self.cfg.base_url:
                kwargs["base_url"] = self.cfg.base_url
            return anthropic.Anthropic(**kwargs)

        if self.provider in _OPENAI_STYLE:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError("缺少 openai 包, 请先 pip install openai") from exc
            kwargs = {"api_key": self.cfg.api_key, "timeout": self.cfg.request_timeout}
            if self.cfg.base_url:
                kwargs["base_url"] = self.cfg.base_url
            return OpenAI(**kwargs)

        raise ValueError(f"未知 provider: {self.provider}")


    def effective_sampling(self) -> dict:
        configured = {
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "top_p": self.cfg.top_p,
        }
        effective = dict(configured)
        effective.update(self._forced_params)
        return {
            "model": self.cfg.model_name,
            "configured": configured,
            "forced_by_model": dict(self._forced_params),
            "effective": effective,
        }

    def complete(self, user_prompt: str) -> str:
        last_err: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                text = self._call_once(user_prompt)
                if text is None:
                    raise RuntimeError("模型返回空内容")
                return text.strip()
            except Exception as exc:
                last_err = exc
                if attempt < self.cfg.max_retries:
                    time.sleep(self.cfg.retry_base_delay * (2 ** attempt))
        assert last_err is not None
        raise last_err

    def _call_once(self, user_prompt: str) -> str | None:
        if self.provider == "anthropic":
            return self._call_anthropic(user_prompt)
        return self._call_openai(user_prompt)

    def _call_openai(self, user_prompt: str) -> str | None:
        messages = []
        if self.cfg.system_prompt:
            messages.append({"role": "system", "content": self.cfg.system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        def _kwargs():
            kw = {
                "model": self.cfg.model_name,
                "messages": messages,
                "temperature": self.cfg.temperature,
                "max_tokens": self.cfg.max_tokens,
            }
            if self.cfg.top_p is not None:
                kw["top_p"] = self.cfg.top_p
            if self.cfg.extra_body:
                kw["extra_body"] = self.cfg.extra_body
            kw.update(self._forced_params)
            return kw


        applied: set = set()
        for _ in range(6):
            try:
                resp = self._client.chat.completions.create(**_kwargs())
                return resp.choices[0].message.content
            except Exception as exc:
                pc = _parse_param_constraint(str(exc))
                if not pc or pc[0] in applied:
                    raise
                applied.add(pc[0])
                if self._forced_params.get(pc[0]) != pc[1]:
                    self._forced_params[pc[0]] = pc[1]
                    logger.warning(
                        "模型 %s 不接受 %s 的设定，已自动改用该模型唯一允许值 %s=%s"
                        "（模型强制；本次在该参数上偏离原文，请在结论中注明）。",
                        self.cfg.model_name, pc[0], pc[0], pc[1],
                    )
        return self._client.chat.completions.create(**_kwargs()).choices[0].message.content

    def _call_anthropic(self, user_prompt: str) -> str | None:
        kwargs = {
            "model": self.cfg.model_name,
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if self.cfg.top_p is not None:
            kwargs["top_p"] = self.cfg.top_p
        if self.cfg.system_prompt:
            kwargs["system"] = self.cfg.system_prompt
        if self.cfg.extra_body:
            kwargs["extra_body"] = self.cfg.extra_body
        resp = self._client.messages.create(**kwargs)
        parts = [
            block.text
            for block in resp.content
            if getattr(block, "type", None) == "text"
        ]
        return "".join(parts)
