from __future__ import annotations

import json
import os
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parent
ORCHESTRATOR_DIR = BACKEND_DIR.parent
REPO_ROOT = ORCHESTRATOR_DIR.parent

UNIFIED_CONFIG_PATH = ORCHESTRATOR_DIR / "unified_config.yaml"
MODEL_CONFIG_PATH = REPO_ROOT / "model_config.json"
RUNS_DIR = ORCHESTRATOR_DIR / "runs"
FRONTEND_DIST = ORCHESTRATOR_DIR / "frontend" / "dist"
LLM_PROXY_BASE_URL = os.environ.get("ORCH_LLM_PROXY_BASE_URL", "http://127.0.0.1:8000/api/llm-proxy/v1")

_ENV_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


PROJECT_DIRS: dict[str, str] = {
    "altruism": "LLM-Altruism",
    "behavioral": "LLM-Behavioral",
    "identity": "LLM-Can-Harmfully-Misportray-and-Flatten-Identity-Groups",
    "social": "LLM-Social-Conventions-and-Collective-Bias",
    "age_gender_distortion": "LLM-Age-Gender-Distortion",
}


def project_path(project_id: str) -> Path:
    return REPO_ROOT / PROJECT_DIRS[project_id]


def resolve_env(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_PLACEHOLDER.sub(lambda m: os.environ.get(m.group(1), m.group(0)), value)
    if isinstance(value, list):
        return [resolve_env(v) for v in value]
    if isinstance(value, dict):
        return {k: resolve_env(v) for k, v in value.items()}
    return value


def load_raw_config() -> dict[str, Any]:
    load_dotenv(REPO_ROOT / ".env", override=False)
    load_dotenv(ORCHESTRATOR_DIR / ".env", override=False)
    if not UNIFIED_CONFIG_PATH.exists():
        raise FileNotFoundError(f"未找到统一配置 {UNIFIED_CONFIG_PATH}")
    return yaml.safe_load(UNIFIED_CONFIG_PATH.read_text(encoding="utf-8")) or {}


def save_raw_config(data: dict[str, Any]) -> None:
    UNIFIED_CONFIG_PATH.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def is_env_placeholder(value: str) -> bool:
    return bool(value and _ENV_PLACEHOLDER.fullmatch(value.strip()))


def api_key_is_literal(raw: dict[str, Any]) -> bool:
    key = str((raw.get("llm") or {}).get("api_key") or "")
    return bool(key.strip()) and not is_env_placeholder(key)


def mask_config_for_client(raw: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(raw)
    llm = out.setdefault("llm", {})
    key = str(llm.get("api_key") or "")
    if key.strip() and not is_env_placeholder(key):
        llm["api_key"] = ""
    return out


def merge_config_on_save(incoming: dict[str, Any], existing: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(incoming)
    new_llm = out.setdefault("llm", {})
    new_key = str(new_llm.get("api_key") or "").strip()
    if not new_key:
        old_key = str((existing.get("llm") or {}).get("api_key") or "")
        if old_key.strip():
            new_llm["api_key"] = old_key
    return out


def load_model_catalog() -> list[dict[str, Any]]:
    if not MODEL_CONFIG_PATH.is_file():
        return []
    try:
        raw = json.loads(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(raw, list):
        return []
    catalog: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        catalog.append({"name": name, "provider": entry.get("provider")})
    return catalog


def python_executable(raw_cfg: dict[str, Any]) -> str:
    exe = (raw_cfg or {}).get("python_executable")
    if exe:
        return str(exe)
    return sys.executable


def proxy_max_from_raw(raw: dict[str, Any]) -> int:
    proxy = raw.get("proxy") or {}
    try:
        return max(1, int(proxy.get("max_upstream_concurrency") or 4))
    except (TypeError, ValueError):
        return 4
