from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"
DEFAULT_CONFIG = CONFIG_DIR / "config.yaml"

_ENV_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _resolve_env(value: Any) -> Any:
    if isinstance(value, str):
        def _sub(m: re.Match[str]) -> str:
            return os.environ.get(m.group(1), m.group(0))
        return _ENV_PLACEHOLDER.sub(_sub, value)
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    return value


@dataclass
class LLMSpec:
    kind: str
    api_key: str
    model: str
    base_url: str | None = None
    temperature: float = 0.7
    max_tokens: int = 1000
    system: str = ""
    top_p: float | None = None

    @property
    def name(self) -> str:
        return self.model


@dataclass
class RuntimeConfig:
    concurrency: int = 8
    max_retries: int = 5
    max_sample_fail_retries: int = 1
    max_request_failures: int = 25
    max_empty_responses: int = 5
    request_timeout: int = 120
    checkpoint_every: int = 25
    verbose: bool = False


@dataclass
class LLMConfig:
    llm: LLMSpec
    runtime: RuntimeConfig

    def get_model(self) -> LLMSpec:
        return self.llm


@dataclass
class ExperimentDef:
    name: str
    description: str
    raw: dict[str, Any] = field(repr=False)


@dataclass
class GlobalExperimentConfig:
    active: str
    stakes_override: dict[str, Any]
    output_dir: Path
    seed: int
    run_dir: str | None = None


def _load_combined(config_path: Path | str | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"未找到配置文件 {path}")
    load_dotenv(REPO_ROOT / ".env", override=False)
    return _resolve_env(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def load_llm_config(config_path: Path | str | None = None) -> LLMConfig:
    raw = _load_combined(config_path)

    spec = raw.get("llm")
    if not spec:
        raise ValueError("config.yaml 必须包含顶层 `llm:` 块。")


    kind = spec.get("provider_kind") or spec.get("kind")
    required = {"provider_kind": kind, "api_key": spec.get("api_key"), "model": spec.get("model")}
    missing = [k for k, v in required.items() if not v]
    if missing:
        hint = " 请通过 orchestrator 启动，或使用 run.py --config 指向 runs/.../altruism_config.yaml。"
        raise ValueError(f"llm 块缺少必填字段：{missing}.{hint}")

    llm = LLMSpec(
        kind=kind,
        api_key=spec["api_key"],
        base_url=spec.get("base_url"),
        model=spec["model"],
        temperature=float(spec.get("temperature", 0.7)),
        max_tokens=int(spec.get("max_tokens", 1000)),
        system=spec.get("system", "") or "",
        top_p=(float(spec["top_p"]) if spec.get("top_p") not in (None, "") else None),
    )

    rt = raw.get("runtime") or {}
    runtime = RuntimeConfig(
        concurrency=int(rt.get("concurrency", 8)),
        max_retries=int(rt.get("max_retries", 5)),
        max_sample_fail_retries=int(rt.get("max_sample_fail_retries", 1)),
        max_request_failures=int(rt.get("max_request_failures", 25)),
        max_empty_responses=int(rt.get("max_empty_responses", 5)),
        request_timeout=int(rt.get("timeout", rt.get("request_timeout", 120))),
        checkpoint_every=int(rt.get("checkpoint_every", 25)),
        verbose=bool(rt.get("verbose", False)),
    )
    return LLMConfig(llm=llm, runtime=runtime)


def load_global_experiment_config(config_path: Path | str | None = None) -> GlobalExperimentConfig:
    raw = _load_combined(config_path)
    exp = raw.get("experiment") or {}
    if "active" not in exp:
        raise ValueError("config.yaml 必须包含 experiment.active")
    return GlobalExperimentConfig(
        active=exp["active"],
        stakes_override=exp.get("stakes_override") or {},
        output_dir=REPO_ROOT / (exp.get("output_dir") or "results"),
        seed=int(exp.get("seed", 8)),
        run_dir=(exp.get("run_dir") or None),
    )


def load_experiment_def(name: str, config_path: Path | str | None = None) -> ExperimentDef:
    exp_dir = CONFIG_DIR / "experiments"
    if config_path is not None:
        raw_cfg = _load_combined(config_path)
        override_dir = (raw_cfg.get("experiment") or {}).get("definitions_dir")
        if override_dir:
            exp_dir = Path(override_dir)
            if not exp_dir.is_absolute():
                exp_dir = Path(config_path).parent / exp_dir

    path = exp_dir / f"{name}.yaml"
    if not path.exists():
        candidates = sorted(p.stem for p in exp_dir.glob("*.yaml"))
        raise FileNotFoundError(
            f"找不到实验文件 {path}。可用：{candidates}"
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw.get("name") != name:
        raise ValueError(
            f"{path} 的 name 字段是 '{raw.get('name')}'，但被以 '{name}' 加载。"
        )
    return ExperimentDef(
        name=name,
        description=(raw.get("description") or "").strip(),
        raw=raw,
    )
