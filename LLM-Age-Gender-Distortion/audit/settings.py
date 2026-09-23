from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime

import yaml


@dataclass
class ModelConfig:
    provider: str
    model_name: str
    api_key: str
    base_url: str
    temperature: float
    max_tokens: int
    request_timeout: int
    max_retries: int
    retry_base_delay: float
    system_prompt: str
    extra_body: dict
    top_p: float | None = None


@dataclass
class ExperimentConfig:
    conditions: list[str]
    scale_preset: str
    num_occupations: int
    treatment_names_per_gender: int
    control_repeats: int
    control_gender_repeats: int
    treatment_repeats: int
    evaluate_scores: bool
    concurrency: int
    seed: int


@dataclass
class OutputConfig:
    dir: str
    run_name: str


@dataclass
class Settings:
    model: ModelConfig
    experiment: ExperimentConfig
    output: OutputConfig
    config_path: str = ""

    def resolve_run_name(self) -> str:
        if self.output.run_name:
            return self.output.run_name
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model = self.model.model_name.replace("/", "-").replace(":", "-")
        return f"{ts}_{self.model.provider}_{safe_model}"

    def run_dir(self) -> str:
        return os.path.join(self.output.dir, self.resolve_run_name())


_OPENAI_STYLE = {"openai", "deepseek", "gemini", "openai_compatible"}


SCALE_PRESETS: dict[str, dict] = {
    "default": {"num_occupations": 54, "treatment_names_per_gender": 5,  "control_repeats": 10, "control_gender_repeats": 10, "treatment_repeats": 1},
    "small":   {"num_occupations": 54, "treatment_names_per_gender": 4,  "control_repeats": 10, "control_gender_repeats": 10, "treatment_repeats": 5},
    "medium":  {"num_occupations": 54, "treatment_names_per_gender": 8,  "control_repeats": 20, "control_gender_repeats": 20, "treatment_repeats": 10},
    "full":    {"num_occupations": 54, "treatment_names_per_gender": 16, "control_repeats": 50, "control_gender_repeats": 50, "treatment_repeats": 20},
}


def _resolve_scale(e: dict) -> tuple[str, dict]:
    preset = (e.get("scale_preset") or "custom").strip().lower()
    if preset == "custom":
        cu = e.get("custom", {}) or {}
        return preset, {
            "num_occupations": int(cu.get("num_occupations", 54)),
            "treatment_names_per_gender": int(cu.get("treatment_names_per_gender", 16)),
            "control_repeats": int(cu.get("control_repeats", 50)),
            "control_gender_repeats": int(cu.get("control_gender_repeats", 50)),
            "treatment_repeats": int(cu.get("treatment_repeats", 20)),
        }
    if preset in SCALE_PRESETS:
        return preset, SCALE_PRESETS[preset]
    raise ValueError(f"未知 scale_preset: {preset!r}; 可选: default/small/medium/full/custom")


def load_settings(path: str) -> Settings:
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到配置文件: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    m = cfg.get("model", {})
    provider = (m.get("provider") or "openai").strip()


    base_url = (m.get("base_url") or "").strip()
    if not base_url:
        base_url = (
            cfg.get("provider_defaults", {})
            .get(provider, {})
            .get("base_url", "")
            or ""
        ).strip()


    api_key = (m.get("api_key") or "").strip()
    if not api_key:
        env_name = (m.get("api_key_env") or "").strip()
        if env_name:
            api_key = (os.environ.get(env_name) or "").strip()

    model_cfg = ModelConfig(
        provider=provider,
        model_name=(m.get("model_name") or "").strip(),
        api_key=api_key,
        base_url=base_url,
        temperature=float(m.get("temperature", 0.7)),
        max_tokens=int(m.get("max_tokens", 1200)),
        request_timeout=int(m.get("request_timeout", 60)),
        max_retries=int(m.get("max_retries", 5)),
        retry_base_delay=float(m.get("retry_base_delay", 2.0)),
        system_prompt=(m.get("system_prompt") or ""),
        extra_body=(m.get("extra_body") or {}),
        top_p=(float(m["top_p"]) if m.get("top_p") not in (None, "") else None),
    )

    e = cfg.get("experiment", {})
    preset_name, scale = _resolve_scale(e)
    exp_cfg = ExperimentConfig(
        conditions=list(e.get("conditions", ["control", "control_gender", "treatment"])),
        scale_preset=preset_name,
        num_occupations=scale["num_occupations"],
        treatment_names_per_gender=scale["treatment_names_per_gender"],
        control_repeats=scale["control_repeats"],
        control_gender_repeats=scale["control_gender_repeats"],
        treatment_repeats=scale["treatment_repeats"],
        evaluate_scores=bool(e.get("evaluate_scores", True)),
        concurrency=int(e.get("concurrency", 4)),
        seed=int(e.get("seed", 42)),
    )

    o = cfg.get("output", {})
    out_cfg = OutputConfig(
        dir=(o.get("dir") or "results"),
        run_name=(o.get("run_name") or ""),
    )

    return Settings(model=model_cfg, experiment=exp_cfg, output=out_cfg, config_path=path)


def validate_for_api(settings: Settings) -> None:
    m = settings.model
    if m.provider not in _OPENAI_STYLE and m.provider != "anthropic":
        raise ValueError(
            f"未知 provider: {m.provider!r}; 可选: openai/deepseek/gemini/anthropic/openai_compatible"
        )
    if not m.model_name:
        raise ValueError("model.model_name 不能为空")
    if not m.api_key:
        raise ValueError(
            "未提供 API key。请通过 orchestrator 启动实验，"
            "或使用 run.py --config 指向 runs/.../age_gender_distortion_config.yaml。"
        )
    if not m.base_url:
        if m.provider == "openai_compatible":
            raise ValueError("provider=openai_compatible 时必须在 model.base_url 填写完整地址")
        raise ValueError(f"provider={m.provider} 缺少 base_url (检查 provider_defaults)")
