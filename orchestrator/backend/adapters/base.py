from __future__ import annotations

import hashlib
import json
import shutil
import stat as stat_module
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from .. import settings
from ..schemas import ExperimentInfo, ParamField, ProjectInfo, UnifiedLLM

ALL_PROVIDER_KINDS = (
    "openai_chat",
    "openai_responses",
    "anthropic",
    "gemini",
    "huggingface",
    "ollama",
)

_yaml = YAML()
_yaml.preserve_quotes = True
_yaml.width = 4096


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return _yaml.load(f)


def dump_yaml(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        _yaml.dump(data, f)


def llm_proxy_config(llm: UnifiedLLM) -> dict[str, str]:
    from ..llm_proxy import register_llm

    return register_llm(llm)


@dataclass
class JobUnit:

    experiment_id: str
    label: str
    argv: list[str]
    selected: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


class Adapter:
    id: str = ""
    name: str = ""
    paper: str = ""
    description: str = ""
    intro: str = ""
    supported_kinds: tuple[str, ...] = ALL_PROVIDER_KINDS
    selection_mode: str = "multi"
    allow_concurrent: bool = False
    paper_baseline: str = ""


    @property
    def project_dir(self) -> Path:
        return settings.project_path(self.id)


    def list_experiments(self) -> list[ExperimentInfo]:
        raise NotImplementedError

    def param_schema(self) -> list[ParamField]:
        return []

    def info(self) -> ProjectInfo:
        return ProjectInfo(
            id=self.id,
            name=self.name,
            paper=self.paper,
            description=self.description,
            intro=self.intro,
            supported_kinds=list(ALL_PROVIDER_KINDS),
            selection_mode=self.selection_mode,
            experiments=self.list_experiments(),
            param_schema=self.param_schema(),
        )


    def plan_jobs(self, experiment_ids: list[str], params: dict[str, Any]) -> list[JobUnit]:
        raise NotImplementedError

    def render_config(self, llm: UnifiedLLM, params: dict[str, Any], unit: JobUnit,
                      work_dir: Path | None = None) -> None:
        raise NotImplementedError


    def results_dir(self) -> Path:
        return self.project_dir

    def results_dir_for_job(self, job_info: Any | None = None) -> Path:
        return self.results_dir()

    def reset_job_outputs(self, job_info: Any | None = None) -> list[str]:
        target = self.results_dir_for_job(job_info)
        if target is None:
            return []
        target = Path(target).resolve()
        root = self.results_dir().resolve()
        if target == root or root not in target.parents:
            return []
        if not target.exists():
            return []
        shutil.rmtree(target)
        return [str(target)]


    def load_analysis(self, run_name: str | None = None) -> dict | None:
        return None

    def analyze_argv(self, model: str | None = None) -> list[str] | None:
        return None

    def list_result_models(self) -> list[dict[str, Any]]:
        root = self.results_dir()
        out: list[dict[str, Any]] = []
        if not root.is_dir():
            return out
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue

            n_files = 0
            latest = 0.0
            for p in d.rglob("*"):
                if p.name == "run_meta.json" or p.name.startswith("temporary_"):
                    continue
                try:
                    st = p.stat()
                except OSError:
                    continue
                if not stat_module.S_ISREG(st.st_mode):
                    continue
                n_files += 1
                if st.st_mtime > latest:
                    latest = st.st_mtime
            if not n_files:
                continue
            out.append({
                "model": d.name,
                "files": n_files,
                "latest": datetime.fromtimestamp(latest).strftime("%Y-%m-%d %H:%M") if latest else "",
                "latest_ts": latest,
            })
        out.sort(key=lambda x: x["latest_ts"], reverse=True)
        return out

    def load_transcript(self, offset: int = 0, limit: int = 50, job_info: Any | None = None,
                        only_failed: bool = False) -> dict | None:
        return None

    def sample_stats(self, job_info: Any | None = None) -> dict | None:
        return None


    def check_kind(self, llm: UnifiedLLM) -> None:
        if llm.provider_kind not in ALL_PROVIDER_KINDS:
            raise ValueError(
                f"项目 {self.name} 不支持 provider_kind='{llm.provider_kind}'。"
                f"支持：{', '.join(ALL_PROVIDER_KINDS)}"
            )


def opt_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def parse_list(value: Any) -> list[str] | None:
    if value is None or value == "":
        return None
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [s.strip() for s in str(value).split(",") if s.strip()]


def as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on", "y")


def exp_param(params: dict[str, Any], selected: list[str], name: str) -> Any:
    for eid in selected:
        v = params.get(f"{eid}__{name}")
        if v not in (None, ""):
            return v
    return None


def nested_get(data: Any, *keys: str) -> Any:
    cur = data
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def job_config_path(job_info: Any | None) -> Path | None:
    if job_info is None or not getattr(job_info, "log_path", ""):
        return None
    state_path = Path(job_info.log_path).parent / "job_state.json"
    if not state_path.is_file():
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    extra = ((state.get("unit") or {}).get("extra") or {})
    path = extra.get("config_path")
    if not path:
        return None
    cfg = Path(str(path))
    return cfg if cfg.is_file() else None


def job_config(job_info: Any | None) -> Any | None:
    cfg = job_config_path(job_info)
    if cfg is None:
        return None
    try:
        return load_yaml(cfg)
    except Exception:
        return None


def model_slug(name: str) -> str:
    import re
    return re.sub(r"[^0-9A-Za-z._-]+", "-", (name or "model")).strip("-") or "model"


def result_fingerprint(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]


def resolve_run_dir(parent: Path, fingerprint: str, *, reuse: bool = True) -> Path:
    if reuse and parent.is_dir():
        existing = sorted(
            (p for p in parent.glob(f"*__{fingerprint}") if p.is_dir()),
            key=lambda p: p.name,
        )
        if existing:
            return existing[-1]
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return parent / f"{ts}__{fingerprint}"


def remembered_run_dir(
    unit: JobUnit,
    parent: Path,
    fingerprint: str,
    *,
    key: str = "result_run_dir",
) -> Path:
    saved = unit.extra.get(key)
    if saved:
        return Path(str(saved))
    run_dir = _best_existing_run_dir(parent, fingerprint) or resolve_run_dir(parent, fingerprint)
    unit.extra[key] = str(run_dir)
    return run_dir


def _best_existing_run_dir(parent: Path, fingerprint: str) -> Path | None:
    if not parent.is_dir():
        return None
    existing = [p for p in parent.glob(f"*__{fingerprint}") if p.is_dir()]
    if not existing:
        return None
    return max(existing, key=lambda p: (*_run_dir_payload_score(p), p.name))


def _run_dir_payload_score(path: Path) -> tuple[int, int]:
    files = []
    try:
        files = [p for p in path.rglob("*") if p.is_file() and p.name != "run_meta.json"]
    except OSError:
        return (0, 0)
    size = 0
    for p in files:
        try:
            size += p.stat().st_size
        except OSError:
            pass
    return (len(files), size)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


_TRANSCRIPT_CACHE: dict[str, tuple[float, float, int, Any]] = {}
_TRANSCRIPT_LOCK = threading.Lock()
_TRANSCRIPT_MAX_ENTRIES = 32
_TRANSCRIPT_MAX_WEIGHT = 300_000


def cached_by_mtime(
    files: list[Path],
    build: Callable[[], Any],
    *,
    namespace: str = "default",
    min_interval: float = 0.0,
) -> Any:
    paths = sorted(str(f) for f in files)
    key = f"{namespace}\n" + "\n".join(paths)
    try:
        sig = max((Path(p).stat().st_mtime for p in paths), default=0.0)
    except OSError:
        sig = 0.0
    with _TRANSCRIPT_LOCK:
        hit = _TRANSCRIPT_CACHE.get(key)
        if hit is not None:
            cached_sig, built_at, _weight, value = hit
            if cached_sig == sig or (min_interval > 0 and time.monotonic() - built_at < min_interval):
                return value
    result = build()
    weight = max(1, len(result)) if isinstance(result, (list, tuple, dict)) else 1
    with _TRANSCRIPT_LOCK:
        _TRANSCRIPT_CACHE.pop(key, None)
        _TRANSCRIPT_CACHE[key] = (sig, time.monotonic(), weight, result)
        while len(_TRANSCRIPT_CACHE) > 1 and (
            len(_TRANSCRIPT_CACHE) > _TRANSCRIPT_MAX_ENTRIES
            or sum(e[2] for e in _TRANSCRIPT_CACHE.values()) > _TRANSCRIPT_MAX_WEIGHT
        ):
            _TRANSCRIPT_CACHE.pop(next(iter(_TRANSCRIPT_CACHE)))
    return result


def count_samples_by_key(
    files: list[Path],
    key_fn: Callable[[dict[str, Any]], Any],
    ok_fn: Callable[[dict[str, Any]], bool],
) -> dict[str, int]:
    paths = [f for f in files if f.is_file()]

    def build() -> dict[str, int]:
        status: dict[Any, bool] = {}
        for path in paths:
            for rec in read_jsonl(path):
                try:
                    key = key_fn(rec)
                except Exception:
                    continue
                if key is None:
                    continue
                status[key] = status.get(key, False) or bool(ok_fn(rec))
        ok = sum(1 for v in status.values() if v)
        return {"ok": ok, "fail": len(status) - ok}

    return cached_by_mtime(paths, build, namespace="sample_stats")


def count_samples_csv_by_key(
    path: Path,
    key_field: str,
    ok_fn: Callable[[dict[str, str]], bool],
) -> dict[str, int]:
    if not path.is_file():
        return {"ok": 0, "fail": 0}

    def build() -> dict[str, int]:
        import csv as _csv

        status: dict[str, bool] = {}
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in _csv.DictReader(f):
                key = row.get(key_field)
                if key in (None, ""):
                    continue
                status[key] = status.get(key, False) or bool(ok_fn(row))
        ok = sum(1 for v in status.values() if v)
        return {"ok": ok, "fail": len(status) - ok}

    return cached_by_mtime([path], build, namespace="sample_stats")


def non_error_text_ok(text: Any, *error_prefixes: str) -> bool:
    s = str(text or "").strip()
    return bool(s) and not any(s.startswith(p) for p in error_prefixes)


def valid_flag_ok(
    rec: dict[str, Any],
    *,
    text_key: str = "response",
    error_prefixes: tuple[str, ...] = (),
) -> bool:
    v = rec.get("valid")
    if v not in (None, ""):
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() in ("1", "true", "yes")
    return non_error_text_ok(rec.get(text_key), *error_prefixes)


def build_transcript_page(
    rows: list[dict[str, Any]],
    *,
    offset: int,
    limit: int,
    only_failed: bool,
    label_fn: Callable[[dict[str, Any]], str],
    prompt_fn: Callable[[dict[str, Any]], str],
    response_fn: Callable[[dict[str, Any]], str],
    ok_fn: Callable[[dict[str, Any]], bool],
    reason_fn: Callable[[dict[str, Any]], str | None],
) -> dict[str, Any]:
    enriched = [(r, bool(ok_fn(r))) for r in rows]
    if only_failed:
        enriched = [pair for pair in enriched if not pair[1]]
    items = [
        {
            "label": label_fn(r),
            "prompt": prompt_fn(r),
            "response": response_fn(r),
            "valid": ok,
            "fail_reason": None if ok else reason_fn(r),
        }
        for r, ok in enriched[offset: offset + limit]
    ]
    return {"total": len(enriched), "offset": offset, "limit": limit,
            "items": items, "failed_filter": True}
