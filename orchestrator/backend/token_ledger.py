from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from typing import Any

from . import settings


_BASELINE_TOTAL = 19_657_792

_LEDGER_PATH = settings.RUNS_DIR / "token_ledger.json"
_FLUSH_INTERVAL = 2.0


_BUCKET_SECONDS = 1800
_MAX_BUCKETS = 336

_LOCK = threading.Lock()
_LOADED = False
_LAST_FLUSH = 0.0
_STATE: dict[str, Any] = {
    "schema": "token_ledger/v1",
    "baseline_total": _BASELINE_TOTAL,
    "observed": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    "by_model": {},
    "timeline": {},
    "updated_at": None,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _write_locked() -> None:
    _STATE["updated_at"] = _now_iso()
    try:
        settings.RUNS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _LEDGER_PATH.with_name(_LEDGER_PATH.name + ".tmp")
        tmp.write_text(json.dumps(_STATE, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(_LEDGER_PATH)
    except Exception:
        pass


def _touch_bucket_locked() -> None:
    bucket = int(time.time() // _BUCKET_SECONDS) * _BUCKET_SECONDS
    by_model = {name: _int(v.get("total_tokens")) for name, v in _STATE["by_model"].items()}
    if not by_model:
        return
    timeline = _STATE["timeline"]
    timeline[str(bucket)] = {"ts": bucket, "by_model": by_model}
    if len(timeline) > _MAX_BUCKETS:
        for key in sorted(timeline, key=lambda k: _int(k))[:-_MAX_BUCKETS]:
            timeline.pop(key, None)


def _maybe_flush_locked() -> None:
    global _LAST_FLUSH
    now = time.monotonic()
    if now - _LAST_FLUSH < _FLUSH_INTERVAL:
        return
    _LAST_FLUSH = now
    _write_locked()


def load() -> None:
    global _LOADED
    with _LOCK:
        if _LOADED:
            return
        data: Any = None
        if _LEDGER_PATH.is_file():
            try:
                data = json.loads(_LEDGER_PATH.read_text(encoding="utf-8"))
            except Exception:
                data = None
        if isinstance(data, dict):
            bt = data.get("baseline_total")
            _STATE["baseline_total"] = _int(bt) if bt is not None else _BASELINE_TOTAL
            obs = data.get("observed") or {}
            _STATE["observed"] = {
                "prompt_tokens": _int(obs.get("prompt_tokens")),
                "completion_tokens": _int(obs.get("completion_tokens")),
                "total_tokens": _int(obs.get("total_tokens")),
            }
            bm = data.get("by_model")
            _STATE["by_model"] = {
                str(k): {
                    "prompt_tokens": _int((v or {}).get("prompt_tokens")),
                    "completion_tokens": _int((v or {}).get("completion_tokens")),
                    "total_tokens": _int((v or {}).get("total_tokens")),
                }
                for k, v in bm.items()
            } if isinstance(bm, dict) else {}
            tl = data.get("timeline")
            _STATE["timeline"] = {
                str(k): {
                    "ts": _int((v or {}).get("ts")),
                    "by_model": {
                        str(mk): _int(mv)
                        for mk, mv in ((v or {}).get("by_model") or {}).items()
                    },
                }
                for k, v in tl.items()
            } if isinstance(tl, dict) else {}
            _STATE["updated_at"] = data.get("updated_at")
        _LOADED = True

        _touch_bucket_locked()
        _write_locked()


def record(model: str, usage: dict[str, int]) -> None:
    prompt = _int(usage.get("prompt_tokens"))
    completion = _int(usage.get("completion_tokens"))
    total = _int(usage.get("total_tokens")) or (prompt + completion)
    if prompt <= 0 and completion <= 0 and total <= 0:
        return
    if not _LOADED:
        load()
    name = (str(model).strip() or "未命名模型") if model else "未命名模型"
    with _LOCK:
        obs = _STATE["observed"]
        obs["prompt_tokens"] += prompt
        obs["completion_tokens"] += completion
        obs["total_tokens"] += total
        slot = _STATE["by_model"].get(name)
        if slot is None:
            slot = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            _STATE["by_model"][name] = slot
        slot["prompt_tokens"] += prompt
        slot["completion_tokens"] += completion
        slot["total_tokens"] += total
        _touch_bucket_locked()
        _maybe_flush_locked()


def flush() -> None:
    with _LOCK:
        _write_locked()


def snapshot() -> dict[str, Any]:
    with _LOCK:
        baseline = _int(_STATE["baseline_total"])
        obs = {k: _int(v) for k, v in _STATE["observed"].items()}
        by_model = [
            {"model": k, **{kk: _int(vv) for kk, vv in v.items()}}
            for k, v in _STATE["by_model"].items()
        ]
        timeline = [
            {
                "ts": _int(b.get("ts")) * 1000,
                "label": datetime.fromtimestamp(_int(b.get("ts"))).strftime("%H:%M"),
                "by_model": {str(mk): _int(mv) for mk, mv in (b.get("by_model") or {}).items()},
            }
            for b in sorted(_STATE["timeline"].values(), key=lambda x: _int(x.get("ts")))
        ]
    by_model.sort(key=lambda m: m.get("total_tokens", 0), reverse=True)
    return {
        "total_tokens": baseline + obs.get("total_tokens", 0),
        "baseline_total": baseline,
        "observed": obs,
        "by_model": by_model,
        "timeline": timeline,
        "updated_at": _STATE.get("updated_at"),
    }
