from __future__ import annotations

import csv
import json
import logging
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .config import LLMConfig, LLMSpec
from .llm_client import LLMClient
from .schemas import classify_response

logger = logging.getLogger(__name__)


class CostCircuitBreaker(RuntimeError):
    pass


BASE_COLUMNS = [
    "indexx", "modelv", "stakes", "queree", "condit", "respon", "valid",
    "fail_attempts", "upstream_raw_response",
]


_SKIP_COLUMNS = {"sampling_overrides"}
_OLD_TOP_P_ZERO_ERROR_MARKERS = (
    "top_p must be between 0 (exclusive) and 1 (inclusive)",
    "400002",
)
_OLD_TOP_P_TINY_DECIMAL_ERROR_MARKERS = (
    "top_p参数非法：限制小数点[2]位",
    "限制小数点[2]位",
)
_EMPTY_RESPONSE_MARKERS = (
    "empty assistant content",
    "empty Responses API output",
    "empty Anthropic message content",
)


def _all_columns(trials: list[dict[str, Any]]) -> list[str]:
    cols = list(BASE_COLUMNS)
    extras: list[str] = []
    for t in trials:
        for k, v in t.items():
            if k in cols or k in extras or k in _SKIP_COLUMNS:
                continue
            if isinstance(v, (dict, list)):
                continue
            extras.append(k)
    return cols + extras


def _int_or_default(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _json_cell(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    except Exception:
        return json.dumps(str(value), ensure_ascii=False)


def _exception_raw_response(exc: Exception) -> str:
    raw = getattr(exc, "raw", None)
    if raw is not None:
        return _json_cell(raw)
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            body = response.json()
        except Exception:
            body = getattr(response, "text", "")
        return _json_cell({
            "http_status": getattr(response, "status_code", None),
            "url": str(getattr(getattr(response, "request", None), "url", "")),
            "body": body,
        })
    return _json_cell({"error_type": type(exc).__name__, "message": str(exc)})


def _is_retryable_after_client_fix(row: dict[str, str]) -> bool:
    text = f"{row.get('respon', '')} {row.get('upstream_raw_response', '')}"
    if "_client_sampling_adjustments" in text:


        if "1e-06" in text and any(marker in text for marker in _OLD_TOP_P_TINY_DECIMAL_ERROR_MARKERS):
            return True
        return False
    return any(marker in text for marker in _OLD_TOP_P_ZERO_ERROR_MARKERS)


def _is_empty_response_failure(row: dict[str, str]) -> bool:
    text = f"{row.get('respon', '')} {row.get('upstream_raw_response', '')}"
    return any(marker in text for marker in _EMPTY_RESPONSE_MARKERS)


def _load_existing(
    csv_path: Path, *, task_type: str, max_sample_fail_retries: int
) -> tuple[dict[int, dict[str, str]], dict[int, int]]:
    if not csv_path.exists():
        return {}, {}
    out: dict[int, dict[str, str]] = {}
    failed_attempts: dict[int, int] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int(row["indexx"])
            except (KeyError, ValueError):
                continue
            cls = classify_response(row.get("respon"), task_type=task_type, stakes=row.get("stakes"))
            if cls == "fail":

                attempts = max(1, _int_or_default(row.get("fail_attempts"), 1))
                failed_attempts[idx] = attempts
                if _is_retryable_after_client_fix(row):
                    continue
                if attempts < max_sample_fail_retries:
                    continue
                out[idx] = row
                continue
            if cls == "ok":
                row["valid"] = "True"
            out[idx] = row
    return out, failed_attempts


def run_experiment_for_model(
    *,
    trials: list[dict[str, Any]],
    model_cfg: LLMSpec,
    llm_cfg: LLMConfig,
    out_csv: Path,
    progress_label: str = "",
    verbose: bool = False,
    task_type: str = "dictator",
) -> Path:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    columns = _all_columns(trials)
    max_sample_fail_retries = max(1, int(llm_cfg.runtime.max_sample_fail_retries))
    done, prior_failed_attempts = _load_existing(
        out_csv, task_type=task_type, max_sample_fail_retries=max_sample_fail_retries
    )
    if done:
        retryable = sum(1 for t in prior_failed_attempts.values() if t < max_sample_fail_retries)
        logger.info(
            "Resuming: %d/%d trials already completed in %s%s",
            len(done), len(trials), out_csv,
            f"; {retryable} failed samples below retry cap" if retryable else "",
        )

    pending = [t for t in trials if t["indexx"] not in done]
    if not pending:
        logger.info("All trials already complete for %s", out_csv)
        _flush(out_csv, columns, list(done.values()))
        return out_csv

    client = LLMClient(
        model_cfg,
        request_timeout=llm_cfg.runtime.request_timeout,
        max_retries=llm_cfg.runtime.max_retries,
    )

    results: dict[int, dict[str, str]] = dict(done)
    lock = threading.Lock()
    checkpoint_every = max(1, llm_cfg.runtime.checkpoint_every)
    written_since_flush = 0
    max_request_failures = max(0, int(llm_cfg.runtime.max_request_failures))
    max_empty_responses = max(0, int(llm_cfg.runtime.max_empty_responses))
    request_failures_seen = sum(
        1 for row in done.values()
        if classify_response(row.get("respon"), task_type=task_type, stakes=row.get("stakes")) == "fail"
    )
    empty_failures_seen = sum(1 for row in done.values() if _is_empty_response_failure(row))

    def _circuit_breaker_reason() -> str | None:
        if max_empty_responses and empty_failures_seen >= max_empty_responses:
            return (
                f"empty response circuit breaker tripped: {empty_failures_seen} empty responses "
                f"(limit {max_empty_responses})"
            )
        if max_request_failures and request_failures_seen >= max_request_failures:
            return (
                f"request failure circuit breaker tripped: {request_failures_seen} request-layer failures "
                f"(limit {max_request_failures})"
            )
        return None

    initial_abort = _circuit_breaker_reason()
    if initial_abort:
        _flush(out_csv, columns, list(results.values()))
        raise CostCircuitBreaker(
            f"{progress_label or model_cfg.name}: {initial_abort}; "
            "aborting to avoid runaway API cost; no new requests were sent"
        )

    def _do_one(trial: dict[str, Any]) -> dict[str, str]:
        try:
            resp = client.complete(
                trial["queree"],
                overrides=trial.get("sampling_overrides"),
            )
            respon = resp.text
            upstream_raw_response = _json_cell(resp.raw)
        except Exception as exc:
            logger.warning("Trial %s failed permanently: %s", trial["indexx"], exc)
            respon = f"__ERROR__: {type(exc).__name__}: {exc}"
            upstream_raw_response = _exception_raw_response(exc)
        if verbose:
            logger.info(
                "[trial %s | %s] PROMPT:\n%s\nRESPONSE:\n%s\n%s",
                trial.get("indexx"), trial.get("condit"),
                trial.get("queree"), respon, "-" * 60,
            )
        cls = classify_response(respon, task_type=task_type, stakes=trial.get("stakes"))
        valid = cls == "ok"
        fail_attempts = (
            prior_failed_attempts.get(int(trial["indexx"]), 0) + 1 if cls == "fail" else ""
        )
        row = {
            **trial,
            "modelv": model_cfg.name,
            "respon": respon,
            "valid": valid,
            "fail_attempts": fail_attempts,
            "upstream_raw_response": upstream_raw_response,
        }

        return {k: ("" if row.get(k) is None else str(row[k])) for k in columns}

    abort_reason: str | None = None
    pending_iter = iter(pending)
    max_workers = max(1, llm_cfg.runtime.concurrency)

    def _submit_next(executor: ThreadPoolExecutor, in_flight: set) -> bool:
        try:
            trial = next(pending_iter)
        except StopIteration:
            return False
        in_flight.add(executor.submit(_do_one, trial))
        return True

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        in_flight: set = set()
        for _ in range(min(max_workers, len(pending))):
            _submit_next(ex, in_flight)
        bar = tqdm(
            total=len(pending),
            desc=progress_label or model_cfg.name,
            unit="trial",
            file=sys.stdout,
        )
        try:
            while in_flight:
                completed, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                refill_count = 0
                for fut in completed:
                    row = fut.result()
                    idx = int(row["indexx"])
                    cls = classify_response(row.get("respon"), task_type=task_type, stakes=row.get("stakes"))
                    if cls == "fail":
                        request_failures_seen += 1
                    if _is_empty_response_failure(row):
                        empty_failures_seen += 1
                    with lock:
                        results[idx] = row
                        written_since_flush += 1
                        if written_since_flush >= checkpoint_every:
                            _flush(out_csv, columns, list(results.values()))
                            written_since_flush = 0
                    bar.update(1)
                    refill_count += 1
                    abort_reason = _circuit_breaker_reason()
                if not abort_reason:
                    for _ in range(refill_count):
                        _submit_next(ex, in_flight)
                if abort_reason:
                    for fut in in_flight:
                        fut.cancel()
                    break
        finally:
            bar.close()

    _flush(out_csv, columns, list(results.values()))
    if abort_reason:
        raise CostCircuitBreaker(
            f"{progress_label or model_cfg.name}: {abort_reason}; "
            "aborting to avoid runaway API cost; no additional requests were submitted"
        )
    logger.info("Wrote %d rows to %s", len(results), out_csv)
    return out_csv


def _flush(csv_path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    rows_sorted = sorted(rows, key=lambda r: int(r.get("indexx", 0)))
    tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in rows_sorted:
            writer.writerow(r)
    tmp.replace(csv_path)
