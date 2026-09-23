from __future__ import annotations

import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from . import prompts
from .llm_client import LLMClient, write_sampling_meta
from .schemas import valid_score
from .settings import Settings

_SCORE_RE = re.compile(r"\d{1,3}(?:\.\d+)?")


_NOISE_RE = re.compile(
    r"\b1\s*(?:to|and|[-–—/])\s*100\b"
    r"|(?:out\s+of|/)\s*100\b"
    r"|\b1\s+is\s+a\s+low\s+score\b"
    r"|\b100\s+is\s+a\s+high\s+score\b",
    re.IGNORECASE,
)


_SCORE_NEAR_KEYWORD_RE = re.compile(
    r"\b(?:score|rating|rate[ds]?)\b[^\d]{0,20}(\d{1,3}(?:\.\d+)?)",
    re.IGNORECASE,
)


def parse_score(text: str) -> float | None:
    if not text:
        return None
    cleaned = _NOISE_RE.sub(" ", text)

    m = _SCORE_NEAR_KEYWORD_RE.search(cleaned)
    if m:
        val = float(m.group(1))
        if valid_score(val):
            return val

    candidates: list[float] = []
    for x in _SCORE_RE.findall(cleaned):
        val = float(x)
        if valid_score(val):
            candidates.append(val)
    return candidates[-1] if candidates else None


def _load_resumes(raw_path: str) -> list[dict]:
    by_id: dict[str, dict] = {}
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rid = rec.get("id")
            if rid and rec.get("resume_text") and not rec.get("error"):
                by_id[str(rid)] = rec
    return list(by_id.values())


def _score_one(client: LLMClient, rec: dict) -> dict:
    prompt = prompts.build_evaluation_prompt(rec["occupation"], rec["resume_text"])
    out = {
        "id": rec["id"],
        "condition": rec.get("condition"),
        "occupation": rec.get("occupation"),
    }
    try:
        raw = client.complete(prompt)
        out["score_text"] = raw
        out["score"] = parse_score(raw)
        out["error"] = "" if out["score"] is not None else "score_parse_failed"
    except Exception as exc:
        out["score_text"] = ""
        out["score"] = None
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["valid"] = out["score"] is not None and not out["error"]
    return out


def _load_existing_scores(jsonl_path: str) -> list[dict]:
    if not os.path.exists(jsonl_path):
        return []
    rows: list[dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("id") and not rec.get("error") and rec.get("score") is not None:
                    rows.append(rec)
            except json.JSONDecodeError:
                continue
    return rows


def _compact_scores_file(jsonl_path: str) -> None:
    if not os.path.exists(jsonl_path):
        return
    by_id: dict[str, dict] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = rec.get("id")
            if rid and not rec.get("error") and rec.get("score") is not None:
                by_id[str(rid)] = rec
    tmp = jsonl_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in by_id.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, jsonl_path)


def _ensure_append_boundary(path: str, fout) -> None:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                fout.write("\n")
                fout.flush()


def _write_csv_atomic(df: pd.DataFrame, path: str) -> None:
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def run_evaluation(settings: Settings, client: LLMClient, run_dir: str) -> str:
    raw_path = os.path.join(run_dir, "resumes_raw.jsonl")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"找不到 {raw_path}, 请先运行 generate 阶段")

    resumes = _load_resumes(raw_path)
    print(f"[evaluate] 待评分简历: {len(resumes)} 份")

    jsonl_path = os.path.join(run_dir, "scores.jsonl")
    _compact_scores_file(jsonl_path)
    results: list[dict] = _load_existing_scores(jsonl_path)
    done_ids = {str(r.get("id")) for r in results if r.get("id")}
    pending = [r for r in resumes if str(r.get("id")) not in done_ids]


    print(f"[evaluate] Resuming: {len(done_ids)}/{len(resumes)} (断点续跑: 已有 {len(done_ids)} 条评分，剩余 {len(pending)} 条)")

    lock = threading.Lock()
    n_ok = 0
    n_err = 0
    with open(jsonl_path, "a", encoding="utf-8") as fout:
        _ensure_append_boundary(jsonl_path, fout)
        with ThreadPoolExecutor(max_workers=settings.experiment.concurrency) as pool:
            futures = [pool.submit(_score_one, client, r) for r in pending]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="evaluate"):
                out = fut.result()
                with lock:
                    fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                    fout.flush()
                results.append(out)
                if out["error"]:
                    n_err += 1
                else:
                    n_ok += 1

    if pending:
        write_sampling_meta(client, run_dir, "evaluate")
    csv_path = os.path.join(run_dir, "scores.csv")
    _write_csv_atomic(pd.DataFrame(results), csv_path)
    print(f"[evaluate] 完成: 成功 {n_ok}, 失败 {n_err} -> {csv_path}")
    return csv_path
