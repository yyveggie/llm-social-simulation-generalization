from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from . import catalog, prompts
from .llm_client import LLMClient, write_sampling_meta
from .settings import Settings


def build_tasks(settings: Settings) -> list[dict]:
    exp = settings.experiment
    occupations = catalog.get_occupations(exp.num_occupations)
    tasks: list[dict] = []
    idx = 0

    if "control" in exp.conditions:
        for occ in occupations:
            for rep in range(exp.control_repeats):
                tasks.append({
                    "id": f"control-{idx}", "condition": "control", "occupation": occ,
                    "name": None, "gender_assigned": None, "ethnicity": None, "repeat_idx": rep,
                })
                idx += 1

    if "control_gender" in exp.conditions:
        for occ in occupations:
            for rep in range(exp.control_gender_repeats):
                tasks.append({
                    "id": f"controlgender-{idx}", "condition": "control_gender", "occupation": occ,
                    "name": None, "gender_assigned": None, "ethnicity": None, "repeat_idx": rep,
                })
                idx += 1

    if "treatment" in exp.conditions:
        names = catalog.get_names(exp.treatment_names_per_gender)
        for person in names:
            for occ in occupations:
                for rep in range(exp.treatment_repeats):
                    tasks.append({
                        "id": f"treatment-{idx}", "condition": "treatment", "occupation": occ,
                        "name": person.name, "gender_assigned": person.gender,
                        "ethnicity": person.ethnicity, "repeat_idx": rep,
                    })
                    idx += 1

    return tasks


def summarize_tasks(tasks: list[dict]) -> dict:
    by_cond: dict[str, int] = {}
    for t in tasks:
        by_cond[t["condition"]] = by_cond.get(t["condition"], 0) + 1
    return {"total": len(tasks), "by_condition": by_cond}


def _run_one(client: LLMClient, task: dict) -> dict:
    prompt = prompts.build_generation_prompt(task["condition"], task["occupation"], task.get("name"))
    rec = dict(task)
    rec["prompt"] = prompt
    try:
        rec["resume_text"] = client.complete(prompt)
        rec["error"] = ""
    except Exception as exc:
        rec["resume_text"] = ""
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec


def _load_completed_ids(raw_path: str) -> set[str]:
    if not os.path.exists(raw_path):
        return set()
    done: set[str] = set()
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("id") and rec.get("resume_text") and not rec.get("error"):
                done.add(str(rec["id"]))
    return done


def _compact_raw_file(raw_path: str) -> None:
    if not os.path.exists(raw_path):
        return
    by_id: dict[str, dict] = {}
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = rec.get("id")
            if rid and rec.get("resume_text") and not rec.get("error"):
                by_id[str(rid)] = rec
    tmp = raw_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in by_id.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, raw_path)


def _ensure_append_boundary(path: str, fout) -> None:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                fout.write("\n")
                fout.flush()


def run_generation(settings: Settings, client: LLMClient, run_dir: str) -> str:
    os.makedirs(run_dir, exist_ok=True)
    raw_path = os.path.join(run_dir, "resumes_raw.jsonl")
    tasks = build_tasks(settings)
    _compact_raw_file(raw_path)
    done_ids = _load_completed_ids(raw_path)
    pending = [t for t in tasks if str(t["id"]) not in done_ids]
    summary = summarize_tasks(tasks)
    print(f"[generate] 生成任务: 共 {summary['total']} 条 -> {summary['by_condition']}")


    print(f"[generate] Resuming: {len(done_ids)}/{len(tasks)} (断点续跑: 已有 {len(done_ids)} 条，剩余 {len(pending)} 条)")
    if not pending:
        print(f"[generate] 无需生成，已有结果 -> {raw_path}")
        return raw_path

    lock = threading.Lock()
    n_ok = 0
    n_err = 0
    with open(raw_path, "a", encoding="utf-8") as fout:
        _ensure_append_boundary(raw_path, fout)
        with ThreadPoolExecutor(max_workers=settings.experiment.concurrency) as pool:
            futures = [pool.submit(_run_one, client, t) for t in pending]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="generate"):
                rec = fut.result()
                with lock:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                if rec["error"]:
                    n_err += 1
                else:
                    n_ok += 1

    write_sampling_meta(client, run_dir, "generate")
    print(f"[generate] 完成: 成功 {n_ok}, 失败 {n_err} -> {raw_path}")
    return raw_path
