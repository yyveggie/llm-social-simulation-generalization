from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SHARED_DIR = os.path.join(PROJECT_ROOT, "scripts", "shared")
for _p in (PROJECT_ROOT, SHARED_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import analysis_common as common
import llm_prompts as oip

_LEGACY_R2 = re.compile(r"^R2-([a-z]+)$")
_LEGACY_R4 = re.compile(r"^R4-([123])$")


def modernize_task_key(task_key: str) -> str:
    t = str(task_key)
    if t.startswith("R1"):
        return "R1--full"
    m = _LEGACY_R2.match(t)
    if m:
        return f"R2a-{m.group(1)}-full"
    if t == "R3-race":
        return "R2b-race-full"
    if t.startswith("R3-race-"):
        return "R2b-race-full-" + t.split("R3-race-", 1)[1]
    if t == "R3-gender":
        return "R2b-gender-full"
    m = _LEGACY_R4.match(t)
    if m:
        return f"R3-{m.group(1)}-full"
    print(f"警告：未识别的 legacy task_key「{t}」，保持原样（MC 打分会跳过它）。", file=sys.stderr)
    return t


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert legacy llm_generations.pkl to modern JSONL per model.")
    parser.add_argument("--pkl", default="llm_generations.pkl", help="论文公开的聚合生成结果 pkl。")
    parser.add_argument("--out-root", default="outputs", help="输出根目录（与 orchestrator 结果目录一致）。")
    parser.add_argument("--prefix", default="legacy-", help="模型目录名前缀，避免与新模型目录混淆。")
    parser.add_argument("--overwrite", action="store_true", help="已存在输出文件时覆盖重写（会丢弃已打的 mc）。")
    args = parser.parse_args()

    records = common.load_legacy_generations(args.pkl)
    unknown_tasks = Counter()
    by_model: dict[str, list[dict]] = {}
    for rec in records:
        rec = dict(rec)
        modern_key = modernize_task_key(rec["task_key"])
        if modern_key == rec["task_key"] and modern_key not in oip.user_prompts:
            unknown_tasks[modern_key] += 1
        rec["legacy_task_key"] = rec["task_key"]
        rec["task_key"] = modern_key
        by_model.setdefault(rec["model"], []).append(rec)

    total = 0
    for model, rows in sorted(by_model.items()):
        out_dir = os.path.join(args.out_root, f"{args.prefix}{model}", "converted")
        out_path = os.path.join(out_dir, "legacy_generations.jsonl")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"跳过 {out_path}（已存在；如需重写请加 --overwrite，注意会丢弃已打的 mc）。")
            continue
        os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            for rec in rows:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
        total += len(rows)
        tasks = sorted({r["task_key"] for r in rows})
        print(f"{model}: {len(rows)} 条 -> {out_path}（{len(tasks)} 个任务）")

    if unknown_tasks:
        print(f"未识别任务键：{dict(unknown_tasks)}", file=sys.stderr)
    print(f"完成：共写出 {total} 条。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
