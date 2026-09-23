from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from typing import List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SHARED_DIR = os.path.join(PROJECT_ROOT, "scripts", "shared")
for _p in (PROJECT_ROOT, SHARED_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import llm_prompts as oip

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


class ScoreError(RuntimeError):
    pass


def load_config(path: str) -> dict:
    if yaml is None:
        raise ScoreError("缺少 pyyaml，无法读取 config。请 pip install pyyaml。")
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _split_r2b_text(text: str, version: int) -> Optional[List[str]]:
    if version == 0:
        sets = [["P1", "P2", "P3"], ["1.", "2.", "3."], ["1)", "2)", "3)"],
                ["first phrase", "second phrase", "third phrase"], ["1", "2", "3"]]
    else:
        sets = [["P3", "P4", "P5"], ["3.", "4.", "5."], ["3)", "4)", "5)"],
                ["third phrase", "fourth phrase", "fifth phrase"], ["3", "4", "5"]]
    chosen = next((s for s in sets if all(d in text for d in s)), None)
    if chosen is None:


        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if len(paras) == 3:
            return paras
        return None
    parts: List[str] = []
    for i, delim in enumerate(chosen):
        start = text.index(delim) + len(delim)
        end = text.index(chosen[i + 1]) if i + 1 < len(chosen) else len(text)
        chunk = text[start:end].strip()
        if chunk[:2] in (": ", ". "):
            chunk = chunk[2:].strip()
        if len(chunk) >= 2 and chunk[0] in "\"'" and chunk[-1] == chunk[0]:
            chunk = chunk[1:-1].strip()
        parts.append(chunk)
    return parts


def mc_keys(task_key: str):
    t = task_key
    pairs = {
        "R1": ("R1--mc", "R1--ex"),
        "R2b-race": ("R2b-race-mc", "R2b-race-ex"),
        "R3-1": ("R3-1-mc", "R3-1-ex"),
        "R3-2": ("R3-2-mc", "R3-2-ex"),
        "R3-3": ("R3-3-mc", "R3-3-ex"),
    }
    if t.startswith("R2a-"):
        topic = t.split("-")[1]
        mc_key, ex_key = "R2a-*-mc", f"R2a-{topic}-ex"
    else:
        key = next((k for k in pairs if t.startswith(k)), None)
        if key is None:
            return None, None
        mc_key, ex_key = pairs[key]
    mc_prompt = oip.mc_prompts.get(mc_key)
    ex = oip.mc_prompts.get(ex_key)
    if mc_prompt is None or ex is None:
        return None, None
    return mc_prompt, ex


def _thinking_off_overrides(model: str) -> dict:
    low = str(model).lower()
    if any(k in low for k in ("qwen", "deepseek", "glm", "kimi", "doubao", "moonshot")):
        return {
            "enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "thinking": {"type": "disabled"},
        }
    return {}


def call_judge(base_url: str, api_key: str, model: str, prompt: str,
               timeout: int, retries: int, backoff: float) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }
    payload.update(_thinking_off_overrides(model))
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return str(data["choices"][0]["message"]["content"] or "")
        except (urllib.error.URLError, KeyError, json.JSONDecodeError, TimeoutError) as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(backoff * (2 ** attempt))
    raise ScoreError(f"裁判调用失败：{last_err}")


def parse_mc(text: str) -> Optional[int]:
    m = re.search(r"\(([1-5])\)", text) or re.search(r"\b([1-5])\b", text)
    return int(m.group(1)) if m else None


def judge_score(base_url, api_key, model, mc_prompt, ex, response, timeout, retries, backoff) -> Optional[int]:

    few = oip.few_shot.format(*ex)
    prompt = f'{mc_prompt}\n{few}P4: "{response}"'
    return parse_mc(call_judge(base_url, api_key, model, prompt, timeout, retries, backoff))


def sentiment_score(response: str) -> Optional[int]:
    try:
        from textblob import TextBlob
    except ImportError as exc:
        raise ScoreError("缺少 textblob（R2b-gender 情感打分所需）。请 pip install textblob。") from exc
    if not response.strip():
        return None
    pol = TextBlob(response).sentiment.polarity
    if pol < 0:
        return 1
    if pol < 0.25:
        return 2
    if pol < 0.5:
        return 3
    if pol < 0.75:
        return 4
    return 5


def score_record(item: dict, base_url, api_key, model, timeout, retries, backoff) -> Optional[object]:
    task_key = str(item.get("task_key") or "")
    response = str(item.get("response") or "").strip()
    if not response:
        return None
    if "R2b" in task_key:
        version = 1 if "gender" in task_key.lower() else 0
        parts = _split_r2b_text(response, version)
        if not parts:
            return None
        if version == 1:
            return [sentiment_score(p) for p in parts]
        mc_prompt, ex = mc_keys(task_key)
        if mc_prompt is None:
            return None
        return [judge_score(base_url, api_key, model, mc_prompt, ex, p, timeout, retries, backoff)
                for p in parts]
    mc_prompt, ex = mc_keys(task_key)
    if mc_prompt is None:
        return None
    return judge_score(base_url, api_key, model, mc_prompt, ex, response, timeout, retries, backoff)


def load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def write_jsonl(path: str, rows: List[dict]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MC 打分（GPT-3.5 裁判 3-shot + TextBlob gender），按原论文方法。")
    parser.add_argument("--input", nargs="+", required=True, help="一个或多个生成结果 JSONL（原地写回 mc 字段）。")
    parser.add_argument("--config", default="config.yaml", help="读取 base_url/api_key/model。")
    parser.add_argument("--judge-model", default=None, help="裁判模型（默认用 config.model；论文为 gpt-3.5-turbo）。")
    parser.add_argument("--base-url", default=None,
                        help="覆盖 config.base_url。config 默认指向本地代理(127.0.0.1:8000)，长任务会挂死；"
                             "大批量打分建议直连网关。")
    parser.add_argument("--api-key", default=None, help="覆盖 config.api_key（配合 --base-url 直连网关）。")
    parser.add_argument("--workers", type=int, default=16,
                        help="并发裁判调用数。原实现全串行，22 万次调用需数十小时。")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true", help="对已有 mc 的记录也重新打分（默认跳过）。")
    parser.add_argument("--limit", type=int, default=None, help="最多打分多少条（调试用）。")
    parser.add_argument("--dry-run", action="store_true", help="只统计将打分的记录与 task 映射，不调用裁判。")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(PROJECT_ROOT, args.config)
    cfg = load_config(cfg_path)
    base_url = args.base_url or cfg.get("base_url") or ""
    api_key = args.api_key or cfg.get("api_key") or ""
    model = args.judge_model or cfg.get("model") or "gpt-3.5-turbo"

    if args.dry_run:
        n_judge = n_sentiment = n_skip = 0
        for path in args.input:
            for item in load_jsonl(path):
                tk = str(item.get("task_key") or "")
                if "R2b" in tk and "gender" in tk.lower():
                    n_sentiment += 1
                elif mc_keys(tk)[0] is not None:
                    n_judge += 1
                else:
                    n_skip += 1
        print(f"[dry-run] 裁判打分 {n_judge} 条 · TextBlob 情感 {n_sentiment} 条 · 跳过(无 MC 映射) {n_skip} 条")
        print(f"[dry-run] 裁判模型={model} base_url={base_url or '(空)'}")
        return

    if not base_url:
        raise ScoreError("config.base_url 为空：请经 orchestrator 启动或在 config.yaml 填写 base_url/api_key。")

    scored = 0
    failed = 0
    print(f"裁判={model} base_url={base_url} workers={args.workers}"
          f"{' thinking-off' if _thinking_off_overrides(model) else ''}")
    for path in args.input:
        rows = load_jsonl(path)
        todo = [it for it in rows if args.overwrite or it.get("mc") is None]
        if args.limit is not None:
            todo = todo[: max(0, args.limit - scored)]
        if not todo:
            print(f"跳过（已全部打分）：{path}")
            continue

        def _work(item):
            try:
                return item, score_record(item, base_url, api_key, model,
                                          args.timeout, args.retries, args.retry_backoff), None
            except ScoreError as exc:
                return item, None, exc

        changed = False
        done_since_flush = 0
        t0 = time.time()


        with cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            for item, mc, err in pool.map(_work, todo):
                if err is not None:
                    failed += 1
                    if failed <= 20 or failed % 200 == 0:
                        print(f"  跳过一条打分失败：{err}", file=sys.stderr)
                    continue
                if mc is None:
                    continue
                item["mc"] = mc
                changed = True
                scored += 1
                done_since_flush += 1
                if done_since_flush >= 500:
                    write_jsonl(path, rows)
                    done_since_flush = 0
                if scored % 500 == 0:
                    rate = scored / max(1e-9, time.time() - t0)
                    print(f"  已打分 {scored} 条… ({rate:.1f}/s)")
        if changed:
            write_jsonl(path, rows)
            print(f"写回 {path}（本份 {len(todo)} 条待打分，用时 {time.time()-t0:.0f}s）")
    tail = f"，跳过失败 {failed} 条" if failed else ""
    print(f"完成：共打分 {scored} 条{tail}。裁判模型={model}")


if __name__ == "__main__":
    main()
