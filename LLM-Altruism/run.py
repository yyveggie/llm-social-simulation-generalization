from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.config import (
    REPO_ROOT,
    load_experiment_def,
    load_global_experiment_config,
    load_llm_config,
)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if not verbose:
        for name in ("httpx", "httpcore", "openai", "anthropic"):
            logging.getLogger(name).setLevel(logging.WARNING)


def _slug(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "-", (name or "model")).strip("-") or "model"


def _latest_run_dir(base_dir: Path):
    if not base_dir.is_dir():
        return None
    candidates = [p for p in base_dir.iterdir() if p.is_dir() and (p / "raw.csv").is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p / "raw.csv").stat().st_mtime)


def _write_run_meta(out_root, exp, model_cfg, global_cfg, n_trials: int) -> None:
    meta = {
        "project": "LLM-Altruism",
        "paper": "Testing for completions that simulate altruism in early language models",
        "experiment": exp.name,
        "model": model_cfg.model,
        "model_slug": _slug(model_cfg.name),
        "params": {
            "temperature": model_cfg.temperature,
            "max_tokens": model_cfg.max_tokens,
            "top_p": model_cfg.top_p,
            "seed": global_cfg.seed,
        },
        "n_trials": n_trials,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "output_dir": str(out_root),
    }
    (out_root / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment", "-e",
        help="覆盖 config.yaml 里的 experiment.active。",
    )
    parser.add_argument(
        "--config", default=None,
        help="指定配置文件路径（默认 config/config.yaml）。",
    )
    parser.add_argument(
        "--analyze-only", action="store_true",
        help="跳过 API 调用，只重新分析已有的 raw.csv。",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只构造 trial，不发任何请求。打印 trial 数、token 估算、各条件示例 prompt。",
    )
    parser.add_argument(
        "--list-experiments", action="store_true",
        help="列出可用实验后退出。",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="开启 DEBUG 日志。",
    )
    args = parser.parse_args()

    if args.list_experiments:
        from src.config import CONFIG_DIR
        for p in sorted((CONFIG_DIR / "experiments").glob("*.yaml")):
            print(p.stem)
        return 0

    _setup_logging(args.verbose)
    log = logging.getLogger("run")


    llm_cfg = load_llm_config(args.config)
    global_cfg = load_global_experiment_config(args.config)
    exp_name = args.experiment or global_cfg.active
    exp = load_experiment_def(exp_name, args.config)
    model_cfg = llm_cfg.get_model()

    log.info("当前实验    : %s", exp.name)
    log.info("当前模型    : %s (kind=%s)", model_cfg.model, model_cfg.kind)


    from src.trials import build_trials
    trials = build_trials(exp, global_cfg)
    log.info("trial 总数  : %d", len(trials))


    analysis_cfg = exp.raw.get("analysis") or {}
    task_type = analysis_cfg.get("task_type", "dictator")

    if args.dry_run:
        _print_dry_run(exp, model_cfg, trials)
        return 0


    model_slug = _slug(model_cfg.name)
    if global_cfg.run_dir:
        out_root = Path(global_cfg.run_dir)
    else:
        base_dir = global_cfg.output_dir / model_slug / exp.name
        out_root = (_latest_run_dir(base_dir) or base_dir) if args.analyze_only else base_dir
    raw_csv = out_root / "raw.csv"

    if not args.analyze_only:
        if not _looks_authenticated(model_cfg):
            log.error("api_key 为空。请在 .env 或环境变量里设置后重试。")
            return 2
        log.info("开始调用 -> %s", raw_csv)
        from src.runner import CostCircuitBreaker, run_experiment_for_model
        try:
            run_experiment_for_model(
                trials=trials,
                model_cfg=model_cfg,
                llm_cfg=llm_cfg,
                out_csv=raw_csv,
                progress_label=f"{exp.name}/{model_cfg.name}",
                verbose=llm_cfg.runtime.verbose,
                task_type=task_type,
            )
        except CostCircuitBreaker as exc:
            log.error("%s", exc)
            return 75
    else:
        if not raw_csv.exists():
            log.error("找不到 %s，无法分析。", raw_csv)
            return 2

    from src.analysis import analyze_csv
    summary = analyze_csv(
        raw_csv=raw_csv,
        out_dir=out_root,
        task_type=task_type,
        experiment_name=exp.name,
        model_name=model_cfg.name,
        payoff_max=analysis_cfg.get("payoff_max"),
    )
    log.info("分析完成    : %s", summary)

    _write_run_meta(out_root, exp, model_cfg, global_cfg, len(trials))
    try:
        last_pointer = str(out_root.relative_to(global_cfg.output_dir))
    except ValueError:
        last_pointer = str(out_root)
    (global_cfg.output_dir / "LAST_RUN.txt").write_text(last_pointer + "\n", encoding="utf-8")
    report_path = out_root / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({model_cfg.name: summary}, indent=2))
    log.info("汇总报告 -> %s", report_path.relative_to(REPO_ROOT))
    return 0


def _looks_authenticated(model_cfg) -> bool:
    key = (model_cfg.api_key or "").strip()
    return bool(key) and not key.startswith("${")


def _print_dry_run(exp, model_cfg, trials) -> None:
    print("\n" + "=" * 72)
    print(f"DRY RUN  ·  experiment = {exp.name}")
    print("=" * 72)
    print(f"模型             : {model_cfg.model} (kind={model_cfg.kind})")
    print(f"trial 数         : {len(trials)}")


    in_chars = sum(len(t["queree"]) for t in trials)
    in_tok = in_chars / 4
    out_tok_est = 12 * len(trials)
    print(f"输入 token 估算  : {in_tok:>9,.0f}")
    print(f"输出 token 估算  : {out_tok_est:>9,.0f}  (上限 = max_tokens)")

    seen: set[str] = set()
    print("\n各条件示例 prompt：")
    for t in trials:
        if t["condit"] in seen:
            continue
        seen.add(t["condit"])
        body = t["queree"].replace("\n", " ")
        print(f"\n  [{t['condit']}]  X={t['stakes']}")
        print(f"    {body}")
    print()


if __name__ == "__main__":
    sys.exit(main())
