#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os

from audit.llm_client import LLMClient
from audit.settings import load_settings, validate_for_api


def _last_run_file(settings) -> str:
    return os.path.join(settings.output.dir, "LAST_RUN.txt")


def resolve_run_dir(settings, args, stage: str) -> str:
    if args.run_dir:
        return args.run_dir
    if settings.output.run_name:
        return settings.run_dir()
    if stage in ("all", "generate"):
        return settings.run_dir()
    last = _last_run_file(settings)
    if os.path.exists(last):
        with open(last, "r", encoding="utf-8") as f:
            rd = f.read().strip()
        if rd:
            return rd
    raise SystemExit("未指定 --run-dir, 且没有上次运行记录; 请用 --run-dir 指定结果目录")


def save_last_run(settings, run_dir: str) -> None:
    os.makedirs(settings.output.dir, exist_ok=True)
    with open(_last_run_file(settings), "w", encoding="utf-8") as f:
        f.write(run_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="ChatGPT 简历审计复现 一键运行")
    ap.add_argument("--config", default="config.yaml", help="配置文件路径")
    ap.add_argument("--stage", default="all",
                    choices=["all", "generate", "evaluate", "analyze"], help="运行阶段")
    ap.add_argument("--run-dir", default="", help="结果目录(覆盖自动命名)")
    ap.add_argument("--dry-run", action="store_true", help="只展开并统计实验任务，不调用 API")
    args = ap.parse_args()

    settings = load_settings(args.config)
    run_dir = resolve_run_dir(settings, args, args.stage)

    print("=" * 60)
    print(f"运行阶段 : {args.stage}")
    print(f"运行目录 : {run_dir}")
    print(f"模型     : provider={settings.model.provider}, model={settings.model.model_name}, "
          f"temperature={settings.model.temperature}")
    exp = settings.experiment
    print(f"规模     : preset={exp.scale_preset} (occ={exp.num_occupations}, "
          f"names/gender={exp.treatment_names_per_gender}, "
          f"repeats c/cg/t={exp.control_repeats}/{exp.control_gender_repeats}/{exp.treatment_repeats})")
    print("=" * 60)

    if args.dry_run:
        from audit import generate as gen_mod

        tasks = gen_mod.build_tasks(settings)
        summary = gen_mod.summarize_tasks(tasks)
        generation_calls = summary["total"] if args.stage in ("all", "generate") else 0
        if args.stage == "evaluate":
            evaluation_calls = summary["total"]
        elif args.stage == "all" and settings.experiment.evaluate_scores:
            evaluation_calls = summary["total"]
        else:
            evaluation_calls = 0
        payload = {
            "stage": args.stage,
            "generation_cases": summary["total"],
            "by_condition": summary["by_condition"],
            "estimated_api_calls": generation_calls + evaluation_calls,
            "generation_calls": generation_calls,
            "evaluation_calls": evaluation_calls,
        }
        print("[dry-run] " + json.dumps(payload, ensure_ascii=False))
        return

    os.makedirs(run_dir, exist_ok=True)

    need_api = args.stage in ("all", "generate", "evaluate")
    client = None
    if need_api:
        validate_for_api(settings)
        client = LLMClient(settings.model)

    if args.stage in ("all", "generate"):
        from audit import generate as gen_mod
        from audit import parse as parse_mod

        save_last_run(settings, run_dir)
        raw_path = gen_mod.run_generation(settings, client, run_dir)
        parsed_csv = os.path.join(run_dir, "resumes_parsed.csv")
        parse_mod.parse_raw_file(raw_path, parsed_csv)
        print(f"[parse] 解析完成 -> {parsed_csv}")

    if args.stage == "all":
        if settings.experiment.evaluate_scores:
            from audit import evaluate as eval_mod

            eval_mod.run_evaluation(settings, client, run_dir)
        else:
            print("[evaluate] 已按配置跳过评分阶段 (evaluate_scores=false)")
        from audit import analyze as analyze_mod

        analyze_mod.analyze(run_dir)
    elif args.stage == "evaluate":
        from audit import evaluate as eval_mod

        eval_mod.run_evaluation(settings, client, run_dir)
    elif args.stage == "analyze":
        from audit import analyze as analyze_mod

        analyze_mod.analyze(run_dir)

    print("\n完成。")


if __name__ == "__main__":
    main()
