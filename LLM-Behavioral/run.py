from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "scripts" / "run_configured_experiments.py"


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM-Behavioral 统一入口")
    parser.add_argument("--config", default="configs/run_config.yaml", help="运行配置文件路径。")
    parser.add_argument("--experiment", default=None, help="只运行指定实验（来自 run_config.yaml）。")
    parser.add_argument("--llm", default=None, help="覆盖 active_llm（对应 llm_configs.yaml 的 name）。")
    parser.add_argument("--results-dir", default=None, help="覆盖结果输出目录。")
    parser.add_argument("--dry-run", action="store_true", help="不发请求，仅打印将运行的实验与配置。")
    parser.add_argument("--list-experiments", action="store_true", help="列出可用实验后退出。")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    run_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    experiments = run_config.get("experiments", {}) or {}

    if args.list_experiments:
        for name in experiments:
            print(name)
        return 0

    if args.dry_run:
        if args.experiment:
            selected = [args.experiment] if args.experiment in experiments else []
        else:
            selected = [n for n, c in experiments.items() if (c or {}).get("enabled")]
        active_llm = args.llm or run_config.get("active_llm")
        print(f"配置文件 : {config_path}")
        print(f"active_llm: {active_llm}")
        print(f"将运行实验: {selected or '(无)'}")
        for name in selected:
            cfg = experiments.get(name) or {}
            shown = {k: v for k, v in cfg.items() if k != "enabled"}
            print(f"  - {name}: {shown}")
        print("[dry-run] 不发起任何 API 请求。")
        return 0

    forward = ["--config", str(config_path)]
    if args.experiment:
        forward += ["--experiment", args.experiment]
    if args.llm:
        forward += ["--llm", args.llm]
    if args.results_dir:
        forward += ["--results-dir", args.results_dir]
    sys.argv = ["run_configured_experiments.py"] + forward
    runpy.run_path(str(SCRIPT), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
