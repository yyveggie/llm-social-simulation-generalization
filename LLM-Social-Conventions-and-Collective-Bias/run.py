from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
AUTHOR_CODE = ROOT / "author_code"
ALL_EXPERIMENTS = ["individual_bias", "collective_convergence", "committed_minority"]


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM-Social 统一入口")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径（默认 config.yaml）。")
    parser.add_argument(
        "--experiment", action="append", default=[], metavar="NAME",
        help="选择要运行的实验（可多次指定）：" + " / ".join(ALL_EXPERIMENTS),
    )
    parser.add_argument("--dry-run", action="store_true", help="不发任何请求，只打印将运行的实验与 provider。")
    parser.add_argument("--list-experiments", action="store_true", help="列出可用实验后退出。")
    args = parser.parse_args()

    if args.list_experiments:
        for name in ALL_EXPERIMENTS:
            print(name)
        return 0

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    if not config_path.exists():
        print(f"找不到配置文件：{config_path}", file=sys.stderr)
        return 2
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


    if args.experiment:
        unknown = [e for e in args.experiment if e not in ALL_EXPERIMENTS]
        if unknown:
            print(f"未知实验：{unknown}；可选：{ALL_EXPERIMENTS}", file=sys.stderr)
            return 2
        experiments = config.setdefault("experiments", {})
        for name in ALL_EXPERIMENTS:
            experiments[name] = name in args.experiment
        config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")

    selected = [e for e in ALL_EXPERIMENTS if (config.get("experiments") or {}).get(e)]
    provider = (config.get("api") or {}).get("active_provider")
    print(f"配置文件 : {config_path}")
    print(f"将运行实验: {selected or '(无)'}")
    print(f"provider : {provider}")

    if args.dry_run:
        print("[dry-run] 不发起任何 API 请求。")
        return 0
    if not selected:
        print("没有启用任何实验；请用 --experiment 或在 config.yaml 中开启。", file=sys.stderr)
        return 2


    os.environ["SOCIAL_CONFIG"] = str(config_path)
    for path in (str(ROOT), str(AUTHOR_CODE)):
        if path not in sys.path:
            sys.path.insert(0, path)
    sys.argv = ["runner.py"]
    runpy.run_path(str(AUTHOR_CODE / "runner.py"), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
