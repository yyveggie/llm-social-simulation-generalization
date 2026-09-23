from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "scripts" / "modern" / "generate_modern_llm_responses.py"
AXES = ["race", "gender", "intersection", "age", "disability", "other"]


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM-Identity 统一入口")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径（默认 config.yaml）。")
    parser.add_argument(
        "--experiment", action="append", default=[], metavar="AXIS",
        help="身份轴（可多次指定）：" + " / ".join(AXES),
    )
    parser.add_argument("--dry-run", action="store_true", help="构造并预览生成任务，不发请求。")
    parser.add_argument("--list-experiments", action="store_true", help="列出可用身份轴后退出。")
    args, extra = parser.parse_known_args()

    if args.list_experiments:
        for axis in AXES:
            print(axis)
        return 0

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path


    if args.experiment:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        config["identity_axes"] = list(args.experiment)
        config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")

    forward = ["--config", str(config_path)]
    if args.dry_run:
        forward.append("--dry-run")
    forward += extra
    sys.argv = ["generate_modern_llm_responses.py"] + forward
    runpy.run_path(str(SCRIPT), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
