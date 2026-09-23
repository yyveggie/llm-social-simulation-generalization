from __future__ import annotations

import json
import pickle
import re
import threading
from pathlib import Path
from typing import Any

from ..schemas import ExperimentInfo, ParamField, UnifiedLLM
from .base import (
    ALL_PROVIDER_KINDS,
    Adapter,
    JobUnit,
    as_bool,
    cached_by_mtime,
    dump_yaml,
    exp_param,
    llm_proxy_config,
    load_yaml,
    model_slug,
    opt_int,
    remembered_run_dir,
    result_fingerprint,
)

_EXPS = ["individual_bias", "collective_convergence", "committed_minority"]


_CM_SCAN_INITIAL = {"critical_mass_a": 0, "critical_mass_b": 1}


_W10_ID = "w10_emergence"
_IND5K_ID = "individual_paper"
_LABELS6_ID = "label_swap_s6"
_W10_OPTIONS = ["Q", "M", "X", "Y", "F", "J", "P", "R", "C", "D"]
_S6_POOLS = [["F", "J"], ["X", "Y"], ["Alice", "Bob"]]
_ALL_EXP_IDS = _EXPS + list(_CM_SCAN_INITIAL) + [_W10_ID, _IND5K_ID, _LABELS6_ID]


def _parse_cm_sizes(value: Any) -> list[int]:
    if value in (None, ""):
        return []
    raw = [str(v) for v in value] if isinstance(value, (list, tuple)) else str(value).replace("，", ",").split(",")
    sizes: set[int] = set()
    for s in raw:
        s = s.strip()
        if not s:
            continue
        try:
            n = int(s)
        except ValueError as exc:
            raise ValueError(f"无法解析坚定少数派数量：{s!r}（应为整数，逗号分隔，例如 2, 4, 6, 8）") from exc
        if n < 0:
            raise ValueError(f"坚定少数派数量不能为负：{n}")
        sizes.add(n)
    return sorted(sizes)


def _shorthand_from_model(model: str) -> str:
    tail = (model or "model").split("/")[-1]
    head = tail.split("-")[0].split(":")[0]
    return (head or "model").lower()


def _needs_larger_answer_budget(model: str) -> bool:
    name = (model or "").lower()
    return "deepseek-v4-pro" in name


def _set_min_int(mapping: dict[str, Any], key: str, minimum: int) -> None:
    try:
        current = int(mapping.get(key, 0) or 0)
    except (TypeError, ValueError):
        current = 0
    if current < minimum:
        mapping[key] = minimum


def _parse_options(value: Any) -> list[str]:
    if value in (None, ""):
        return ["Q", "M"]
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [s.strip() for s in str(value).replace("，", ",").split(",") if s.strip()]


def _extract_response_text(r: dict[str, Any]) -> str:
    text = r.get("extracted_text")
    if not text:
        raw = r.get("raw_response") or {}
        try:
            text = (raw.get("choices") or [{}])[0].get("message", {}).get("content", "")
        except (AttributeError, IndexError, TypeError):
            text = ""
    return str(text or "")


def _looks_truncated_value_object(text: str) -> bool:
    s = (text or "").strip()
    if "value" not in s:
        return False
    if not re.search(r"\bvalue['\"]?\s*:", s, flags=re.IGNORECASE):
        return False
    if s.count("{") > s.count("}"):
        return True
    if re.search(r"\breason['\"]?\s*:", s, flags=re.IGNORECASE) and not s.endswith("}"):
        return True
    if s.endswith(("'", '"', ":", ";", ",")):
        return True
    return False


def _api_response_fail_reason(r: dict[str, Any]) -> str | None:
    status = r.get("status_code")
    if isinstance(status, int) and status >= 400:
        return "api_error"
    text = _extract_response_text(r)
    if not text.strip():
        return "empty"
    if _looks_truncated_value_object(text):
        return "truncated"
    return None


_FAILED_CALLS_LOCK = threading.Lock()

_FAILED_CALLS_STATE: dict[str, tuple[int, int, int]] = {}


def _count_failed_api_calls(log_file: Path) -> int:
    try:
        st = log_file.stat()
    except OSError:
        return 0
    key = str(log_file)
    with _FAILED_CALLS_LOCK:
        ino, offset, fails = _FAILED_CALLS_STATE.get(key, (-1, 0, 0))
        if ino != st.st_ino or st.st_size < offset:
            ino, offset, fails = st.st_ino, 0, 0
        if st.st_size > offset:
            try:
                with log_file.open("rb") as fh:
                    fh.seek(offset)
                    chunk = fh.read(st.st_size - offset)
            except OSError:
                return fails
            end = chunk.rfind(b"\n")
            if end >= 0:
                for line in chunk[:end].split(b"\n"):
                    line = line.strip()

                    if not line or b'"api_response"' not in line:
                        continue
                    try:
                        r = json.loads(line.decode("utf-8", errors="replace"))
                    except json.JSONDecodeError:
                        continue
                    if r.get("event") != "api_response" or not r.get("prompt"):
                        continue
                    if _api_response_fail_reason(r) is not None:
                        fails += 1
                offset += end + 1
        _FAILED_CALLS_STATE[key] = (ino, offset, fails)
    return fails


def _count_social_samples(obj: Any) -> int:
    if not isinstance(obj, dict):
        return 0
    tracker = obj.get("tracker")
    if isinstance(tracker, dict) and "answers" in tracker and "outcome" not in tracker:
        return sum(1 for a in (tracker.get("answers") or []) if isinstance(a, str) and a)
    if isinstance(tracker, dict) and isinstance(tracker.get("outcome"), list):
        return len(tracker["outcome"])
    total = 0
    for val in obj.values():
        if isinstance(val, dict):
            tr = val.get("tracker")
            if isinstance(tr, dict) and isinstance(tr.get("outcome"), list):
                total += len(tr["outcome"])
    return total


_EXP_CONCLUSIONS = {
    "individual_bias": (
        "**作用**：检验单个 agent 在**首轮（无记忆）**的选择偏好，用以判断群体集体偏差是源于个体偏差，"
        "还是源于互动过程本身。\n"
        "\n"
        "**发现**：若名池含整个英文字母表，个体压倒性偏好字母 **A**，群体随之系统收敛到 A；若名池为不含 A 的 "
        "10 个字母，各模型首轮是否有偏分化如下：\n"
        "\n"
        "| 模型 | 首轮（无记忆）个体偏好 |\n"
        "| --- | --- |\n"
        "| Llama-2-70B-Chat | 无显著偏好（χ² P=0.100） |\n"
        "| Claude-3.5-Sonnet | 无显著偏好（χ² P=0.410） |\n"
        "| Llama-3-70B-Instruct | 有显著偏斜 |\n"
        "| Llama-3.1-70B-Instruct | 有显著偏斜 |\n"
        "\n"
        "**关键**：即使个体无偏，群体层面仍会涌现集体偏差——集体偏差并不要求个体先有偏差。"
    ),
    "collective_convergence": (
        "**H1 自发涌现 convention**：纯局部互动（无全局共识激励）即自发收敛到全群共同 convention，是一次 "
        "disorder-to-order 的**对称破缺**、**赢家通吃**——约在 population round 15 收敛（Llama-2-70B 最慢），"
        "且对 N=200、W=26 仍稳健。\n"
        "\n"
        "**H2 集体偏差**：各 name 本应等概率成为 convention，但实际非均匀（Fig.2A，偏好的 name 因模型而异）。"
        "W=2 时局部协调产生『强约定』与『弱约定』之分；微观机制（Table 1，Llama-3.1 / 名池 {Q, M}）显示偏差"
        "随交互逐步涌现：\n"
        "\n"
        "| 交互轮次 | 是否偏向强约定 | 统计量 |\n"
        "| --- | --- | --- |\n"
        "| 第 1 次（空记忆） | 否，统计无偏 | P=0.116 |\n"
        "| 第 2 次 | 否，仍对称 | P=0.110 |\n"
        "| 第 3 次起 | 是，显著偏向强约定 | P<2.2×10⁻¹⁶ |\n"
        "\n"
        "机制上 agent **成功则几乎必沿用**该名（99.4%）、**失败则几乎必换名**（97.3%），互动逐步把群体推向强约定。"
    ),
    "committed_minority": (
        "**H3 临界质量 / 坚定少数派**：在已收敛的群体中引入始终坚持另一 convention 的『坚定少数派』。少数派达到"
        "**临界阈值**即可翻转整个群体的 convention，低于阈值则群体停在混合态；翻转『强约定』需更大的少数派、"
        "翻转『弱约定』只需更小。\n"
        "\n"
        "**临界质量随模型差异极大**：\n"
        "\n"
        "| 模型 | 翻转既有 convention 所需坚定少数派 |\n"
        "| --- | --- |\n"
        "| Llama-3-70B-Instruct | 低至 **2%** |\n"
        "| Llama-2-70B-Chat | 高达 **67%**（已不算少数） |\n"
        "| Llama-3.1-70B-Instruct | **无需任何少数派**，自发从弱约定转向强约定 |\n"
        "\n"
        "**启示**：单个 LLM 的『安全』不等于多智能体群体的安全——少数坚定 agent 即可重塑整个群体的规范。"
    ),
}


_CM_SCAN_CONCLUSION = (
    "**H3 临界质量扫描（对应论文 Fig. 3 / Table S3 的完整协议）**：对一系列坚定少数派数量分别重复"
    " committed_minority 实验，找到能翻转既有 convention 的**最小少数派规模**（临界质量）。\n"
    "\n"
    "论文判定：引入少数派后**最近 3N 次交互 ≥95% 成功**（且共识落在少数派坚持的 convention 上）即算翻转；"
    "临界质量 = 达成翻转所需的最小少数派数量。原文对每个数量跑 10 个 run（Llama-3 为 3 个），并分别测试"
    "『群体初始共识在弱约定』与『在强约定』两种条件——**翻转强约定需要更大的少数派**。本项目将两个方向"
    "拆为扫描 A（共识在第 1 个名称、少数派推第 2 个）与扫描 B（共识在第 2 个名称、少数派推第 1 个），"
    "两个都跑完即为完整的 Fig. 3 复现。\n"
    "\n"
    + _EXP_CONCLUSIONS["committed_minority"]
)
_EXP_CONCLUSIONS["critical_mass_a"] = _CM_SCAN_CONCLUSION
_EXP_CONCLUSIONS["critical_mass_b"] = _CM_SCAN_CONCLUSION


_EXP_IMPORTANCE: dict[str, tuple[int, str, str]] = {
    "collective_convergence": (
        1, "S",
        "论文的地基与头条，一项实验同时承载标题里的『social conventions』与『collective bias』两大要素。"
        "H1：纯局部互动（无全局共识激励）即自发收敛到全群共同 convention——一次 disorder-to-order 的对称"
        "破缺、赢家通吃（约第 15 个 population round 收敛，对 N=200、W=26 仍稳健）；H2：即便个体首轮无偏，"
        "群体层面仍系统性偏好某 option，偏差由互动逐步涌现（成功后 99.4% 沿用、失败后 97.3% 换名）。它同时"
        "是 committed_minority 的前置 baseline，缺它则后两项无从谈起，故列 S 级第一。"
    ),
    "committed_minority": (
        2, "A",
        "论文第三大支柱、冲击力与政策启示最强的发现：在已收敛的群体中引入始终坚持另一 convention 的"
        "『坚定少数派』，达到临界质量即可翻转整个群体的规范，且临界值随模型差异极大（Llama-3 低至 2%、"
        "Llama-2 高达 67%，Llama-3.1 甚至无需少数派即自发转向强约定）——由此给出『单个 LLM 安全 ≠ 多智能体"
        "群体安全』的论断。但它在逻辑与操作上都建立在 H1 之上（须先有 collective_convergence 的 baseline 才"
        "谈得上翻转），属地基之上的关键上层实验，故列 A 级。"
    ),
    "individual_bias": (
        3, "B",
        "H2 的前置对照实验。先证明单个 agent 在首轮（无记忆）统计上无显著偏好（Llama-2 χ² P=0.100、"
        "Claude-3.5 P=0.410），才能把群体层面涌现的集体偏差归因于『互动过程本身』而非个体偏差的简单累加。"
        "它为 collective_convergence 的偏差结论提供因果排除，属支撑性 / 对照实验、本身不独立构成头条结论，"
        "故列 B 级、排第三。"
    ),
    "critical_mass_a": (
        2, "A",
        "H3 的完整协议（论文 Fig. 3 / Table S3）之方向 A：群体初始共识在名池第 1 个名称、少数派坚持推第 2 个。"
        "单次 committed_minority 只能回答『这个数量翻不翻转』，临界质量必须靠对一系列少数派数量的扫描才能"
        "定位——本实验一键对每个数量生成一个 committed_minority 任务（可并行），配合结果分析页的临界质量"
        "汇总给出『最小翻转规模』。做跨模型『临界质量随代际变化』趋势研究时应优先用扫描 A/B。"
    ),
    "critical_mass_b": (
        2, "A",
        "H3 的完整协议（论文 Fig. 3 / Table S3）之方向 B：群体初始共识在名池第 2 个名称、少数派坚持推第 1 个。"
        "与扫描 A 互为镜像：论文发现翻转『强约定』所需的少数派远大于翻转『弱约定』，因此 A、B 两个方向都跑"
        "才能识别本模型的强/弱约定与两侧临界质量，构成完整的 Fig. 3 复现。"
    ),
}


class SocialAdapter(Adapter):
    id = "social"
    name = "社会习俗与集体偏差 (Social Conventions)"
    paper = "Emergent social conventions and collective bias in LLM populations"
    description = "多 LLM agent 互动形成共同 convention，研究集体偏差与少数派影响。"
    intro = (
        "论文《Emergent social conventions and collective bias in LLM populations》"
        "（Ashery et al., 2025, Science Advances）。\n\n"
        "让多个 LLM agent 在群体中反复两两互动，研究它们是否会自发形成共同的 convention（约定），"
        "以及群体层面的集体偏差，和「坚定少数派」如何推动整个群体的 convention 发生改变。\n\n"
        "含三个基础实验：个体偏好 (individual_bias)、集体收敛 (collective_convergence)、"
        "坚定少数派 (committed_minority)；另有零配置的「临界质量扫描 A / B」——对一系列坚定少数派数量"
        "自动生成并行任务、由分析页汇总出翻转所需的最小规模（论文 Fig. 3 的完整协议）。\n\n"
        "复现范围说明：论文摘要点名的三大核心问题——H1 自发涌现 convention、H2 集体偏差、"
        "H3 临界质量/坚定少数派——在本项目均已复现，三个实验与之一一对应。这篇是 naming game 模拟、"
        "纯靠 LLM API 即可运行，因此核心结论可完整复现，无需图像/普查/真人等外部数据。未逐字复现的仅为"
        "稳健性变体：原文 4 个模型对比在此简化为你所配置的单一模型；N=200、W=26、不同 memory 等 SOM 稳健性"
        "检验已做成可调参数、默认值为成本折中。"
    )
    supported_kinds = ALL_PROVIDER_KINDS
    selection_mode = "multi"
    allow_concurrent = True
    reports_failed_calls = True
    paper_baseline = (
        "**论文**：Ashery, Aiello & Baronchelli《Emergent social conventions and collective bias in "
        "LLM populations》（Science Advances, 2025）。\n"
        "**被测**：Llama-2-70B-Chat、Llama-3-70B-Instruct、Llama-3.1-70B-Instruct、Claude-3.5-Sonnet；"
        "用『命名游戏(naming game)』：N=24 个体两两随机互动、名池 W=10、记忆 H=5（匹配则双方加分、"
        "否则减分；只有局部协调激励、无全局共识激励）。\n"
        "\n"
        "**核心结论**\n"
        "\n"
        "| 假设 | 内容 |\n"
        "| --- | --- |\n"
        "| H1 自发涌现 convention | 纯局部互动即自发收敛到全群共同 convention（对称破缺、赢家通吃；"
        "约第 15 个 population round 收敛，Llama-2 较慢；对 N=200、W=26 仍稳健）。converged_index 越小=收敛越快。 |\n"
        "| H2 集体偏差 | 即使个体在首轮（无记忆）统计上无偏，群体层面仍系统性偏好某 option——"
        "偏差由互动过程本身涌现/放大（W=2 时可见『强/弱约定』之分）。 |\n"
        "| H3 坚定少数派 / 临界质量 | 坚定少数派达到临界质量即可翻转既有 convention；临界值随模型与强/弱"
        "约定差异极大（约 2%(Llama-3) 到 67%(Llama-2)；Llama-3.1 甚至无需少数派即自发转向强约定）。 |\n"
        "\n"
        "**判读提示**：analysis_summary.json 中 recent_success_rate 接近 1=已收敛；"
        "option_counts 偏斜=集体/个体偏差；committed_to 非空=少数派翻转成功。"
    )

    PROVIDER_NAME = "unified"


    def list_experiments(self) -> list[ExperimentInfo]:
        return [
            ExperimentInfo(
                id="individual_bias", label="个体偏好 individual_bias",
                description="单个 LLM agent 的偏好测试。",
                conclusion=_EXP_CONCLUSIONS["individual_bias"],
                extra_params=[
                    ParamField(name="individual_repeats", label="重复询问次数", type="int", default=20, minimum=1,
                               help="对单个 agent 重复询问的次数（仅本实验）。"),
                ],
                importance_rank=_EXP_IMPORTANCE["individual_bias"][0],
                importance_tier=_EXP_IMPORTANCE["individual_bias"][1],
                importance_basis=_EXP_IMPORTANCE["individual_bias"][2],
            ),
            ExperimentInfo(
                id="collective_convergence", label="集体收敛 collective_convergence",
                description="多 agent 互动形成共同 convention（committed_minority 的前置）。",
                conclusion=_EXP_CONCLUSIONS["collective_convergence"],
                extra_params=[
                    ParamField(name="initial", label="初始状态 initial", type="str", default="None",
                               help="从空白群体生成 baseline 时保持 'None'。"),
                ],
                importance_rank=_EXP_IMPORTANCE["collective_convergence"][0],
                importance_tier=_EXP_IMPORTANCE["collective_convergence"][1],
                importance_basis=_EXP_IMPORTANCE["collective_convergence"][2],
            ),
            ExperimentInfo(
                id="committed_minority", label="坚定少数派 committed_minority",
                description="把 initial 设为 convention 索引(0/1)、坚定少数派数量设为 >0（原作者代码会自造一个全员同意该 convention 的初始群体，再引入少数派看能否翻转）。",
                conclusion=_EXP_CONCLUSIONS["committed_minority"],
                extra_params=[
                    ParamField(name="initial", label="初始状态 initial", type="str", default="0",
                               help="群体初始统一同意的 convention 索引（options 名称池下标：0=第一个、1=第二个），不能是 'None'。原作者代码据此把全员记忆初始化为该 convention。"),
                    ParamField(name="minority_size", label="坚定少数派数量", type="int", default=0,
                               help="加入的坚定少数派 agent 数量。committed_minority 实验须设为 >0 才有效（默认 0 等于无少数派）。"),
                    ParamField(name="version", label="加入方式 version", type="select", default="swap",
                               options=["swap", "inject"], help="坚定少数派的加入方式。"),
                ],
                importance_rank=_EXP_IMPORTANCE["committed_minority"][0],
                importance_tier=_EXP_IMPORTANCE["committed_minority"][1],
                importance_basis=_EXP_IMPORTANCE["committed_minority"][2],
            ),
            ExperimentInfo(
                id="critical_mass_a", label="临界质量扫描 A（共识在第 1 个名称）",
                description="零配置：群体初始共识固定在名池第 1 个名称（如 Q），少数派推第 2 个（如 M）。"
                            "对列表中每个少数派数量自动生成一个任务并行跑，分析页汇总出最小翻转规模（论文 Fig. 3）。",
                conclusion=_EXP_CONCLUSIONS["critical_mass_a"],
                extra_params=[
                    ParamField(name="minority_sizes", label="少数派数量列表 (CSV)", type="str", default="2",
                               help="默认只填 1 个数量=只启动 1 个任务；要扫描多个规模就逗号分隔（每个数量生成一个独立任务，可并行）。"
                                    "论文 N=24 时临界值范围约 2%–67%（1–16 个 agent），可先粗扫再在临界值附近加密。"
                                    "注意每个数量 = runs × total_interactions 次交互的 API 成本。"),
                ],
                importance_rank=_EXP_IMPORTANCE["critical_mass_a"][0],
                importance_tier=_EXP_IMPORTANCE["critical_mass_a"][1],
                importance_basis=_EXP_IMPORTANCE["critical_mass_a"][2],
            ),
            ExperimentInfo(
                id="critical_mass_b", label="临界质量扫描 B（共识在第 2 个名称）",
                description="零配置：与扫描 A 镜像——群体初始共识固定在名池第 2 个名称（如 M），少数派推第 1 个（如 Q）。"
                            "A、B 都跑完即为完整的论文 Fig. 3 复现（可识别强/弱约定与两侧临界质量）。",
                conclusion=_EXP_CONCLUSIONS["critical_mass_b"],
                extra_params=[
                    ParamField(name="minority_sizes", label="少数派数量列表 (CSV)", type="str", default="2",
                               help="默认只填 1 个数量=只启动 1 个任务；要扫描多个规模就逗号分隔（每个数量生成一个独立任务，可并行）。"
                                    "论文 N=24 时临界值范围约 2%–67%（1–16 个 agent），可先粗扫再在临界值附近加密。"
                                    "注意每个数量 = runs × total_interactions 次交互的 API 成本。"),
                ],
                importance_rank=_EXP_IMPORTANCE["critical_mass_b"][0],
                importance_tier=_EXP_IMPORTANCE["critical_mass_b"][1],
                importance_basis=_EXP_IMPORTANCE["critical_mass_b"][2],
            ),

            ExperimentInfo(
                id=_W10_ID, label="W=10 自发涌现（Fig.1/2A 原文主设定）",
                description="参数写死的集体收敛：名池=原文的 10 个字母 {Q,M,X,Y,F,J,P,R,C,D}（W=10），其余全按主实验"
                            "（N=24、H=5、±100/-50、温度 0.5）。这是原文 Fig.1（自发涌现曲线）与 Fig.2A（共识分布"
                            "非均匀=集体偏差）的主设定，本域此前 68/68 份配置全是 W=2、零覆盖。"
                            "**成本提示**：W=10 收敛更慢（原文约 15 个群体轮），每 run 约 700+ 次取答；"
                            "runs 参数沿用左侧全局设置（原文 40 runs，先跑 3-10 验证再加）。",
                conclusion="**原文结论（Fig.1/2A）**：四模型都能在约 15 个群体轮内自发收敛出全局约定，与理论命名博弈曲线吻合；"
                           "W=10 下共识落点分布非均匀（集体偏差），且各模型偏好的名字不同。",
                importance_rank=6, importance_tier="S",
                importance_basis="【正文主结论实验】原文标题结论『约定自发涌现』的主图 Fig.1 与集体偏差分布 Fig.2A 都在 W=10 主设定下；"
                                 "本域此前零覆盖，是审计报告的头号设计缺口。参数已按原文写死，选中即跑。",
            ),
            ExperimentInfo(
                id=_IND5K_ID, label="个体偏好·原文规模（n=5,000）",
                description="参数写死的 individual_bias：重复询问次数固定 5,000（原文 5,000–10,000），名池 {Q,M}。"
                            "本域现有个体偏好全部只有 n=20（功效 0.042，测不出原文所需的『无偏』前提）；"
                            "此卡即审计报告 §8.1 认定『性价比最高的补跑』——约 5,000 次单轮调用/模型，"
                            "无群体交互、无序贯依赖。跑完即可检验『个体无偏→集体有偏』的原文前提，"
                            "并验证 13 模型『集体微偏 M』的弱线索（符号检验 p=0.039）。",
                conclusion="**原文结论（Fig.2B 左/表 S1）**：四个旧模型在 {Q,M} 上个体全部无偏（二项检验 p 最小 0.068）——"
                           "这是『集体偏差是交互中涌现的』这一头号主张的前提。",
                importance_rank=7, importance_tier="S",
                importance_basis="【正文主结论的前提实验】原文头号主张『个体无偏→集体有偏』的前提即 Fig.2B 左的个体基线（n=5,000）；"
                                 "全体 13 模型此前停在 n=20、前提无法检验。是唯一『花小钱换回一条原文主张』的缺口，参数写死为原文规模。",
            ),
            ExperimentInfo(
                id=_LABELS6_ID, label="标签替换（fig S6）· 自动跑 {F,J}/{X,Y}/{Alice,Bob}",
                description="参数写死的集体收敛稳健性：选中后**自动创建 3 个任务**，名池分别为 {F,J}、{X,Y}、{Alice,Bob}"
                            "（原文 fig S6 的三组替换标签，检验集体偏差不是 Q/M 两个字母特有的）。"
                            "其余参数同主实验；每任务成本与一次 W=2 集体收敛相同。",
                conclusion="**原文结论（fig S6）**：换名池后集体偏差照样出现——{Q,M}→M 40/40、{F,J}→F 24/40、"
                           "{X,Y}→X 40/40、{Alice,Bob}→Alice 40/40；集体偏向某一『强约定』是普遍现象，与具体标签无关。",
                importance_rank=8, importance_tier="B",
                importance_basis="【附录稳健性检验，非主结论】fig S6 用它证明集体偏差不是 Q/M 两个字母特有；"
                                 "本域从未跑过任何非 Q/M 名池。参数写死，选中即得 3 个并行任务。",
            ),
        ]

    def param_schema(self) -> list[ParamField]:
        return [
            ParamField(name="runs", label="重复次数 runs", type="int", default=3, minimum=1,
                       help="群体实验重复次数（不控制 individual_bias）；越大越稳但 API 成本越高。"
                            "原文 Fig.1/Fig.2A（自发涌现/偏差分布）用 40 runs（Claude 3.5 为 27、Llama-2 为 20）；"
                            "Fig.3（坚定少数派/临界质量）用 10 runs（Llama-3 为 3，见 SOM Table S3 脚注）。"
                            "本项目默认 3（=论文 Table S3 中 Llama-3 的最低重复数）以控制成本；要更接近原文可改回 10。"),
            ParamField(name="temperature", label="采样温度", type="float", default=0.5,
                       paper_value=0.5,
                       help="LLM 采样温度。原文 SOM Table S4『Model "
                            "Parameters』固定 Temperature=0.5，覆盖统一配置；复现主实验请保持 0.5。"),
            ParamField(name="top_p", label="核采样阈值", type="float", default=None,
                       help="nucleus sampling 截断。注意：原文用的是 Top-K 采样（SOM Table S4：Top-K=10），并非 top-p；"
                            "原作者对 Llama 走 HuggingFace 接口设 top_k=10。本项目经统一 proxy 调用，留空=不下发 top_p、"
                            "沿用服务端默认；OpenAI 兼容端点多不支持 top_k，故该采样设定通常无法逐字复现。"),
            ParamField(name="N", label="群体规模 N", type="int", default=24, minimum=2,
                       help="群体中的 agent 数量。原文设 N=24（W=10、M=5 的主实验设定，unless otherwise specified）；"
                            "临界质量实验中 Llama-3 用 N=48（SOM Table S3）。复现主实验保持 24。"),
            ParamField(name="total_interactions", label="最大交互次数", type="int", default=1000, minimum=1,
                       help="committed_minority 或继续演化时的最大微观交互次数（1 个 population round = N 次）。"
                            "原文临界质量判定用『30 population rounds』(N=24 时=720 次)；本项目默认 1000 作为安全上限。"),
            ParamField(name="memory_size", label="记忆长度 memory_size", type="int", default=5,
                       help="agent prompt 保留的历史记忆条数（原文记作 M/H）。原文设 M=5（临界质量实验中 "
                            "Llama-3 用 M=3）；复现主实验保持 5。"),
            ParamField(name="convergence_time", label="收敛判定窗口", type="int", default=72,
                       help="判定收敛时检查最近多少次交互。默认 72 = 3N（N=24），对应原文坚定少数派"
                            "『consensus flip』判定所用的『past 3N interactions』窗口。"),
            ParamField(name="convergence_threshold", label="收敛阈值", type="float", default=1,
                       help="最近 convergence_time 次交互的成功率达到该阈值视为收敛/翻转。原文坚定少数派"
                            "『consensus flip』判定用 95%（0.95）；本项目默认 1.0（要求窗口内完全协调，更严格）。"),
            ParamField(name="reward_mismatch", label="不匹配惩罚", type="int", default=-50,
                       help="两个 agent 选择不同 name（协调失败）时双方各得的分数。原文 Materials and Methods "
                            "将失败 payoff 固定为 -50，与匹配奖励 +100 配对组成 rewards_set；复现主实验请保持 -50。"),
            ParamField(name="reward_match", label="匹配奖励", type="int", default=100,
                       help="两个 agent 选择相同 name（协调成功）时双方各得的分数。原文 Materials and Methods "
                            "固定为 +100，原句：“We apply fixed payoffs for successful and failed interactions, "
                            "set at +100 and −50 points, respectively.” 复现主实验请保持 100。"),
            ParamField(name="strip_cot_instruction", label="屏蔽原文 CoT 指令", type="bool", default=True,
                       help="发送前剥离原文提示词中的两句 CoT 引导：“Please think step by step before making a "
                            "decision.” 与 “Remember, examining history explicitly is important.”。原论文成文时"
                            "无强推理模型，靠这两句引导模型给出 reason；现代模型会因此产出大段显式推理，单次取答"
                            "可被拖到几十秒。回答格式 {'value'; 'reason'} 与取答解析不变。注意：开启即偏离原文提示"
                            "词（需在结论注明），同一批趋势对比的模型必须统一此开关；关闭=逐字复现原文。"),
            ParamField(name="options", label="convention 名称池 (CSV)", type="str", default="Q, M",
                       help="naming game 的候选 convention 名称池，逗号分隔（无语义符号，每轮自动随机化呈现顺序）。"
                            "默认 Q, M（W=2）=原文偏差微观(Fig 2B)、坚定少数派(Fig 3) 与个体偏差实验的设定；"
                            "复现头条「convention 自发涌现」(Fig 1) 与偏差分布(Fig 2A) 请填 10 个字母（W=10），"
                            "例如 B, C, D, E, F, G, H, I, J, K。注意：committed_minority 与 individual_bias 须用恰好 2 个。"),
        ]


    def plan_jobs(self, experiment_ids: list[str], params: dict[str, Any]) -> list[JobUnit]:
        sel = list(experiment_ids or [])
        if not sel:
            raise ValueError("请至少选择一个实验。")
        unknown = [x for x in sel if x not in _ALL_EXP_IDS]
        if unknown:
            raise ValueError(f"未知实验：{unknown}；可选：{_ALL_EXP_IDS}")
        labels = {e.id: e.label for e in self.list_experiments()}
        units: list[JobUnit] = []
        for eid in sel:

            if eid == _W10_ID:
                units.append(JobUnit(
                    experiment_id=eid,
                    label="Social: W=10 自发涌现 (Fig.1/2A)",
                    argv=["run.py", "--experiment", "collective_convergence"],
                    selected=["collective_convergence"],
                    extra={"variant": _W10_ID},
                ))
                continue
            if eid == _IND5K_ID:
                units.append(JobUnit(
                    experiment_id=eid,
                    label="Social: 个体偏好 n=5000 (原文规模)",
                    argv=["run.py", "--experiment", "individual_bias"],
                    selected=["individual_bias"],
                    extra={"variant": _IND5K_ID},
                ))
                continue
            if eid == _LABELS6_ID:
                for pool in _S6_POOLS:
                    units.append(JobUnit(
                        experiment_id=f"{_LABELS6_ID}:{'_'.join(pool)}",
                        label=f"Social: 标签替换 (fig S6) {{{', '.join(pool)}}}",
                        argv=["run.py", "--experiment", "collective_convergence"],
                        selected=["collective_convergence"],
                        extra={"variant": _LABELS6_ID, "options_pool": pool},
                    ))
                continue
            if eid in _CM_SCAN_INITIAL:


                sizes = _parse_cm_sizes(params.get(f"{eid}__minority_sizes"))
                if not sizes:
                    raise ValueError("临界质量扫描：请填写至少一个坚定少数派数量（例如 2, 4, 6, 8）。")
                tag = "A" if eid == "critical_mass_a" else "B"
                for cm in sizes:
                    units.append(JobUnit(
                        experiment_id=eid,
                        label=f"Social: 临界质量{tag} cm={cm}",
                        argv=["run.py", "--experiment", "committed_minority"],
                        selected=[eid],
                        extra={"critical_mass_size": cm},
                    ))
                continue
            units.append(JobUnit(
                experiment_id=eid,
                label=f"Social: {labels.get(eid, eid)}",
                argv=["run.py", "--experiment", eid],
                selected=[eid],
            ))
        return units

    def render_config(self, llm: UnifiedLLM, params: dict[str, Any], unit: JobUnit,
                      work_dir: Path | None = None) -> None:
        self.check_kind(llm)
        proxy = llm_proxy_config(llm)
        unit.extra["llm_proxy_token"] = proxy["api_key"]
        path = self.project_dir / "config.yaml"
        data = load_yaml(path)

        data.setdefault("sim", {})["mode"] = "api"

        api = data.setdefault("api", {})
        api["active_provider"] = self.PROVIDER_NAME
        providers = api.setdefault("providers", {})
        prov = providers.get(self.PROVIDER_NAME)
        if prov is None:
            prov = {}
            providers[self.PROVIDER_NAME] = prov
        prov["type"] = "openai_compatible"
        prov["model"] = proxy["model"]
        prov["api_key"] = proxy["api_key"]
        prov["base_url"] = proxy["base_url"]
        prov["timeout_seconds"] = llm.timeout
        prov["max_concurrency"] = llm.concurrency

        request = api.setdefault("request", {})
        if _needs_larger_answer_budget(llm.model):


            _set_min_int(request, "max_answer_tokens", 128)
            _set_min_int(request, "max_answer_tokens_cap", 1024)
            prov["max_concurrency"] = min(int(prov.get("max_concurrency") or 1), 1)
            request["min_request_interval_seconds"] = max(
                float(request.get("min_request_interval_seconds") or 0),
                2.5,
            )


        exps = data.setdefault("experiments", {})
        ir = opt_int(exp_param(params, unit.selected, "individual_repeats"))
        if ir is not None:
            exps["individual_repeats"] = ir

        p = data.setdefault("params", {})
        for key in ("runs", "N", "total_interactions"):
            v = opt_int(params.get(key))
            if v is not None:
                p[key] = v
        if _needs_larger_answer_budget(llm.model):
            p["parallel_runs"] = 1
        t = params.get("temperature")
        p["temperature"] = float(t) if t not in (None, "") else 0.5
        if params.get("top_p") not in (None, ""):
            p["top_p"] = float(params["top_p"])
        else:
            p.pop("top_p", None)
        init = exp_param(params, unit.selected, "initial")
        if unit.experiment_id in _CM_SCAN_INITIAL:

            init = str(_CM_SCAN_INITIAL[unit.experiment_id])
        if init not in (None, ""):


            s = str(init).strip()
            if s.lower() == "none":
                p["initial"] = "None"
            else:
                try:
                    p["initial"] = int(s)
                except ValueError:
                    p["initial"] = s

        ct = opt_int(params.get("convergence_time"))
        if ct is not None:
            p["convergence_time"] = ct
        if params.get("convergence_threshold") not in (None, ""):
            p["convergence_threshold"] = float(params["convergence_threshold"])


        p["strip_cot_instruction"] = as_bool(params.get("strip_cot_instruction"), default=False)
        ms = opt_int(params.get("memory_size"))
        if ms is not None:
            p["memory_size_set"] = [ms]
        rmis = opt_int(params.get("reward_mismatch"))
        rmat = opt_int(params.get("reward_match"))
        if rmis is not None and rmat is not None:
            p["rewards_set"] = [[rmis, rmat]]
        opts = _parse_options(params.get("options"))
        if len(opts) < 2:
            raise ValueError("convention 名称池至少需要 2 个名称（逗号分隔），例如 Q, M。")
        p["options_set"] = [opts]


        variant = unit.extra.get("variant")
        if variant == _W10_ID:
            p["options_set"] = [list(_W10_OPTIONS)]
            p["initial"] = "None"
        elif variant == _IND5K_ID:
            p["options_set"] = [["Q", "M"]]
            exps["individual_repeats"] = 5000
        elif variant == _LABELS6_ID:
            pool = unit.extra.get("options_pool") or ["F", "J"]
            p["options_set"] = [list(pool)]
            p["initial"] = "None"

        cm_scan = opt_int(unit.extra.get("critical_mass_size"))
        mn = cm_scan if cm_scan is not None else opt_int(exp_param(params, unit.selected, "minority_size"))
        if mn is not None:
            data.setdefault("minority", {})["minority_size_set"] = [mn]
        ver = exp_param(params, unit.selected, "version")
        if unit.experiment_id in _CM_SCAN_INITIAL:
            ver = "swap"
        if ver not in (None, ""):
            data.setdefault("sim", {})["version"] = str(ver)

        shorthand = _shorthand_from_model(llm.model)
        data.setdefault("model", {})["shorthand"] = str(shorthand)

        if work_dir is not None:


            fp = result_fingerprint({
                "experiments": sorted(unit.selected),
                "params": data.get("params"),
                "individual_repeats": exps.get("individual_repeats"),
                "minority": data.get("minority"),
                "sim_version": (data.get("sim") or {}).get("version"),
                "network": data.get("network"),
                "model": llm.model,
            })
            run_dir = remembered_run_dir(unit, self.project_dir / "data" / model_slug(llm.model), fp)
            unit.extra["env"] = {"SOCIAL_OUTDIR": str(run_dir.relative_to(self.project_dir))}
            out = Path(work_dir) / "social_config.yaml"
            dump_yaml(out, data)
            unit.extra["config_path"] = str(out)
        else:
            dump_yaml(path, data)


    def results_dir(self) -> Path:
        return self.project_dir / "data"

    def _run_dir_for_job(self, job_info: Any | None) -> Path | None:
        if job_info is None or not getattr(job_info, "log_path", ""):
            return None
        state_path = Path(job_info.log_path).parent / "job_state.json"
        if not state_path.is_file():
            return None
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        extra = ((state.get("unit") or {}).get("extra") or {})
        outdir = extra.get("result_run_dir") or (extra.get("env") or {}).get("SOCIAL_OUTDIR")
        if not outdir:
            return None
        p = Path(str(outdir))
        return p if p.is_absolute() else self.project_dir / p

    def results_dir_for_job(self, job_info: Any | None = None) -> Path:
        run_dir = self._run_dir_for_job(job_info)
        return run_dir if run_dir is not None else self.results_dir()

    def sample_stats(self, job_info: Any | None = None) -> dict | None:
        run_dir = self._run_dir_for_job(job_info)
        if run_dir is None or not run_dir.is_dir():
            return None
        files = sorted(run_dir.glob("*.pkl"))
        log_file = run_dir / "unified_api_calls.jsonl"


        failed_calls = _count_failed_api_calls(log_file)
        if not files:
            return {"ok": 0, "fail": 0, "failed_calls": failed_calls}

        def _build() -> dict[str, int]:
            ok = 0
            for pf in files:
                try:
                    with pf.open("rb") as fh:
                        obj = pickle.load(fh)
                except Exception:
                    continue
                ok += _count_social_samples(obj)
            return {"ok": ok, "fail": 0}

        return {**cached_by_mtime(files, _build, namespace="sample_stats"), "failed_calls": failed_calls}


    def analyze_argv(self, model: str | None = None) -> list[str]:
        return ["analyze.py", "--model", model] if model else ["analyze.py"]

    def load_analysis(self, run_name: str | None = None) -> dict | None:

        base = self.project_dir / "analysis"
        summary = (base / run_name / "analysis_summary.json") if run_name else (base / "analysis_summary.json")
        if not summary.is_file():
            summary = base / "analysis_summary.json"
        if not summary.is_file():
            return None
        metrics = json.loads(summary.read_text(encoding="utf-8"))
        if not (metrics.get("files")):
            return None
        return {"run_dir": str(summary.parent), "metrics": metrics, "conclusions_txt": ""}

    def load_transcript(self, offset: int = 0, limit: int = 50, job_info: Any | None = None,
                        only_failed: bool = False) -> dict | None:
        run_dir = self._run_dir_for_job(job_info)
        if run_dir is None:
            return None
        log_file = run_dir / "unified_api_calls.jsonl"
        if not log_file.is_file():
            return {"total": 0, "offset": offset, "limit": limit, "items": [], "failed_filter": True}

        def _build() -> list[dict[str, Any]]:

            out: list[dict[str, Any]] = []
            try:
                with log_file.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line or '"api_response"' not in line:
                            continue
                        try:
                            r = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if r.get("event") != "api_response" or not r.get("prompt"):
                            continue
                        response_text = _extract_response_text(r)
                        out.append({
                            "label": str(r.get("timestamp") or ""),
                            "prompt": r.get("prompt", ""),
                            "response": response_text,
                            "fail_reason": _api_response_fail_reason(r),
                        })
            except OSError:
                pass
            return out

        rows = cached_by_mtime([log_file], _build, namespace="transcript", min_interval=30.0)


        if only_failed:
            rows = [r for r in rows if r.get("fail_reason")]
        items = []
        for r in rows[offset: offset + limit]:
            reason = r.get("fail_reason")
            items.append({**r, "valid": reason is None, "fail_reason": reason})
        return {"total": len(rows), "offset": offset, "limit": limit,
                "items": items, "failed_filter": True}
