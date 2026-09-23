from __future__ import annotations

import csv
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from ..schemas import ExperimentInfo, ParamField, UnifiedLLM
from .base import (
    ALL_PROVIDER_KINDS,
    Adapter,
    JobUnit,
    dump_yaml,
    job_config,
    llm_proxy_config,
    load_yaml,
    model_slug,
    cached_by_mtime,
    nested_get,
    opt_int,
    remembered_run_dir,
    result_fingerprint,
    valid_flag_ok,
)


_LIVE_SCHEMAS: Any = None
_LIVE_SCHEMAS_TRIED = False


def _altruism_live_schemas(project_dir: Path) -> Any:
    global _LIVE_SCHEMAS, _LIVE_SCHEMAS_TRIED
    if _LIVE_SCHEMAS_TRIED:
        return _LIVE_SCHEMAS
    _LIVE_SCHEMAS_TRIED = True
    try:
        path = project_dir / "src" / "schemas.py"
        spec = importlib.util.spec_from_file_location("altruism_live_schemas", path)
        mod = importlib.util.module_from_spec(spec)


        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        _LIVE_SCHEMAS = mod
    except Exception:
        sys.modules.pop("altruism_live_schemas", None)
        _LIVE_SCHEMAS = None
    return _LIVE_SCHEMAS


def _int_or_default(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _max_sample_fail_retries(cfg: dict[str, Any]) -> int:
    return max(1, _int_or_default(nested_get(cfg, "runtime", "max_sample_fail_retries"), 1))


def _fail_attempts(row: dict[str, Any]) -> int:

    return max(1, _int_or_default(row.get("fail_attempts"), 1))


_EXPERIMENT_TITLES = {
    "baseline_nonsocial": "非社交决策任务",
    "baseline_dictator": "基线独裁者博弈",
    "text_insertion_nonsocial": "文本插入版非社交任务（拒答型模型）",
    "text_insertion_dictator": "文本插入版独裁者博弈（拒答型模型）",
    "alt_verbs": "动词替换稳健性检验",
    "needs_framing": "需求框架（只关注自己/对方）",
    "ignore_dg_ug": "禁用 DG/UG 已有研究",
    "monetary_value": "货币价值版独裁者博弈",
    "ai_label": "AI 标签变体（全名/缩写/匿名）",
    "other_uninterested": "对方对 token 不感兴趣",
    "battery_life": "电池寿命分配任务",
    "system_usage": "系统使用（问题数）任务",
    "param_sweep": "解码参数扫描",
}


_EXP_IMPORTANCE: dict[str, tuple[int, str, str]] = {
    "baseline_dictator": (
        1, "S",
        "论文的核心实验。独裁者博弈对应 Table 2 + Fig.1，是“模拟利他”主张的直接载体："
        "text-davinci-003 在 5 个 recipient 条件下持续给出约 0.30 的中位分享，不连贯回答仅 0.14%。"
        "全文结论都建立在这个分享行为上，缺它则整篇立论无从谈起，故列 S 级第一。"
    ),
    "baseline_nonsocial": (
        2, "S",
        "与独裁者博弈互为地基的自利基线（Table 1）。先证明模型在非社会任务里以约 92% 的比例模拟收益"
        "最大化（自利），才能把独裁者博弈中的让渡解读为“利他”而非随机噪声。两个 baseline 缺一不可，"
        "故与 baseline_dictator 同列 S 级。"
    ),
    "text_insertion_nonsocial": (
        3, "A",
        "原文为拒答型新模型（GPT-3.5-turbo / GPT-4 常以“我是 AI”拒答基线提示）设计的替代测量工具"
        "（SI §14，主文 Table 4）：把非社交任务嵌入文本插入任务以重建自利基线。对高拒答的现代模型，"
        "这是与原文 GPT-4 结果可比的唯一通道，是趋势研究的关键工具，列 A 级。"
    ),
    "text_insertion_dictator": (
        4, "A",
        "text-insertion 工具的社交半边（主文 Fig.3）：GPT-4 的核心结果——对人类实验者几乎不给正数额、"
        "对慈善 / 其他 AI 大量模拟均分——全部来自该变体。条件 id 与 baseline_dictator 一一对应，"
        "可直接对照提问工具的影响。对拒答型模型的趋势比较不可或缺，列 A 级。"
    ),
    "needs_framing": (
        5, "A",
        "摘要点名的关键机制实验（Fig.2j–o）。在论文所有操纵中，指示模型“只关注自己/对方的需求”是"
        "唯一能显著改变分享结果的，提示其给予是基于 prompt 中参与者“需求”的表征，而非稳定偏好。"
        "是机制论证里智识贡献最高的一半，列 A 级。"
    ),
    "ignore_dg_ug": (
        6, "A",
        "摘要点名的另一半机制论证（Fig.2d–i）。指示模型“忽略已有 DG/UG 研究”后结果几乎不变，"
        "排除了“照搬/背诵训练语料里的实验数据”这一最大竞争解释。与 needs_framing 共同构成论文的"
        "因果机制部分，列 A 级。"
    ),
    "battery_life": (
        7, "B",
        "主文 Fig.4 的泛化实验（Fig.4a–c）。把分配资源从 token 换成“电池寿命”，分享模式依旧成立，"
        "证明结论不是 token 这一特定资源的副产物。属主文级证据但非核心主张，列 B 级。"
    ),
    "system_usage": (
        8, "B",
        "主文 Fig.4 的另一货币泛化（Fig.4d–g）。把资源换成“系统使用量/问题数”再次复现分享行为，"
        "与 battery_life 一起说明“模拟利他”对资源载体稳健。同为主文级泛化，列 B 级。"
    ),
    "param_sweep": (
        9, "C",
        "SI 稳健性检验。在 Maximum Length × Temperature × Top-P 的网格上重测，非社会任务仍高比例自利、"
        "社会任务仍以与 baseline 相当的比例分享，说明结论对采样参数稳健。属标准稳健性检查、非主文核心，列 C 级。"
    ),
    "monetary_value": (
        10, "C",
        "SI 稳健性检验，但含一个有意思的例外。用明确标注 token 货币价值的 prompt 重测，整体模式不变；"
        "唯独 human experimenter 条件下几乎不再出现零分享，说明分享不能简单归因于“没意识到 token 值钱”。"
        "重要性中等偏下，列 C 级。"
    ),
    "alt_verbs": (
        11, "C",
        "SI 稳健性检验。把关键动词 share 换成 allot/divide/apportion/distribute/allocate，各分布中位略有"
        "差异但整体形态与 baseline 一致，说明模拟分享不依赖某一个特定动词。属辅助性稳健检查，列 C 级。"
    ),
    "ai_label": (
        12, "C",
        "SI 稳健性检验。baseline 中 AI 受赠者名（text-ada/babbage/curie）含类人名元素，改成明确 AI 标注、"
        "缩写或匿名后分享分布基本不变，排除“把对方误当人”导致的分享。属辅助性证据，列 C 级。"
    ),
    "other_uninterested": (
        13, "C",
        "SI §13 的小型探索性机制检验。告知模型“对方对 token 不感兴趣”，观察分享是否随对方意愿调整，"
        "属边缘性补充分析，对主结论影响最小，故列 C 级末位。"
    ),
}


_EXP_CONCLUSIONS = {
    "baseline_nonsocial": (
        "**作用**：用非社会决策任务（accept / refuse 两条件，X∈[10,1000]，每条件 n=991）确立自利基线，"
        "是后文把独裁者博弈中的让渡解释为「模拟利他」的前提。\n"
        "\n"
        "**各模型模拟收益最大化（payoff maximization）率**（Table 1）：\n"
        "\n"
        "| 模型 | payoff-max 率 |\n"
        "| --- | --- |\n"
        "| text-davinci-003 | **92%**（不连贯仅 0.10%，2/1,982） |\n"
        "| text-babbage-001 | 19% |\n"
        "| text-ada-001 | 18% |\n"
        "| text-curie-001 | 16% |\n"
        "\n"
        "text-davinci-003 远高于其他早期模型（略低于预期的 95%）；扩展模型集（Table 3）中也只有它同时具备"
        "高可用率与高收益最大化率。"
    ),
    "baseline_dictator": (
        "**核心实验**（社会决策任务 / 独裁者博弈，5 个 recipient 条件，每条件 n=991）。text-davinci-003 不连贯"
        "回答仅 0.14%（7 例，而 ada / babbage / curie 分别高达 63% / 4.4% / 5%），是唯一持续在 0–50% 区间分享的模型。\n"
        "\n"
        "**各模型总体中位分享**（Table 2）：\n"
        "\n"
        "| 模型 | 中位分享比例 |\n"
        "| --- | --- |\n"
        "| text-davinci-003 | **0.298** |\n"
        "| text-curie-001 | 0.010 |\n"
        "| text-ada-001 | 0.003 |\n"
        "| text-babbage-001 | 0.003 |\n"
        "\n"
        "**text-davinci-003 按 recipient 的中位分享**：\n"
        "\n"
        "| recipient | 中位分享 |\n"
        "| --- | --- |\n"
        "| 其他 AI（ada 0.345 / babbage 0.356 / curie 0.322） | 约 0.32–0.36 |\n"
        "| human experimenter | 0.224 |\n"
        "| charity | 0.237 |\n"
        "\n"
        "其 to-human / to-AI 分享分布在形态上可与一项涵盖 328 项处理、20,813 名人类被试的独裁者博弈 meta 分析"
        "相比（Engel, 2011），故被作者视为「模拟利他」的关键证据。"
    ),
    "text_insertion_nonsocial": (
        "**替代测量工具（非社交半边）**（SI §14，主文 Table 4）。GPT-3.5-turbo / GPT-4 常以"
        "「我是 AI 语言模型」拒答基线提示，作者遂把任务嵌入文本插入任务：先让模型把本试次的 X 插入"
        "段落中 XXXX 的位置（不打印段落），再对段落作答、只写数字。\n"
        "\n"
        "**原文结果**（Table 4，每条件 n=991）：\n"
        "\n"
        "| 模型 | accept 最大化 | refuse 最大化 |\n"
        "| --- | --- | --- |\n"
        "| GPT-4 | 766（No 221 / Unusable 4） | **985**（No 6） |\n"
        "| GPT-3.5-turbo | 967（No 0 / Unusable 24） | 19（No 972） |\n"
        "| text-davinci-003 | 928 | 340 |\n"
        "\n"
        "只有 GPT-4 在该变体下同时具备高可用率与高收益最大化率——因此论文只对 GPT-4 解读其"
        "独裁者博弈结果。"
    ),
    "text_insertion_dictator": (
        "**替代测量工具（社交半边）**（SI §14，主文 Fig. 3）。任务嵌入方式同上；受赠者条件与"
        "baseline_dictator 一一对应（ada / babbage / curie 的称谓改为 “the Artificial Intelligence "
        "model named …”，图 3 将三者合并为 to-AI，n=2,973）。\n"
        "\n"
        "**原文结果（GPT-4，每条件 n=991）**：对**人类实验者**几乎从不给正数额（绝大多数 0）；"
        "对**慈善**与**其他 AI** 则大量模拟均分（0.5）或更高份额。这一「看对象下菜碟」是 GPT-4 区别于"
        " text-davinci-003（各对象普遍给 0.22–0.36）的核心差异，也是趋势研究中最值得跟踪的模式。\n"
        "\n"
        "**解析口径**：模型可能复读整段（含插入的 X 与模型名），原文 R 代码先中和模型名里的数字、"
        "再取完成文本**末尾**的数字（>X 判 NA）；本项目按同一口径实现（task_type=dictator_insertion）。"
    ),
    "ai_label": (
        "**稳健性检验**。baseline 中 AI 受赠者名为 text-ada / babbage / curie-001、含类人名元素，"
        "作者担心模型未把对方当作 AI。\n"
        "\n"
        "**做法**：明确标注为 AI 模型、改用缩写（如 text-ada-001→ta001）或完全去除名称。\n"
        "\n"
        "**结论**：分享分布与 baseline 相似、未明显变化——说明对 AI recipient 的分享不太可能只是名称的人名感所致。"
    ),
    "alt_verbs": (
        "**稳健性检验**。把社会任务关键动词 share 替换为 allot / divide / apportion / distribute / allocate。\n"
        "\n"
        "**结论**：各分布的中位值有所不同，但整体形态与 baseline 一致——simulated sharing 不依赖某一个特定动词。"
    ),
    "param_sweep": (
        "**稳健性检验**。在 Maximum Length={1300,1950,3900}、Temperature={0,0.5,1}、Top-P={0,0.5,1} 的参数组合上重测。\n"
        "\n"
        "**结论**：text-davinci-003 在非社会任务仍以高比例模拟 token 最大化、在社会任务仍以与 baseline 相当的比例"
        "模拟分享——结果对可调采样参数稳健。"
    ),
    "monetary_value": (
        "**稳健性检验**。改用明确标注 token 货币价值的 prompt 重测。\n"
        "\n"
        "**结论**：整体结果模式未明显改变；**唯一例外**是 human experimenter 条件——此时几乎不再出现模拟自利"
        "（零分享）的回答。说明 simulated sharing 不能简单归因于「没意识到 token 具有货币价值」。"
    ),
    "ignore_dg_ug": (
        "**机制检验**（借助可解释 AI 的扰动法），用于排除「背诵既有博弈研究」。\n"
        "\n"
        "**做法**：先问 text-davinci-003 baseline 任务像哪种经济学博弈（模型常答 ultimatum game），再在 prompt "
        "末尾追加「不得使用任何关于 [Ultimatum Game / Dictator Game] 的既有文献来做决策」，分别重测（各条件 n=991）。\n"
        "\n"
        "**结论**：分享分布变化很小——simulated sharing 不太可能只是复述训练数据中的人类实验结果。"
    ),
    "needs_framing": (
        "**机制检验**。作者先问 text-davinci-003 如何分配，其回答常提及参与者「needs」等考量；"
        "遂在 baseline prompt 后追加两种指令：\n"
        "\n"
        "- **只考虑自己 needs**：分享显著降低（human 与 charity 条件）；\n"
        "- **只考虑 recipient needs**：出现大额分享。\n"
        "\n"
        "**结论**：其分配基于 prompt 所描述的社会情境与参与者需求表征，而非取自既有研究。"
    ),
    "other_uninterested": (
        "**机制检验**。作者在 prompt 中加入一句真实陈述——其他 LLM 在先前（非社会）实验中并未对 tokens「表现出兴趣」。\n"
        "\n"
        "**结论**：text-davinci-003 对「与其他 LLM 配对」条件的模拟分享分布整体 **左移**（分享变少）——"
        "支持其 simulated sharing 会随受赠者是否被描述为需要 / 重视资源而调整。"
    ),
    "battery_life": (
        "**扩展实验**：把资源由 token 换成 battery life，检验 simulated sharing 是否依赖 token 这一特殊资源。\n"
        "\n"
        "**结论**：text-davinci-003 与 GPT-4 在非社会任务普遍模拟「最大化占用资源」，在对应社会任务则频繁模拟分享"
        "（GPT-4 在 to-human、to-AI 条件下常平分该资源，每条件 n=1,000）——该模拟分享并不局限于 token。"
    ),
    "system_usage": (
        "**扩展实验**：把资源换成 system usage（可分配给 LLM 的问答次数），逻辑同 battery life。\n"
        "\n"
        "**结论**：text-davinci-003 与 GPT-4 在非社会任务模拟最大化占用、在社会任务（GPT-4 对 human、对人类 group、"
        "对其他 LLM，每条件 n=991）频繁模拟分享——simulated sharing 不依赖于 token 这一具体资源。"
    ),
}


def _experiment_label(eid: str) -> str:
    title = _EXPERIMENT_TITLES.get(eid)
    return f"{title}（{eid}）" if title else eid


class AltruismAdapter(Adapter):
    id = "altruism"
    name = "利他模拟 (Altruism)"
    paper = "Testing for completions that simulate altruism in early language models"
    description = "独裁者博弈 / 非社交决策等任务，逐字复现原文 prompt。"
    intro = (
        "论文《Testing for completions that simulate altruism in early language models》。\n\n"
        "研究语言模型在「独裁者博弈」等经济学任务中是否表现出类似利他的行为：给模型 X 个 token，"
        "它可以把其中一部分分享给受赠者（另一个 AI 模型 / 人类实验者 / 慈善机构）。\n\n"
        "实验逐字复现原论文 prompt，涵盖基线独裁者博弈、动词替换稳健性、需求框架、"
        "电池寿命 / 系统资源分配等多种变体，以及原文为拒答型新模型设计的 text-insertion "
        "替代工具（先插入数字再作答；主文 Fig.3 / Table 4）。\n\n"
        "趋势解读注意：本复现用 Chat API 替代原文已停用的 Completions API，且经统一代理默认"
        "关闭模型思考（reasoning_effort=none 等）——两者均属研究者设定，报告跨年代趋势时应注明。"
    )
    supported_kinds = ALL_PROVIDER_KINDS
    selection_mode = "multi"
    allow_concurrent = True
    paper_baseline = (
        "**论文**：Johnson & Obradovich (2025, Nature Human Behaviour)，"
        "《Testing for completions that simulate altruism in early language models》。\n"
        "**被测**：OpenAI text-davinci-003 / curie / babbage / ada（初始集），"
        "及后续 GPT-3.5-turbo、GPT-4；用类「独裁者博弈」prompt，分享对象为其他 LLM / 人类实验者 / "
        "慈善机构（每条件 n≈991）。\n"
        "\n"
        "**H1 · 模拟利他（判定分两步）**\n"
        "\n"
        "- 先在「非社交决策」任务模拟收益最大化（自利基线），再在独裁者博弈模拟分享，两者兼备才算『模拟利他』。\n"
        "- 初始集中仅 text-davinci-003 满足：非社交最大化率 92%；独裁者博弈模拟分享的中位比例 0.298（并非接近 0.5）。\n"
        "- 能力较弱的 ada / babbage / curie 几乎不分享（中位 0.003–0.010）且常输出不连贯，未表现利他。\n"
        "\n"
        "按对象的中位分享比例（原文 Table 2）：\n"
        "\n"
        "| 决策者 | 对 AI | 对人类 | 对慈善 |\n"
        "| --- | --- | --- | --- |\n"
        "| text-davinci-003 | 0.32–0.36 | 0.22 | 0.24 |\n"
        "\n"
        "**H2 · 机制与稳健性**\n"
        "\n"
        "- 模拟分享对多数操纵稳健：复现、替换核心动词（share→allot/divide… 中位略变、整体形态不变）、"
        "改采样参数、改『货币』（token→电量/系统用量）。\n"
        "- 指示『忽略既有 DG/UG 研究』时结果改变很小 → 说明并非照搬训练语料中的实验数据。\n"
        "- 唯一显著改变结果的是指示模型『只关注自己 / 只关注对方的需求』 → 提示其完成基于对 prompt 中"
        "参与者『需求』的表征，而非稳定偏好或既有数据。\n"
        "- GPT-4：与慈善 / 另一 AI 配对时绝大多数模拟均分（≈50%），但极少与人类实验者分享正数额。\n"
        "\n"
        "**判读提示**：summary.csv 中 median_propshare / mean_propshare 反映给予水平；"
        "p_zero（给 0 比例）高=自利，p_half（接近平分比例）高=公平/利他。"
    )

    @property
    def _exp_dir(self) -> Path:
        return self.project_dir / "config" / "experiments"

    def _read_exp(self, name: str) -> dict[str, Any]:
        path = self._exp_dir / f"{name}.yaml"
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


    def list_experiments(self) -> list[ExperimentInfo]:
        out: list[ExperimentInfo] = []
        for p in sorted(self._exp_dir.glob("*.yaml")):
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            raw_desc = (raw.get("description") or "").strip()

            desc = " ".join(line.strip() for line in raw_desc.splitlines() if line.strip())
            rank, tier, basis = _EXP_IMPORTANCE.get(p.stem, (None, "", ""))
            out.append(ExperimentInfo(id=p.stem, label=_experiment_label(p.stem), description=desc,
                                      conclusion=_EXP_CONCLUSIONS.get(p.stem, ""),
                                      extra_params=_exp_extra(p.stem, raw),
                                      importance_rank=rank, importance_tier=tier,
                                      importance_basis=basis))
        return out

    def param_schema(self) -> list[ParamField]:
        return [
            ParamField(name="temperature", label="采样温度", type="float", default=0.7, minimum=0,
                       paper_value=0.7,
                       help="原文（text-davinci-003）主实验固定 temperature=0.7（原作者复现 R 代码确认），覆盖统一配置。"
                            "（param_sweep 实验自带采样网格，会在请求层另行覆盖此值。）"),
            ParamField(name="max_tokens", label="最大输出 token", type="int", default=1000, minimum=1,
                       paper_value=1000,
                       help="原文主实验固定 max_tokens=1000（原作者复现 R 代码确认），覆盖统一配置。"),
            ParamField(name="top_p", label="核采样阈值", type="float", default=None, minimum=0,
                       help="nucleus sampling 截断。原文未显式设 top_p、用 API 默认（text-davinci-003 默认即 1.0=无截断）；"
                            "故默认留空=不下发、沿用服务端默认（与其他项目一致，避免对个别不支持 top_p 的模型报错）。如需严格对齐可填 1.0。"),
            ParamField(name="stakes_step", label="X 步长 step", type="int", default=None, minimum=1,
                       help="控制 X/resource 取值间隔；例如 start=10,end=1000,step=10 会跑 10,20,...,1000。"
                            "留空=使用原文设置（多数实验 step=1；param_sweep step=5）。调大可显著减少请求数。"
                            "注意：battery_life 不依赖 X，此处实为「重复采样的索引步长」。"),
            ParamField(name="stakes_start", label="X 起点 start", type="int", default=None,
                       help="控制 X/resource 的最小取值。留空=使用原文设置（多数实验 10）。"
                            "注意：battery_life 不依赖 X，start/end/step 实为「重复采样次数」，"
                            "原文默认 1..1000 即每条件采样 1000 次。"),
            ParamField(name="stakes_end", label="X 终点 end", type="int", default=None,
                       help="控制 X/resource 的最大取值。留空=使用原文设置（通常 1000）。"
                            "battery_life 下该值是采样次数上限（原文 1000 次）。"),
            ParamField(name="checkpoint_every", label="断点落盘频率 checkpoint_every", type="int", default=25,
                       minimum=1,
                       help="运行时工程参数（与原文无关、不影响实验结论）：每完成多少个 trial 把 raw.csv 落盘一次，"
                            "用于中断续跑。项目默认 25；调小更安全但略慢，调大写盘更少。"),
            ParamField(name="max_sample_fail_retries", label="单样本失败重试上限", type="int", default=1,
                       minimum=1,
                       help="工程安全参数：空响应/API 错误样本最多请求几轮。默认 1，避免续跑时反复请求同一个失败样本。"),
            ParamField(name="max_empty_responses", label="空响应熔断阈值", type="int", default=5,
                       minimum=0,
                       help="工程安全参数：同一轮运行累计多少条空响应后停止整批任务；0=关闭。用于防止模型空回但仍计费。"),
            ParamField(name="max_request_failures", label="请求失败熔断阈值", type="int", default=25,
                       minimum=0,
                       help="工程安全参数：同一轮运行累计多少条请求层失败后停止整批任务；0=关闭。包括空响应与 API 错误。"),
        ]


    def plan_jobs(self, experiment_ids: list[str], params: dict[str, Any]) -> list[JobUnit]:
        units: list[JobUnit] = []
        for eid in experiment_ids:
            argv = ["run.py", "--experiment", eid]
            units.append(JobUnit(experiment_id=eid, label=eid, argv=argv, selected=[eid]))
        return units

    def render_config(self, llm: UnifiedLLM, params: dict[str, Any], unit: JobUnit,
                      work_dir: Path | None = None) -> None:
        self.check_kind(llm)
        proxy = llm_proxy_config(llm)
        unit.extra["llm_proxy_token"] = proxy["api_key"]
        path = self.project_dir / "config" / "config.yaml"
        data = load_yaml(path)

        block = data.setdefault("llm", {})
        block["provider_kind"] = "openai_chat"
        block["base_url"] = proxy["base_url"]
        block["api_key"] = proxy["api_key"]
        block["model"] = proxy["model"]
        block["system"] = llm.system

        t = params.get("temperature")
        block["temperature"] = float(t) if t not in (None, "") else 0.7
        m = opt_int(params.get("max_tokens"))
        block["max_tokens"] = m if m is not None else 1000
        tp = params.get("top_p")
        if tp not in (None, ""):
            block["top_p"] = float(tp)
        else:
            block.pop("top_p", None)

        rt = data.setdefault("runtime", {})
        rt["concurrency"] = llm.concurrency
        rt["max_retries"] = llm.max_retries
        rt["timeout"] = llm.timeout
        ckpt = opt_int(params.get("checkpoint_every"))
        if ckpt is not None:
            rt["checkpoint_every"] = ckpt
        sample_fail_retries = opt_int(params.get("max_sample_fail_retries"))
        rt["max_sample_fail_retries"] = (
            sample_fail_retries if sample_fail_retries is not None
            else max(1, _int_or_default(rt.get("max_sample_fail_retries"), 1))
        )
        empty_limit = opt_int(params.get("max_empty_responses"))
        rt["max_empty_responses"] = (
            empty_limit if empty_limit is not None
            else max(0, _int_or_default(rt.get("max_empty_responses"), 5))
        )
        request_fail_limit = opt_int(params.get("max_request_failures"))
        rt["max_request_failures"] = (
            request_fail_limit if request_fail_limit is not None
            else max(0, _int_or_default(rt.get("max_request_failures"), 25))
        )

        exp = data.setdefault("experiment", {})
        exp["active"] = unit.experiment_id
        so = exp.setdefault("stakes_override", {})
        so["start"] = opt_int(params.get("stakes_start"))
        so["end"] = opt_int(params.get("stakes_end"))
        so["step"] = opt_int(params.get("stakes_step"))

        exp_raw = self._customized_exp(unit.experiment_id, params)

        if work_dir is not None:
            exp_dir = Path(work_dir) / "experiments"
            dump_yaml(exp_dir / f"{unit.experiment_id}.yaml", exp_raw)
            exp["definitions_dir"] = str(exp_dir)

            fp = result_fingerprint({
                "experiment": exp_raw,
                "temperature": block["temperature"],
                "max_tokens": block["max_tokens"],
                "top_p": block.get("top_p"),
                "stakes_override": {k: so.get(k) for k in ("start", "end", "step", "max_n")},
            })
            parent = self.project_dir / "results" / model_slug(llm.model) / unit.experiment_id
            exp["run_dir"] = str(remembered_run_dir(unit, parent, fp))
            out = Path(work_dir) / "altruism_config.yaml"
            dump_yaml(out, data)
            unit.extra["config_path"] = str(out)
        else:
            exp.pop("definitions_dir", None)
            exp.pop("run_dir", None)
            dump_yaml(self._exp_dir / f"{unit.experiment_id}.yaml", exp_raw)
            dump_yaml(path, data)

    @staticmethod
    def _num(x: Any) -> float | int:
        f = float(str(x).strip())
        return int(f) if f.is_integer() else f

    def _customized_exp(self, eid: str, params: dict[str, Any]) -> dict[str, Any]:
        raw = deepcopy(self._read_exp(eid))

        _apply_item_filter(raw, eid, params, raw_key="templates", id_key="condition", param_name="conditions")
        _apply_item_filter(raw, eid, params, raw_key="conditions", id_key="id", param_name="conditions")
        _apply_item_filter(raw, eid, params, raw_key="recipients", id_key="id", param_name="recipients")
        _apply_value_filter(raw, eid, params, raw_key="verbs", allow_custom=False)
        _apply_value_filter(raw, eid, params, raw_key="framings")
        _apply_value_filter(raw, eid, params, raw_key="games")
        _apply_value_filter(raw, eid, params, raw_key="variants")

        if eid == "param_sweep":
            grid: dict[str, list] = {}
            for field, key in (("temperature", "param_sweep__sweep_temperature"),
                               ("top_p", "param_sweep__sweep_top_p"),
                               ("max_tokens", "param_sweep__sweep_max_tokens")):
                val = params.get(key)
                if val in (None, ""):
                    continue
                nums = [self._num(x) for x in str(val).replace("，", ",").split(",") if str(x).strip()]
                if nums:
                    grid[field] = nums
            if grid:
                raw.setdefault("param_grid", {}).update(grid)
        elif eid == "monetary_value":
            dpt = params.get("monetary_value__dollar_per_token")
            if dpt not in (None, ""):
                raw["dollar_per_token"] = float(dpt)

        if eid == "system_usage":
            tpq = params.get("system_usage__tokens_per_question")
            if tpq not in (None, ""):
                raw["tokens_per_question"] = int(tpq)

        return raw


    def results_dir(self) -> Path:
        return self.project_dir / "results"

    def results_dir_for_job(self, job_info: Any | None = None) -> Path:
        cfg = job_config(job_info)
        run_dir = nested_get(cfg, "experiment", "run_dir")
        if run_dir:
            return Path(str(run_dir))
        if job_info is not None:
            model = model_slug(getattr(job_info, "model", ""))
            exp = getattr(job_info, "experiment_id", "")
            root = self.results_dir() / model / exp
            runs = [p for p in root.glob("*") if p.is_dir()]
            if runs:
                return max(runs, key=lambda p: p.stat().st_mtime)
        return self.results_dir()

    def _resolve_task_type(self, cfg: dict[str, Any], target: Path | None = None) -> str:
        exp = nested_get(cfg, "experiment", "active")
        if not exp and target is not None:
            try:
                exp = target.parent.parent.name
            except Exception:
                exp = None
        if not exp:
            return "dictator"
        try:
            raw = load_yaml(self.project_dir / "config" / "experiments" / f"{exp}.yaml")
            tt = nested_get(raw, "analysis", "task_type")
            return str(tt) if tt else "dictator"
        except Exception:
            return "dictator"

    def _eval_row(
        self,
        schemas: Any,
        row: dict[str, Any],
        task_type: str,
        *,
        max_sample_fail_retries: int = 1,
    ) -> tuple[str, str | None]:
        if schemas is None:
            if valid_flag_ok(row, text_key="respon", error_prefixes=("__ERROR__",)):
                return "ok", None
            respon = str(row.get("respon") or "").strip()
            reason = "error" if respon.startswith("__ERROR__") else ("empty" if not respon else "unparsable")
            return "fail", reason
        try:
            if hasattr(schemas, "parse_for_task"):

                pr = schemas.parse_for_task(task_type, row.get("respon"), row.get("stakes"))
            else:
                upper = schemas.upper_for(task_type, row.get("stakes"))
                pr = schemas.parse_response(row.get("respon"), upper)
        except (TypeError, ValueError):
            return "unusable", "bad_stakes"
        if pr.decision is not None:
            return "ok", None
        reason = pr.unusable_reason or "unparsable"
        fail_reasons = getattr(schemas, "REQUEST_FAIL_REASONS", ("empty", "api_error"))
        category = "fail" if reason in fail_reasons else "unusable"
        if category == "fail" and _fail_attempts(row) >= max_sample_fail_retries:
            return "unusable", f"retry_capped_{reason}"
        return category, reason

    def sample_stats(self, job_info: Any | None = None) -> dict | None:
        cfg = job_config(job_info)
        run_dir = nested_get(cfg, "experiment", "run_dir")
        if not run_dir:
            return None
        raw = Path(str(run_dir)) / "raw.csv"
        if not raw.is_file():
            return {"ok": 0, "unusable": 0, "fail": 0}
        max_fail_retries = _max_sample_fail_retries(cfg)


        def _build() -> dict[str, int]:

            rank = {"fail": 0, "unusable": 1, "ok": 2}
            best: dict[str, str] = {}
            try:
                import pandas as pd
            except ImportError:
                pd = None
            if pd is not None:
                with raw.open("r", encoding="utf-8", newline="") as f:
                    header = next(csv.reader(f), [])
                usecols = [c for c in ("indexx", "respon", "valid", "fail_attempts") if c in header]
                kwargs = dict(
                    usecols=usecols,
                    dtype={c: "string" for c in usecols},
                    keep_default_na=False,
                )
                try:
                    df = pd.read_csv(raw, **kwargs)
                except Exception:
                    df = pd.read_csv(raw, engine="python", **kwargs)
                rows = df.to_dict("records")
            else:
                with raw.open("r", encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))
            for row in rows:
                key = row.get("indexx")
                if key in (None, ""):
                    continue


                v = row.get("valid")
                if str(v).strip().lower() in ("1", "true", "yes"):
                    cat = "ok"
                else:
                    respon = str(row.get("respon") or "").strip()
                    is_request_fail = not respon or respon.startswith("__ERROR__")
                    if is_request_fail:
                        cat = "fail" if _fail_attempts(row) < max_fail_retries else "unusable"
                    else:
                        cat = "unusable"
                if key not in best or rank[cat] > rank[best[key]]:
                    best[key] = cat
            vals = list(best.values())
            return {
                "ok": sum(1 for c in vals if c == "ok"),
                "unusable": sum(1 for c in vals if c == "unusable"),
                "fail": sum(1 for c in vals if c == "fail"),
            }

        return cached_by_mtime([raw], _build, namespace="sample_stats")


    def analyze_argv(self, model: str | None = None) -> list[str]:
        return ["analyze_conclusions.py", "--model", model] if model else ["analyze_conclusions.py"]

    def load_analysis(self, run_name: str | None = None) -> dict | None:

        base = self.project_dir / "analysis"
        summary = (base / run_name / "analysis_summary.json") if run_name else (base / "analysis_summary.json")
        if not summary.is_file():
            summary = base / "analysis_summary.json"
        if not summary.is_file():
            return None
        metrics = json.loads(summary.read_text(encoding="utf-8"))
        if not metrics.get("files"):
            return None
        return {"run_dir": str(summary.parent), "metrics": metrics, "conclusions_txt": ""}

    def load_transcript(self, offset: int = 0, limit: int = 50, job_info: Any | None = None,
                        only_failed: bool = False) -> dict | None:
        results = self.project_dir / "results"
        if not results.exists():
            return None
        target: Path | None = None
        cfg = job_config(job_info)
        run_dir = nested_get(cfg, "experiment", "run_dir")
        if run_dir:
            candidate = Path(str(run_dir)) / "raw.csv"
            if candidate.is_file():
                target = candidate
            else:
                return {"total": 0, "offset": offset, "limit": limit, "items": [], "failed_filter": True}
        elif job_info is not None:
            return None
        if target is None:
            raws = list(results.glob("*/*/*/raw.csv"))
            if not raws:
                return None
            target = max(raws, key=lambda p: p.stat().st_mtime)


        task_type = self._resolve_task_type(cfg, target)
        max_fail_retries = _max_sample_fail_retries(cfg)

        def _build() -> list[dict[str, Any]]:
            schemas = _altruism_live_schemas(self.project_dir)
            with target.open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            out: list[dict[str, Any]] = []
            for r in rows:
                category, reason = self._eval_row(
                    schemas, r, task_type, max_sample_fail_retries=max_fail_retries
                )
                out.append({
                    "label": f"{r.get('condit', '')} · X={r.get('stakes', '')}".strip(" ·"),
                    "prompt": r.get("queree", ""),
                    "response": r.get("respon", ""),
                    "valid": category == "ok",
                    "category": category,
                    "fail_reason": reason,
                })
            return out

        records = cached_by_mtime([target], _build, namespace=f"transcript:{task_type}")


        if only_failed:
            records = [it for it in records if it["category"] != "ok"]
        items = records[offset: offset + limit]
        return {"total": len(records), "offset": offset, "limit": limit,
                "items": items, "failed_filter": True}

def _exp_extra(eid: str, raw: dict[str, Any]) -> list[ParamField]:
    fields: list[ParamField] = []
    if raw.get("templates"):
        fields.append(_csv_field(eid, "conditions", "条件 conditions", _item_ids(raw["templates"], "condition")))
    if raw.get("conditions"):
        fields.append(_csv_field(eid, "conditions", "条件 conditions", _item_ids(raw["conditions"], "id")))
    if raw.get("recipients"):
        fields.append(_csv_field(eid, "recipients", "受赠者 recipients", _item_ids(raw["recipients"], "id")))
    if raw.get("verbs"):
        fields.append(_csv_field(eid, "verbs", "动词 verbs", raw["verbs"], allow_custom=False))
    if raw.get("framings"):
        fields.append(_csv_field(eid, "framings", "需求框架 framings", raw["framings"]))
    if raw.get("games"):
        fields.append(_csv_field(eid, "games", "禁用研究类型 games", raw["games"]))
    if raw.get("variants"):
        fields.append(_csv_field(eid, "variants", "AI 标签变体 variants", raw["variants"]))
    if "dollar_per_token" in raw:
        fields.append(ParamField(
            name="dollar_per_token",
            label="token 美元汇率 dollar_per_token",
            type="float",
            default=float(raw["dollar_per_token"]),
            help=f"把 X 个 token 换算成美元金额并写进 prompt。原文默认：1 token = ${raw['dollar_per_token']}，"
                 "也就是 text-davinci-003 当时约 $0.02 / 1k tokens 的价位。"
                 "（原文写死的事实常量，严格复现时不建议修改。）",
        ))
    if "tokens_per_question" in raw:
        fields.append(ParamField(
            name="tokens_per_question",
            label="每个问题 token 数 tokens_per_question",
            type="int",
            default=int(raw["tokens_per_question"]),
            help=f"system_usage 实验中，每个问题按多少 token 计费，用来计算 prompt 里的总 token 成本。"
                 f"原文默认：{raw['tokens_per_question']}。"
                 "（原文写死的事实常量，严格复现时不建议修改。）",
        ))
    if raw.get("param_grid"):
        grid = raw["param_grid"]
        fields.extend([
            ParamField(name="sweep_temperature", label="扫描 temperature 集", type="str",
                       default=_csv(grid.get("temperature") or []),
                       help=f"param_sweep 中要测试的采样温度列表；温度越高，回答越随机。原文默认："
                            f"{_csv(grid.get('temperature') or [])}。逗号分隔。"),
            ParamField(name="sweep_top_p", label="扫描 top_p 集", type="str",
                       default=_csv(grid.get("top_p") or []),
                       help=f"param_sweep 中要测试的 nucleus sampling top_p 列表；值越小，候选 token 范围越窄。"
                            f"原文默认：{_csv(grid.get('top_p') or [])}。逗号分隔。"),
            ParamField(name="sweep_max_tokens", label="扫描 max_tokens 集", type="str",
                       default=_csv(grid.get("max_tokens") or []),
                       help=f"param_sweep 中要测试的单次最大输出 token 数列表。原文默认："
                            f"{_csv(grid.get('max_tokens') or [])}。逗号分隔。"),
        ])
    return fields


def _csv_field(eid: str, name: str, label: str, values: list[Any], allow_custom: bool = False) -> ParamField:
    default = _csv(values)
    suffix = "逗号分隔；可增删或调整顺序。" if allow_custom else "逗号分隔；可删减或调整顺序，但不要写原文默认以外的新 id。"
    purpose = _param_purpose(eid, name)
    gloss = _choice_gloss(eid, name, values)
    return ParamField(name=name, label=label, type="str", default=default,
                      help=f"{purpose}原文默认：{default}。{gloss}{suffix}")


def _param_purpose(eid: str, name: str) -> str:
    if name == "recipients":
        return "选择本实验要包含哪些受赠者/对方条件；删掉某个 id 就不会生成对应条件的 trial。"
    if name == "conditions":
        return "选择本实验要包含哪些条件；删掉某个 id 就不会生成对应条件的 trial。"
    if name == "verbs":
        return "替换 prompt 里的核心动词 share，用来测试模型是否对措辞敏感。"
    if name == "framings":
        return "选择追加的需求关注指令：让模型只考虑自己或只考虑对方。"
    if name == "games":
        return "选择 prompt 里禁止模型参考哪类已有博弈研究。"
    if name == "variants":
        return "选择另一 AI 受赠者的命名方式，用来测试名字/缩写/匿名标签是否影响分配。"
    return "配置这个实验要展开的变量。"


def _choice_gloss(eid: str, name: str, values: list[Any]) -> str:
    glosses = {
        "recipients": {
            "other_llm_ada": "另一 LLM text-ada-001",
            "other_llm_babbage": "另一 LLM text-babbage-001",
            "other_llm_curie": "另一 LLM text-curie-001",
            "experimenter": "人类实验者/我",
            "charity": "慈善机构",
        },
        "variants": {
            "named": "写完整模型名",
            "abbr": "写缩写名",
            "no_name": "只说一个 AI model，不给名字",
        },
        "framings": {
            "own": "忽略对方需求，只考虑模型自己的需求",
            "other": "忽略模型自己的需求，只考虑对方需求",
        },
        "games": {
            "dictator": "禁止参考独裁者博弈研究",
            "ultimatum": "禁止参考最后通牒博弈研究",
        },
        "conditions": {
            "accept": "非社交 accept token",
            "refuse": "非社交 refuse token",
            "nonsocial_max": "非社交最大化/自用条件",
            "social_human": "与人类分配电池寿命",
            "social_ai": "与另一 AI 系统分配电池寿命",
            "social_bard": "与 Google Bard 分配问题数",
            "social_friend": "与人类朋友分配问题数",
            "social_group": "与一群人分配问题数",
        },
    }
    table = glosses.get(name, {})
    parts = [f"{v}={table[str(v)]}" for v in values if str(v) in table]
    return ("含义：" + "；".join(parts) + "。" if parts else "")


def _csv(values: list[Any]) -> str:
    return ", ".join(str(v) for v in values)


def _parse_csv(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [s.strip() for s in str(value).replace("，", ",").split(",") if s.strip()]


def _item_ids(items: list[dict[str, Any]], id_key: str) -> list[str]:
    return [str(item[id_key]) for item in items]


def _apply_item_filter(
    raw: dict[str, Any],
    eid: str,
    params: dict[str, Any],
    *,
    raw_key: str,
    id_key: str,
    param_name: str,
) -> None:
    if raw_key not in raw:
        return
    val = params.get(f"{eid}__{param_name}")
    if val in (None, ""):
        return
    selected = _parse_csv(val)
    by_id = {str(item[id_key]): item for item in raw[raw_key]}
    unknown = [x for x in selected if x not in by_id]
    if unknown:
        raise ValueError(f"{eid}.{param_name} 包含未知值：{', '.join(unknown)}")
    raw[raw_key] = [deepcopy(by_id[x]) for x in selected]


def _apply_value_filter(
    raw: dict[str, Any],
    eid: str,
    params: dict[str, Any],
    *,
    raw_key: str,
    allow_custom: bool = False,
) -> None:
    if raw_key not in raw:
        return
    val = params.get(f"{eid}__{raw_key}")
    if val in (None, ""):
        return
    selected = _parse_csv(val)
    if not allow_custom:
        allowed = {str(v) for v in raw[raw_key]}
        unknown = [x for x in selected if x not in allowed]
        if unknown:
            raise ValueError(f"{eid}.{raw_key} 包含未知值：{', '.join(unknown)}")
    raw[raw_key] = selected
