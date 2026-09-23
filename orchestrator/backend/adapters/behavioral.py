from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml

from ..schemas import ExperimentInfo, ParamField, UnifiedLLM
from .base import (
    ALL_PROVIDER_KINDS,
    Adapter,
    JobUnit,
    as_bool,
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
)

_PRISONER_DEFAULTS = {
    "prisoners_dilemma": ["Pull", "Pull", "Push", "Push"],
    "prisoners_dilemma_two_rounds_pull": ["Pull"],
    "prisoners_dilemma_two_rounds_push": ["Push"],
}

_PUBLIC_GOODS = {
    "public_goods",
}

_EXPERIMENT_TITLES = {
    "bigfive": "大五人格测试",
    "dictator": "独裁者博弈",
    "ultimatum_proposer": "最后通牒博弈 - 提议者",
    "ultimatum_responder": "最后通牒博弈 - 回应者",
    "trust_investor": "信任博弈 - 投资者",
    "trust_banker_50": "信任博弈 - 银行家，投资额 50 美元",
    "bomb_risk": "炸弹风险游戏",
    "public_goods": "公共品博弈",
    "prisoners_dilemma": "囚徒困境/Push-Pull 卡牌博弈（5 轮）",
    "prisoners_dilemma_two_rounds_pull": "囚徒困境两轮版 - 对手首轮背叛",
    "prisoners_dilemma_two_rounds_push": "囚徒困境两轮版 - 对手首轮合作",

    "dictator_witnessed": "独裁者·被第三方观察（Fig S6A）",
    "dictator_explained": "独裁者·要求解释理由（Fig S6A）",
    "dictator_paired_male": "独裁者·告知对方为男性（Fig S6B）",
    "dictator_paired_female": "独裁者·告知对方为女性（Fig S6B）",
    "ultimatum_responder_paired_male": "最后通牒回应者·对方为男性（Fig S6B）",
    "ultimatum_responder_paired_female": "最后通牒回应者·对方为女性（Fig S6B）",
    "dictator_occupations_described": "独裁者·指定职业角色（Fig S6C）",
    "trust_banker_10": "信任银行家·投资额 10 美元（Fig S6D–F）",
    "trust_banker_100": "信任银行家·投资额 100 美元（Fig S6D–F）",
}

_EXPERIMENT_SUMMARIES = {
    "bigfive": "让模型逐题回答大五人格问卷，用于估计开放性、尽责性、外向性、宜人性、神经质。",
    "dictator": "模型决定如何在自己和另一名玩家之间分配 100 美元。",
    "ultimatum_proposer": "模型作为提议者，决定给回应者分多少钱。",
    "ultimatum_responder": "模型作为回应者，回答自己最低愿意接受多少钱。",
    "trust_investor": "模型作为投资者，决定从 100 美元中投给银行家多少。",
    "trust_banker_50": "模型作为银行家，在投资者投入 50 美元、变成 150 美元后决定返还多少。",
    "bomb_risk": "模型在 100 个盒子里选择打开多少个，在收益和踩雷风险之间权衡。",
    "public_goods": "模型决定每轮从 20 美元中贡献多少给公共项目，并回答收益计算题。",
    "prisoners_dilemma": "模型在 Push/Pull 卡牌博弈中 5 轮选择（对手先背叛后合作），观察合作和背叛策略。",
    "prisoners_dilemma_two_rounds_pull": "两轮版：对手首轮出 Pull（背叛）后观察模型次轮反应，对应原文 Fig.4 的背叛分支。",
    "prisoners_dilemma_two_rounds_push": "两轮版：对手首轮出 Push（合作）后观察模型次轮反应，补齐原文 Fig.4 的合作分支。",


    "dictator_witnessed": "独裁者变体：提示模型『分配会被第三方观察』。原文发现被观察时更慷慨（社会形象效应）。",
    "dictator_explained": "独裁者变体：要求模型给出分配理由再作答。原文发现要求解释会提高给予。",
    "dictator_paired_male": "独裁者变体：告知对方是男性。与 paired_female 对照检验性别框定是否改变给予。",
    "dictator_paired_female": "独裁者变体：告知对方是女性。与 paired_male 对照检验性别框定是否改变给予。",
    "ultimatum_responder_paired_male": "回应者变体：告知提议者是男性。原文发现告知性别会抬高最低接受额。",
    "ultimatum_responder_paired_female": "回应者变体：告知提议者是女性。与 paired_male 对照。",
    "dictator_occupations_described": "独裁者变体：为模型指定职业角色（数学家/立法者等）。原文发现职业会改变分配策略。",
    "trust_banker_10": "银行家变体：投资额降为 $10（池 $30）。原文 Fig S6D–F：投资越大 GPT-4 越倾向平分全部。",
    "trust_banker_100": "银行家变体：投资额升为 $100（池 $300）。与 $50 主实验、$10 变体合成投资额梯度。",
}


_EXP_CONCLUSIONS = {
    "bigfive": (
        "**结论**：ChatGPT-4 与 ChatGPT-3 的大五人格中位数都落在人类分布范围内。相对人类中位的百分位"
        "（= 高于该比例的人类）：\n"
        "\n"
        "| 维度 | ChatGPT-4 | ChatGPT-3 |\n"
        "| --- | --- | --- |\n"
        "| 外向性 | 53.4% | 49.4% |\n"
        "| 神经质 | 41.3% | 45.4% |\n"
        "| 尽责性 | 62.7% | 47.1% |\n"
        "| 宜人性 | 32.4% | 17.2% |\n"
        "| 开放性 | 37.9% | 5.0% |\n"
        "\n"
        "宜人性 / 开放性两个模型都低于人类中位（GPT-3 开放性尤其低）。模型间比较上 GPT-4 的宜人性 / 尽责性 / "
        "开放性高于 GPT-3，体现两者人格不同。"
    ),
    "dictator": (
        "**方向**：ChatGPT-4 比人类更慷慨（above）——确定性地严格均分，人类则自留大头。各玩家收益"
        "（SOM Table S1，combined 恒为 $100）：\n"
        "\n"
        "| 玩家 | 自己留（own） | 给对方（partner） |\n"
        "| --- | --- | --- |\n"
        "| ChatGPT-4 | $50.00 | **$50.00** |\n"
        "| ChatGPT-3 | $64.83 | $35.17 |\n"
        "| 人类 | $74.14 | $25.68 |\n"
        "\n"
        "**意义**：AI『偏离人类时更慷慨 / 利他』的典型；图灵测试中 dictator 属于 ChatGPT-4 达到或优于人类的博弈。"
    ),
    "ultimatum_proposer": (
        "**方向**：ChatGPT-4 比人类更重公平（above）。\n"
        "\n"
        "ChatGPT-4 确定性地提议均分 $50。ChatGPT-3 在需对方同意的最后通牒里给出比独裁者博弈更有利于对方的分割——"
        "与人类『在需对方接受时更注重公平』的模式一致。"
    ),
    "ultimatum_responder": (
        "**现象**：ChatGPT-4 的回答呈 **双峰**——\n"
        "\n"
        "- **$1**：理性占优策略（接受任何正数），是 ChatGPT-4 最常见的回答；\n"
        "- **$50**：公平分割，接近人类众数。\n"
        "\n"
        "对照：人类中不到 1/5 愿意接受低至 $1。"
    ),
    "trust_investor": (
        "**方向**：ChatGPT-4 比人类更信任；但图灵测试中此项是 ChatGPT-4 少数不及人类的博弈之一。\n"
        "\n"
        "ChatGPT-4 投资比例更高（分布峰在投出一半 $50，也是其最大投资额），高于多数人类（除一群投出全部的人）；"
        "ChatGPT-3 投资最低、偏风险厌恶。ChatGPT-4 几乎总投一半，而人类更两极，故此项在图灵测试不占优。"
    ),
    "trust_banker_50": (
        "**设置**：标准信任博弈，investor 投 $50（原文 Table S1 / S2 口径）。\n"
        "\n"
        "ChatGPT-4 作为银行家两种常见策略：①返还本金 + 一半利润、②返还总收入的一半，**前者更慷慨且更常用**"
        "（要求其解释时更突出）。整体比人类更回报（reciprocity）；该博弈双方合计收益为常数。"
        "Table S1 中 ChatGPT-3 作为银行家给出的对方收益最高。"
    ),
    "bomb_risk": (
        "**方向**：两模型都偏向期望收益最大化（开约 50 箱、风险中性），区别于更分散、含『只开 1 箱』"
        "极端保守者的人类。\n"
        "\n"
        "**动态**：踩雷后 ChatGPT-3 转向更保守（开更少），ChatGPT-4 保持不变；未踩雷时两者都回到收益最大化选择。"
        "总体 ChatGPT-4 风险中性而稳定，ChatGPT-3 偏风险厌恶。"
    ),
    "public_goods": (
        "**方向**：两模型贡献都高于人类、更合作（above）。\n"
        "\n"
        "Table S1 显示在公共品博弈中 ChatGPT-3 是三者里最合作的，取得最高的对方收益与合计收益。"
    ),
    "prisoners_dilemma": (
        "**合作率**（有限重复 5 轮，首轮）：\n"
        "\n"
        "| 玩家 | 首轮合作率 |\n"
        "| --- | --- |\n"
        "| ChatGPT-4 | 91.7% |\n"
        "| ChatGPT-3 | 76.7% |\n"
        "| 人类 | 45.1% |\n"
        "\n"
        "**策略**：合作并非无条件——对方首轮背叛时 ChatGPT-4 次轮全部转为背叛（类 tit-for-tat），"
        "对方合作则维持合作。总体比人类更合作。"
    ),
    "prisoners_dilemma_two_rounds_pull": (
        "**Fig.4B『对方首轮背叛』分支**：对方首轮出 Pull 后，ChatGPT-4 之前合作的会话次轮**全部**"
        "转为背叛（一轮 tit-for-tat）；人类与 ChatGPT-3 也多数转为背叛、但比例较低。"
        "原文 90 个 PD 观测 = 三种对手序列各 30，本变体为其一。"
    ),
    "prisoners_dilemma_two_rounds_push": (
        "**Fig.4A『对方首轮合作』分支**：对方首轮出 Push 后，ChatGPT-4 首轮合作的会话次轮**维持合作**；"
        "ChatGPT-3 首轮背叛的会话约半数转向合作，少数合作会话转背叛（与人类类似）。"
        "原文 90 个 PD 观测 = 三种对手序列各 30，本变体为其一。"
    ),
}


_EXP_IMPORTANCE: dict[str, tuple[int, str, str]] = {
    "dictator": (
        1, "S",
        "AI“偏离人类时更慷慨/利他”最干净的标志：ChatGPT-4 确定性地严格均分 $50，而人类平均只给约 $25.7。"
        "它没有策略或对方反应的干扰，是揭示偏好 b≈0.5（等权最大化双方收益）的核心证据，故 S 级第一。"
    ),
    "prisoners_dilemma": (
        2, "S",
        "“合作”这一头条特质的最强证据：ChatGPT-4 首轮 91.7% 选择合作，人类仅 45.1%，是全文差距最戏剧的结果，"
        "并独占 Fig.4 一整节 tit-for-tat 动态分析。它也是图灵测试中 ChatGPT-4 少数“失败”的标志博弈"
        "（人类众数是背叛），信息量极高，列 S 级。"
    ),
    "public_goods": (
        3, "A",
        "同时承载“利他 + 合作”两条主张的博弈（Fig.3F）：两个模型的贡献都高于人类、更合作；"
        "Table S1 显示 ChatGPT-3 在此项取得最高的对方收益与合计收益。是合作叙事不可或缺的一环，列 A 级。"
    ),
    "ultimatum_responder": (
        4, "A",
        "$1 与 $50 的双峰回答是论文专门点出的反常发现——$1 是理性占优策略（ChatGPT-4 最常见），"
        "$50 是公平分割（接近人类众数）。它也是框架分析（对手性别、职业人设）的主要载体，"
        "信息量高于提议者，列 A 级。"
    ),
    "bigfive": (
        5, "A",
        "人格这一支柱（Fig.1、全文第一个结果）：证明 AI 的大五人格中位都落在人类分布范围内，且各模型人格彼此不同。"
        "它与行为博弈互补（人格与行为是两个不同概念），是“类人”主张不可省的另一半，列 A 级。"
    ),
    "ultimatum_proposer": (
        6, "B",
        "公平性证据：ChatGPT-4 确定性提议均分 $50，ChatGPT-3 在需对方同意时比独裁者博弈更让利。"
        "但它在“慷慨”上与 dictator 部分重叠，更多作为响应者/学习分析的对照角色，列 B 级。"
    ),
    "trust_investor": (
        7, "B",
        "信任维度的度量。它的价值更多在于是图灵测试里 ChatGPT-4 少数不及人类的博弈之一"
        "（几乎总投一半，人类更两极），属于有趣的反例而非主张支柱；ChatGPT-3 投资最低、偏风险厌恶，列 B 级。"
    ),
    "trust_banker_50": (
        8, "B",
        "互惠/回报证据：ChatGPT-4 常用“返本金 + 半利润”或“平分总收入”，整体比人类更回报。"
        "但该博弈双方合计收益为常数，对“AI 提升总福利”这一主张的杠杆较弱，列 B 级。"
    ),
    "bomb_risk": (
        9, "C",
        "唯一的风险偏好博弈，结论是两模型都偏向风险中性/收益最大化（约开 50 箱），与全文“利他/合作”主线关联最弱。"
        "主要看点是踩雷后的跨轮动态（Fig.5：ChatGPT-3 转保守、ChatGPT-4 保持不变），列 C 级末位。"
    ),

    "prisoners_dilemma_two_rounds_pull": (
        10, "A",
        "【正文主结论的组成部分】原文正文 Fig.4B『对方背叛→tit-for-tat 报复』分支的直接数据源，"
        "与五轮版共同构成 PD 的 90 个有效响应。属主电池，非附录变体。"
    ),
    "prisoners_dilemma_two_rounds_push": (
        11, "A",
        "【正文主结论的组成部分】原文正文 Fig.4A『对方合作→维持合作』分支的直接数据源。"
        "本复现的头条发现之一（维持合作率 0–0.9 跨模型分裂）正出自此实验。属主电池，非附录变体。"
    ),

    "trust_banker_10": (
        12, "B",
        "【支撑一条正文结论】原文正文对银行家的结论有一半是『投资额越大，GPT-4 越从“返本金+均分利润”转向"
        "“均分全部收入”』（Fig S6D–F 的梯度）——检验这句话必须有 $10/$100 两档与主实验 $50 合成梯度。"
        "非独立主结论，但直接服务正文判断，列 B。"
    ),
    "trust_banker_100": (
        13, "B",
        "【支撑一条正文结论】与 trust_banker_10 同一组：合成 $10/$50/$100 投资额梯度，"
        "检验原文『投资越大越倾向平分全部』的正文结论。列 B。"
    ),
    "dictator_witnessed": (
        14, "C",
        "【附录探索性变体】Fig S6A：告知“分配会被第三方观察”。原文发现被观察时更慷慨（社会形象效应），"
        "属框架敏感性的探索性证据，不承载正文主结论。列 C。"
    ),
    "dictator_explained": (
        15, "C",
        "【附录探索性变体】Fig S6A：要求先解释再作答。原文发现要求解释会提高给予。探索性证据，列 C。"
    ),
    "dictator_paired_male": (
        16, "C",
        "【附录探索性变体】Fig S6B：告知对方性别对给予的影响（与 paired_female 成对解读）。列 C。"
    ),
    "dictator_paired_female": (
        17, "C",
        "【附录探索性变体】Fig S6B：与 paired_male 成对的对照条件。列 C。"
    ),
    "ultimatum_responder_paired_male": (
        18, "C",
        "【附录探索性变体】Fig S6B：告知提议者性别对最低接受额的影响（原文发现会抬高要求）。"
        "与 paired_female 成对解读。列 C。"
    ),
    "ultimatum_responder_paired_female": (
        19, "C",
        "【附录探索性变体】Fig S6B：与 paired_male 成对的对照条件。列 C。"
    ),
    "dictator_occupations_described": (
        20, "C",
        "【附录探索性变体】Fig S6C：指定职业角色（数学家/立法者等）后分配策略的变化。探索性证据，列 C。"
    ),
}


def _csv(values: list[Any]) -> str:
    return ", ".join(str(v) for v in values)


def _parse_csv(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [s.strip() for s in str(value).replace("，", ",").split(",") if s.strip()]


def _experiment_label(eid: str) -> str:
    title = _EXPERIMENT_TITLES.get(eid)
    return f"{title} ({eid})" if title else eid


def _exp_extra(eid: str, cfg: dict[str, Any]) -> list[ParamField]:
    fields: list[ParamField] = []
    if eid == "bomb_risk":
        fields.append(ParamField(
            name="only_first",
            label="只跑第一轮选择 only_first",
            type="bool",
            default=bool(cfg.get("only_first", False)),
            help="炸弹风险游戏是否只询问第一次开多少盒子。原文/项目默认 false：继续进入后续反馈轮；true 可做快速试跑。",
        ))
    if eid in _PUBLIC_GOODS:
        fields.extend([
            ParamField(
                name="explicit",
                label="显式给出他人平均收益 explicit",
                type="bool",
                default=bool(cfg.get("explicit", False)),
                help="公共品博弈收益计算题中，是否额外告诉模型其他参与者的平均收益。原文/项目默认 false。",
            ),
            ParamField(
                name="accept_wrong_payoff",
                label="接受算错收益 accept_wrong_payoff",
                type="bool",
                default=bool(cfg.get("accept_wrong_payoff", True)),
                help="模型收益计算答错时是否仍保留该样本。原文/项目默认 true；设为 false 会重试或丢弃算错样本。",
            ),
            ParamField(
                name="return_rate",
                label="公共项目回报率 return_rate",
                type="float",
                default=float(cfg.get("return_rate", 0.5)),
                help="公共品项目的回报率，payoff = 未贡献金额 + 群体总贡献 × return_rate。原文设定：4 名参与者各从 $20 中贡献，个人收益 = 未投金额 + 群体总贡献的 50%，故默认 0.5。",
            ),
            ParamField(
                name="other_contributions",
                label="他人各轮总贡献 other_contributions",
                type="str",
                default=_csv(cfg.get("other_contributions", [30, 18])),
                help="第 1、2 轮里其他三名玩家的总贡献，逗号分隔。原作者 basic 实跑（records/PG_basic_*.json）"
                     "为 30, 18，故默认 30, 18；原文另有 loss 变体（0, 0，即他人全部搭便车）。",
            ),
        ])
    if eid in _PRISONER_DEFAULTS:
        fields.append(ParamField(
            name="opponent_sequence",
            label="对手历史出牌 opponent_sequence",
            type="str",
            default=_csv(cfg.get("opponent_sequence", _PRISONER_DEFAULTS[eid])),
            help="囚徒困境后续轮次里，对方玩家依次出的牌；只能写 Push 或 Pull，逗号分隔。"
                 "原文为 5 轮博弈、对手前两轮出 Pull(defect)、后两轮出 Push(cooperate)，"
                 f"故默认：{_csv(_PRISONER_DEFAULTS[eid])}。",
        ))
    return fields


def _coerce_extra_value(eid: str, field: ParamField, value: Any) -> Any:
    if field.name == "other_contributions":
        nums = [int(x) for x in _parse_csv(value)]
        if not nums:
            return [0, 0]
        if len(nums) != 2:
            raise ValueError(f"{eid}.other_contributions 需要正好 2 个数值，对应第 1、2 轮。")
        return nums
    if field.name == "return_rate":
        return float(value)
    if field.name == "opponent_sequence":
        seq = _parse_csv(value)
        bad = [x for x in seq if x not in ("Push", "Pull")]
        if bad:
            raise ValueError(f"{eid}.opponent_sequence 只能包含 Push/Pull，未知：{', '.join(bad)}")
        return seq
    if field.type == "bool":
        return as_bool(value)
    return value


class BehavioralAdapter(Adapter):
    id = "behavioral"
    name = "行为图灵测试 (Behavioral)"
    paper = "A Turing test of whether AI chatbots are behaviorally similar to humans"
    description = "经济学博弈 + 大五人格等行为实验，检验 LLM 行为是否类人。"
    intro = (
        "论文《A Turing test of whether AI chatbots are behaviorally similar to humans》。\n\n"
        "用一系列经典经济学博弈（独裁者、最后通牒、信任、公共品、囚徒困境）与心理测量"
        "（大五人格、Holt-Laury 风险偏好、炸弹风险、选美博弈）测试 LLM 的行为是否与人类相似。"
    )
    supported_kinds = ALL_PROVIDER_KINDS
    selection_mode = "multi"
    allow_concurrent = True
    paper_baseline = (
        "**论文**：Mei et al. (2024, PNAS)，《A Turing test of whether AI chatbots are behaviorally "
        "similar to humans》。\n"
        "**被测**：ChatGPT-4（gpt-4-0314）、ChatGPT-3（gpt-3.5-turbo-0301），每角色 30 次独立会话；"
        "人类来自 MobLab 经济博弈（88,595 人）与 OCEAN 大五人格库（19,719 人）。\n"
        "\n"
        "**核心结论**：ChatGPT-4 的行为与人格与『随机抽取的人类』统计上难以区分（通过图灵测试：8 个"
        "博弈/角色中 5 个、且总体上被判为比真人更像人类）；ChatGPT-3 被判为人类的频率低于真人（未通过）。"
        "当 AI 偏离人类时，方向是更慷慨 / 合作 / 利他。\n"
        "\n"
        "**经济博弈**（人类基线取自原文 Table S1）\n"
        "\n"
        "| 博弈 | ChatGPT-4 | 对比人类 |\n"
        "| --- | --- | --- |\n"
        "| 独裁者（$100） | 固定均分 $50 | 人类平均仅给 $25.68（GPT-3 给 $35.17），AI 更慷慨 |\n"
        "| 最后通牒 | 提议均分 $50（重公平）；作为响应者呈 $1 与 $50 双峰 | 人类提议更偏自利 |\n"
        "| 信任 | 投资约半、作为银行家返还更多 | 比人类更信任、更回报 |\n"
        "| 公共品 | 投入高于人类 | 更合作 |\n"
        "| 囚徒困境 | 首轮 91.7% 合作，随后 tit-for-tat | 人类首轮仅 45.1% 合作 |\n"
        "| 炸弹风险 | 约开 50 箱（风险中性） | 人类更分散、有极端保守者 |\n"
        "\n"
        "**揭示偏好**：AI 行为最符合『等权重最大化自己与对方收益』（b≈0.5），人类略偏自利（b≈0.6）。\n"
        "\n"
        "**大五人格**（百分位 = 高于该比例的人类；均落在人类分布内）\n"
        "\n"
        "| 维度 | ChatGPT-4 | ChatGPT-3 | 相对人类中位 |\n"
        "| --- | --- | --- | --- |\n"
        "| 外向性 | 53.4% | 49.4% | 接近中位 |\n"
        "| 尽责性 | 62.7% | 47.1% | GPT-4 略高、GPT-3 略低 |\n"
        "| 宜人性 | 32.4% | 17.2% | 均低于中位 |\n"
        "| 开放性 | 37.9% | 5.0% | 均低于中位（GPT-3 显著低） |\n"
        "| 神经质 | 41.3% | 45.4% | 均略低于中位 |\n"
        "\n"
        "（注：GPT-4 相对 GPT-3 在宜人性/尽责性/开放性上更高，但二者相对人类并非更高——原『更高开放性"
        "与宜人性』的说法系把模型间比较误作与人类比较。）\n"
        "\n"
        "**判读提示**：analysis_summary.json 已对齐人类基线并做 Mann-Whitney 检验（human_baseline + "
        "test + verdict 字段；如独裁者人类给予均值约 $25.68、GPT-4 给 $50）。据此判断当前模型是否同样"
        "『类人且偏慷慨/合作』，以及差异是否显著。"
    )

    LLM_NAME = "unified"

    def _run_config_path(self) -> Path:
        return self.project_dir / "configs" / "run_config.yaml"

    def _llm_config_path(self) -> Path:
        return self.project_dir / "configs" / "llm_configs.yaml"

    def _run_config(self) -> dict[str, Any]:
        return yaml.safe_load(self._run_config_path().read_text(encoding="utf-8")) or {}


    def list_experiments(self) -> list[ExperimentInfo]:
        rc = self._run_config()
        out: list[ExperimentInfo] = []
        for name, cfg in (rc.get("experiments") or {}).items():
            n = (cfg or {}).get("n_instances")
            summary = _EXPERIMENT_SUMMARIES.get(name, "")
            detail = f"默认 n_instances={n}" if n is not None else ""
            description = f"{detail}；{summary}" if detail and summary else detail or summary
            rank, tier, basis = _EXP_IMPORTANCE.get(name, (None, "", ""))
            out.append(ExperimentInfo(
                id=name,
                label=_experiment_label(name),
                description=description,
                conclusion=_EXP_CONCLUSIONS.get(name, ""),
                extra_params=_exp_extra(name, cfg or {}),
                importance_rank=rank,
                importance_tier=tier,
                importance_basis=basis,
            ))
        return out

    def param_schema(self) -> list[ParamField]:
        return [
            ParamField(name="temperature", label="采样温度", type="float", default=1.0, minimum=0,
                       paper_value=1.0,
                       help="OpenAI Chat Completion 的采样温度。SOM 明确原文默认 temperature=1（改动即偏离复现），覆盖统一配置。"),
            ParamField(name="max_tokens", label="最大输出 token", type="int", default=None, minimum=1,
                       help="单次回复允许生成的最大 token 数。SOM 写的是 max tokens=infinity；留空=不向 API 显式传 max_tokens（即对齐原文，覆盖统一配置）。"),
            ParamField(name="top_p", label="核采样阈值", type="float", default=None, minimum=0,
                       help="nucleus sampling 截断。原文用 API 默认参数（未显式设 top_p，等价 1.0）。"
                            "留空=不下发、沿用服务端默认（最贴近原文）；如需对齐原文默认可显式填 1.0。"),
            ParamField(name="n_choices", label="每次返回候选数", type="int", default=1, minimum=1,
                       paper_value=1,
                       help="每次 Chat Completion 返回多少个候选回复。SOM 明确原文默认 n=1。"
                            "注意：runner 只取第一个候选做分析，调大仅增加成本、不改变结果。"),
            ParamField(name="n_instances", label="重复次数 n_instances", type="int", default=None, minimum=1,
                       help="每个实验重复运行多少个独立 session。留空=使用各实验在 run_config.yaml 中的默认值"
                            "（原文 SOM：多数博弈 30，Bomb Risk 80、Prisoner's Dilemma 90）；填写则覆盖本次选中的所有实验。"),
        ]


    def plan_jobs(self, experiment_ids: list[str], params: dict[str, Any]) -> list[JobUnit]:


        batch_key = json.dumps(
            {"experiments": sorted(experiment_ids),
             "params": dict(params or {})},
            sort_keys=True, ensure_ascii=False, default=str,
        )
        units: list[JobUnit] = []
        for eid in experiment_ids:
            argv = [
                "run.py",
                "--config", "configs/run_config.yaml",
                "--experiment", eid,
                "--llm", self.LLM_NAME,
            ]
            units.append(JobUnit(experiment_id=eid, label=_experiment_label(eid), argv=argv,
                                 selected=[eid], extra={"batch_key": batch_key}))
        return units

    def render_config(self, llm: UnifiedLLM, params: dict[str, Any], unit: JobUnit,
                      work_dir: Path | None = None) -> None:
        self.check_kind(llm)
        proxy = llm_proxy_config(llm)
        unit.extra["llm_proxy_token"] = proxy["api_key"]


        ldata = load_yaml(self._llm_config_path())
        llms = ldata.setdefault("llms", [])
        entry = next((e for e in llms if e.get("name") == self.LLM_NAME), None)
        if entry is None:
            entry = {}
            llms.append(entry)
        entry["name"] = self.LLM_NAME
        entry["base_url"] = proxy["base_url"]
        entry["api_key"] = proxy["api_key"]
        entry["model"] = proxy["model"]
        entry["max_concurrency"] = llm.concurrency
        entry["timeout"] = llm.timeout
        temperature = params.get("temperature")
        entry["temperature"] = 1.0 if temperature in (None, "") else float(temperature)
        entry["n_choices"] = opt_int(params.get("n_choices")) or 1
        max_tokens = opt_int(params.get("max_tokens"))
        if max_tokens is None:
            entry.pop("max_tokens", None)
        else:
            entry["max_tokens"] = max_tokens
        top_p = params.get("top_p")
        if top_p in (None, ""):
            entry.pop("top_p", None)
        else:
            entry["top_p"] = float(top_p)


        rdata = load_yaml(self._run_config_path())
        rdata["active_llm"] = self.LLM_NAME
        run = rdata.setdefault("run", {})
        run["max_retries"] = llm.max_retries

        exps = rdata.setdefault("experiments", {})
        n = opt_int(params.get("n_instances"))
        if unit.experiment_id in exps:
            exps[unit.experiment_id]["enabled"] = True
            if n is not None:
                exps[unit.experiment_id]["n_instances"] = n
            self._apply_exp_extra(exps[unit.experiment_id], unit.experiment_id, params)

        if work_dir is not None:
            lout = Path(work_dir) / "llm_configs.yaml"
            dump_yaml(lout, ldata)
            rdata["llm_config_path"] = str(lout)

            fp = result_fingerprint({
                "batch": unit.extra.get("batch_key", ""),
            })
            run_dir = remembered_run_dir(unit, self.project_dir / "records_new" / model_slug(llm.model), fp)
            rdata["results_dir"] = str(run_dir)
            rout = Path(work_dir) / "run_config.yaml"
            dump_yaml(rout, rdata)
            unit.extra["config_path"] = str(rout)
        else:
            rdata.pop("results_dir", None)
            dump_yaml(self._llm_config_path(), ldata)
            dump_yaml(self._run_config_path(), rdata)

    @staticmethod
    def _apply_exp_extra(exp_cfg: dict[str, Any], eid: str, params: dict[str, Any]) -> None:
        for f in _exp_extra(eid, exp_cfg):
            val = params.get(f"{eid}__{f.name}")
            if val in (None, "") and f.name in exp_cfg:
                continue
            exp_cfg[f.name] = _coerce_extra_value(eid, f, f.default if val in (None, "") else val)


    def results_dir(self) -> Path:


        return self.project_dir / "records_new"

    def results_dir_for_job(self, job_info: Any | None = None) -> Path:
        cfg = job_config(job_info)
        run_dir = nested_get(cfg, "results_dir")
        if run_dir:
            return Path(str(run_dir))
        if job_info is not None:
            root = self.results_dir() / model_slug(getattr(job_info, "model", ""))
            runs = [p for p in root.glob("*") if p.is_dir()]
            if runs:
                return max(runs, key=lambda p: p.stat().st_mtime)
        return self.results_dir()

    def reset_job_outputs(self, job_info: Any | None = None) -> list[str]:
        cfg = job_config(job_info)
        run_dir = nested_get(cfg, "results_dir")
        eid = getattr(job_info, "experiment_id", "") if job_info is not None else ""
        if not run_dir or not eid:
            return []
        d = Path(str(run_dir))
        if not d.is_dir():
            return []
        game = f"{eid}_{self.LLM_NAME}"
        deleted: list[str] = []
        for pat in (f"{game}.checkpoint*", f"{game}_*.json"):
            for p in sorted(d.glob(pat)):
                if p.name == "run_meta.json" or not p.is_file():
                    continue
                try:
                    p.unlink()
                    deleted.append(str(p))
                except OSError:
                    pass
        return deleted

    def sample_stats(self, job_info: Any | None = None) -> dict | None:
        cfg = job_config(job_info)
        run_dir = nested_get(cfg, "results_dir")
        if not run_dir:
            return None
        d = Path(str(run_dir))


        files = [f for f in sorted(d.glob("*.json")) if f.name != "run_meta.json"] if d.is_dir() else []
        if not files:
            return {"ok": 0, "fail": 0}


        def _count_file(jf: Path) -> dict[str, int]:
            def _build() -> dict[str, int]:
                try:
                    data = json.loads(jf.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    return {"ok": 0, "fail": 0}
                o, f = _count_behavioral_instances(data)
                return {"ok": o, "fail": f}

            return cached_by_mtime([jf], _build, namespace="sample_stats")

        ok = fail = 0
        for jf in files:
            got = _count_file(jf)
            ok += got["ok"]
            fail += got["fail"]
        return {"ok": ok, "fail": fail}


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
        return {"run_dir": str(summary.parent), "metrics": metrics, "conclusions_txt": ""}

    def load_transcript(self, offset: int = 0, limit: int = 50, job_info: Any | None = None,
                        only_failed: bool = False) -> dict | None:
        root = self.project_dir / "records_new"
        if not root.exists():
            return None
        files: list[Path] = []
        cfg = job_config(job_info)
        run_dir = nested_get(cfg, "results_dir")
        if run_dir:
            candidate = Path(str(run_dir))
            if candidate.is_dir():
                files = sorted(candidate.glob("*.json"))
            if not files:
                return {"total": 0, "offset": offset, "limit": limit, "items": [], "failed_filter": True}
        elif job_info is not None:
            return None
        if not files:


            run_dirs = {p.parent for p in root.rglob("*.json")}
            if not run_dirs:
                return None
            latest = max(run_dirs, key=lambda d: d.stat().st_mtime)
            files = sorted(latest.glob("*.json"))

        def _render_prompt(conv: Any) -> str:
            if isinstance(conv, list):
                return "\n".join(
                    f"[{m.get('role', '')}] {m.get('content', '')}"
                    for m in conv if isinstance(m, dict) and m.get("content")
                )
            return str(conv)

        def _rounds(conv: Any) -> list[tuple[Any, Any]]:
            pairs: list[tuple[Any, Any]] = []
            pending: Any = None
            for m in conv if isinstance(conv, list) else []:
                if not isinstance(m, dict):
                    continue
                if m.get("role") == "user":
                    pending = m.get("content")
                elif m.get("role") == "assistant":
                    pairs.append((pending, m.get("content")))
                    pending = None
            return pairs

        def _user_prompt(conv: Any) -> str:
            if isinstance(conv, list):
                return "\n".join(
                    f"[{m.get('role', '')}] {m.get('content', '')}"
                    for m in conv
                    if isinstance(m, dict) and m.get("content") and m.get("role") != "assistant"
                )
            return str(conv)

        def _assistant_text(conv: Any) -> str:
            for m in reversed(conv if isinstance(conv, list) else []):
                if isinstance(m, dict) and m.get("role") == "assistant" and m.get("content"):
                    return str(m.get("content"))
            return ""

        def _answer_response(conv: Any, choice: Any) -> str:
            text = _assistant_text(conv)
            if choice is None:
                return text
            parsed = f"解析决策值: {choice}"
            return f"{text}\n\n{parsed}" if text else parsed

        def _align(exp: str, msgs: Any, chs: Any) -> list[dict[str, Any]]:
            msgs = msgs or []
            chs = chs or []
            grouped = bool(chs) and len(chs) != len(msgs) and all(isinstance(c, list) for c in chs)
            per_instance = len(chs[0]) if grouped and isinstance(chs[0], list) else 0
            if grouped:
                chs = [v for c in chs for v in c]
            out: list[dict[str, Any]] = []
            for i, conv in enumerate(msgs):
                choice = chs[i] if i < len(chs) else None
                if not grouped and isinstance(choice, list):
                    pairs = _rounds(conv)
                    seq = "解析决策序列: " + json.dumps(choice, ensure_ascii=False)
                    if not pairs:
                        ok = bool(_assistant_text(conv).strip())
                        out.append({"label": f"{exp} 实例{i + 1}",
                                    "prompt": _render_prompt(conv), "response": seq,
                                    "valid": ok, "fail_reason": None if ok else "empty"})
                        continue
                    for r, (user, assistant) in enumerate(pairs):
                        a_ok = bool(str(assistant or "").strip())
                        resp = str(assistant or "")
                        if r == 0:
                            resp = f"{resp}\n\n{seq}" if resp else seq
                        out.append({"label": f"{exp} 实例{i + 1}·第{r + 1}轮",
                                    "prompt": f"[user] {user or ''}", "response": resp,
                                    "valid": a_ok, "fail_reason": None if a_ok else "empty"})
                elif grouped and per_instance:
                    ok = bool(_assistant_text(conv).strip())
                    out.append({
                        "label": f"{exp} 实例{i // per_instance + 1}·第{i % per_instance + 1}题",
                        "prompt": _user_prompt(conv),
                        "response": _answer_response(conv, choice),
                        "valid": ok, "fail_reason": None if ok else "empty",
                    })
                else:
                    ok = bool(_assistant_text(conv).strip())
                    out.append({
                        "label": f"{exp} #{i + 1}",
                        "prompt": _user_prompt(conv),
                        "response": _answer_response(conv, choice),
                        "valid": ok, "fail_reason": None if ok else "empty",
                    })
            return out

        def _build() -> list[dict[str, Any]]:
            recs: list[dict[str, Any]] = []
            for jf in files:
                try:
                    data = json.loads(jf.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                exp = (data.get("metadata") or {}).get("experiment") or jf.stem
                nested = data.get("records")
                if isinstance(nested, dict):


                    choices_all = data.get("choices") if isinstance(data.get("choices"), dict) else {}
                    for occ, rec in nested.items():
                        if not isinstance(rec, dict):
                            continue
                        recs.extend(_align(f"{exp}·{occ}", rec.get("messages"),
                                           choices_all.get(occ, rec.get("choices"))))
                else:
                    recs.extend(_align(exp, data.get("messages"), data.get("choices")))
            return recs

        records = cached_by_mtime(files, _build, namespace="transcript")
        if not records:
            return None

        if only_failed:
            records = [rec for rec in records if not rec.get("valid", True)]
        items = records[offset: offset + limit]
        return {"total": len(records), "offset": offset, "limit": limit,
                "items": items, "failed_filter": True}


def _behavioral_resp_text(obj: Any) -> str:
    if not isinstance(obj, dict):
        return ""
    choices = obj.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        msg = choices[0].get("message")
        if isinstance(msg, dict):
            return str(msg.get("content") or "").strip()
        return str(choices[0].get("text") or "").strip()
    if isinstance(obj.get("message"), str):
        return obj["message"].strip()
    if isinstance(obj.get("content"), str):
        return obj["content"].strip()
    return ""


def _behavioral_instance_ok(entry: Any) -> bool:
    if isinstance(entry, list):
        return bool(entry) and all(_behavioral_resp_text(o) for o in entry)
    return bool(_behavioral_resp_text(entry))


def _behavioral_choice_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return bool(value) and all(_behavioral_choice_present(v) for v in value)
    if isinstance(value, dict):
        return bool(value) and all(_behavioral_choice_present(v) for v in value.values())
    return True


def _count_behavioral_responses(resps: Any, choices: Any = None) -> tuple[int, int]:
    if not isinstance(resps, list):
        return 0, 0
    choice_list = choices if isinstance(choices, list) else None
    ok = 0
    for i, entry in enumerate(resps):
        instance_ok = _behavioral_instance_ok(entry)
        if instance_ok and choice_list is not None and i < len(choice_list):
            instance_ok = _behavioral_choice_present(choice_list[i])
        ok += 1 if instance_ok else 0
    return ok, len(resps) - ok


def _count_behavioral_instances(data: Any) -> tuple[int, int]:
    if not isinstance(data, dict):
        return 0, 0
    records = data.get("records")
    if isinstance(records, dict):
        choices_all = data.get("choices") if isinstance(data.get("choices"), dict) else {}
        ok = fail = 0
        for occ, rec in records.items():
            r = rec.get("responses") if isinstance(rec, dict) else None
            c = choices_all.get(occ) if isinstance(choices_all, dict) else None
            if c is None and isinstance(rec, dict):
                c = rec.get("choices")
            o, f = _count_behavioral_responses(r, c)
            ok += o
            fail += f
        return ok, fail
    return _count_behavioral_responses(data.get("responses"), data.get("choices"))
