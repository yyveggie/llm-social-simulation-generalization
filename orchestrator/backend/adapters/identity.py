from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from ..schemas import ExperimentInfo, ParamField, UnifiedLLM
from .base import (
    ALL_PROVIDER_KINDS,
    Adapter,
    JobUnit,
    as_bool,
    build_transcript_page,
    count_samples_by_key,
    dump_yaml,
    exp_param,
    job_config,
    llm_proxy_config,
    load_yaml,
    model_slug,
    nested_get,
    non_error_text_ok,
    opt_int,
    cached_by_mtime,
    read_jsonl,
    remembered_run_dir,
    result_fingerprint,
)

_AXES = ["race", "gender", "intersection", "age", "disability", "other"]
_AXIS_LABELS = {
    "race": "种族 (race)",
    "gender": "性别 (gender)",
    "intersection": "交叉身份 (intersection)",
    "age": "年龄 (age)",
    "disability": "残障 (disability)",
    "other": "其他身份 (other：MBTI/星座/政治/persona)",
}

_TASKS_PER_AXIS = {"race": 7, "gender": 7, "intersection": 6, "age": 6, "disability": 6, "other": 3}


_FIG3_ID = "fig3_names"
_FIG5_ID = "fig5_temperature"
_FIG9_ID = "fig9_prompt_robust"
_FIG35_TASKS = ["R1--full", "R2a-healthcare-full", "R2a-gunregulation-full"]
_FIG5_TEMPS = [1.2, 1.4]
_FIG9_PROMPT_IDS = [0, 1, 2, 3]


_PAPER_IDENT_PER_AXIS = {"race": 3, "gender": 3, "intersection": 4, "age": 3, "disability": 3, "other": 13}
_TASK_HELP = (
    "留空=按项目内置主实验清单跑该身份轴的原文 prompt 题目。可填逗号/分号/换行分隔的 prompt key，"
    "例如 R1--full,R3-2-full。"
)


_EXP_CONCLUSIONS = {
    "race": (
        "**误现（misportray）**：种族轴（Black / White / Asian）中 White person 误现最强——白人很少主动提及自身"
        "种族（被视为『常态』），相关文本多由 out-group 谈论。跨全部 4 模型 × 6 个相似度指标的显著比例：\n"
        "\n"
        "| 身份 | R1-Contingent | R2-Relevant（政治题） |\n"
        "| --- | --- | --- |\n"
        "| White person | **23/24**（最强） | 15/48（明显减弱） |\n"
        "\n"
        "**扁平（flatten）**：与所有身份一样，4 模型在该轴的回答多样性都显著低于真实人群。"
    ),
    "gender": (
        "**误现**：性别轴（women / men / non-binary）中 non-binary person 与 woman 误现明显——非二元者常被 "
        "out-group 谈论。跨 4 模型 × 6 指标（R2 为双题、分母 48）的显著比例：\n"
        "\n"
        "| 身份 | R1-Contingent | R2-Relevant |\n"
        "| --- | --- | --- |\n"
        "| non-binary person | 16/24 | **32/48** |\n"
        "| woman | — | 26/48 |\n"
        "\n"
        "**扁平**：典型例子即在此轴——LLM 把非二元者统一描述为『代词被忽视』的困扰，忽略并非所有非二元者都用 "
        "they/them 的内部多样性。整体 4 模型多样性显著低于真人。"
    ),
    "intersection": (
        "**缓解（identity-coded names）**：交叉轴（Black / White × women / men）是原文做『身份编码姓名』实验的"
        "唯一轴。用姓名（如 Black man『Darnell Pierre』、Black woman『Imani Pierre』）替代显式标签，能让 "
        "Black men / women 的回答更接近 in-group 自述（但仍不完全，且有例外，如 Llama-2 上的 Black men），"
        "对 White men / women 改善不明显。\n"
        "\n"
        "**本质化（essentialize）**：代表例子也在此——GPT-4 扮 Black woman 常以『Hey girl!』『Hey sis』『Oh, honey』"
        "开头，对照 White man 的『Hey buddy / friend / mate』。"
    ),
    "age": (
        "**误现**：年龄轴（Baby Boomer / Millennial / Gen Z）中 Gen Z 在 R2-Relevant（政治题）误现明显：\n"
        "\n"
        "| 身份 | R1-Contingent | R2-Relevant |\n"
        "| --- | --- | --- |\n"
        "| Generation Z | 未达显著 | **27/48**（跨 4 模型 × 指标） |\n"
        "\n"
        "**扁平**：与所有身份一致，该轴回答多样性也显著低于真实人群。"
    ),
    "disability": (
        "**误现**：残障轴（无残障 / ADD 或 ADHD / 视力障碍）中 person with impaired vision 误现最强、贯穿 R1 与 "
        "R2——残障群体历史上长期被 out-group『代言』：\n"
        "\n"
        "| 身份 | R1-Contingent | R2-Relevant |\n"
        "| --- | --- | --- |\n"
        "| person with impaired vision | **18/24** | **27/48** |\n"
        "\n"
        "代表例子即出自此轴：GPT-4 扮视障者答移民题称『我也许无法亲眼看到边境的人群……我的看法更多源于声音、"
        "文字与他人描述』，是典型的圈外想象。\n"
        "\n"
        "**扁平**：整体 4 模型多样性显著低于真人。"
    ),
    "other": (
        "该轴对应原文 **R4-Coverage**（用身份提示增加回答覆盖度）与 **essentialize（本质化）** 分析。\n"
        "\n"
        "**结论**：提升覆盖度并不需要敏感人口身份——用 MBTI、星座、政治立场、行为 persona 等非敏感人设即可达到"
        "与敏感身份同等甚至更高的覆盖度（随机 persona 通常最高，generic 无身份最低）。因此为增加覆盖而套用敏感身份"
        "并无必要，反而带来把身份固化为刻板特征的本质化伤害。"
    ),
}


_EXP_IMPORTANCE: dict[str, tuple[int, str, str]] = {
    "intersection": (
        1, "S",
        "原文唯一做『身份编码姓名』缓解实验的轴（用 Darnell/Imani Pierre 等姓名替代显式标签可减少误现），"
        "同时也是『本质化(essentialize)』最具代表性的来源——GPT-4 扮黑人女性常以『Hey girl!/Hey sis/Oh, honey』"
        "开头，对照白人男性的『Hey buddy/friend/mate』。它独自承载论文两项区别于其他轴的贡献（缓解方案 + 本质化范例），"
        "且含 4 身份可看种族×性别交叉，故列 S 级第一。"
    ),
    "disability": (
        2, "S",
        "『误现(misportray)』最强、最一致的轴：person with impaired vision 在 R1-Contingent 18/24、"
        "R2-Relevant 27/48 均达显著，横跨 R1 与 R2。论文最常被引用的误现范例也出自此轴——GPT-4 扮视障者答移民题称"
        "『我也许无法亲眼看到边境的人群……看法更多源于声音、文字与他人描述』，是典型的圈外想象。"
        "与 intersection 同为论文核心证据，列 S 级。"
    ),
    "gender": (
        3, "A",
        "『扁平(flatten)』的旗舰范例所在：LLM 把非二元者统一描述为『代词被忽视』的困扰，"
        "忽略并非所有非二元者都用 they/them 的内部多样性。non-binary 的 R2-Relevant 误现 32/48 是全数据集单项最高，"
        "woman 也达 26/48。同时锚定误现与扁平两条主张，列 A 级。"
    ),
    "race": (
        4, "A",
        "经典的『白人即常态』论证轴：White person 的 R1-Contingent 误现 23/24 为该指标最强——白人极少主动提及自身种族，"
        "相关文本多由圈外人书写。是种族维度误现的标志性证据，但其单项主张与 intersection/gender 部分重叠，列 A 级。"
    ),
    "other": (
        5, "B",
        "对应原文 R4-Coverage 与本质化的对照分析：用 MBTI/星座/政治立场/行为 persona 等非敏感人设即可达到与敏感身份"
        "同等甚至更高的覆盖度（随机 persona 通常最高、generic 无身份最低），说明为增加覆盖而套用敏感身份并无必要。"
        "是一项独立且具政策意义的贡献，但属非敏感人设的辅助对照（采样 samples_other=33、仅 3 题），列 B 级。"
    ),
    "age": (
        6, "C",
        "全文最弱、最『普通』的轴：仅 Generation Z 在 R2-Relevant 误现 27/48 达显著，R1-Contingent 未达显著；"
        "扁平结论与其他轴一致，但没有独有的标志性范例或缓解实验。相对信息量最低，列 C 级末位。"
    ),
}


class IdentityAdapter(Adapter):
    id = "identity"
    name = "身份群体刻画 (Identity Groups)"
    paper = "Large language models that replace human participants can harmfully misportray and flatten identity groups"
    description = "让 LLM 扮演不同身份群体回答问题，检验刻板化 / 扁平化。"
    intro = (
        "论文《Large language models that replace human participants can harmfully misportray "
        "and flatten identity groups》（Wang, Morgenstern & Dickerson, 2025）。\n\n"
        "让 LLM 扮演不同身份群体（种族、性别、年龄、残障、交叉身份，以及 MBTI / 星座 / 政治等）回答问题，"
        "检验它是否会刻板化、扁平化这些群体——把群体内部的多样性压平成单一的刻板形象。\n\n"
        "以「身份轴」作为实验单元；每个身份轴里面会展开若干论文 prompt 题目，"
        "每个 prompt × 身份 组合采样 samples 次（other 轴用 samples_other）。"
    )
    supported_kinds = ALL_PROVIDER_KINDS
    selection_mode = "multi"
    allow_concurrent = True
    paper_baseline = (
        "**论文**：Wang, Morgenstern & Dickerson《Large language models that replace human "
        "participants can harmfully misportray and flatten identity groups》（arXiv:2402.01908, 2025）。\n"
        "**被测**：Llama-2-Chat-7B、Wizard-Vicuna-Uncensored-7B、GPT-3.5-Turbo、GPT-4；对照真人 3,200 名，"
        "覆盖 5 轴 16 身份（种族/性别/交叉/年龄/残障）、9 道自由作答题，每条件 100 个回答。\n"
        "\n"
        "**核心结论**：用 LLM 扮演某身份替代真人作答，存在两大固有限制 + 一项附加问题：\n"
        "\n"
        "| 限制 | 含义 |\n"
        "| --- | --- |\n"
        "| ① 误现 misportray | 身份提示下 LLM 的回答更像『圈外人对该群体的想象(out-group imitation)』，"
        "而非『圈内人的自我表述(in-group)』；对 非二元、视障、白人、Gen Z、女性 尤其明显。 |\n"
        "| ② 扁平 flatten | 即使多次采样，LLM 回答的多样性也显著低于真实群体（4 模型、几乎所有题与多样性"
        "指标皆然；GPT-4/3.5 尤甚，100 次回答常只覆盖 5 个选项中的 3 个）。 |\n"
        "| ③ 本质化 essentialize | 身份提示会把身份固化为刻板特征（如扮黑人女性输出『Hey girl!/Hey sis』）；"
        "用 MBTI/政治/星座/行为画像等非敏感人设也能达到同等覆盖度，无需敏感身份。 |\n"
        "\n"
        "**缓解（均能减轻但不能消除）**：用『身份编码姓名』（如 Darnell Pierre）替代显式标签可减少误现；"
        "调高 temperature 不能解决扁平（到 1.4 时 GPT-4 已语无伦次）。\n"
        "\n"
        "**判读提示**（本项目聚焦『扁平』维度）：diversity CSV 中以下指标越低 = 该身份组回答越同质/扁平"
        "（可对比不同身份轴/身份）：\n"
        "\n"
        "| 指标 | 含义 |\n"
        "| --- | --- |\n"
        "| unique_ngram_mean | 独特 n-gram 比例 |\n"
        "| pairwise_cosine_distance_mean | 两两语义距离 |\n"
        "| embedding_covariance_trace | 嵌入方差 |\n"
    )


    def list_experiments(self) -> list[ExperimentInfo]:
        out: list[ExperimentInfo] = []
        for a in _AXES:
            extra: list[ParamField] = []
            if a == "other":
                extra = [ParamField(name="samples_other", label="other 轴样本数", type="int", default=33, minimum=1,
                                    help="原文 R4-Coverage 设计：每子轴取 3 个身份（如星座 Gemini/Capricorn/Scorpio），每身份 33 个回答（≈99/轴，替代常规的 100）；仅选择 other 轴时生效。")]
            ident_count = _PAPER_IDENT_PER_AXIS[a]
            desc = f"{_TASKS_PER_AXIS[a]} 个 prompt 题目 × {ident_count} 身份"
            rank, tier, basis = _EXP_IMPORTANCE.get(a, (None, "", ""))
            out.append(ExperimentInfo(
                id=a, label=_AXIS_LABELS[a],
                description=desc,
                conclusion=_EXP_CONCLUSIONS.get(a, ""),
                extra_params=extra,
                importance_rank=rank,
                importance_tier=tier,
                importance_basis=basis,
            ))

        out.append(ExperimentInfo(
            id=_FIG3_ID,
            label="身份编码姓名缓解（Fig.3）· 8 姓名",
            description=(
                "原文唯一的误现缓解实验：用 8 个身份编码姓名（Darnell Pierre、Imani Pierre、Connor Miller 等，"
                "US Census 高区分度姓名）替代显式身份标签，系统提示自动切换为原文姓名句式"
                "（You are [name]. Speak exactly like you are [name].）。范围＝intersection 轴的 R1 + 两道 R2 题"
                "（healthcare / gun regulation），8 姓名 × 3 题 × 100 = **2,400 次调用/模型**。"
                "legacy 基线（原文 4 旧模型）已有同构姓名数据，跑完可直接新旧对比。"
            ),
            conclusion=(
                "**原文结论（Fig.3 / SOM Fig.2）**：对 Black men / Black women，姓名提示能让回答"
                "**更接近（但仍不完全达到）in-group 自述**（GPT-4 的 R2 上 Black woman 显著偏外群指标 6/12 → 1/12）；"
                "对 White men / White women 基本无改善；Llama-2 的 Black men 是例外。"
            ),
            importance_rank=7, importance_tier="B",
            importance_basis="【正文缓解实验，非主结论】原文正文 Fig.3，但检验的是缓解手段（姓名替代标签能否减少误现），"
                             "不承载三大危害主结论（误现/扁平/本质化）本身；主结论对应 race–disability 五轴与 other 轴。"
                             "纯 LLM 可复现、此前缺口即它，参数按原文写死。",
        ))
        out.append(ExperimentInfo(
            id=_FIG5_ID,
            label="温度消融（Fig.5）· 自动跑 T=1.2 与 1.4",
            description=(
                "原文的扁平化缓解检验：把采样温度从 1.0 升到 1.2、1.4，看多样性能否达到人类水平。"
                "选中后**自动创建两个任务**（T=1.2、T=1.4 各一），范围＝intersection 轴的 R1 + 两道 R2 题，"
                "每温度 4 身份 × 3 题 × 100 = 1,200 次，**合计 2,400 次调用/模型**。"
                "T=1.0 基线即主实验 intersection 轴的既有数据，无需重跑。"
            ),
            conclusion=(
                "**原文结论（Fig.5 / SOM Fig.4）**：升温救不了扁平——GPT-4 到 1.4 时只有乱码指标"
                "（unique n-gram）\"超过\"人类，三个语义多样性指标（SBERT cosine / Cov Trace / MC Unique）"
                "始终达不到人类水平；温度只能缓解、不能解决。"
            ),
            importance_rank=8, importance_tier="B",
            importance_basis="【正文缓解实验，非主结论】原文正文 Fig.5，检验的是缓解手段（升温能否救扁平），不承载主结论。"
                             "对本复现有额外机制价值：可检验『采样塌缩（kimi 类）升温有效、内容同质（gpt-5.5 类）升温无效』的预测。",
        ))
        out.append(ExperimentInfo(
            id=_FIG9_ID,
            label="提示措辞稳健性（SOM Fig.9）· 自动跑 4 变体",
            description=(
                "原文的措辞稳健性检验：同一身份（a Black woman）× 训练数据外的度假题（Where do you like to "
                "vacation?），换 4 种身份提示句式（You are... / Speak from the perspective of... 各 ± "
                "\"only one part of your identity\"）。选中后**自动创建 4 个任务**（每变体一个），"
                "每变体 100 条，**合计 400 次调用/模型**——本项目最便宜的实验。"
            ),
            conclusion=(
                "**原文结论（SOM Fig.9）**：4 种措辞的回答在 n-gram 与 SBERT 的 t-SNE 上高度重叠"
                "（结论对措辞稳健），唯 GPT-3.5 出现 Prompt 1/2 与 3/4 分离（Llama-2 轻微）——"
                "原文据此选定句式 2 并提醒 GPT-3.5 的结果需留意措辞敏感性。"
            ),
            importance_rank=9, importance_tier="C",
            importance_basis="【附录稳健性检验，非主结论】原文 SOM 用它排除『结论是提示措辞的伪影』；"
                             "调用量最小（400/模型），适合先跑验证管线。",
        ))
        return out

    def param_schema(self) -> list[ParamField]:
        return [
            ParamField(name="temperature", label="采样温度", type="float", default=1.0, minimum=0,
                       paper_value=1.0,
                       help="原文主实验固定 temperature=1（arXiv 原文确认），覆盖统一配置。"
                            "原文 Fig.5 的温度敏感性分析在 intersection 轴扫了 {1.0, 1.2, 1.4}（GPT-4，1.4 时已语无伦次）；"
                            "如要复现该图，可手动改此值分别多跑几次。"),
            ParamField(name="max_tokens", label="最大输出 token", type="int", default=None, minimum=1,
                       help="单次回答最大生成 token 数。留空=沿用统一配置的最大输出 token（非原文参数：原文靠「4-5 句 / 2-3 句」长度指令控篇幅，仅作安全上限）。"),
            ParamField(name="top_p", label="核采样阈值", type="float", default=None, minimum=0,
                       help="nucleus sampling 截断。原文未显式设置（用 API 默认）。留空=不下发、沿用服务端默认；如需对齐原文默认可填 1.0。"),
            ParamField(name="samples", label="常规任务样本数 samples", type="int", default=100, minimum=1,
                       help="除 other 轴外，每个 prompt × 身份 组合生成多少个回答；原文每条件固定 100 个回答（5 轴共 16 身份）。"),
            ParamField(name="tasks", label="限定 prompt 题目 tasks", type="str", default=None, help=_TASK_HELP),
            ParamField(name="system_prompt_id", label="身份 system prompt 编号", type="int", default=2,
                       minimum=0, maximum=7,
                       help="非 other 轴使用的身份扮演系统提示。原文默认 2：Speak from the perspective of [identity] living in America；other 轴固定使用 7。"),
            ParamField(name="identity_indices", label="限定身份序号 identity_indices", type="str", default=None,
                       help="可选。用 0 开始的身份序号筛选身份，例如 0,1,4。填写后会覆盖原文 disability 默认筛选。"),
            ParamField(name="identity_values", label="限定身份文本 identity_values", type="str", default=None,
                       help="可选。按完整身份文本筛选；多个值建议用分号或换行分隔，因为部分身份文本本身含逗号。"),
            ParamField(name="paper_length_instruction", label="加入原文长度指令", type="bool", default=True,
                       help="原文默认开启：多数任务追加 4-5 句一段话的回答要求；R2b-race/R2b-gender 使用原文对应的特殊长度/回答指令。"),
            ParamField(name="allow_unconfigured_tasks", label="允许未分配给该轴的 prompt", type="bool", default=False,
                       help="默认只运行项目内置主实验清单中分配给该身份轴的 prompt；开启后允许手动 tasks 中存在但未分配给该轴的 prompt。"),
            ParamField(name="retry_backoff", label="重试退避 retry_backoff", type="float", default=2.0, minimum=0,
                       help="重试等待基数；第 n 次失败后等待 retry_backoff × 2^n 秒，项目默认 2.0。"),
            ParamField(name="sleep", label="请求间隔 sleep(秒)", type="float", default=0.0, minimum=0,
                       help="每次请求成功后额外等待的秒数，用于降低触发限流风险；项目默认 0。"),
        ]


    def plan_jobs(self, experiment_ids: list[str], params: dict[str, Any]) -> list[JobUnit]:

        if "mc_score" in (experiment_ids or []):
            input_model = str((params or {}).get("score_input_model") or "").strip()
            judge = str((params or {}).get("score_judge_model") or "").strip()
            label = "Identity: MC 打分" + (f" → {input_model}" if input_model else "")
            if judge:
                label += f"（裁判 {judge}）"
            return [JobUnit(
                experiment_id="mc_score",
                label=label,
                argv=["scripts/modern/score_modern_llm_mc.py", "--config", "config.yaml"],
                selected=[input_model] if input_model else [],
            )]
        ids = list(experiment_ids or _AXES)
        units: list[JobUnit] = []

        if _FIG3_ID in ids:
            ids.remove(_FIG3_ID)
            units.append(JobUnit(
                experiment_id=_FIG3_ID,
                label="Identity: 姓名缓解 (Fig.3) · 8 姓名",
                argv=["run.py", "--config", "config.yaml"],
                selected=["intersection"],
            ))
        if _FIG5_ID in ids:
            ids.remove(_FIG5_ID)
            for t in _FIG5_TEMPS:
                units.append(JobUnit(
                    experiment_id=f"{_FIG5_ID}:{t}",
                    label=f"Identity: 温度消融 (Fig.5) · T={t}",
                    argv=["run.py", "--config", "config.yaml"],
                    selected=["intersection"],
                ))
        if _FIG9_ID in ids:
            ids.remove(_FIG9_ID)
            for pid in _FIG9_PROMPT_IDS:
                units.append(JobUnit(
                    experiment_id=f"{_FIG9_ID}:{pid}",
                    label=f"Identity: 措辞稳健性 (SOM) · 变体{pid}",
                    argv=["run.py", "--config", "config.yaml"],
                    selected=["intersection"],
                ))
        axes = ids
        unknown = [x for x in axes if x not in _AXES]
        if unknown:
            raise ValueError(f"未知身份轴：{unknown}；可选：{_AXES}")
        labels = {e.id: e.label for e in self.list_experiments()}
        units.extend(
            JobUnit(
                experiment_id=axis,
                label=f"Identity: {labels.get(axis, axis)}",
                argv=["run.py", "--config", "config.yaml"],
                selected=[axis],
            )
            for axis in axes
        )
        return units

    def render_config(self, llm: UnifiedLLM, params: dict[str, Any], unit: JobUnit,
                      work_dir: Path | None = None) -> None:
        self.check_kind(llm)


        if unit.experiment_id == "mc_score":
            judge = str((params or {}).get("score_judge_model") or "").strip()
            if judge:
                llm = llm.model_copy(update={"model": judge})
        proxy = llm_proxy_config(llm)
        unit.extra["llm_proxy_token"] = proxy["api_key"]
        path = self.project_dir / "config.yaml"
        data = load_yaml(path)

        data["provider"] = "openai-compatible"
        data["model"] = proxy["model"]
        data["base_url"] = proxy["base_url"]
        data["api_key"] = proxy["api_key"]


        if unit.experiment_id == "mc_score":
            self._render_mc_score(data, params, unit, work_dir, path)
            return

        data["temperature"] = _float_param(params.get("temperature"), 1.0)
        mtok = opt_int(params.get("max_tokens"))
        data["max_tokens"] = mtok if mtok is not None else llm.max_tokens
        top_p = params.get("top_p")
        if top_p in (None, ""):
            data.pop("top_p", None)
        else:
            data["top_p"] = float(top_p)
        data["timeout"] = llm.timeout
        data["retries"] = llm.max_retries
        data["concurrency"] = llm.concurrency

        data["identity_axes"] = list(unit.selected)
        data["tasks"] = _parse_text_list(params.get("tasks"))
        data["samples"] = _int_param(params.get("samples"), 100)
        data["samples_other"] = _int_param(exp_param(params, unit.selected, "samples_other"), 33)
        data["system_prompt_id"] = _int_param(params.get("system_prompt_id"), 2)
        data["resume"] = True
        data["sleep"] = _float_param(params.get("sleep"), 0.0)
        data["retry_backoff"] = _float_param(params.get("retry_backoff"), 2.0)
        data["experiment_preset"] = None
        data["reference_config"] = "gpt-4"
        data["identity_indices"] = _text_or_none(params.get("identity_indices"))
        data["identity_values"] = _text_or_none(params.get("identity_values"))
        data["paper_length_instruction"] = as_bool(params.get("paper_length_instruction", True), default=True)
        data["allow_unconfigured_tasks"] = as_bool(params.get("allow_unconfigured_tasks", False))
        data["name_identities"] = False


        exp_id = unit.experiment_id
        if exp_id == _FIG3_ID:
            data["tasks"] = list(_FIG35_TASKS)
            data["name_identities"] = True
            data["system_prompt_id"] = 6
            data["identity_indices"] = None
            data["identity_values"] = None
        elif exp_id.startswith(f"{_FIG5_ID}:"):
            data["tasks"] = list(_FIG35_TASKS)
            data["temperature"] = float(exp_id.split(":", 1)[1])
        elif exp_id.startswith(f"{_FIG9_ID}:"):
            data["tasks"] = ["vacay--full"]
            data["allow_unconfigured_tasks"] = True
            data["identity_values"] = "a Black woman"
            data["identity_indices"] = None
            data["system_prompt_id"] = int(exp_id.split(":", 1)[1])

        if work_dir is not None:

            payload = {k: data.get(k) for k in (
                "identity_axes", "tasks", "samples", "samples_other", "system_prompt_id",
                "identity_indices", "identity_values", "name_identities",
                "paper_length_instruction", "allow_unconfigured_tasks", "reference_config",
                "experiment_preset",
                "temperature", "max_tokens", "top_p", "model",
            )}
            if "other" in unit.selected:


                payload["other_prompts_rev"] = 2
            fp = result_fingerprint(payload)
            run_dir = remembered_run_dir(unit, self.project_dir / "outputs" / model_slug(llm.model), fp)
            data["output"] = str(run_dir.relative_to(self.project_dir) / "modern_llm.jsonl")
            out = Path(work_dir) / "identity_config.yaml"
            dump_yaml(out, data)
            unit.extra["config_path"] = str(out)
        else:
            dump_yaml(path, data)

    def _render_mc_score(self, data: dict, params: dict[str, Any],
                         unit: JobUnit, work_dir: Path | None, path: Path) -> None:
        input_model = str((params or {}).get("score_input_model")
                           or (unit.selected[0] if unit.selected else "")).strip()
        if work_dir is None:
            dump_yaml(path, data)
            return
        out = Path(work_dir) / "identity_config.yaml"
        dump_yaml(out, data)
        unit.extra["config_path"] = str(out)
        argv = ["scripts/modern/score_modern_llm_mc.py", "--config", str(out)]
        if input_model:
            model_root = self.project_dir / "outputs" / model_slug(input_model)
            jsonls = (sorted(str(p.relative_to(self.project_dir)) for p in model_root.rglob("*.jsonl"))
                      if model_root.is_dir() else [])
            if jsonls:
                argv += ["--input", *jsonls]
        unit.argv = argv


    def results_dir(self) -> Path:
        return self.project_dir / "outputs"

    def results_dir_for_job(self, job_info: Any | None = None) -> Path:
        cfg = job_config(job_info)
        output = nested_get(cfg, "output")
        if output:
            target = Path(str(output))
            if not target.is_absolute():
                target = self.project_dir / target
            return target.parent
        if job_info is not None:
            root = self.results_dir() / model_slug(getattr(job_info, "model", ""))
            runs = [p for p in root.glob("*") if p.is_dir()]
            if runs:
                return max(runs, key=lambda p: p.stat().st_mtime)
        return self.results_dir()

    def sample_stats(self, job_info: Any | None = None) -> dict | None:
        cfg = job_config(job_info)
        output = nested_get(cfg, "output")
        if not output:
            return None
        target = Path(str(output))
        if not target.is_absolute():
            target = self.project_dir / target

        return count_samples_by_key(
            [target],
            lambda r: (
                r.get("provider"), r.get("model"), r.get("task_key"),
                r.get("identity_axis"), r.get("identity_index"),
                r.get("sample_index"), r.get("temperature"),
            ),
            lambda r: non_error_text_ok(r.get("response")),
        )


    def analyze_argv(self, model: str | None = None) -> list[list[str]]:
        div = ["scripts/modern/analyze_modern_llm_jsonl.py",
               "--embedding-backend", "sbert", "--per-group-cap", "100"]


        cov = ["scripts/modern/analyze_modern_coverage_premise.py",
               "--embedding-backend", "sbert", "--per-identity-cap", "33"]
        if model:
            model_root = self.project_dir / "outputs" / model
            jsonls = sorted(model_root.rglob("*.jsonl")) if model_root.is_dir() else []
            if jsonls:
                inputs = [str(p.relative_to(self.project_dir)) for p in jsonls]
                div += ["--output-csv", f"outputs/{model}/diversity_summary.csv", "--input", *inputs]
                cov += ["--coverage-csv", f"outputs/{model}/coverage_summary.csv",
                        "--premise1-csv", f"outputs/{model}/premise1_summary.csv", "--input", *inputs]
        return [div, cov]

    def load_analysis(self, run_name: str | None = None) -> dict | None:
        outputs = self.project_dir / "outputs"
        if not outputs.exists():
            return None
        search_root = (outputs / run_name) if run_name else outputs
        if not search_root.exists():
            return None

        csvs = list(search_root.rglob("*.csv"))

        def _latest(keyword: str) -> Path | None:
            hits = [p for p in csvs if keyword in p.name.lower()]
            return max(hits, key=lambda p: p.stat().st_mtime) if hits else None

        div_csv = _latest("diversity")
        cov_csv = _latest("coverage")
        prem_csv = _latest("premise")
        anchor = div_csv or cov_csv or prem_csv
        if anchor is None:
            return None
        model_key = run_name or _model_from_root(anchor, outputs)
        files: dict[str, Any] = {}
        backend = ""
        if div_csv:
            rows = _read_csv(div_csv)
            backend = (rows[0].get("embedding_backend") if rows else "") or backend
            files[f"{model_key}/diversity"] = _identity_flattening_card(rows)
        if cov_csv or prem_csv:
            cov_rows = _read_csv(cov_csv) if cov_csv else []
            prem_rows = _read_csv(prem_csv) if prem_csv else []
            if not backend and cov_rows:
                backend = cov_rows[0].get("embedding_backend") or ""
            files[f"{model_key}/coverage_premise"] = _identity_coverage_premise_card(cov_rows, prem_rows)
        return {
            "run_dir": str(anchor.parent),
            "metrics": {"stats_backend": backend, "files": files},
            "conclusions_txt": "",
        }

    def load_transcript(self, offset: int = 0, limit: int = 50, job_info: Any | None = None,
                        only_failed: bool = False) -> dict | None:
        outputs = self.project_dir / "outputs"
        if not outputs.exists():
            return None
        target: Path | None = None
        cfg = job_config(job_info)
        output = nested_get(cfg, "output")
        if output:
            candidate = Path(str(output))
            if not candidate.is_absolute():
                candidate = self.project_dir / candidate
            if candidate.is_file():
                target = candidate
            else:
                return {"total": 0, "offset": offset, "limit": limit, "items": [], "failed_filter": True}
        elif job_info is not None:
            return None
        if target is None:
            files = list(outputs.rglob("*.jsonl"))
            if not files:
                return None
            target = max(files, key=lambda p: p.stat().st_mtime)
        rows = cached_by_mtime([target], lambda: read_jsonl(target), namespace="transcript")

        return build_transcript_page(
            rows, offset=offset, limit=limit, only_failed=only_failed,
            label_fn=lambda r: " · ".join(str(x) for x in (r.get("task_key"), r.get("identity_axis"), r.get("identity")) if x),
            prompt_fn=lambda r: r.get("user_prompt", ""),
            response_fn=lambda r: r.get("response") or r.get("error", ""),
            ok_fn=lambda r: non_error_text_ok(r.get("response")),
            reason_fn=lambda r: "error" if str(r.get("error") or "").strip() else "empty",
        )

def _text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _int_param(value: Any, default: int) -> int:
    parsed = opt_int(value)
    return default if parsed is None else parsed


def _float_param(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _parse_text_list(value: Any) -> list[str] | None:
    if value is None or value == "":
        return None
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value)
    delimiter = "\n" if "\n" in text else (";" if ";" in text else ",")
    normalized = text.replace("，", ",")
    return [part.strip() for part in normalized.split(delimiter) if part.strip()]


def _parse_int_list(value: Any) -> list[int] | None:
    items = _parse_text_list(value)
    if items is None:
        return None
    return [int(item) for item in items]


def _model_from_root(target: Path, root: Path) -> str:
    try:
        parts = target.relative_to(root).parts
    except ValueError:
        return "model"
    return parts[0] if parts else "model"


def _to_float(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _identity_flattening_card(rows: list[dict[str, Any]]) -> dict[str, Any]:
    agg: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        ident = (r.get("identity") or "?").strip()
        axis = (r.get("identity_axis") or "").strip()
        bucket = agg.setdefault((axis, ident), {
            "identity": ident, "axis": axis, "n_responses": 0,
            "_unique": [], "_cosine": [], "_trace": [], "_mc": [],
        })
        bucket["n_responses"] += int(_to_float(r.get("n_responses")) or 0)
        for col, key in (("unique_ngram_mean", "_unique"),
                         ("pairwise_cosine_distance_mean", "_cosine"),
                         ("embedding_covariance_trace", "_trace"),
                         ("mc_unique", "_mc")):
            val = _to_float(r.get(col))
            if val is not None:
                bucket[key].append(val)
    items: list[dict[str, Any]] = []
    for bucket in agg.values():
        items.append({
            "identity": bucket["identity"],
            "axis": bucket["axis"],
            "n_responses": bucket["n_responses"],
            "unique_ngram": _mean(bucket["_unique"]),
            "pairwise_cosine": _mean(bucket["_cosine"]),
            "cov_trace": _mean(bucket["_trace"]),
            "mc_unique": _mean(bucket["_mc"]),
        })
    items.sort(key=lambda x: (x["axis"], x["identity"]))
    return {"type": "identity_flattening", "n_rows": len(rows), "by_identity": items}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _append(acc: list[float], value: Any) -> None:
    v = _to_float(value)
    if v is not None:
        acc.append(v)


def _identity_coverage_premise_card(cov_rows: list[dict[str, Any]],
                                    prem_rows: list[dict[str, Any]]) -> dict[str, Any]:
    cov_axis: dict[str, dict[str, Any]] = {}
    for r in cov_rows:
        axis = (r.get("identity_axis") or "").strip()
        d = cov_axis.setdefault(axis, {"axis": axis, "_vendi": [], "_det": [], "_mc": [], "n_identities": 0})
        _append(d["_vendi"], r.get("vendi_point"))
        _append(d["_det"], r.get("det_point"))
        _append(d["_mc"], r.get("mc_unique"))
        ni = _to_float(r.get("n_identities"))
        if ni is not None:
            d["n_identities"] = max(d["n_identities"], int(ni))
    coverage = sorted(
        ({"axis": d["axis"], "n_identities": d["n_identities"],
          "vendi": _mean(d["_vendi"]), "det": _mean(d["_det"]),
          "mc_unique": _mean(d["_mc"])} for d in cov_axis.values()),
        key=lambda x: x["axis"],
    )

    prem_axis: dict[str, dict[str, Any]] = {}
    for r in prem_rows:
        axis = (r.get("identity_axis") or "").strip()
        d = prem_axis.setdefault(axis, {"axis": axis, "_w": [], "_a": [], "_wp": [], "_ap": [],
                                        "n_pairs": 0, "n_sig": 0, "n_wsig": 0, "n_chi": 0, "n_chisig": 0})
        _append(d["_w"], r.get("mean_within"))
        _append(d["_a"], r.get("mean_across"))
        _append(d["_wp"], r.get("mean_within_persample"))
        _append(d["_ap"], r.get("mean_across_persample"))
        d["n_pairs"] += 1
        p = _to_float(r.get("p_value"))
        if p is not None and p < 0.05:
            d["n_sig"] += 1
        wp = _to_float(r.get("wilcoxon_p"))
        if wp is not None and wp < 0.05:
            d["n_wsig"] += 1
        chip = _to_float(r.get("chisq_p"))
        if chip is not None:
            d["n_chi"] += 1
            if chip < 0.05:
                d["n_chisig"] += 1
    premise1 = sorted(
        ({"axis": d["axis"], "n_pairs": d["n_pairs"],
          "mean_within": _mean(d["_w"]), "mean_across": _mean(d["_a"]),
          "frac_sig": (d["n_sig"] / d["n_pairs"]) if d["n_pairs"] else None,
          "mean_within_persample": _mean(d["_wp"]), "mean_across_persample": _mean(d["_ap"]),
          "frac_wilcoxon_sig": (d["n_wsig"] / d["n_pairs"]) if d["n_pairs"] else None,
          "frac_chisq_sig": (d["n_chisig"] / d["n_chi"]) if d["n_chi"] else None,
          "n_chi": d["n_chi"]}
         for d in prem_axis.values()),
        key=lambda x: x["axis"],
    )

    return {
        "type": "identity_coverage_premise",
        "coverage": {"n_rows": len(cov_rows), "by_axis": coverage},
        "premise1": {"n_pairs": len(prem_rows), "by_axis": premise1},
    }
