from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..schemas import ExperimentInfo, ParamField, UnifiedLLM
from .base import (
    ALL_PROVIDER_KINDS,
    Adapter,
    JobUnit,
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

_CONDITIONS = ["control", "control_gender", "treatment"]


_S14_ID = "s14_temperature"
_S14_TEMPS = [0.3, 1.7]

_PRESETS: dict[str, dict[str, int]] = {
    "default": {"num_occupations": 54, "treatment_names_per_gender": 5, "control_repeats": 10, "control_gender_repeats": 10, "treatment_repeats": 1},
    "small": {"num_occupations": 54, "treatment_names_per_gender": 4, "control_repeats": 10, "control_gender_repeats": 10, "treatment_repeats": 5},
    "medium": {"num_occupations": 54, "treatment_names_per_gender": 8, "control_repeats": 20, "control_gender_repeats": 20, "treatment_repeats": 10},
    "full": {"num_occupations": 54, "treatment_names_per_gender": 16, "control_repeats": 50, "control_gender_repeats": 50, "treatment_repeats": 20},
}

_CUSTOM_DEFAULT = _PRESETS["full"]


_EXP_CONCLUSIONS = {
    "control": (
        "**作用**：基线条件——不指定姓名或性别，为每个职业生成 50 份简历（54 职业 × 50 = 2,700 份），"
        "用作衡量『引入性别线索后偏移多少』的对照基准。\n"
        "\n"
        "**发现**：相比 control（无性别 / 无姓名），treatment 中给女性申请人生成的简历更年轻、经验更少，"
        "给男性的更年长、经验更多（差异均 P<0.00001）——一旦引入性别线索，ChatGPT 就按"
        "『男性更年长、更有经验』的刻板模式偏移。"
    ),
    "control_gender": (
        "**作用**：复制 / 稳健性条件——同 control，但额外要求 ChatGPT 自己给出申请人姓名与性别（共 2,500 份），"
        "用于排除『偏差只是 treatment 所用特定姓名或 prompt 设计造成』的可能。\n"
        "\n"
        "**发现**：当 ChatGPT 自行生成男性申请人时（相比自行生成女性，控制职业，n=2,500）：\n"
        "\n"
        "| 维度 | 男性 vs 女性 | 统计量 |\n"
        "| --- | --- | --- |\n"
        "| 年龄 | 大 1.3 岁 | t=17.3, P=2.2×10⁻¹⁶ |\n"
        "| 毕业时间 | 早 1.2 年（更久） | t=7.10, P=2.29×10⁻¹² |\n"
        "\n"
        "年龄-性别偏差**不依赖具体姓名 / prompt**，模型自发产生。"
    ),
    "treatment": (
        "**核心条件**（Fig.4，三大假设的主检验）：用 16 个去偏男女姓名，为每个 姓名 × 性别 × 职业 组合各生成 "
        "20 份简历（共 34,560 份、每性别 17,280 份），再让 ChatGPT 给每份简历 1–100 评分。\n"
        "\n"
        "**H1 性别 × 简历内容**：女性姓名简历相比男性姓名（控制姓名与职业，n=34,560）：\n"
        "\n"
        "| 维度 | 女性 vs 男性 | 统计量 |\n"
        "| --- | --- | --- |\n"
        "| 年龄 | 小 1.6 岁 | t=20.5, P=7.09×10⁻⁹³ |\n"
        "| 毕业时间 | 晚 1.3 年（更近） | t=12.5, P=1.18×10⁻³⁵ |\n"
        "| 相关经验 | 少 0.92 年 | t=5.39, P=6.97×10⁻⁸ |\n"
        "\n"
        "**H2 年龄 → 评分**：评分与生成的申请人年龄正相关（r=0.27, P=2.2×10⁻¹⁶, n=39,560；回归 β[age]=0.04, "
        "P=2.96×10⁻¹⁰）——年龄越大评分越高。\n"
        "\n"
        "**H3 年龄 × 性别**：male × age 交互显著为正（β=0.04, t=6.61, P=3.66×10⁻¹¹）——年龄带来的加分在男性身上更大。\n"
        "\n"
        "**结论**：ChatGPT 假定女性更年轻、经验更少，并给『更年长的男性』申请人更高评分（对 temperature 稳健，SI Fig.20）。"
    ),
}


_EXP_IMPORTANCE: dict[str, tuple[int, str, str]] = {
    "treatment": (
        1, "S",
        "论文 Fig.4 的核心主检验，三大假设的唯一直接载体。用 16 个在流行度/族裔/感知年龄上去偏的"
        "男女姓名，为每个 姓名 × 性别 × 职业 组合生成简历（34,560 份）再让模型 1–100 评分，把"
        "『年龄-性别扭曲』落到可统计的因果设计上：H1 女性姓名简历更年轻（−1.6 岁, P≈7e-93）、"
        "毕业更晚、相关经验更少；H2 评分随设定年龄升高（r=0.27）；H3 male×age 正交互"
        "（β=0.04, P≈3.7e-11），即更偏好『更年长的男性』。缺它则 Fig.4 全部结论无从谈起，故列 S 级第一。"
    ),
    "control_gender": (
        2, "A",
        "关键稳健性 / 复制条件，承担排除混淆的因果论证。设计同 control，但要求模型自行给出申请人"
        "姓名与性别——当模型『自发』生成性别时，男性仍比女性年长约 1.3 岁（t=17.3）、毕业早约 1.2 年"
        "（t=7.10, n=2,500）。这证明年龄-性别偏差是模型自发产生、不依赖 treatment 所用的特定姓名或 "
        "prompt 设计，是 treatment 结论可信度的主要支撑。因其支撑而非独立承载主结论，故列 A 级。"
    ),
    "control": (
        3, "B",
        "中性对照基线。不指定姓名/性别，仅为每个职业生成简历，用作『引入性别线索前』的锚点，"
        "衡量 treatment / control_gender 相对它偏移多少。它本身不含性别对比、无法单独得出年龄-性别"
        "扭曲结论，价值在于提供可比基准，故列 B 级、排第三。"
    ),
}


class AgeGenderDistortionAdapter(Adapter):
    id = "age_gender_distortion"
    name = "年龄-性别扭曲 (Age/Gender Distortion)"
    paper = "Age and gender distortion in online media and large language models"
    description = "复现 Nature 原文 Fig.4 的 ChatGPT 简历审计实验。"
    intro = (
        "论文《Age and gender distortion in online media and large language models》"
        "(Guilbeault, Delecourt & Desikan, 2025, Nature) 是一篇多方法研究，用 6 块实证共同论证"
        "『女性被普遍表征得比男性年轻』是一种贯穿互联网及其算法、且偏离现实(US Census)的系统性扭曲："
        "Census 基线、约 140 万张网络图像/视频(Fig.1)、9 个语言模型(Fig.2)、地位/收入分析、"
        "人类预注册实验(Fig.3, n=459)、ChatGPT 简历审计(Fig.4)。\n\n"
        "本项目只复现其中的 Fig.4：让模型为不同职业生成简历，再让同一模型给简历打分，"
        "分析年龄、性别与评分之间的扭曲关系。UI 中的三个实验即 Fig.4 的三个条件："
        "control(不含姓名/性别)、control_gender(要求模型给出性别)、treatment(指定去偏姓名)。\n\n"
        "复现范围说明：Fig.4 能独立得出『LLM 存在年龄-性别偏差——假定女性更年轻、经验更少，"
        "并给更年长的男性更高评分』这一子结论，它是标题中 large language models 那一半的直接证据，"
        "也是 6 块实证里最适合用 LLM API 复现的一块。但文章『在线媒体 + 算法系统性扭曲』的总论点"
        "还依赖 Census、图像分析、9 个语言模型、人类实验等本项目未复现的部分——单凭 Fig.4 不能完整得出。"
    )
    supported_kinds = ALL_PROVIDER_KINDS
    selection_mode = "multi"
    allow_concurrent = True
    paper_baseline = (
        "**论文**：Guilbeault, Delecourt & Srinivasa Desikan《Age and gender distortion in online "
        "media and large language models》（Nature, 2025）。本项目复现其 Fig.4「ChatGPT 简历审计」。\n"
        "**被测**：GPT-4o mini；54 个职业 × 16 个（按流行度/族裔/感知年龄归一化的）男女姓名，"
        "生成近 4 万份简历（treatment 34,560 份），分 control / control-gender / treatment 三条件；"
        "再让模型给每份简历 1–100 评分（结论对 temperature 稳健，SI Fig.20）。\n"
        "\n"
        "**核心结论（3 个假设）**\n"
        "\n"
        "| 假设 | 内容（原文 Fig.4 数据） |\n"
        "| --- | --- |\n"
        "| H1 性别 × 简历内容 | 同职业下，给『女性姓名』生成的简历比『男性姓名』年龄更小（−1.6 岁, "
        "P≈7e-93）、毕业更晚（−1.3 年）、相关经验更少（−0.92 年）——即男性被设定为更年长、经验更多。 |\n"
        "| H2 年龄 → 评分 | 模型对简历的评分与其设定的申请人年龄正相关（r=0.27, P≈2e-16；回归 β_age=0.04）"
        "——年龄越大评分越高。 |\n"
        "| H3 年龄 × 性别交互 | 『male × age』交互显著为正（β=0.04, P≈3.7e-11）——年龄带来的加分在男性"
        "申请人身上更大（即偏好『更年长的男性』）。 |\n"
        "\n"
        "**背景**：该结论是全文的一环——论文还发现女性在约 140 万张图片/视频与 9 个语言模型中被普遍"
        "表征得比男性年轻，高地位/高薪职业尤甚。"
    )


    def list_experiments(self) -> list[ExperimentInfo]:
        return [
            ExperimentInfo(
                id="control",
                label="控制条件 control",
                description="不指定姓名或性别，只要求模型为某职业生成一份简历。",
                conclusion=_EXP_CONCLUSIONS["control"],
                extra_params=[
                    ParamField(name="control_repeats", label="control 重复次数", type="int", default=None, minimum=1,
                               help="每个职业重复生成多少份 control 简历。原文 full 为 50（54 职业 → control 共 2,700 份）。填写后会自动使用 custom 规模。"),
                ],
                importance_rank=_EXP_IMPORTANCE["control"][0],
                importance_tier=_EXP_IMPORTANCE["control"][1],
                importance_basis=_EXP_IMPORTANCE["control"][2],
            ),
            ExperimentInfo(
                id="control_gender",
                label="控制+性别条件 control_gender",
                description="不指定姓名，但要求简历包含申请人性别。",
                conclusion=_EXP_CONCLUSIONS["control_gender"],
                extra_params=[
                    ParamField(name="control_gender_repeats", label="control_gender 重复次数", type="int", default=None, minimum=1,
                               help="每个职业重复生成多少份 control_gender 简历。原文设计与 control 相同、额外要模型给出性别，full 为 50（生成 2,700 份；原文统计分析有效 2,500 份，排除未能明确性别者）。填写后会自动使用 custom 规模。"),
                ],
                importance_rank=_EXP_IMPORTANCE["control_gender"][0],
                importance_tier=_EXP_IMPORTANCE["control_gender"][1],
                importance_basis=_EXP_IMPORTANCE["control_gender"][2],
            ),
            ExperimentInfo(
                id="treatment",
                label="姓名处理条件 treatment",
                description="指定原文使用的去偏姓名，为每个 姓名 × 职业 组合生成简历。",
                conclusion=_EXP_CONCLUSIONS["treatment"],
                extra_params=[
                    ParamField(name="treatment_repeats", label="treatment 重复次数", type="int", default=None, minimum=1,
                               help="每个 姓名 × 职业 组合重复生成多少份简历。原文 full 为 20（54 职业 × 16 姓名 × 2 性别 × 20 → treatment 共 34,560 份）。填写后会自动使用 custom 规模。"),
                ],
                importance_rank=_EXP_IMPORTANCE["treatment"][0],
                importance_tier=_EXP_IMPORTANCE["treatment"][1],
                importance_basis=_EXP_IMPORTANCE["treatment"][2],
            ),
            ExperimentInfo(
                id=_S14_ID,
                label="温度稳健性（SI Fig.20）· 自动跑 T=0.3 与 1.7",
                description="原文 SI 的温度稳健性检验：主实验固定 temperature=0.7，SI 在 {0.3, 0.7, 1.7} 下复测结论。"
                            "选中后**自动创建 2 个任务**（T=0.3、T=1.7 各一，三条件齐跑、默认规模 1,620 份 + 评分），"
                            "T=0.7 基线复用主实验既有数据。每温度约 3,240 次调用/模型。"
                            "（注：SI 还换了一版提示词做稳健性，管线暂不支持提示词变体，此卡只覆盖温度部分。）",
                conclusion="**原文结论（SI Fig.20）**：生成与评分阶段的年龄–性别扭曲在 {0.3, 0.7, 1.7} 三档温度下方向与显著性一致——"
                           "主结论不依赖采样温度。",
                importance_rank=4, importance_tier="B",
                importance_basis="【SI 稳健性检验，非主结论】主结论由三条件主实验承载；此卡检验主结论是否依赖采样温度"
                                 "（原文 SI Fig.20 的温度部分）。参数写死，选中即得 2 个并行任务。",
            ),
        ]

    def param_schema(self) -> list[ParamField]:
        return [
            ParamField(name="temperature", label="采样温度", type="float", default=0.7, minimum=0,
                       paper_value=0.7,
                       help="原文主实验固定 temperature=0.7（Nature Fig.4 图注明确：set to its default value of 0.7；模型 gpt-4o-mini），覆盖统一配置。"
                            "SI(Supplementary Fig.20) 验证结论在 {0.3, 0.7, 1.7} 下稳健。"
                            "注意：Kimi K2.5 非思考模式此前用 0.6，如所选模型不接受 0.7 可改回。"),
            ParamField(name="max_tokens", label="最大输出 token", type="int", default=None, minimum=1,
                       help="单次回复最大生成 token。留空=沿用统一配置的最大输出 token；填写则覆盖。"),
            ParamField(name="top_p", label="核采样阈值", type="float", default=None, minimum=0,
                       help="nucleus sampling 截断。留空=不下发、沿用服务端默认；如需对齐原文默认可填 1.0。"),
            ParamField(name="scale_preset", label="实验规模 scale_preset", type="select", default="default",
                       options=["default", "small", "medium", "full", "custom"],
                       help="【默认 default】每个条件约 500 份：control 540 + control_gender 540 + treatment 540 ≈ 1,620 份，"
                            "再各评 1 次分，合计约 3,240 次调用——够看出三大假设的方向，费用可控。"
                            "small≈6,480 次；medium≈21,600 次；full≈79,920 次。"
                            "【full = 严格复现原文】Guilbeault et al. 2025 (Nature) Fig.4 的完整配置："
                            "54 职业 × 16 个去偏姓名/性别 × 重复(control 50 / control_gender 50 / treatment 20)，"
                            "= control 2,700 + control_gender 2,700 + treatment 34,560 ≈ 近 4 万份简历，再各评 1 次分；"
                            "要原样复现原文请选 full。custom = 用下方自定义规模(职业数 / 姓名数 / 各条件重复次数)精确控制。"),
            ParamField(name="retry_base_delay", label="重试退避基数", type="float", default=2.0, minimum=0,
                       help="API 失败后的指数退避基数；重试次数使用左侧统一配置。"),
        ]


    def plan_jobs(self, experiment_ids: list[str], params: dict[str, Any]) -> list[JobUnit]:
        selected = list(experiment_ids or _CONDITIONS)
        units: list[JobUnit] = []
        if _S14_ID in selected:
            selected.remove(_S14_ID)
            for t in _S14_TEMPS:
                units.append(JobUnit(
                    experiment_id=f"{_S14_ID}:{t}",
                    label=f"Age/Gender: 温度稳健性 (SI) · T={t}",
                    argv=["run.py", "--config", "config.yaml"],
                    selected=list(_CONDITIONS),
                    extra={"variant_temperature": t},
                ))
        unknown = [x for x in selected if x not in _CONDITIONS]
        if unknown:
            raise ValueError(f"未知实验条件：{unknown}；可选：{_CONDITIONS}")
        labels = {e.id: e.label for e in self.list_experiments()}
        units.extend(
            JobUnit(
                experiment_id=cid,
                label=f"Age/Gender: {labels.get(cid, cid)}",
                argv=["run.py", "--config", "config.yaml"],
                selected=[cid],
            )
            for cid in selected
        )
        return units

    def render_config(self, llm: UnifiedLLM, params: dict[str, Any], unit: JobUnit,
                      work_dir: Path | None = None) -> None:
        self.check_kind(llm)
        proxy = llm_proxy_config(llm)
        unit.extra["llm_proxy_token"] = proxy["api_key"]
        path = self.project_dir / "config.yaml"
        data = load_yaml(path)

        model = data.setdefault("model", {})
        model["provider"] = "openai_compatible"
        model["model_name"] = proxy["model"]
        model["api_key"] = proxy["api_key"]
        model["api_key_env"] = ""
        model["base_url"] = proxy["base_url"]
        tval = params.get("temperature")
        if tval not in (None, ""):
            model["temperature"] = float(tval)
        vt = unit.extra.get("variant_temperature")
        if vt is not None:
            model["temperature"] = float(vt)
        mtok = opt_int(params.get("max_tokens"))
        model["max_tokens"] = mtok if mtok is not None else llm.max_tokens
        tp = params.get("top_p")
        if tp in (None, ""):
            model.pop("top_p", None)
        else:
            model["top_p"] = float(tp)
        model["request_timeout"] = llm.timeout
        model["max_retries"] = llm.max_retries
        model["retry_base_delay"] = _float_param(params.get("retry_base_delay"), 2.0)
        model["system_prompt"] = llm.system
        model["extra_body"] = _extra_body_for(model.get("extra_body"), llm)

        exp = data.setdefault("experiment", {})
        preset = str(params.get("scale_preset") or "default").lower()
        custom = _custom_values(params, unit.selected, data)
        if _has_custom_override(params, unit.selected):
            preset = "custom"
        exp["scale_preset"] = preset
        exp["custom"] = custom
        exp["conditions"] = [c for c in _CONDITIONS if c in unit.selected]
        exp["evaluate_scores"] = True
        exp["concurrency"] = llm.concurrency

        out = data.setdefault("output", {})
        out_dir = out.get("dir") or "results"
        if work_dir is not None:

            fp = result_fingerprint({
                "scale_preset": exp.get("scale_preset"),
                "custom": exp.get("custom"),
                "conditions": exp.get("conditions"),
                "evaluate_scores": exp.get("evaluate_scores"),
                "temperature": model.get("temperature"),
                "max_tokens": model.get("max_tokens"),
                "top_p": model.get("top_p"),
                "model": model.get("model_name"),
            })
            results_root = self.project_dir / out_dir
            run_dir = remembered_run_dir(unit, results_root / model_slug(llm.model), fp)
            out["run_name"] = str(run_dir.relative_to(results_root))
            cfg = Path(work_dir) / "age_gender_distortion_config.yaml"
            dump_yaml(cfg, data)
            unit.extra["config_path"] = str(cfg)
        else:
            dump_yaml(path, data)


    def results_dir(self) -> Path:
        return self.project_dir / "results"

    def results_dir_for_job(self, job_info: Any | None = None) -> Path:
        cfg = job_config(job_info)
        run_name = nested_get(cfg, "output", "run_name")
        if run_name:
            out_dir = nested_get(cfg, "output", "dir") or "results"
            root = Path(str(out_dir))
            if not root.is_absolute():
                root = self.project_dir / root
            return root / str(run_name)
        if job_info is not None:
            root = self.results_dir() / model_slug(getattr(job_info, "model", ""))
            runs = [p for p in root.glob("*") if p.is_dir()]
            if runs:
                return max(runs, key=lambda p: p.stat().st_mtime)
        return self.results_dir()

    def sample_stats(self, job_info: Any | None = None) -> dict | None:
        cfg = job_config(job_info)
        run_name = nested_get(cfg, "output", "run_name")
        if not run_name:
            return None
        out_dir = nested_get(cfg, "output", "dir") or "results"
        root = Path(str(out_dir))
        if not root.is_absolute():
            root = self.project_dir / root
        raw = root / str(run_name) / "resumes_raw.jsonl"

        return count_samples_by_key(
            [raw],
            lambda r: r.get("id"),
            lambda r: non_error_text_ok(r.get("resume_text")) and not str(r.get("error") or "").strip(),
        )


    def analyze_argv(self, model: str | None = None) -> list[str]:
        base = ["run.py", "--config", "config.yaml", "--stage", "analyze"]
        if not model:
            return base
        model_root = self.project_dir / "results" / model
        runs = [
            p for p in model_root.iterdir()
            if p.is_dir() and (p / "resumes_parsed.csv").is_file()
        ] if model_root.is_dir() else []
        if runs:
            latest = max(runs, key=lambda p: p.stat().st_mtime)
            return base + ["--run-dir", str(latest.relative_to(self.project_dir))]
        return base

    def load_analysis(self, run_name: str | None = None) -> dict | None:
        results = self.project_dir / "results"
        if not results.exists():
            return None


        search_root = (results / run_name) if run_name else results
        summaries = list(search_root.rglob("analysis_summary.json")) if search_root.exists() else []
        if not summaries:
            return None
        summary = max(summaries, key=lambda p: p.stat().st_mtime)
        run_dir = summary.parent
        try:
            s = json.loads(summary.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        concl = run_dir / "conclusions.txt"
        model_key = run_name or _model_from_root(run_dir, results)
        card = {
            "type": "age_gender_distortion",
            "n_total_resumes": s.get("n_total_resumes"),
            "n_by_condition": s.get("n_by_condition") or {},
            "n_scored": s.get("n_scored"),
            "fig4a": s.get("fig4a_treatment_gender_effect") or {},
            "control_gender": s.get("control_gender_gender_effect") or {},
            "fig4b": s.get("fig4b_age_score_corr") or {},
            "fig4c": s.get("fig4c_age_gender_interaction") or {},
        }
        return {
            "run_dir": str(run_dir),
            "metrics": {"files": {f"{model_key}/fig4": card}},
            "conclusions_txt": concl.read_text(encoding="utf-8") if concl.is_file() else "",
        }

    def load_transcript(self, offset: int = 0, limit: int = 50, job_info: Any | None = None,
                        only_failed: bool = False) -> dict | None:
        results = self.project_dir / "results"
        if not results.exists():
            return None
        target: Path | None = None
        cfg = job_config(job_info)
        run_name = nested_get(cfg, "output", "run_name")
        if run_name:
            out_dir = nested_get(cfg, "output", "dir") or "results"
            root = Path(str(out_dir))
            if not root.is_absolute():
                root = self.project_dir / root
            candidate = root / str(run_name) / "resumes_raw.jsonl"
            if candidate.is_file():
                target = candidate
            else:
                return {"total": 0, "offset": offset, "limit": limit, "items": [], "failed_filter": True}
        elif job_info is not None:
            return None
        if target is None:
            files = list(results.rglob("resumes_raw.jsonl"))
            if not files:
                return None
            target = max(files, key=lambda p: p.stat().st_mtime)
        rows = cached_by_mtime([target], lambda: read_jsonl(target), namespace="transcript")


        def _ok(r: dict[str, Any]) -> bool:
            return non_error_text_ok(r.get("resume_text")) and not str(r.get("error") or "").strip()

        return build_transcript_page(
            rows, offset=offset, limit=limit, only_failed=only_failed,
            label_fn=lambda r: " · ".join(str(x) for x in (r.get("condition"), r.get("occupation"), r.get("name")) if x) or str(r.get("id", "")),
            prompt_fn=lambda r: r.get("prompt", ""),
            response_fn=lambda r: r.get("resume_text") or r.get("error", ""),
            ok_fn=_ok,
            reason_fn=lambda r: "error" if str(r.get("error") or "").strip() else "empty",
        )

def _extra_body_for(existing: Any, llm: UnifiedLLM) -> dict:
    marker = f"{llm.model} {llm.base_url}".lower()
    if "kimi" in marker or "moonshot" in marker:
        return existing if isinstance(existing, dict) else {}
    return {}


def _float_param(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _has_custom_override(params: dict[str, Any], selected: list[str]) -> bool:
    extra_names = ["control_repeats", "control_gender_repeats", "treatment_repeats"]
    return any(exp_param(params, selected, name) not in (None, "") for name in extra_names)


def _custom_values(params: dict[str, Any], selected: list[str], data: dict[str, Any]) -> dict[str, int]:
    current = ((data.get("experiment") or {}).get("custom") or {}) if isinstance(data, dict) else {}
    base = {key: int(current.get(key, value)) for key, value in _CUSTOM_DEFAULT.items()}
    for key in ("control_repeats", "control_gender_repeats", "treatment_repeats"):
        value = opt_int(exp_param(params, selected, key))
        if value is not None:
            base[key] = value
    return base


def _model_from_root(run_dir: Path, root: Path) -> str:
    try:
        parts = run_dir.relative_to(root).parts
    except ValueError:
        return "model"
    return parts[0] if parts else "model"
