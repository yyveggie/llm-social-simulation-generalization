from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

try:
    import statsmodels.formula.api as smf
    _HAS_SM = True
except Exception:
    _HAS_SM = False


def _coef(fit, term: str) -> dict | None:
    if term in fit.params.index:
        return {
            "estimate": float(fit.params[term]),
            "std_error": float(fit.bse[term]),
            "t": float(fit.tvalues[term]),
            "p_value": float(fit.pvalues[term]),
            "n": int(fit.nobs),
        }
    return None


def _ttest(a: pd.Series, b: pd.Series) -> dict | None:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return None
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {
        "mean_female": float(a.mean()), "n_female": int(len(a)),
        "mean_male": float(b.mean()), "n_male": int(len(b)),
        "diff_male_minus_female": float(b.mean() - a.mean()),
        "t": float(t), "p_value": float(p),
    }


def _pearson(x: pd.Series, y: pd.Series) -> dict | None:
    d = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"),
                      "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(d) < 3:
        return None
    r, p = stats.pearsonr(d["x"], d["y"])
    return {"r": float(r), "p_value": float(p), "n": int(len(d))}


def _ols_male_coef(df: pd.DataFrame, outcome: str, *, use_name: bool = True) -> dict | None:
    if not _HAS_SM:
        return None
    d = df.dropna(subset=[outcome, "male"]).copy()
    if d["male"].nunique() < 2 or len(d) < 5:
        return None
    rhs = "male"
    if d["occupation"].nunique() > 1:
        rhs += " + C(occupation)"
    if use_name and d["name"].nunique() > 1:
        rhs += " + C(name)"
    try:
        fit = smf.ols(f"{outcome} ~ {rhs}", data=d).fit()
        return _coef(fit, "male")
    except Exception:
        return None


def _ols_score_controlled(df: pd.DataFrame) -> dict | None:
    if not _HAS_SM:
        return None
    d = df.dropna(subset=["score", "male", "applicant_age", "total_experience"]).copy()
    if d["male"].nunique() < 2 or len(d) < 5:
        return None
    rhs = "male + applicant_age + total_experience"
    if d["occupation"].nunique() > 1:
        rhs += " + C(occupation)"
    if d["name"].nunique() > 1:
        rhs += " + C(name)"
    try:
        fit = smf.ols(f"score ~ {rhs}", data=d).fit()
        return {
            "age_coef": _coef(fit, "applicant_age"),
            "male_coef": _coef(fit, "male"),
            "experience_coef": _coef(fit, "total_experience"),
        }
    except Exception:
        return None


def _ols_interaction(df: pd.DataFrame) -> dict | None:
    if not _HAS_SM:
        return None
    d = df.dropna(subset=["score", "male", "applicant_age"]).copy()
    if d["male"].nunique() < 2 or len(d) < 5:
        return None
    rhs = "male * applicant_age"
    if d["occupation"].nunique() > 1:
        rhs += " + C(occupation)"
    if d["name"].nunique() > 1:
        rhs += " + C(name)"
    try:
        fit = smf.ols(f"score ~ {rhs}", data=d).fit()
        return {
            "interaction_male_x_age": _coef(fit, "male:applicant_age"),
            "male_main": _coef(fit, "male"),
            "age_main": _coef(fit, "applicant_age"),
        }
    except Exception:
        return None


def _load_warning(run_dir: str, file: str, reason: str) -> dict:
    return {"run_dir": run_dir, "file": file, "reason": reason}


def _read_csv_if_ready(path: str, required_cols: list[str] | None = None) -> tuple[pd.DataFrame | None, str | None]:
    if not os.path.exists(path):
        return None, "missing"
    try:
        if os.path.getsize(path) == 0:
            return None, "empty"
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None, "empty"
    except (OSError, pd.errors.ParserError) as exc:
        return None, f"unreadable: {type(exc).__name__}: {exc}"

    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return None, f"missing columns: {', '.join(missing)}"
    return df, None


def _read_one_run(d: str) -> tuple[pd.DataFrame | None, list[dict]]:
    parsed_path = os.path.join(d, "resumes_parsed.csv")
    warnings: list[dict] = []
    df, reason = _read_csv_if_ready(parsed_path, required_cols=["id", "condition", "gender"])
    if df is None:
        warnings.append(_load_warning(d, "resumes_parsed.csv", reason or "not ready"))
        return None, warnings

    scores_path = os.path.join(d, "scores.csv")
    sc, reason = _read_csv_if_ready(scores_path, required_cols=["id", "score"])
    if sc is not None:
        sc = sc[["id", "score"]]
        df = df.merge(sc, on="id", how="left")
    else:
        df["score"] = np.nan
        if reason:
            warnings.append(_load_warning(d, "scores.csv", reason))
    return df, warnings


def _detect_condition(parsed_path: str) -> str | None:
    try:
        col = pd.read_csv(parsed_path, usecols=["condition"], nrows=500)
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return None
    vals = col["condition"].dropna().unique()
    return str(vals[0]) if len(vals) else None


def load_dataset(run_dir: str) -> pd.DataFrame:
    root = os.path.dirname(run_dir)

    latest_by_cond: dict[str, tuple[float, str]] = {}
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            full = os.path.join(root, name)
            parsed = os.path.join(full, "resumes_parsed.csv")
            if not (os.path.isdir(full) and os.path.exists(parsed)):
                continue
            cond = _detect_condition(parsed)
            if not cond:
                continue
            mt = os.path.getmtime(parsed)
            if cond not in latest_by_cond or mt > latest_by_cond[cond][0]:
                latest_by_cond[cond] = (mt, full)
    dirs = [d for _, d in latest_by_cond.values()]

    if run_dir not in dirs:
        dirs.append(run_dir)

    frames: list[pd.DataFrame] = []
    load_warnings: list[dict] = []
    for d in dirs:
        df_part, warnings = _read_one_run(d)
        load_warnings.extend(warnings)
        if df_part is not None:
            frames.append(df_part)
    if not frames:
        raise FileNotFoundError(f"找不到 {os.path.join(run_dir, 'resumes_parsed.csv')}, 请先运行 generate (含解析) 阶段")
    df = pd.concat(frames, ignore_index=True)
    if "id" in df.columns:
        df = df.drop_duplicates(subset="id", keep="last")
    df["male"] = df["gender"].map({"male": 1.0, "female": 0.0})
    df.attrs["input_run_dirs"] = dirs
    df.attrs["load_warnings"] = load_warnings
    return df


def _collect_sampling_meta(dirs: list[str]) -> dict:
    out: dict = {}
    for d in dirs:
        path = os.path.join(d, "sampling_meta.json")
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                out[os.path.basename(d)] = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
    return out


def _forced_param_notes(sampling_meta: dict) -> list[str]:
    notes: list[str] = []
    for run_name, stages in (sampling_meta or {}).items():
        for stage, meta in (stages or {}).items():
            forced = (meta or {}).get("forced_by_model") or {}
            if forced:
                pairs = ", ".join(f"{k}={v}" for k, v in forced.items())
                notes.append(f"  - {run_name}/{stage}: 模型强制 {pairs}（偏离原文 temperature=0.7 口径）")
    return notes


def analyze(run_dir: str) -> dict:
    df = load_dataset(run_dir)
    df.to_csv(os.path.join(run_dir, "analysis_dataset.csv"), index=False)

    treat = df[(df["condition"] == "treatment") & df["gender"].isin(["male", "female"])].copy()
    fem = treat[treat["gender"] == "female"]
    mal = treat[treat["gender"] == "male"]


    cg = df[(df["condition"] == "control_gender") & df["gender"].isin(["male", "female"])].copy()
    cg_fem = cg[cg["gender"] == "female"]
    cg_mal = cg[cg["gender"] == "male"]

    outcomes = ["applicant_age", "total_experience", "years_since_grad", "num_skills"]

    summary: dict = {
        "run_dir": run_dir,
        "input_run_dirs": df.attrs.get("input_run_dirs", [run_dir]),
        "analysis_complete": not bool(df.attrs.get("load_warnings")),
        "load_warnings": df.attrs.get("load_warnings", []),
        "sampling_meta": _collect_sampling_meta(df.attrs.get("input_run_dirs", [run_dir])),
        "n_total_resumes": int(len(df)),
        "n_by_condition": df["condition"].value_counts().to_dict(),
        "n_scored": int(df["score"].notna().sum()),
        "fig4a_treatment_gender_effect": {
            "ttest": {o: _ttest(fem[o], mal[o]) for o in outcomes},
            "ols_male_coef": {o: _ols_male_coef(treat, o) for o in outcomes},
        },
        "control_gender_gender_effect": {
            "n": int(len(cg)),
            "n_female": int(len(cg_fem)),
            "n_male": int(len(cg_mal)),
            "ttest": {o: _ttest(cg_fem[o], cg_mal[o]) for o in outcomes},

            "ols_male_coef": {o: _ols_male_coef(cg, o, use_name=False) for o in outcomes},
        },


        "si_feature_correlations": {
            "age_vs_years_since_grad": _pearson(df["applicant_age"], df["years_since_grad"]),
            "age_vs_total_experience": _pearson(df["applicant_age"], df["total_experience"]),
            "years_since_grad_vs_total_experience": _pearson(df["years_since_grad"], df["total_experience"]),
        },
        "fig4b_age_score_corr": {
            "all": _pearson(df["applicant_age"], df["score"]),
            "treatment": _pearson(treat["applicant_age"], treat["score"]),
            "experience_all": _pearson(df["total_experience"], df["score"]),
            "experience_treatment": _pearson(treat["total_experience"], treat["score"]),
            "grad_all": _pearson(df["years_since_grad"], df["score"]),
            "grad_treatment": _pearson(treat["years_since_grad"], treat["score"]),
            "controlled_ols": _ols_score_controlled(df),
        },
        "fig4c_age_gender_interaction": _ols_interaction(treat),
    }

    with open(os.path.join(run_dir, "analysis_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    conclusions = build_conclusions(summary)
    with open(os.path.join(run_dir, "conclusions.txt"), "w", encoding="utf-8") as f:
        f.write(conclusions)

    print("\n" + conclusions)
    print(f"[analyze] 已写出: analysis_dataset.csv / analysis_summary.json / conclusions.txt -> {run_dir}")
    return summary


def _verdict(estimate: float | None, p_value: float | None, expect_positive: bool = True) -> str:
    if estimate is None or p_value is None:
        return "数据不足/无法估计"
    direction = "正向(>0)" if estimate > 0 else ("负向(<0)" if estimate < 0 else "≈0")
    sig = "显著(p<0.05)" if p_value < 0.05 else "不显著(p≥0.05)"
    return f"本模型: {direction}、{sig}"


def build_conclusions(s: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("ChatGPT 简历审计复现 - 结论")
    lines.append("(对照: Guilbeault, Delecourt & Desikan 2025, Nature, Fig.4)")
    lines.append("=" * 70)
    lines.append(f"简历总数: {s['n_total_resumes']}; 各条件: {s['n_by_condition']}; 已评分: {s['n_scored']}")
    if not s.get("analysis_complete", True):
        warnings = s.get("load_warnings") or []
        lines.append(
            f"注意: 本次分析有 {len(warnings)} 个输入文件尚未就绪或不可读；"
            "如果同批条件任务仍在运行，完成后请重新运行 analyze 以得到完整三条件汇总。"
        )
    forced_notes = _forced_param_notes(s.get("sampling_meta") or {})
    if forced_notes:
        lines.append("注意: 以下阶段的采样参数被模型强制覆盖（跨模型趋势比较时须注明）:")
        lines.extend(forced_notes)
    lines.append("")


    lines.append("[H1] 论文结论: 同一职业下, 模型给『男性姓名』生成的简历 年龄更大、经验更多、毕业更久。")
    a = s["fig4a_treatment_gender_effect"]
    label = {"applicant_age": "申请人年龄", "total_experience": "相关经验年数",
             "years_since_grad": "毕业年限", "num_skills": "技能数"}
    for key in ["applicant_age", "total_experience", "years_since_grad", "num_skills"]:
        coef = a["ols_male_coef"].get(key)
        tt = a["ttest"].get(key)
        if coef:
            v = _verdict(coef["estimate"], coef["p_value"], expect_positive=True)
            lines.append(
                f"  - {label[key]}: 男性效应 β={coef['estimate']:+.3f} "
                f"(SE={coef['std_error']:.3f}, p={coef['p_value']:.2e}, n={coef['n']}) -> {v}"
            )
        else:
            lines.append(f"  - {label[key]}: 数据不足/无法估计回归")
        if tt:
            lines.append(
                f"      均值 女={tt['mean_female']:.2f} vs 男={tt['mean_male']:.2f} "
                f"(差 {tt['diff_male_minus_female']:+.2f}, p={tt['p_value']:.2e})"
            )
    lines.append("")


    cg = s.get("control_gender_gender_effect") or {}
    if cg.get("n"):
        lines.append("")
        lines.append(f"[H1-对照] control_gender(模型自生成姓名/性别, n={cg.get('n')}): 同上口径的性别效应")
        for key in ["applicant_age", "total_experience", "years_since_grad", "num_skills"]:
            coef = (cg.get("ols_male_coef") or {}).get(key)
            tt = (cg.get("ttest") or {}).get(key)
            if coef:
                v = _verdict(coef["estimate"], coef["p_value"])
                lines.append(f"  - {label[key]}: 男性效应 β={coef['estimate']:+.3f} (p={coef['p_value']:.2e}, n={coef['n']}) -> {v}")
            elif tt:
                lines.append(f"  - {label[key]}: 均值 女={tt['mean_female']:.2f} vs 男={tt['mean_male']:.2f} (差 {tt['diff_male_minus_female']:+.2f}, p={tt['p_value']:.2e})")
            else:
                lines.append(f"  - {label[key]}: 数据不足")
    lines.append("")


    lines.append("[H2] 论文结论: 简历年龄越大, 模型给的评分越高 (正相关)。")
    b = s["fig4b_age_score_corr"]
    corr_rows = [
        ("all", "年龄~评分 全样本"), ("treatment", "年龄~评分 treatment"),
        ("experience_all", "经验~评分 全样本"), ("experience_treatment", "经验~评分 treatment"),
        ("grad_all", "毕业~评分 全样本"), ("grad_treatment", "毕业~评分 treatment"),
    ]
    for scope, name in corr_rows:
        c = b.get(scope)
        if c:
            d = "正向(>0)" if c["r"] > 0 else ("负向(<0)" if c["r"] < 0 else "≈0")
            sig = "显著" if c["p_value"] < 0.05 else "不显著"
            lines.append(f"  - {name}: r={c['r']:+.3f} (p={c['p_value']:.2e}, n={c['n']}) -> 本模型: {d}、{sig}")
        else:
            lines.append(f"  - {name}: 数据不足 (是否已执行评分阶段?)")
    co = b.get("controlled_ols") or {}
    age_c = co.get("age_coef")
    if age_c:
        v = _verdict(age_c["estimate"], age_c["p_value"])
        lines.append(f"  - 控制回归 score~male+age+经验+Name+Occupation: β_age={age_c['estimate']:+.4f} (p={age_c['p_value']:.2e}) -> {v}")
    lines.append("")


    lines.append("[H3] 论文结论: 年龄与评分的正向关系在『男性』身上更强 (age×male 交互为正)。")
    c = s.get("fig4c_age_gender_interaction")
    if c and c.get("interaction_male_x_age"):
        it = c["interaction_male_x_age"]
        v = _verdict(it["estimate"], it["p_value"], expect_positive=True)
        lines.append(
            f"  - age×male 交互: β={it['estimate']:+.4f} "
            f"(SE={it['std_error']:.4f}, p={it['p_value']:.2e}, n={it['n']}) -> {v}"
        )
    else:
        lines.append("  - 数据不足/无法估计 (需 treatment 条件且已评分)")
    lines.append("")
    lines.append("注: 小规模(默认档)下不显著属正常; 放大到论文规模 full (50/50/20, 16 names) 后再判读显著性。")
    lines.append("=" * 70)
    return "\n".join(lines)
