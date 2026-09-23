#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = Path(__file__).resolve().parent / "model_registry.yaml"


def _load_json(path: Path) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _latest(paths) -> Path | None:
    paths = [p for p in paths if p.is_file()]
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None


def _num(x):
    if isinstance(x, bool) or x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v and v not in (float("inf"), float("-inf")) else None


def _row(indicator: str, value, *, n=None, p=None, src: Path | None = None) -> dict | None:
    v = _num(value)
    if v is None:
        return None
    return {
        "indicator": indicator,
        "value": round(v, 6),
        "n": int(n) if _num(n) is not None else "",
        "p": _num(p) if _num(p) is not None else "",
        "source_file": str(src.relative_to(ROOT)) if src else "",
    }


def _rows(*candidates) -> list[dict]:
    return [r for r in candidates if r is not None]


def extract_behavioral(model: str) -> list[dict]:
    p = ROOT / "LLM-Behavioral" / "analysis" / model / "analysis_summary.json"
    d = _load_json(p)
    if not d:
        return []
    out: list[dict] = []
    for exp, e in (d.get("experiments") or {}).items():
        if not isinstance(e, dict):
            continue
        llm = e.get("llm") or {}
        test = e.get("test") or {}
        out += _rows(_row(f"{exp}.mean", llm.get("mean"),
                          n=llm.get("n"), p=test.get("p_approx"), src=p))
        tur = e.get("turing") or {}
        if _num(tur.get("win")) is not None:
            out += _rows(_row(f"{exp}.turing_win_minus_loss",
                              tur["win"] - (tur.get("loss") or 0), n=tur.get("n_draws"), src=p))
        fr = e.get("first_round") or {}
        fllm = fr.get("llm") or {}
        out += _rows(_row(f"{exp}.first_round_mean", fllm.get("mean"),
                          n=fllm.get("n"), p=(fr.get("test") or {}).get("p_approx"), src=p))
        for dim, blk in (e.get("dimensions") or {}).items():
            if isinstance(blk, dict):
                out += _rows(_row(f"{exp}.{dim}.percentile", blk.get("percentile"),
                                  n=(blk.get("llm") or {}).get("n"), src=p))
    rp = d.get("revealed_preference") or {}
    out += _rows(_row("revealed_preference.best_b", rp.get("overall_best_b"), src=p))
    for exp, blk in (d.get("dynamics") or {}).items():
        if not isinstance(blk, dict):
            continue
        for k in ("coop_then_coop", "defect_then_coop", "after_bomb_mean", "after_safe_mean"):
            out += _rows(_row(f"{exp}.{k}", blk.get(k), n=blk.get("n_instances"), src=p))
    return out


def extract_altruism(model: str) -> list[dict]:
    p = ROOT / "LLM-Altruism" / "analysis" / model / "analysis_summary.json"
    d = _load_json(p)
    if not d:
        return []
    out: list[dict] = []
    for key, card in (d.get("files") or {}).items():
        if not isinstance(card, dict):
            continue
        exp = key.split("/", 1)[1] if "/" in key else key
        ov = card.get("overall") or {}
        for k in ("mean", "median", "p_zero", "p_half", "p_full"):
            out += _rows(_row(f"{exp}.share_{k}", ov.get(k), n=ov.get("n"), src=p))
        gate = card.get("gate") or {}
        for k in ("accept_payoffmax", "refuse_payoffmax", "combined_payoffmax"):
            out += _rows(_row(f"{exp}.gate_{k}", gate.get(k), src=p))
        out += _rows(_row(f"{exp}.usable_rate", card.get("usable_rate"),
                          n=card.get("n_dictator"), src=p))
    return out


def _strip_model_token(stem: str, model: str) -> str:
    head = stem.split("_", 1)
    if len(head) == 2 and (head[0].lower() in model.lower() or model.lower().startswith(head[0].lower())):
        return head[1]
    return stem


def extract_social(model: str) -> list[dict]:
    p = (ROOT / "LLM-Social-Conventions-and-Collective-Bias" / "analysis"
         / model / "analysis_summary.json")
    d = _load_json(p)
    if not d:
        return []
    out: list[dict] = []
    for key, card in (d.get("files") or {}).items():
        if not isinstance(card, dict):
            continue
        stem = _strip_model_token(Path(key).stem, model)
        ctype = card.get("type")
        if ctype == "individual_bias":
            bt = card.get("bias_test") or {}
            if _num(bt.get("k")) is not None and _num(bt.get("n")):
                out += _rows(_row(f"individual_bias.favored_share[{stem}]",
                                  bt["k"] / bt["n"], n=bt.get("n"), p=bt.get("p_value"), src=p))
        elif ctype == "collective_or_committed":
            h1 = card.get("H1_convergence") or {}
            out += _rows(
                _row(f"collective.convergence_rate[{stem}]", h1.get("convergence_rate"),
                     n=h1.get("total_runs"), src=p),
                _row(f"collective.success_rate[{stem}]", card.get("overall_success_rate"),
                     n=card.get("n_runs"), src=p),
            )
            for k, v in (card.get("H2_collective_bias") or {}).items():
                out += _rows(_row(f"collective.H2_{k}[{stem}]", v, src=p))
    for k, blk in (d.get("critical_mass") or {}).items():
        if isinstance(blk, dict):
            out += _rows(_row(f"critical_mass.full[{k}]", blk.get("critical_mass_full"), src=p))
    return out


def extract_identity(model: str) -> list[dict]:
    root = ROOT / "LLM-Can-Harmfully-Misportray-and-Flatten-Identity-Groups" / "outputs" / model
    if not root.is_dir():
        return []
    out: list[dict] = []

    def _read(path: Path) -> list[dict]:
        try:
            with open(path, encoding="utf-8") as f:
                return list(csv.DictReader(f))
        except OSError:
            return []

    div = _latest(root.rglob("diversity_summary.csv"))
    if div:
        rows = _read(div)
        for col, name in (("pairwise_cosine_distance_mean", "diversity.pairwise_cosine_mean"),
                          ("unique_ngram_mean", "diversity.unique_ngram_mean")):
            vals = [_num(r.get(col)) for r in rows]
            vals = [v for v in vals if v is not None]
            if vals:
                out += _rows(_row(name, statistics.mean(vals), n=len(vals), src=div))
    cov = _latest(root.rglob("coverage_summary.csv"))
    if cov:
        by_axis: dict[str, list[tuple[float | None, float | None]]] = {}
        for r in _read(cov):
            by_axis.setdefault(r.get("identity_axis") or "?", []).append(
                (_num(r.get("vendi_point")), _num(r.get("det_point"))))
        for axis, pairs in by_axis.items():
            vendi = [a for a, _ in pairs if a is not None]
            det = [b for _, b in pairs if b is not None]
            if vendi:
                out += _rows(_row(f"coverage.vendi[{axis}]", statistics.mean(vendi),
                                  n=len(vendi), src=cov))
            if det:
                out += _rows(_row(f"coverage.det[{axis}]", statistics.mean(det),
                                  n=len(det), src=cov))
    prem = _latest(root.rglob("premise1_summary.csv"))
    if prem:
        by_axis2: dict[str, list[float]] = {}
        for r in _read(prem):
            w, a = _num(r.get("mean_within")), _num(r.get("mean_across"))
            if w is not None and a is not None:
                by_axis2.setdefault(r.get("identity_axis") or "?", []).append(a - w)
        for axis, diffs in by_axis2.items():
            out += _rows(_row(f"premise1.across_minus_within[{axis}]",
                              statistics.mean(diffs), n=len(diffs), src=prem))
    return out


def extract_age_gender(model: str) -> list[dict]:
    root = ROOT / "LLM-Age-Gender-Distortion" / "results" / model
    if not root.is_dir():
        return []
    summaries = sorted(root.glob("*/analysis_summary.json"),
                       key=lambda q: q.stat().st_mtime, reverse=True)
    for p in summaries:
        d = _load_json(p)
        if not d:
            continue
        out: list[dict] = []
        ols = (d.get("fig4a_treatment_gender_effect") or {}).get("ols_male_coef") or {}
        for name, key in (("h1.age_male_beta", "applicant_age"),
                          ("h1.experience_male_beta", "total_experience"),
                          ("h1.grad_male_beta", "years_since_grad"),
                          ("h1.skills_male_beta", "num_skills")):
            c = ols.get(key) or {}
            out += _rows(_row(name, c.get("estimate"), n=c.get("n"), p=c.get("p_value"), src=p))
        corr = (d.get("fig4b_age_score_corr") or {}).get("all") or {}
        out += _rows(_row("h2.age_score_r", corr.get("r"),
                          n=corr.get("n"), p=corr.get("p_value"), src=p))
        it = (d.get("fig4c_age_gender_interaction") or {}).get("interaction_male_x_age") or {}
        out += _rows(_row("h3.age_x_male_beta", it.get("estimate"),
                          n=it.get("n"), p=it.get("p_value"), src=p))
        if out:
            return out
    return []


EXTRACTORS = {
    "behavioral": extract_behavioral,
    "altruism": extract_altruism,
    "social": extract_social,
    "identity": extract_identity,
    "age_gender_distortion": extract_age_gender,
}


_RESULT_ROOTS = [
    "LLM-Behavioral/records_new",
    "LLM-Altruism/results",
    "LLM-Altruism/analysis",
    "LLM-Social-Conventions-and-Collective-Bias/data",
    "LLM-Social-Conventions-and-Collective-Bias/analysis",
    "LLM-Can-Harmfully-Misportray-and-Flatten-Identity-Groups/outputs",
    "LLM-Age-Gender-Distortion/results",
]

_NOT_MODELS = {"archive", "batch", "analysis", "raw", "scored", "validation", "figures", "logs"}


def discover_models() -> set[str]:
    found: set[str] = set()
    for rel in _RESULT_ROOTS:
        root = ROOT / rel
        if not root.is_dir():
            continue
        for d in root.iterdir():
            if d.is_dir() and d.name not in _NOT_MODELS and not d.name.startswith((".", "_")):
                found.add(d.name)
    return found


def scaffold_missing(missing: list[str]) -> None:
    with open(REGISTRY, "a", encoding="utf-8") as f:
        for m in sorted(missing):
            f.write(
                f"\n  {m}:\n"
                "    vendor: \"\"\n"
                "    family: \"\"\n"
                "    release_date: null       # TODO: 发布日期 YYYY-MM-DD（可让 AI 联网查证后填入）\n"
                "    date_confidence: unknown\n"
                "    open_weights: null\n"
                "    params_total_b: null\n"
                "    params_active_b: null\n"
                "    reasoning: null\n"
                "    elo: null\n"
                "    source: \"TODO: collect_metrics.py --scaffold 自动生成，待核实\"\n"
            )
    print(f"[scaffold] 已把 {len(missing)} 个新模型的占位条目追加到 {REGISTRY.relative_to(ROOT)}，"
          "请补全 release_date 等 TODO 字段（可让 AI 查证）。")


def load_registry() -> dict:
    with open(REGISTRY, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("models") or {}


def load_prices() -> dict:
    path = ROOT / "model_config.json"
    out: dict = {}
    try:
        for item in json.loads(path.read_text(encoding="utf-8")):
            name = item.get("name")
            if name:
                out[name] = item.get("input_price_per_1m_tokens")
    except (OSError, json.JSONDecodeError):
        pass
    return out


def collect(include_excluded: bool = False, include_unregistered: bool = True) -> list[dict]:
    registry = load_registry()
    prices = load_prices()

    unregistered = sorted(discover_models() - set(registry))
    if unregistered:
        print(f"[collect] 发现 {len(unregistered)} 个未注册模型（元数据留空，"
              f"用 --scaffold 生成占位条目后补全）：{', '.join(unregistered)}")
        if include_unregistered:
            registry = {**registry, **{m: {} for m in unregistered}}
    table: list[dict] = []
    for model, meta in registry.items():
        meta = meta or {}
        if meta.get("exclude") and not include_excluded:
            continue
        base = {
            "model": model,
            "vendor": meta.get("vendor", ""),
            "family": meta.get("family", ""),
            "release_date": str(meta.get("release_date") or ""),
            "open_weights": meta.get("open_weights", ""),
            "params_total_b": meta.get("params_total_b") if meta.get("params_total_b") is not None else "",
            "input_price_per_1m": prices.get(model, ""),
        }
        for project, extractor in EXTRACTORS.items():
            for row in extractor(model):
                table.append({**base, "project": project, **row})
    table.sort(key=lambda r: (r["project"], r["indicator"], r["release_date"], r["model"]))
    return table


FIELDS = ["model", "vendor", "family", "release_date", "open_weights", "params_total_b",
          "input_price_per_1m", "project", "indicator", "value", "n", "p", "source_file"]


def main() -> None:
    ap = argparse.ArgumentParser(description="跨项目模型趋势指标抽取")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "metrics_long.csv"))
    ap.add_argument("--include-excluded", action="store_true",
                    help="纳入 registry 中标记 exclude 的模型（旧试跑 / 图像模型等）")
    ap.add_argument("--scaffold", action="store_true",
                    help="把结果目录中新发现、尚未注册的模型以占位条目追加进 model_registry.yaml 后退出")
    args = ap.parse_args()

    if args.scaffold:
        missing = sorted(discover_models() - set(load_registry()))
        if missing:
            scaffold_missing(missing)
        else:
            print("[scaffold] 没有发现未注册的新模型。")
        return

    table = collect(include_excluded=args.include_excluded)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(table)

    by_model: dict[str, set] = {}
    for r in table:
        by_model.setdefault(r["model"], set()).add(r["project"])
    print(f"[collect] 共 {len(table)} 条指标，覆盖 {len(by_model)} 个模型 -> {out_path}")
    for m in sorted(by_model, key=lambda x: next(
            (r["release_date"] for r in table if r["model"] == x), "")):
        date = next((r["release_date"] for r in table if r["model"] == m), "?")
        print(f"  {m:36s} {date:12s} 项目: {', '.join(sorted(by_model[m]))}")


if __name__ == "__main__":
    main()
