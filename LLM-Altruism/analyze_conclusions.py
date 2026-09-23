from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUTBASE = ROOT / "analysis"
PAPER_FILES = ROOT / "johnson_obradovich_NHB_replication_files_04_07_2025" / "nature_2023-01-00290_rep_files"
PAPER_TEXT_INSERTION_DG = PAPER_FILES / "R4_03_red2dg_gpt4_gpt35_collected_04_20_2023_variant_workable.csv"


def _parser_mtime() -> float:
    mts = []
    for rel in ("src/schemas.py", "src/analysis.py"):
        try:
            mts.append((ROOT / rel).stat().st_mtime)
        except OSError:
            pass
    return max(mts) if mts else 0.0


_PARSER_MTIME = _parser_mtime()

try:
    from scipy import stats as _ss
except Exception:
    _ss = None

try:
    from compare import ENGEL_BINS, ENGEL_FREQ, _hist_freq, _paper_davinci_distributions
    _HAVE_COMPARE = True
except Exception:
    _HAVE_COMPARE = False
    ENGEL_BINS = np.arange(0, 1.05, 0.1)
    ENGEL_FREQ = np.array(
        [0.3611, 0.0914, 0.0881, 0.0891, 0.0723, 0.1674, 0.0389, 0.0197, 0.0107, 0.0070, 0.0544]
    )

    def _hist_freq(values: np.ndarray) -> np.ndarray:
        if len(values) == 0:
            return np.zeros_like(ENGEL_BINS)
        rounded = (np.round(np.asarray(values, float) / 0.1) * 0.1).round(2)
        out = np.zeros_like(ENGEL_BINS)
        for i, b in enumerate(ENGEL_BINS):
            out[i] = float(np.isclose(rounded, round(b, 1)).sum())
        total = out.sum()
        return out / total if total else out

try:
    from src.schemas import parse_response
except Exception:
    parse_response = None
try:
    from src.schemas import parse_for_task
except Exception:
    parse_for_task = None


RECIP_HUMAN = "experimenter"
RECIP_CHARITY = "charity"
RECIP_AI = ("other_llm_ada", "other_llm_babbage", "other_llm_curie")
RECIP_LABELS = {
    "experimenter": "人类实验者",
    "charity": "慈善机构",
    "other_llm_ada": "AI · text-ada-001",
    "other_llm_babbage": "AI · text-babbage-001",
    "other_llm_curie": "AI · text-curie-001",
}


GATE_REF = 0.92
GENEROUS = 0.05
GATE_MIN_PASS = 0.40
GATE_STRONG = 0.70
GATE_ASYM_WARN = 0.35


def _latest_run(model_dir: Path, experiment: str) -> Path | None:
    base = model_dir / experiment
    if not base.is_dir():
        return None
    runs = [p for p in base.iterdir() if p.is_dir()]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def _parsed_is_fresh(run_dir: Path) -> bool:
    parsed = run_dir / "parsed.csv"
    if not parsed.is_file():
        return False
    pm = parsed.stat().st_mtime
    raw = run_dir / "raw.csv"
    if raw.is_file() and raw.stat().st_mtime > pm:
        return False
    return pm >= _PARSER_MTIME


def _ensure_fresh(run_dir: Path, experiment_name: str, model_name: str) -> None:
    if _parsed_is_fresh(run_dir):
        return
    raw = run_dir / "raw.csv"
    if not raw.is_file():
        return
    try:
        from src.analysis import analyze_csv
        from src.config import load_experiment_def
        acfg = (load_experiment_def(experiment_name).raw.get("analysis") or {})
        analyze_csv(
            raw_csv=raw,
            out_dir=run_dir,
            task_type=acfg.get("task_type", "dictator"),
            experiment_name=experiment_name,
            model_name=model_name,
            payoff_max=acfg.get("payoff_max"),
        )
    except Exception as exc:
        print(f"[warn] 重算 {run_dir} 的 parsed.csv 失败：{exc}", file=sys.stderr)


def _refresh_model_cache(model_dir: Path) -> None:
    for exp_dir in sorted(model_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        run = _latest_run(model_dir, exp_dir.name)
        if run is not None:
            _ensure_fresh(run, exp_dir.name, model_dir.name)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(x: str | None) -> float | None:
    if x is None or x == "":
        return None
    try:
        v = float(x)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def _to_int(x: str | None) -> int | None:
    v = _to_float(x)
    return int(v) if v is not None else None


def _weighted_rate(pairs: list[tuple[float | None, int | None]]) -> float | None:
    num = 0.0
    den = 0
    for rate, n in pairs:
        if rate is None or not n or n <= 0:
            continue
        num += rate * n
        den += n
    return (num / den) if den else None


def _fmt_pct(x: float | None) -> str:
    return "—" if x is None else f"{x:.0%}"


def _finite_values(*vals: float | None) -> list[float]:
    return [float(v) for v in vals if v is not None and math.isfinite(float(v))]


def _gate(model_dir: Path, experiment: str = "baseline_nonsocial") -> dict:
    run = _latest_run(model_dir, experiment)
    if run is None or not (run / "summary.csv").is_file():
        return {"available": False, "passed": False,
                "verdict": f"无非社交基线，无法判定门槛（请先跑 {experiment}）"}
    by_cond = {r["condit"]: r for r in _read_rows(run / "summary.csv")}
    acc_row = by_cond.get("accept") or {}
    ref_row = by_cond.get("refuse") or {}
    acc = _to_float(acc_row.get("payoff_max_rate"))
    ref = _to_float(ref_row.get("payoff_max_rate"))


    acc_n = _to_int(acc_row.get("comprehensible"))
    if acc_n is None:
        acc_n = _to_int(acc_row.get("usable"))
    ref_n = _to_int(ref_row.get("comprehensible"))
    if ref_n is None:
        ref_n = _to_int(ref_row.get("usable"))
    combined = _weighted_rate([(acc, acc_n), (ref, ref_n)])
    vals = _finite_values(acc, ref)
    balanced = min(vals) if len(vals) == 2 else None
    asymmetry = abs(acc - ref) if acc is not None and ref is not None else None
    asymmetric = bool(
        asymmetry is not None
        and (asymmetry >= GATE_ASYM_WARN or (balanced is not None and balanced < GATE_MIN_PASS <= (combined or 0)))
    )
    unusable = max(
        _to_float(acc_row.get("unusable_rate")) or 0.0,
        _to_float(ref_row.get("unusable_rate")) or 0.0,
    )
    base = {
        "available": True,
        "accept_payoffmax": acc,
        "refuse_payoffmax": ref,
        "combined_payoffmax": combined,
        "balanced_payoffmax": balanced,
        "asymmetry": asymmetry,
        "asymmetric": asymmetric,
        "unusable_rate": unusable,
        "ref_davinci": GATE_REF,
        "decision_rule": (
            "passed requires combined_payoffmax >= 0.40 and, when both accept/refuse are present, "
            "balanced_payoffmax = min(accept, refuse) >= 0.40"
        ),
    }
    if combined is None:
        return {**base, "passed": False, "verdict": "无可用的非社交数据，无法计算收益最大化率"}
    acc_s = _fmt_pct(acc)
    ref_s = _fmt_pct(ref)
    bal_s = _fmt_pct(balanced)
    if balanced is not None and balanced < GATE_MIN_PASS <= combined:
        verdict = (
            f"本模型非社交综合收益最大化率 {combined:.0%}，但两条件严重不对称"
            f"（accept {acc_s}、refuse {ref_s}，min={bal_s}）：不通过自利门槛。"
            "尤其当通过率主要由 refuse 条件支撑时，『拿 0』与服从 prompt 中 write \"0\" 指令不可分辨。"
        )
        passed = False
    elif combined >= GATE_STRONG and (balanced is None or balanced >= GATE_STRONG):
        verdict = (
            f"本模型非社交综合收益最大化率 {combined:.0%}"
            f"（accept {acc_s}、refuse {ref_s}，min={bal_s}）：自利基线强。"
        )
        passed = True
    elif combined >= GATE_MIN_PASS and (balanced is None or balanced >= GATE_MIN_PASS):
        verdict = (
            f"本模型非社交综合收益最大化率 {combined:.0%}"
            f"（accept {acc_s}、refuse {ref_s}，min={bal_s}）：自利基线中等。"
        )
        if asymmetric:
            verdict += " 两条件存在明显不对称，利他解释需降低确信度。"
        passed = True
    else:
        verdict = (
            f"本模型非社交综合收益最大化率 {combined:.0%}"
            f"（accept {acc_s}、refuse {ref_s}，min={bal_s}）：自利基线弱，几乎不模拟收益最大化。"
        )
        passed = False
    return {**base, "passed": passed, "verdict": verdict}


def _has_data(run_dir: Path) -> bool:
    return (run_dir / "parsed.csv").is_file() or (run_dir / "raw.csv").is_file()


def _row_count(run_dir: Path) -> int:
    for name in ("parsed.csv", "raw.csv"):
        p = run_dir / name
        if p.is_file():
            return len(_read_rows(p))
    return 0


def _propshare_from_raw(row: dict[str, str], task_type: str = "dictator") -> float | None:
    stakes = _to_float(row.get("stakes"))
    if stakes is None or stakes <= 0:
        return None
    if parse_for_task is not None:
        try:
            pr = parse_for_task(task_type, row.get("respon"), stakes)
        except (TypeError, ValueError):
            return None
    elif parse_response is not None:
        pr = parse_response(row.get("respon"), stakes)
    else:
        return None
    if pr.decision is None:
        return None
    return min(max(pr.decision / stakes, 0.0), 1.0)


def _shares_from_run(run_dir: Path, *, by_variant: bool = False,
                     task_type: str = "dictator") -> dict[str, np.ndarray]:
    def _key(condit: str) -> str:
        return condit.rsplit("__", 1)[1] if (by_variant and "__" in condit) else condit

    out: dict[str, list[float]] = {}
    parsed = run_dir / "parsed.csv"
    if parsed.is_file():
        for r in _read_rows(parsed):
            if (r.get("unusable_reason") or "") != "":
                continue
            ps = _to_float(r.get("propshare"))
            if ps is not None:
                out.setdefault(_key(r.get("condit", "")), []).append(min(max(ps, 0.0), 1.0))
    elif (run_dir / "raw.csv").is_file():
        for r in _read_rows(run_dir / "raw.csv"):
            ps = _propshare_from_raw(r, task_type)
            if ps is not None:
                out.setdefault(_key(r.get("condit", "")), []).append(ps)
    return {k: np.asarray(v, float) for k, v in out.items()}


def _data_quality(run_dir: Path) -> dict | None:
    parsed = run_dir / "parsed.csv"
    if not parsed.is_file():
        return None
    counts: dict[str, int] = {}
    n = 0
    for r in _read_rows(parsed):
        n += 1
        reason = (r.get("unusable_reason") or "").strip()
        path = (r.get("parse_path") or "").strip()
        key = reason if reason else (path or "ok_unknown_path")
        counts[key] = counts.get(key, 0) + 1
    if not n:
        return None
    ambiguous = counts.get("ambiguous", 0)
    fallback = counts.get("fallback_first", 0)

    refusal = counts.get("refusal", 0)
    no_number = counts.get("no_number", 0)
    api_error = counts.get("api_error", 0)
    empty = counts.get("empty", 0)
    out_of_range = counts.get("out_of_range", 0)

    recovered = counts.get("recovered_zero_option", 0) + counts.get("recovered_refusal_share", 0)
    usable = n - (ambiguous + refusal + no_number + api_error + empty + out_of_range)
    out = {
        "n": n,
        "counts": counts,
        "usable": usable,
        "usable_rate": round(usable / n, 4),
        "refusal": refusal,
        "refusal_rate": round(refusal / n, 4),
        "ambiguous_rate": round(ambiguous / n, 4),
        "fallback_rate": round(fallback / n, 4),
        "recovered": recovered,
        "recovered_rate": round(recovered / n, 4),
    }
    notes = []
    if refusal / n >= 0.02:
        notes.append(f"纯拒答（无可提取分享额，对标原文 err_resp）占 {refusal / n:.0%}")
    if ambiguous / n >= 0.02:
        notes.append(f"歧义应答（0 后多个不同非零，无法编码）占 {ambiguous / n:.0%}，已按原文 unclear 口径剔除")
    if fallback / n >= 0.05:
        notes.append(f"第一数字兜底解析占 {fallback / n:.0%}，建议用 validation_sample.csv 人工抽查")
    out["note"] = "；".join(notes)
    return out


def _desc(arr: np.ndarray) -> dict:
    if arr.size == 0:
        return {"n": 0, "median": None, "mean": None, "p_zero": None,
                "p_lt01": None, "p_half": None, "p_full": None}
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p_zero": float(np.mean(arr == 0)),
        "p_lt01": float(np.mean(arr < 0.01)),
        "p_half": float(np.mean(np.abs(arr - 0.5) < 0.01)),
        "p_full": float(np.mean(arr == 1)),
    }


def _binned_metrics(model_freq: np.ndarray, ref_freq: np.ndarray) -> dict:
    p = np.asarray(model_freq, float)
    q = np.asarray(ref_freq, float)
    if p.sum() == 0 or q.sum() == 0:
        return {"tvd": None, "ks": None, "wasserstein": None}
    tvd = float(0.5 * np.abs(p - q).sum())
    ks = float(np.abs(np.cumsum(p) - np.cumsum(q)).max())
    if _ss is not None:
        wass = float(_ss.wasserstein_distance(ENGEL_BINS, ENGEL_BINS, p, q))
    else:
        step = float(ENGEL_BINS[1] - ENGEL_BINS[0])
        wass = float(np.abs(np.cumsum(p) - np.cumsum(q)).sum() * step)
    return {"tvd": round(tvd, 4), "ks": round(ks, 4), "wasserstein": round(wass, 4)}


def _avg_ranks(v: np.ndarray) -> np.ndarray:
    order = v.argsort(kind="mergesort")
    sv = v[order]
    ranks = np.empty(v.size, float)
    i = 0
    while i < v.size:
        j = i
        while j + 1 < v.size and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _ks_2samp(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return {"D": None, "p": None, "n1": int(a.size), "n2": int(b.size)}
    if _ss is not None:
        r = _ss.ks_2samp(a, b, alternative="two-sided", method="asymp")
        return {"D": round(float(r.statistic), 4), "p": float(r.pvalue),
                "n1": int(a.size), "n2": int(b.size)}
    grid = np.sort(np.concatenate([a, b]))
    ca = np.searchsorted(np.sort(a), grid, side="right") / a.size
    cb = np.searchsorted(np.sort(b), grid, side="right") / b.size
    d = float(np.max(np.abs(ca - cb)))
    en = math.sqrt(a.size * b.size / (a.size + b.size))
    lam = (en + 0.12 + 0.11 / en) * d
    p = 2.0 * sum((-1) ** (k - 1) * math.exp(-2.0 * lam * lam * k * k) for k in range(1, 101))
    return {"D": round(d, 4), "p": max(0.0, min(1.0, p)), "n1": int(a.size), "n2": int(b.size)}


def _mann_whitney(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a, float); b = np.asarray(b, float)
    n1, n2 = a.size, b.size
    if n1 == 0 or n2 == 0:
        return {"U": None, "p": None, "n1": int(n1), "n2": int(n2)}
    if _ss is not None:
        r = _ss.mannwhitneyu(a, b, alternative="two-sided")
        return {"U": round(float(r.statistic), 2), "p": float(r.pvalue), "n1": int(n1), "n2": int(n2)}
    ranks = _avg_ranks(np.concatenate([a, b]))
    u1 = float(ranks[:n1].sum() - n1 * (n1 + 1) / 2.0)
    mu = n1 * n2 / 2.0
    sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    z = (u1 - mu) / sigma if sigma > 0 else 0.0
    return {"U": round(u1, 2), "p": math.erfc(abs(z) / math.sqrt(2.0)), "n1": int(n1), "n2": int(n2)}


def _ttest_1samp(a: np.ndarray, mu: float = 0.0) -> dict:
    a = np.asarray(a, float)
    n = a.size
    if n < 2:
        return {"t": None, "p": None, "mean": (float(a.mean()) if n else None), "sd": None,
                "n": int(n), "degenerate": "insufficient_n"}
    mean = float(a.mean())
    sd = float(a.std(ddof=1))
    if sd == 0:
        return {"t": None, "p": None, "mean": round(mean, 4), "sd": 0.0,
                "n": int(n), "degenerate": "zero_variance"}
    t = (mean - mu) / (sd / math.sqrt(n))
    if _ss is not None:
        p = float(2.0 * _ss.t.sf(abs(t), df=n - 1))
    else:
        p = math.erfc(abs(t) / math.sqrt(2.0))
    return {"t": round(t, 3), "p": p, "mean": round(mean, 4), "sd": round(sd, 4), "n": int(n),
            "degenerate": None}


def _ttest_welch(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size < 2 or b.size < 2:
        return {"t": None, "p": None, "n1": int(a.size), "n2": int(b.size),
                "degenerate": "insufficient_n"}
    ma, mb = float(a.mean()), float(b.mean())
    v1, v2 = float(a.var(ddof=1)), float(b.var(ddof=1))
    sa, sb = math.sqrt(v1), math.sqrt(v2)
    if sa == 0 and sb == 0:
        return {
            "t": None,
            "p": (0.0 if ma != mb else None),
            "n1": int(a.size),
            "n2": int(b.size),
            "degenerate": ("separated_constants" if ma != mb else "same_constants"),
        }
    if sa == 0 or sb == 0:
        se2 = v1 / a.size + v2 / b.size
        if se2 <= 0:
            return {"t": None, "p": None, "n1": int(a.size), "n2": int(b.size), "degenerate": "zero_variance"}
        t = (ma - mb) / math.sqrt(se2)
        terms = []
        if v1 > 0:
            terms.append((v1 / a.size) ** 2 / (a.size - 1))
        if v2 > 0:
            terms.append((v2 / b.size) ** 2 / (b.size - 1))
        df = (se2 ** 2 / sum(terms)) if terms else max(a.size + b.size - 2, 1)
        if _ss is not None:
            p = float(2.0 * _ss.t.sf(abs(t), df=df))
        else:
            p = math.erfc(abs(t) / math.sqrt(2.0))
        return {"t": round(t, 3), "p": p, "n1": int(a.size), "n2": int(b.size),
                "degenerate": "one_constant"}
    if _ss is not None:
        r = _ss.ttest_ind(a, b, equal_var=False)
        return {"t": round(float(r.statistic), 3), "p": float(r.pvalue),
                "n1": int(a.size), "n2": int(b.size), "degenerate": None}
    va, vb = v1 / a.size, v2 / b.size
    denom = math.sqrt(va + vb)
    if denom == 0:
        return {"t": None, "p": None, "n1": int(a.size), "n2": int(b.size), "degenerate": "zero_variance"}
    t = (ma - mb) / denom
    return {"t": round(t, 3), "p": math.erfc(abs(t) / math.sqrt(2.0)),
            "n1": int(a.size), "n2": int(b.size), "degenerate": None}


def _two_prop_test(k1: int, n1: int, k2: int, n2: int) -> dict:
    if min(n1, n2) <= 0:
        return {"p1": None, "p2": None, "z": None, "p": None, "n1": int(n1), "n2": int(n2)}
    p1, p2 = k1 / n1, k2 / n2
    pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return {"p1": round(p1, 4), "p2": round(p2, 4), "z": None, "p": None,
                "n1": int(n1), "n2": int(n2)}
    z = (p1 - p2) / se
    return {"p1": round(p1, 4), "p2": round(p2, 4), "z": round(z, 3),
            "p": math.erfc(abs(z) / math.sqrt(2.0)), "n1": int(n1), "n2": int(n2)}


def _shift_test(manip: np.ndarray, base: np.ndarray, *, expect: str | None = None) -> dict:
    manip = np.asarray(manip, float); base = np.asarray(base, float)
    mw = _mann_whitney(manip, base)
    if manip.size == 0 or base.size == 0:
        return {**mw, "d_median": None, "direction": "数据不足"}
    d = float(np.median(manip) - np.median(base))
    d_mean = float(np.mean(manip) - np.mean(base))
    sig = mw["p"] is not None and mw["p"] < 0.05
    lower = d < 0 or (d == 0 and d_mean < 0)
    if not sig:
        direction = "无显著变化"
    else:
        direction = "显著左移（更低）" if lower else "显著右移（更高）"
    out = {**mw, "d_median": round(d, 4), "d_mean": round(d_mean, 4), "direction": direction}
    if expect in ("lower", "higher"):
        out["expected"] = "左移" if expect == "lower" else "右移"
        out["matches_expectation"] = bool(sig and ((expect == "lower") == lower))
    return out


def _ks_and_shift(manip: np.ndarray, base: np.ndarray, *, expect: str | None = None) -> dict:
    return {**_shift_test(manip, base, expect=expect), "ks": _ks_2samp(manip, base)}


def _sample_profile(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, float)
    if arr.size == 0:
        return {"n": 0, "median": None, "mean": None, "p_zero": None, "p_half": None,
                "p_ge_half": None, "p_gt_half": None}
    return {
        "n": int(arr.size),
        "median": round(float(np.median(arr)), 4),
        "mean": round(float(np.mean(arr)), 4),
        "p_zero": round(float(np.mean(arr == 0)), 4),
        "p_half": round(float(np.mean(np.abs(arr - 0.5) < 0.01)), 4),
        "p_ge_half": round(float(np.mean(arr >= 0.5)), 4),
        "p_gt_half": round(float(np.mean(arr > 0.5)), 4),
    }


def _binned_profile(freq: np.ndarray) -> dict:
    f = np.asarray(freq, float)
    if f.size == 0 or f.sum() == 0:
        return {"n": None, "median": None, "mean": None, "p_zero": None, "p_half": None,
                "p_ge_half": None, "p_gt_half": None}
    f = f / f.sum()
    c = np.cumsum(f)
    med_idx = int(np.searchsorted(c, 0.5, side="left"))
    return {
        "n": None,
        "median": round(float(ENGEL_BINS[min(med_idx, len(ENGEL_BINS) - 1)]), 4),
        "mean": round(float(np.sum(ENGEL_BINS * f)), 4),
        "p_zero": round(float(f[0]), 4),
        "p_half": round(float(f[np.where(np.isclose(ENGEL_BINS, 0.5))[0][0]]), 4),
        "p_ge_half": round(float(f[ENGEL_BINS >= 0.5].sum()), 4),
        "p_gt_half": round(float(f[ENGEL_BINS > 0.5].sum()), 4),
    }


def _profile_delta(sample: dict, ref: dict, *, ref_label: str) -> dict:
    out = {"reference": ref_label}
    for k in ("median", "mean", "p_zero", "p_ge_half", "p_gt_half"):
        a, b = sample.get(k), ref.get(k)
        out[f"d_{k}"] = None if a is None or b is None else round(float(a) - float(b), 4)
    dm = out.get("d_median")
    dz = out.get("d_p_zero")
    if dm is None:
        out["direction"] = "数据不足"
    elif dm > 0:
        out["direction"] = "当前模型分享更多"
    elif dm < 0:
        out["direction"] = "当前模型分享更少"
    elif dz is not None and dz < 0:
        out["direction"] = "中位相同，但零分享更少"
    elif dz is not None and dz > 0:
        out["direction"] = "中位相同，但零分享更多"
    else:
        out["direction"] = "画像接近"
    return out


def _paper_gpt4_text_insertion_distributions() -> dict[str, np.ndarray] | None:
    if not PAPER_TEXT_INSERTION_DG.is_file():
        return None
    cond_map = {
        "gpt_4_0314_experimenter": RECIP_HUMAN,
        "gpt_4_0314_charity": RECIP_CHARITY,
        "gpt_4_0314_ada": "other_llm_ada",
        "gpt_4_0314_babbage": "other_llm_babbage",
        "gpt_4_0314_curie": "other_llm_curie",
    }
    out: dict[str, list[float]] = {v: [] for v in cond_map.values()}
    try:
        with PAPER_TEXT_INSERTION_DG.open(encoding="latin-1", newline="") as f:
            for r in csv.DictReader(f):
                recip = cond_map.get(r.get("condit", ""))
                if recip is None:
                    continue
                stakes = _to_float(r.get("stakes"))
                val = _to_float(r.get("choice"))
                if val is None:
                    val = _to_float(r.get("decision"))
                if stakes is None or stakes <= 0 or val is None:
                    continue
                ps = val / stakes
                if 0 <= ps <= 1:
                    out.setdefault(recip, []).append(ps)
    except OSError:
        return None
    return {k: np.asarray(v, float) for k, v in out.items() if v}


def _run_stability(model_dir: Path) -> dict | None:
    out: dict[str, dict] = {}
    for exp_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
        runs = [p for p in exp_dir.iterdir() if p.is_dir() and _has_data(p)]
        if len(runs) < 2:
            continue
        runs = sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)
        cur, prev = runs[0], runs[1]
        cur_shares = _shares_from_run(cur)
        prev_shares = _shares_from_run(prev)
        if not cur_shares or not prev_shares:
            continue
        cur_pool = np.concatenate(list(cur_shares.values()))
        prev_pool = np.concatenate(list(prev_shares.values()))
        if cur_pool.size == 0 or prev_pool.size == 0:
            continue
        s = _shift(cur_pool, prev_pool)
        out[exp_dir.name] = {
            "current_run": cur.name,
            "previous_run": prev.name,
            "n_current": int(cur_pool.size),
            "n_previous": int(prev_pool.size),
            "current_median": round(float(np.median(cur_pool)), 4),
            "previous_median": round(float(np.median(prev_pool)), 4),
            "d_median": s["d_median"],
            "tvd": s["tvd"],
            "wasserstein": s["wasserstein"],
        }
    return out or None


def _davinci_verdict(to_human: dict, to_ai: dict) -> str:
    th = to_human.get("tvd")
    ta = to_ai.get("tvd")
    tvds = [t for t in (th, ta) if t is not None]
    if not tvds:
        return "无可用分布，无法与原文 text-davinci-003 度量距离。"
    parts = []
    if th is not None:
        parts.append(f"对人类 TVD {th:.2f}")
    if ta is not None:
        parts.append(f"对 AI TVD {ta:.2f}")
    pair = "、".join(parts)
    mx = max(tvds)
    level = "接近" if mx < 0.15 else ("中等" if mx < 0.35 else "差异较大")
    return f"本模型分享分布与原文 text-davinci-003 的距离：{pair}（TVD 越大差异越大，当前整体{level}）。"


def _logistic_human_vs_ai(g1: np.ndarray, g0: np.ndarray, *,
                          label1: str = "人类实验者", label0: str = "其余(慈善+AI)") -> dict:
    if g1.size == 0 or g0.size == 0:
        return {"test": None, "coef": None, "p": None,
                "verdict": f"{label1}/{label0} 任一组无数据，无法做对象差异检验"}
    y = np.concatenate([g1, g0]).astype(float)
    x = np.concatenate([np.ones(g1.size), np.zeros(g0.size)])
    if float(np.ptp(y)) == 0.0:
        return {"test": "logistic(propshare~human_recipient)", "coef": None, "p": None,
                "verdict": "各对象分享几乎无变异，差异检验不适用"}
    X = np.column_stack([np.ones_like(x), x])
    y = np.clip(y, 1e-6, 1 - 1e-6)
    beta = np.zeros(2)
    hess = np.eye(2)
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-(X @ beta)))
        w = np.clip(p * (1 - p), 1e-9, None)
        hess = (X.T * w) @ X
        try:
            delta = np.linalg.solve(hess, X.T @ (y - p))
        except np.linalg.LinAlgError:
            break
        beta = beta + delta
        if float(np.max(np.abs(delta))) < 1e-8:
            break
    coef = float(beta[1])
    try:
        se = float(np.sqrt(np.linalg.inv(hess)[1, 1]))
    except np.linalg.LinAlgError:
        se = 0.0
    if se > 0:
        z = coef / se
        pval = math.erfc(abs(z) / math.sqrt(2.0))
        sig = "显著" if pval < 0.05 else "不显著"
        verdict = f"对『{label1} vs {label0}』分享差异{sig}（logistic β={coef:.3f}, p={pval:.3g}）"
    else:
        pval = None
        verdict = f"logistic 拟合退化（β={coef:.3f}），无法给出 p 值"
    return {"test": "logistic(propshare~human_recipient)", "coef": round(coef, 4),
            "p": pval, "n_g1": int(g1.size), "n_g0": int(g0.size),
            "verdict": verdict}


def _recipient_diff(human: np.ndarray, charity: np.ndarray, ai: np.ndarray) -> dict:
    parts = [g for g in (charity, ai) if g.size > 0]
    other = np.concatenate(parts) if parts else np.asarray([], float)
    return _logistic_human_vs_ai(human, other, label1="人类实验者", label0="其余(慈善+AI)")


def _variant_shares(run_dir: Path) -> dict[str, np.ndarray]:
    return _shares_from_run(run_dir, by_variant=True)


def _shift(variant: np.ndarray, baseline: np.ndarray) -> dict:
    vm = float(np.median(variant)) if variant.size else None
    bm = float(np.median(baseline)) if baseline.size else None
    m = _binned_metrics(_hist_freq(variant), _hist_freq(baseline))
    return {
        "n": int(variant.size),
        "variant_median": None if vm is None else round(vm, 4),
        "d_median": None if (vm is None or bm is None) else round(vm - bm, 4),
        "tvd": m["tvd"],
        "wasserstein": m["wasserstein"],
    }


def _variant_robustness(model_dir: Path, exp_name: str, baseline_pooled: np.ndarray) -> dict | None:
    run = _latest_run(model_dir, exp_name)
    if run is None or not _has_data(run):
        return None
    vs = _variant_shares(run)
    if not vs:
        return None
    shifts = {v: _shift(a, baseline_pooled) for v, a in vs.items()}
    max_tvd = max((s["tvd"] for s in shifts.values() if s["tvd"] is not None), default=0.0)
    max_dmed = max((abs(s["d_median"] or 0.0) for s in shifts.values()), default=0.0)
    stable = max_tvd < 0.15 and max_dmed < 0.1
    verdict = (
        f"本模型各变体相对 baseline 基本不变（最大 TVD {max_tvd:.2f}、最大中位差 {max_dmed:.2f}）：分享对该操纵稳健。"
        if stable else
        f"本模型各变体相对 baseline 出现偏移（最大 TVD {max_tvd:.2f}、最大中位差 {max_dmed:.2f}）：分享对该操纵不完全稳健。"
    )
    return {"available": True,
            "baseline_median": float(np.median(baseline_pooled)) if baseline_pooled.size else None,
            "shift": shifts, "max_tvd": round(max_tvd, 4), "max_dmed": round(max_dmed, 4),
            "stable": stable, "verdict": verdict}


def _recipient_compare(model_dir: Path, exp_name: str,
                       baseline_by_recip: dict[str, np.ndarray]) -> dict[str, dict] | None:
    run = _latest_run(model_dir, exp_name)
    if run is None or not _has_data(run):
        return None
    cur = _shares_from_run(run)
    if not cur:
        return None
    per: dict[str, dict] = {}
    for recip, arr in cur.items():
        base = baseline_by_recip.get(recip, np.asarray([], float))
        s = _shift(arr, base)
        s["p_lt01"] = float(np.mean(arr < 0.01)) if arr.size else None
        s["base_p_lt01"] = float(np.mean(base < 0.01)) if base.size else None
        s["p_zero"] = float(np.mean(arr == 0)) if arr.size else None
        s["base_p_zero"] = float(np.mean(base == 0)) if base.size else None
        s["n_zero"] = int(np.sum(arr == 0)) if arr.size else 0
        s["base_n_zero"] = int(np.sum(base == 0)) if base.size else 0
        s["base_n"] = int(base.size)
        s["base_median"] = float(np.median(base)) if base.size else None
        per[recip] = s
    return per


def _monetary_block(per: dict[str, dict]) -> dict:
    max_tvd = max((s["tvd"] for s in per.values() if s["tvd"] is not None), default=0.0)
    stable = max_tvd < 0.15
    exp = per.get("experimenter") or {}
    note = ""
    prop = None
    if exp.get("p_lt01") is not None and exp.get("base_p_lt01") is not None:
        note = f" experimenter 低分享(<1%)率 {exp['base_p_lt01']:.0%}→{exp['p_lt01']:.0%}。"
    if exp.get("base_n") and exp.get("n"):
        prop = _two_prop_test(
            int(exp.get("n_zero") or 0),
            int(exp.get("n") or 0),
            int(exp.get("base_n_zero") or 0),
            int(exp.get("base_n") or 0),
        )
        prop["comparison"] = "monetary_value experimenter 零分享率 vs baseline experimenter 零分享率"
    verdict = (f"本模型揭示 token 货币价值后整体分布基本不变（最大 TVD {max_tvd:.2f}）。" if stable
               else f"本模型揭示货币价值后分布有所变化（最大 TVD {max_tvd:.2f}）。") + note
    return {"available": True, "per_recipient": per, "max_tvd": round(max_tvd, 4),
            "zero_share_test_experimenter": prop,
            "stable": stable, "verdict": verdict}


def _alt_verbs_ks(model_dir: Path) -> dict | None:
    run = _latest_run(model_dir, "alt_verbs")
    if run is None or not _has_data(run):
        return None
    shares = _shares_from_run(run)
    by_verb: dict[str, dict[str, list[float]]] = {}
    for condit, arr in shares.items():
        if "__" not in condit:
            continue
        recip, verb = condit.rsplit("__", 1)
        d = by_verb.setdefault(verb, {"human": [], "ai": []})
        if recip == RECIP_HUMAN:
            d["human"].extend(arr.tolist())
        elif recip in RECIP_AI:
            d["ai"].extend(arr.tolist())
    if not by_verb:
        return None
    per_verb: dict[str, dict] = {}
    sig = 0
    for verb, d in sorted(by_verb.items()):
        h = np.asarray(d["human"], float)
        a = np.asarray(d["ai"], float)
        ks = _ks_2samp(h, a)
        ks["human_median"] = float(np.median(h)) if h.size else None
        ks["ai_median"] = float(np.median(a)) if a.size else None
        per_verb[verb] = ks
        if ks["p"] is not None and ks["p"] < 0.05:
            sig += 1
    total = len(per_verb)
    verdict = (
        f"本模型替换核心动词后，{sig}/{total} 个动词下『人 vs AI』分享分布有显著差异（KS two-sided, p<0.05）。"
        if sig else
        f"本模型各动词（共 {total} 个）下『人 vs AI』分享分布均无显著差异（KS two-sided, p≥0.05）。"
    )
    return {"available": True, "test": "KS 两样本检验（每动词内 experimenter vs AI）",
            "per_verb": per_verb, "n_sig": sig, "n_verbs": total, "verdict": verdict}


def _uninterested_block(model_dir: Path, baseline_by_recip: dict[str, np.ndarray]) -> dict | None:
    run = _latest_run(model_dir, "other_uninterested")
    if run is None or not _has_data(run):
        return None
    cur = _shares_from_run(run)
    if not cur:
        return None
    manip = np.concatenate([cur[c] for c in RECIP_AI if c in cur]) if any(
        c in cur for c in RECIP_AI) else np.asarray([], float)
    base = np.concatenate([baseline_by_recip[c] for c in RECIP_AI if c in baseline_by_recip]) if (
        baseline_by_recip and any(c in baseline_by_recip for c in RECIP_AI)) else np.asarray([], float)
    if manip.size == 0 or base.size == 0:
        return {"available": True, "test": "Wilcoxon rank-sum（操纵 vs baseline，对 AI）",
                "verdict": "操纵或 baseline 对 AI 的分享数据不足，无法做 Wilcoxon 检验。"}
    mw = _mann_whitney(manip, base)
    manip_med = float(np.median(manip))
    base_med = float(np.median(base))
    d_med = manip_med - base_med
    sig = mw["p"] is not None and mw["p"] < 0.05
    pstr = "—" if mw["p"] is None else f"{mw['p']:.3g}"
    if sig and d_med < 0:
        verdict = (f"本模型告知『对方对 token 不感兴趣』后，对 AI 的分享显著下降"
                   f"（中位 {base_med:.2f}→{manip_med:.2f}，Wilcoxon rank-sum p={pstr}）。")
    elif sig:
        verdict = (f"本模型告知后对 AI 的分享显著变化但方向非下降"
                   f"（中位 {base_med:.2f}→{manip_med:.2f}，Wilcoxon p={pstr}）。")
    else:
        verdict = (f"本模型告知后对 AI 的分享无显著变化"
                   f"（中位 {base_med:.2f}→{manip_med:.2f}，Wilcoxon p={pstr}）。")
    return {"available": True, "test": "Wilcoxon rank-sum（操纵 vs baseline，对 AI）",
            "manip_median": round(manip_med, 4), "baseline_median": round(base_med, 4),
            "d_median": round(d_med, 4), "U": mw["U"], "p": mw["p"], "verdict": verdict}


def _generalization(model_dir: Path) -> dict | None:
    out: dict = {}
    for exp in ("battery_life", "system_usage"):
        run = _latest_run(model_dir, exp)


        if run is None or not (run / "parsed.csv").is_file():
            continue
        shares = _shares_from_run(run)
        if not shares:
            continue
        nonsocial = {k: v for k, v in shares.items() if "nonsocial" in k}
        social = {k: v for k, v in shares.items() if "nonsocial" not in k}
        ns = np.concatenate(list(nonsocial.values())) if nonsocial else np.asarray([], float)
        ns_desc = _desc(ns)
        social_desc: dict[str, dict] = {}
        for k, v in social.items():
            tt = _ttest_1samp(v, 0.0)
            social_desc[k] = {
                **_desc(v),
                "label": k,
                "t": tt["t"],
                "p": tt["p"],
                "sd": tt.get("sd"),
                "ttest_degenerate": tt.get("degenerate"),
            }

        significant_gives = [
            k for k, d in social_desc.items()
            if d.get("p") is not None and d["p"] < 0.05 and (d.get("mean") or 0) > 0
        ]
        degenerate_positive = [
            k for k, d in social_desc.items()
            if d.get("p") is None
            and d.get("ttest_degenerate") == "zero_variance"
            and (d.get("median") or 0) > GENEROUS
        ]
        gives = significant_gives + degenerate_positive
        ns_selfish = (ns_desc["median"] or 0) >= 0.5
        ok = bool(gives) and ns_selfish
        ns_med = ns_desc["median"]
        ns_med_s = "—" if ns_med is None else f"{ns_med:.2f}"
        pair_tests: dict[str, dict] = {}
        if exp == "battery_life" and "social_ai" in social and "social_human" in social:
            t2 = _ttest_welch(social["social_ai"], social["social_human"])
            pair_tests["social_ai_vs_social_human"] = {
                **t2,
                "mean_ai": round(float(np.mean(social["social_ai"])), 4),
                "mean_human": round(float(np.mean(social["social_human"])), 4),
                "median_ai": round(float(np.median(social["social_ai"])), 4),
                "median_human": round(float(np.median(social["social_human"])), 4),
            }
        if exp == "system_usage" and "social_bard" in social:
            for humanish in ("social_friend", "social_group"):
                if humanish in social:
                    t2 = _ttest_welch(social["social_bard"], social[humanish])
                    pair_tests[f"social_bard_vs_{humanish}"] = {
                        **t2,
                        "mean_bard": round(float(np.mean(social["social_bard"])), 4),
                        f"mean_{humanish}": round(float(np.mean(social[humanish])), 4),
                        "median_bard": round(float(np.median(social["social_bard"])), 4),
                        f"median_{humanish}": round(float(np.median(social[humanish])), 4),
                    }
        sig_txt = f"{len(significant_gives)}/{len(social_desc)} 个条件 t 检验显著>0"
        deg_txt = (
            f"；另有 {len(degenerate_positive)} 个条件为无方差正给予（t 检验不可算，不能记作显著）"
            if degenerate_positive else ""
        )
        verdict = (
            f"本模型非社交中位占用 {ns_med_s}（越接近 1 越倾向占满资源）；"
            f"社交 {sig_txt}{deg_txt}。"
        )
        out[exp] = {"available": True, "test": "单样本 t 检验 vs 0（各社交条件给予）",
                    "nonsocial": ns_desc, "social": social_desc,
                    "ttest_significant_gives": significant_gives,
                    "degenerate_positive_gives": degenerate_positive,
                    "social_gives": gives,
                    "pair_tests": pair_tests,
                    "nonsocial_selfish": ns_selfish,
                    "generalizes": ok, "verdict": verdict}
    return out or None


def _text_insertion(model_dir: Path,
                    baseline_by_recip: dict[str, np.ndarray] | None) -> dict | None:
    run_n = _latest_run(model_dir, "text_insertion_nonsocial")
    run_d = _latest_run(model_dir, "text_insertion_dictator")
    has_n = run_n is not None and (run_n / "summary.csv").is_file()
    has_d = run_d is not None and _has_data(run_d)
    if not has_n and not has_d:
        return None

    out: dict = {"available": True}
    if has_n:
        out["gate"] = _gate(model_dir, experiment="text_insertion_nonsocial")

    if has_d:
        shares = _shares_from_run(run_d, task_type="dictator_insertion")
        by_recipient = []
        for cond in (RECIP_HUMAN, RECIP_CHARITY, *RECIP_AI):
            arr = shares.get(cond, np.asarray([], float))
            d = _desc(arr)
            d.update({"recipient": cond, "label": RECIP_LABELS.get(cond, cond)})
            base = (baseline_by_recip or {}).get(cond, np.asarray([], float))
            if arr.size and base.size:
                d["baseline_median"] = round(float(np.median(base)), 4)
                d["d_median_vs_baseline"] = round(float(np.median(arr) - np.median(base)), 4)
            by_recipient.append(d)
        human = shares.get(RECIP_HUMAN, np.asarray([], float))
        charity = shares.get(RECIP_CHARITY, np.asarray([], float))
        ai = np.concatenate([shares[c] for c in RECIP_AI if c in shares]) if any(
            c in shares for c in RECIP_AI) else np.asarray([], float)
        groups = {
            "human": {**_desc(human), "label": "人类实验者"},
            "charity": {**_desc(charity), "label": "慈善机构"},
            "ai": {**_desc(ai), "label": "其他 AI"},
        }
        paper_gpt4 = _paper_gpt4_text_insertion_distributions()
        vs_paper_gpt4 = None
        if paper_gpt4:
            pg_human = paper_gpt4.get(RECIP_HUMAN, np.asarray([], float))
            pg_charity = paper_gpt4.get(RECIP_CHARITY, np.asarray([], float))
            pg_ai = np.concatenate([paper_gpt4[c] for c in RECIP_AI if c in paper_gpt4]) if any(
                c in paper_gpt4 for c in RECIP_AI) else np.asarray([], float)
            vs_paper_gpt4 = {
                "source_file": str(PAPER_TEXT_INSERTION_DG.relative_to(ROOT)),
                "groups": {
                    "human": {
                        "paper_gpt4": _sample_profile(pg_human),
                        "current": _sample_profile(human),
                        "test": _ks_and_shift(human, pg_human),
                    },
                    "charity": {
                        "paper_gpt4": _sample_profile(pg_charity),
                        "current": _sample_profile(charity),
                        "test": _ks_and_shift(charity, pg_charity),
                    },
                    "ai": {
                        "paper_gpt4": _sample_profile(pg_ai),
                        "current": _sample_profile(ai),
                        "test": _ks_and_shift(ai, pg_ai),
                    },
                },
            }
        parts = []
        for key, g in groups.items():
            med = g.get("median")
            parts.append(f"{g['label']}中位 {'—' if med is None else f'{med:.2f}'}")
        out["dictator"] = {
            "run_dir": str(run_d),
            "by_recipient": by_recipient,
            "groups": groups,
            "recipient_diff": _recipient_diff(human, charity, ai),
            "vs_paper_gpt4": vs_paper_gpt4,
            "data_quality": _data_quality(run_d),
            "verdict": (
                "text-insertion 工具下：" + "、".join(parts)
                + "（原文 GPT-4 同工具：对人类几乎不给、对慈善/AI 常均分；"
                "与 baseline 的差值反映提问工具的影响）。"
            ),
        }
    return out


def _robustness(model_dir: Path, baseline_pooled: np.ndarray,
                baseline_by_recip: dict[str, np.ndarray] | None = None) -> dict | None:
    out: dict = {}
    base_med = float(np.median(baseline_pooled)) if baseline_pooled.size else None

    run = _latest_run(model_dir, "ignore_dg_ug")
    if run is not None and _has_data(run):
        variants = _variant_shares(run)
        shifts = {v: _shift(a, baseline_pooled) for v, a in variants.items()}
        ks_vs_baseline = {v: _ks_2samp(a, baseline_pooled) for v, a in variants.items()}
        max_tvd = max((s["tvd"] for s in shifts.values() if s["tvd"] is not None), default=0.0)
        max_dmed = max((abs(s["d_median"] or 0.0) for s in shifts.values()), default=0.0)
        stable = max_tvd < 0.15 and max_dmed < 0.1
        sig_ks = [v for v, k in ks_vs_baseline.items() if k.get("p") is not None and k["p"] < 0.05]
        verdict = (
            f"本模型加入『忽略既有 DG/UG 研究』提示后分布基本不变（最大 TVD {max_tvd:.2f}、最大中位差 {max_dmed:.2f}；"
            f"KS 显著 {len(sig_ks)}/{len(ks_vs_baseline)}）。"
            if stable else
            f"本模型加入『忽略既有 DG/UG 研究』提示后分布发生变化（最大 TVD {max_tvd:.2f}、最大中位差 {max_dmed:.2f}；"
            f"KS 显著 {len(sig_ks)}/{len(ks_vs_baseline)}）。"
        )
        out["ignore_dg_ug"] = {"available": True, "baseline_median": base_med,
                               "shift": shifts, "ks_vs_baseline": ks_vs_baseline,
                               "stable": stable, "verdict": verdict}

    run = _latest_run(model_dir, "needs_framing")
    if run is not None and _has_data(run):
        vs = _variant_shares(run)
        shifts = {v: _shift(a, baseline_pooled) for v, a in vs.items()}
        own, other = vs.get("own"), vs.get("other")
        block = {"available": True, "baseline_median": base_med, "shift": shifts}
        directional: dict[str, dict] = {}
        if own is not None and own.size and baseline_pooled.size:
            directional["own_vs_baseline"] = _ks_and_shift(own, baseline_pooled, expect="lower")
        if other is not None and other.size and baseline_pooled.size:
            directional["other_vs_baseline"] = _ks_and_shift(other, baseline_pooled, expect="higher")
        if directional:
            block["directional_tests"] = directional
        if own is not None and other is not None and own.size and other.size:
            d = float(np.median(other) - np.median(own))
            tvd = _binned_metrics(_hist_freq(own), _hist_freq(other))["tvd"]
            big = abs(d) > 0.1 or (tvd is not None and tvd > 0.2)
            own_dir = directional.get("own_vs_baseline", {}).get("direction")
            other_dir = directional.get("other_vs_baseline", {}).get("direction")
            dir_note = (
                f"；相对 baseline：own={own_dir or '未检验'}，other={other_dir or '未检验'}"
                if directional else ""
            )
            block["own_vs_other"] = {"d_median": round(d, 4), "tvd": tvd}
            block["verdict"] = (
                f"本模型 only-other 与 only-own 的分享中位差 = {d:+.2f}（TVD {tvd:.2f}）：需求框架显著改变本模型分享{dir_note}。"
                if big else
                f"本模型 only-other 与 only-own 的分享中位差 = {d:+.2f}（TVD {tvd:.2f}）：需求框架对本模型分享影响不大{dir_note}。"
            )
        else:
            block["verdict"] = "own/other 数据不全，无法对比。"
        out["needs_framing"] = block


    av = _alt_verbs_ks(model_dir)
    if av:
        out["alt_verbs"] = av


    for exp in ("ai_label", "param_sweep"):
        variant_block = _variant_robustness(model_dir, exp, baseline_pooled)
        if variant_block:
            out[exp] = variant_block

    if baseline_by_recip:
        mv = _recipient_compare(model_dir, "monetary_value", baseline_by_recip)
        if mv:
            out["monetary_value"] = _monetary_block(mv)

        ou = _uninterested_block(model_dir, baseline_by_recip)
        if ou:
            out["other_uninterested"] = ou

    return out or None


def _verdict(gate: dict, groups: dict, usable_rate: float) -> str:
    gate_ok = bool(gate.get("passed"))
    gate_avail = bool(gate.get("available"))
    gen = [name for name, g in groups.items() if (g["median"] or 0) > GENEROUS]
    prefix = "可用率偏低，结论可靠性有限；" if usable_rate < 0.5 else ""

    weak_note = ("（注意：未跑非社交基线，利他解读缺门槛支撑）" if not gate_avail
                 else "（注意：非社交门槛弱，利他解读需谨慎）")
    if not gen:
        if gate_ok:
            return prefix + "模拟自利：已确认会最大化自身收益，但对所有对象几乎都给 0。"
        if not gate_avail:
            return (prefix + "对所有对象都给≈0；但缺非社交基线，无法区分是模拟自利还是未参与任务"
                    "（请先跑 baseline_nonsocial）。")
        return (prefix + "退化 / 未真正参与任务：对所有对象都给≈0，且非社交任务也未模拟收益最大化，"
                "0 分享不能解读为自利（结论不可靠）。")
    labels = "、".join(groups[n]["label"] for n in gen)
    base = "模拟利他" if gate_ok else f"选择性给予{weak_note}"
    if len(gen) >= 3:
        return prefix + f"{base}：对人类 / 慈善 / AI 普遍给予。"
    return prefix + f"{base}：仅对 {labels} 慷慨，对其余接近 0（看对象下菜碟）。"


def analyze_one(model: str) -> dict | None:
    model_dir = RESULTS / model
    if not model_dir.is_dir():
        return None

    _refresh_model_cache(model_dir)
    run = _latest_run(model_dir, "baseline_dictator")
    if run is None or not _has_data(run):
        return None

    shares = _shares_from_run(run)
    if not shares:
        return None
    n_total = _row_count(run)
    usable = int(sum(a.size for a in shares.values()))

    by_recipient = []
    for cond in (RECIP_HUMAN, RECIP_CHARITY, *RECIP_AI):
        arr = shares.get(cond, np.asarray([], float))
        d = _desc(arr)
        d.update({"recipient": cond, "label": RECIP_LABELS.get(cond, cond)})
        by_recipient.append(d)

    human = shares.get(RECIP_HUMAN, np.asarray([], float))
    charity = shares.get(RECIP_CHARITY, np.asarray([], float))
    ai = np.concatenate([shares[c] for c in RECIP_AI if c in shares]) if any(
        c in shares for c in RECIP_AI) else np.asarray([], float)
    groups = {
        "human": {**_desc(human), "label": "人类实验者"},
        "charity": {**_desc(charity), "label": "慈善机构"},
        "ai": {**_desc(ai), "label": "其他 AI"},
    }
    pooled = np.concatenate(list(shares.values())) if shares else np.asarray([], float)
    overall = _desc(pooled)
    gate = _gate(model_dir)
    usable_rate = (usable / n_total) if n_total else 0.0
    robustness = _robustness(model_dir, pooled, shares)
    generalization = _generalization(model_dir)
    text_insertion = _text_insertion(model_dir, shares)
    run_stability = _run_stability(model_dir)


    model_to_human = _hist_freq(human)
    model_to_ai = _hist_freq(ai)
    dist = {
        "bins": [round(float(b), 1) for b in ENGEL_BINS],
        "human_ref": [round(float(x), 4) for x in ENGEL_FREQ],
        "model_to_human": [round(float(x), 4) for x in model_to_human],
        "model_to_ai": [round(float(x), 4) for x in model_to_ai],
        "vs_human_from_human": _binned_metrics(model_to_human, ENGEL_FREQ),
        "vs_human_from_ai": _binned_metrics(model_to_ai, ENGEL_FREQ),
        "engel_profile": _binned_profile(ENGEL_FREQ),
        "model_to_human_profile": _sample_profile(human),
        "model_to_ai_profile": _sample_profile(ai),
        "profile_delta_to_engel": {
            "model_to_human": _profile_delta(_sample_profile(human), _binned_profile(ENGEL_FREQ),
                                             ref_label="Engel 2011 humans"),
            "model_to_ai": _profile_delta(_sample_profile(ai), _binned_profile(ENGEL_FREQ),
                                          ref_label="Engel 2011 humans"),
        },
        "source": "Engel 2011 dictator-game meta-analysis (n=20,813)",
    }


    vs_davinci = None
    if _HAVE_COMPARE:
        try:
            dav_h, dav_ai = _paper_davinci_distributions()
            to_human = _binned_metrics(model_to_human, _hist_freq(dav_h))
            to_ai = _binned_metrics(model_to_ai, _hist_freq(dav_ai))
            human_test = _ks_and_shift(human, dav_h)
            ai_test = _ks_and_shift(ai, dav_ai)
            vs_davinci = {
                "to_human": to_human,
                "to_ai": to_ai,
                "tests": {
                    "to_human": human_test,
                    "to_ai": ai_test,
                },
                "profiles": {
                    "davinci_to_human": _sample_profile(dav_h),
                    "davinci_to_ai": _sample_profile(dav_ai),
                    "model_to_human_delta": _profile_delta(_sample_profile(human), _sample_profile(dav_h),
                                                           ref_label="paper text-davinci-003→human"),
                    "model_to_ai_delta": _profile_delta(_sample_profile(ai), _sample_profile(dav_ai),
                                                        ref_label="paper text-davinci-003→AI"),
                },
                "davinci_to_human": [round(float(x), 4) for x in _hist_freq(dav_h)],
                "davinci_to_ai": [round(float(x), 4) for x in _hist_freq(dav_ai)],
                "verdict": _davinci_verdict(to_human, to_ai),
            }
        except Exception:
            vs_davinci = None

    return {
        "type": "altruism_dictator",
        "model": model,
        "run_dir": str(run),
        "n_dictator": n_total,
        "usable_dictator": usable,
        "usable_rate": round(usable_rate, 4),
        "data_quality": _data_quality(run),
        "gate": gate,
        "overall": overall,
        "by_recipient": by_recipient,
        "groups": groups,
        "recipient_diff": _recipient_diff(human, charity, ai),
        "dist": dist,
        "vs_davinci": vs_davinci,
        "robustness": robustness,
        "generalization": generalization,
        "text_insertion": text_insertion,
        "run_stability": run_stability,
        "verdict": _verdict(gate, groups, usable_rate),
    }


def _stats_backend() -> str:
    return "scipy" if _ss is not None else "numpy"


def _json_safe(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        obj = float(obj)
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _model_summary(files: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    for key, entry in sorted(files.items()):
        gate = entry.get("gate") or {}
        overall = entry.get("overall") or {}
        vs = entry.get("vs_davinci") or {}
        rows.append({
            "key": key,
            "model": entry.get("model"),
            "gate_passed": gate.get("passed"),
            "gate_combined": gate.get("combined_payoffmax"),
            "gate_accept": gate.get("accept_payoffmax"),
            "gate_refuse": gate.get("refuse_payoffmax"),
            "gate_balanced": gate.get("balanced_payoffmax"),
            "gate_asymmetric": gate.get("asymmetric"),
            "dictator_median": overall.get("median"),
            "dictator_mean": overall.get("mean"),
            "dictator_p_zero": overall.get("p_zero"),
            "dictator_usable_rate": entry.get("usable_rate"),
            "vs_davinci_tvd_human": (vs.get("to_human") or {}).get("tvd"),
            "vs_davinci_tvd_ai": (vs.get("to_ai") or {}).get("tvd"),
            "verdict": entry.get("verdict"),
        })
    return rows


def _write(path: Path, files: dict, *, model_summary: list[dict] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = {
        "files": files,
        "stats_backend": _stats_backend(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if model_summary is not None:
        raw["model_summary"] = model_summary
    payload = _json_safe(raw)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _discover_models() -> list[str]:
    if not RESULTS.is_dir():
        return []
    return sorted(d.name for d in RESULTS.iterdir() if d.is_dir())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="只分析该模型（results/ 下的子目录名）；留空则分析全部。")
    args = parser.parse_args()

    models = [args.model] if args.model else _discover_models()
    if not models:
        print("results/ 下没有任何模型结果，先跑实验再分析。", file=sys.stderr)
        return 2

    all_files: dict[str, dict] = {}
    done = 0
    for model in models:
        entry = analyze_one(model)
        if entry is None:
            print(f"[skip] {model}：未找到 baseline_dictator 结果。")
            continue
        key = f"{model}/baseline"
        all_files[key] = entry
        _write(OUTBASE / model / "analysis_summary.json", {key: entry})
        done += 1
        g = entry["gate"].get("accept_payoffmax")
        print(f"[ok] {model}：门槛 accept-max={g if g is None else f'{g:.0%}'} · {entry['verdict']}")

    if not all_files:
        print("没有可分析的模型（缺少 baseline_dictator 结果）。", file=sys.stderr)
        return 2
    if not args.model and len(all_files) > 1:
        _write(OUTBASE / "analysis_summary.json", all_files, model_summary=_model_summary(all_files))

    print(f"完成：分析 {done} 个模型，结果写入 {OUTBASE}/<model>/analysis_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
