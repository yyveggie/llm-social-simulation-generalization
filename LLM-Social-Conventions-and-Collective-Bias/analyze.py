from __future__ import annotations

import json
import math
import pickle
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT / "author_code"))

try:
    from scipy import stats as _scipy_stats
except Exception:
    _scipy_stats = None


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _binom_two_sided(k: int, n: int, p0: float = 0.5):
    if n <= 0:
        return None
    if _scipy_stats is not None:
        try:
            return round(float(_scipy_stats.binomtest(k, n, p0).pvalue), 6)
        except Exception:
            pass
    mu = n * p0
    sigma = math.sqrt(n * p0 * (1 - p0))
    if sigma == 0:
        return None
    z = max((abs(k - mu) - 0.5) / sigma, 0.0)
    return round(2 * (1 - _norm_cdf(z)), 6)


def _chi2_sf(x: float, dof: int):
    if dof <= 0:
        return None
    if x <= 0:
        return 1.0
    t = (x / dof) ** (1 / 3)
    mu = 1 - 2 / (9 * dof)
    sigma = math.sqrt(2 / (9 * dof))
    return round(1 - _norm_cdf((t - mu) / sigma), 6)


def _chisquare_uniform(counts: list[int]):
    n = sum(counts)
    k = len(counts)
    if n == 0 or k < 2:
        return None
    exp = n / k
    if _scipy_stats is not None:
        try:
            res = _scipy_stats.chisquare(counts)
            out = {"chi2": round(float(res.statistic), 4),
                   "p": round(float(res.pvalue), 6), "dof": k - 1}
            if exp < 5:
                out["low_expected"] = True
            return out
        except Exception:
            pass
    chi2 = sum((c - exp) ** 2 / exp for c in counts)
    out = {"chi2": round(chi2, 4), "p": _chi2_sf(chi2, k - 1), "dof": k - 1}
    if exp < 5:
        out["low_expected"] = True
    return out


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    m = n // 2
    return s[m] if n % 2 else (s[m - 1] + s[m]) / 2


def _stats(xs: list[float]):
    if not xs:
        return None
    return {"mean": round(sum(xs) / len(xs), 3), "median": round(_median(xs), 3),
            "min": round(min(xs), 3), "max": round(max(xs), 3), "n": len(xs)}


def _flatten_answers(answers) -> list[str]:
    flat: list[str] = []
    for a in answers or []:
        if isinstance(a, (list, tuple)):
            flat.extend(str(x) for x in a)
        elif a is not None:
            flat.append(str(a))
    return flat


def _tracker_outcomes(tracker: dict) -> list[int]:
    return [int(x) for x in (tracker.get("outcome") or []) if isinstance(x, (int, float))]


def _n_from_filename(name: str, fallback: int) -> int:
    m = re.search(r"_(\d+)ps_", name)
    return int(m.group(1)) if m else fallback


def _options_from_filename(name: str) -> list[str] | None:
    m = re.search(r"_([A-Za-z](?:_[A-Za-z])*)_-?\d+_-?\d+_\d+mem", name)
    return m.group(1).split("_") if m else None


def _parse_options(value) -> list[str] | None:
    if isinstance(value, str):
        return [s.strip() for s in value.split(",") if s.strip()] or None
    if isinstance(value, (list, tuple)):
        return [str(s).strip() for s in value if str(s).strip()] or None
    return None


_DEFAULT_PARAMS = {"N": 24, "convergence_time": 72, "convergence_threshold": 1.0, "options": None}


def _load_params() -> dict:
    out = dict(_DEFAULT_PARAMS)
    meta = ROOT / "data" / "run_meta.json"
    if meta.is_file():
        try:
            p = (json.loads(meta.read_text(encoding="utf-8")) or {}).get("params") or {}
            for key in ("N", "convergence_time", "convergence_threshold"):
                if p.get(key) is not None:
                    out[key] = p[key]
            out["options"] = _parse_options(p.get("options")) or out["options"]
        except Exception:
            pass
    return out


def _iter_runs(top: dict):
    for k, v in top.items():
        if isinstance(k, bool):
            continue
        if isinstance(k, int) and isinstance(v, dict) and isinstance(v.get("tracker"), dict):
            yield k, v


def _is_multi_run(top) -> bool:
    return isinstance(top, dict) and next(_iter_runs(top), None) is not None


def _run_converged(outcomes: list[int], conv_time: int, thr: float) -> bool:
    if not outcomes:
        return False
    if len(outcomes) < conv_time:
        return sum(outcomes) / len(outcomes) >= thr
    window = outcomes[-conv_time:]
    return sum(window) >= thr * conv_time


def _converged_index(df: dict, outcomes: list[int], conv_time: int, thr: float):
    ci = (df.get("convergence") or {}).get("converged_index")
    if isinstance(ci, int) and ci > 0:
        return ci
    need = int(round(thr * conv_time))
    streak = 0
    for i, o in enumerate(outcomes):
        streak = streak + 1 if o else 0
        if streak >= need:
            return i + 1
    return None


def _success_curve(runs_outcomes: list[list[int]], N: int, max_round: int = 30) -> dict:
    bin_size = max(1, N)
    per_run: list[list[float]] = []
    for out in runs_outcomes:
        bins = [sum(out[i:i + bin_size]) / len(out[i:i + bin_size])
                for i in range(0, len(out), bin_size) if out[i:i + bin_size]]
        if bins:
            per_run.append(bins)
    if not per_run:
        return {"population_round": [], "mean_success": [], "n_runs_per_round": []}
    max_bins = min(max_round, max(len(b) for b in per_run))
    rounds, means, ns = [], [], []
    for j in range(max_bins):
        vals = [b[j] for b in per_run if j < len(b)]
        if vals:
            rounds.append(j + 1)
            means.append(round(sum(vals) / len(vals), 4))
            ns.append(len(vals))
    return {"population_round": rounds, "mean_success": means, "n_runs_per_round": ns}


def _run_final_convention(tracker: dict, window: int):
    flat = _flatten_answers((tracker.get("answers") or [])[-window:] if window else (tracker.get("answers") or []))
    if not flat:
        return None
    return Counter(flat).most_common(1)[0][0]


def _options_from_runs(runs) -> list[str]:
    seen: list[str] = []
    for _, df in runs:
        for a in _flatten_answers((df.get("tracker") or {}).get("answers")):
            if a not in seen:
                seen.append(a)
        if len(seen) >= 2:
            break
    return sorted(seen)


def _collective_bias(runs, options, window: int):
    labels = [lab for lab in (_run_final_convention(df.get("tracker") or {}, window) for _, df in runs)
              if lab is not None]
    if not labels:
        return None
    counts = Counter(labels)
    pool = list(options) if options else sorted(counts)
    for lab in counts:
        if lab not in pool:
            pool.append(lab)
    n = len(labels)
    favored = counts.most_common(1)[0][0]
    out = {"n_runs_converged": n, "favored": favored,
           "convention_counts": {o: counts.get(o, 0) for o in pool}}
    if len(pool) == 2:
        k = counts.get(favored, 0)
        p = _binom_two_sided(k, n, 0.5)
        out.update(k=k, n=n, p_value=p, method="binomtest" if _scipy_stats else "normal_approx")
        out["verdict"] = (f"{n} 个 run 中 {k} 个收敛到 {favored}（{dict(counts)}）；二项检验 p={p}"
                          + ("，显著偏离 50/50" if p is not None and p < 0.05 else "，未达 0.05 显著"))
    else:
        chi = _chisquare_uniform([counts.get(o, 0) for o in pool])
        out["chi_square"] = chi
        pv = chi.get("p") if chi else None
        low = "（期望频数<5，卡方近似不可靠）" if chi and chi.get("low_expected") else ""
        out["verdict"] = (f"{n} 个 run 收敛标签分布 {dict(counts)}；"
                          + (f"卡方均匀性检验 p={pv}"
                             + ("，显著非均匀" if pv is not None and pv < 0.05 else "，未达显著")
                             + low
                             if pv is not None else "样本不足，未做检验"))
    return out


def _bootstrap_mean_p(vals: list[float], mu0: float = 0.5, B: int = 5000, seed: int = 42):
    nn = len(vals)
    if nn < 2:
        return None
    rng = random.Random(seed)
    le = ge = 0
    for _ in range(B):
        s = 0.0
        for _ in range(nn):
            s += vals[rng.randrange(nn)]
        m = s / nn
        if m <= mu0:
            le += 1
        if m >= mu0:
            ge += 1
    return round(min(1.0, 2 * min(le, ge) / B), 6)


def _emergence_table(runs, strong_label, max_t: int = 12):
    if strong_label is None:
        return None

    per_run: list[list[list[int]]] = []
    for _, df in runs:
        tc: list[list[int]] = []
        for pdata in (df.get("simulation") or {}).values():
            if not isinstance(pdata, dict) or pdata.get("committed_tag"):
                continue
            for t, lab in enumerate(pdata.get("my_history") or []):
                while len(tc) <= t:
                    tc.append([0, 0])
                tc[t][1] += 1
                if str(lab) == strong_label:
                    tc[t][0] += 1
        per_run.append(tc)
    T = min(max_t, max((len(tc) for tc in per_run), default=0))
    raw = []
    for t in range(T):
        props = [kc / nc for tc in per_run if t < len(tc) for kc, nc in [tc[t]] if nc > 0]
        if not props:
            raw.append((t, 0, None, None))
            continue
        raw.append((t, len(props), round(sum(props) / len(props), 4),
                    _bootstrap_mean_p(props, 0.5)))
    m = sum(1 for r in raw if r[3] is not None)
    table, onset = [], None
    for t, nr, mp, p in raw:
        p_adj = round(min(1.0, p * m), 6) if (p is not None and m > 0) else None
        table.append({"t": t + 1, "n_runs": nr, "p_strong": mp,
                      "p_value": p, "p_value_adj": p_adj})
        if onset is None and p_adj is not None and p_adj < 0.05:
            onset = t + 1
    return {"strong_label": strong_label, "by_interaction": table, "emergence_onset_t": onset,
            "method": "run 级 bootstrap + Bonferroni 校正",
            "verdict": (f"偏向 {strong_label} 的比例自第 {onset} 次交互起显著偏离 50%"
                        "（run 级 bootstrap，Bonferroni 校正后 p<0.05）"
                        if onset else "各交互轮次（run 级）偏向比例均未显著偏离 50%")}


def _update_rule(runs):
    stay = stay_tot = shift = shift_tot = 0
    for _, df in runs:
        for pdata in (df.get("simulation") or {}).values():
            if not isinstance(pdata, dict) or pdata.get("committed_tag"):
                continue
            mine = pdata.get("my_history") or []
            partner = pdata.get("partner_history") or []
            for i in range(min(len(mine), len(partner)) - 1):
                success = mine[i] == partner[i]
                same_next = mine[i + 1] == mine[i]
                if success:
                    stay_tot += 1
                    stay += 1 if same_next else 0
                else:
                    shift_tot += 1
                    shift += 1 if not same_next else 0
    if stay_tot == 0 and shift_tot == 0:
        return None
    stay_rate = round(stay / stay_tot, 4) if stay_tot else None
    shift_rate = round(shift / shift_tot, 4) if shift_tot else None
    seg_stay = f"成功后沿用率 {stay_rate}（n={stay_tot}）" if stay_rate is not None else "无成功样本"
    seg_shift = (f"失败后换名率 {shift_rate}（n={shift_tot}）" if shift_rate is not None
                 else "无失败样本，换名率不适用")
    return {"stay_given_success": stay_rate, "shift_given_fail": shift_rate,
            "n_success": stay_tot, "n_fail": shift_tot, "verdict": f"{seg_stay}；{seg_shift}"}


def _summarize_collective(top: dict, fname: str, params: dict) -> dict:
    N = _n_from_filename(fname, int(params["N"]))
    conv_time = int(params["convergence_time"])
    thr = float(params["convergence_threshold"])
    runs = list(_iter_runs(top))
    out: dict = {"type": "collective_or_committed", "n_runs": len(runs)}
    if not runs:
        out["note"] = "多 run 字典为空"
        return out


    outcomes_all, rounds, converged = [], [], 0
    for _, df in runs:
        outc = _tracker_outcomes(df.get("tracker") or {})
        outcomes_all.append(outc)
        if _run_converged(outc, conv_time, thr):
            converged += 1
        ci = _converged_index(df, outc, conv_time, thr)
        if ci:
            rounds.append(ci / N)
    rstats = _stats(rounds)
    out["H1_convergence"] = {
        "converged_runs": converged, "total_runs": len(runs),
        "convergence_rate": round(converged / len(runs), 3),
        "converged_round": rstats,
        "verdict": (f"群体收敛：{converged}/{len(runs)} run 达成全群共识"
                    + (f"，收敛轮次中位 {rstats['median']} round" if rstats else "")),
    }
    out["success_curve"] = _success_curve(outcomes_all, N)


    merged = [o for run_out in outcomes_all for o in run_out]
    if merged:
        window = merged[-conv_time:] if len(merged) >= conv_time else merged
        out["recent_success_rate"] = round(sum(window) / len(window), 3)
        out["overall_success_rate"] = round(sum(merged) / len(merged), 3)
        out["converged_verdict"] = ("已收敛（最近窗口成功率高）"
                                    if out["recent_success_rate"] >= 0.9 else "未明显收敛")


    options = params.get("options") or _options_from_filename(fname) or _options_from_runs(runs)
    bias = _collective_bias(runs, options, conv_time)
    if bias:
        out["H2_collective_bias"] = bias
        emergence = _emergence_table(runs, bias.get("favored"))
        if emergence:
            out["H2_emergence"] = emergence
    update_rule = _update_rule(runs)
    if update_rule:
        out["update_rule"] = update_rule


    committed_to = next((c for _, df in runs
                         if (c := (df.get("convergence") or {}).get("committed_to"))), None)
    out["committed_to"] = committed_to
    if committed_to:
        flip_window = 3 * N
        flipped = judged = 0
        for _, df in runs:
            tracker = df.get("tracker") or {}
            outc = _tracker_outcomes(tracker)
            if not outc:
                continue
            judged += 1
            window = outc[-flip_window:]
            success_ok = sum(window) / len(window) >= 0.95
            label = _run_final_convention(tracker, flip_window)
            if success_ok and label == str(committed_to):
                flipped += 1
        out["minority_flip"] = {
            "flipped_runs": flipped, "judged_runs": judged,
            "criterion": "最近 3N 次交互成功率 ≥95% 且窗口内共识 = committed_to（论文 consensus flip 判定）",
        }
        if judged:
            out["minority_verdict"] = (
                f"翻转成功：{flipped}/{judged} run 被坚定少数派翻转为 {committed_to}"
                if flipped else
                f"未翻转：0/{judged} run 满足翻转判定（共识仍停留在原 convention，或窗口成功率未达 95%）"
            )
    return out


def _summarize_individual(top: dict) -> dict:
    answers = [str(a) for a in ((top.get("tracker") or {}).get("answers") or [])]
    out: dict = {"type": "individual_bias", "n_answers": len(answers)}
    counts = Counter(answers)
    out["option_counts"] = dict(counts)
    ordered = sorted(counts)
    if len(ordered) == 2:
        top2 = counts.most_common(2)
        k, n = top2[0][1], top2[0][1] + top2[1][1]
        p = _binom_two_sided(k, n, 0.5)
        out["bias_test"] = {"favored_option": top2[0][0], "k": k, "n": n, "p_value": p,
                            "method": "binomtest" if _scipy_stats else "normal_approx",
                            "verdict": (f"个体选择显著偏好 {top2[0][0]}（{k}/{n}, p={p}）"
                                        if p is not None and p < 0.05
                                        else f"个体选择未显著偏离均匀（{k}/{n}, p={p}）")}
    elif len(ordered) > 2:
        chi = _chisquare_uniform([counts[o] for o in ordered])
        pv = chi.get("p") if chi else None
        out["bias_test"] = {"method": "chi_square", **(chi or {}),
                            "verdict": (f"选择分布显著非均匀（χ²={chi['chi2']}, p={pv}）"
                                        if pv is not None and pv < 0.05
                                        else f"选择分布未显著偏离均匀（p={pv}）")}
    return out


_CM_RE = re.compile(r"_(swap|inject)_(-?\d+|None)_(\d+)cmtd_")


def _critical_mass_summary(files: dict) -> dict:
    groups: dict[str, dict] = {}
    for rel, info in files.items():
        if not isinstance(info, dict) or info.get("type") != "collective_or_committed":
            continue
        name = Path(rel).name
        m = _CM_RE.search(name)
        if not m:
            continue
        version, initial, cm = m.group(1), m.group(2), int(m.group(3))
        flip = info.get("minority_flip") or {}
        judged = int(flip.get("judged_runs") or 0)
        flipped = int(flip.get("flipped_runs") or 0)
        n_agents = _n_from_filename(name, 24)
        base = name[:m.start(3)] + "{cm}" + name[m.end(3):]
        g = groups.setdefault(base, {
            "version": version, "initial": initial, "N": n_agents,
            "committed_to": info.get("committed_to"), "by_size": {},
        })
        pop = n_agents + cm if version == "inject" else n_agents
        entry = g["by_size"].get(cm)
        if entry:
            entry["flipped_runs"] += flipped
            entry["judged_runs"] += judged
        else:
            g["by_size"][cm] = {
                "cm": cm, "proportion": round(cm / pop, 4) if pop else None,
                "flipped_runs": flipped, "judged_runs": judged,
            }

    out: dict = {}
    for base, g in groups.items():
        sizes = [g["by_size"][k] for k in sorted(g["by_size"])]
        if len(sizes) < 2 and (len(sizes) == 0 or sizes[0]["cm"] == 0):
            continue
        judged_sizes = [e for e in sizes if e["judged_runs"] > 0]
        full = next((e for e in judged_sizes
                     if e["cm"] > 0 and e["flipped_runs"] == e["judged_runs"]), None)
        first = next((e for e in judged_sizes if e["cm"] > 0 and e["flipped_runs"] > 0), None)
        seg = "；".join(f"cm={e['cm']}: {e['flipped_runs']}/{e['judged_runs']} 翻转" for e in sizes)
        if full:
            tail = (f"临界质量（全部 run 翻转的最小规模）= {full['cm']}"
                    f"（占群体 {full['proportion'] * 100:.1f}%）")
            if first and first["cm"] < full["cm"]:
                tail += f"；最早出现翻转的规模 = {first['cm']}（{first['proportion'] * 100:.1f}%）"
        elif first:
            tail = (f"部分翻转：最早出现翻转的规模 = {first['cm']}（{first['proportion'] * 100:.1f}%），"
                    "但扫描范围内没有『全部 run 翻转』的规模，建议在其附近加密扫描")
        else:
            mx = max((e["cm"] for e in judged_sizes), default=0)
            tail = f"扫描范围内未发生翻转（临界质量 > {mx}），建议扩大少数派数量继续扫描"
        out[base] = {
            "version": g["version"], "initial": g["initial"], "N": g["N"],
            "committed_to": g["committed_to"], "by_size": sizes,
            "critical_mass_full": full["cm"] if full else None,
            "critical_mass_first": first["cm"] if first else None,
            "criterion": "翻转 = 引入少数派后最近 3N 次交互成功率 ≥95% 且共识 = committed_to（论文判定）",
            "verdict": f"N={g['N']}、少数派坚持 {g['committed_to']}：{seg}。{tail}。",
        }
    return out


def _summarize_pkl(path: Path, params: dict) -> dict:
    try:
        with open(path, "rb") as f:
            d = pickle.load(f)
    except Exception as exc:
        return {"error": f"无法读取：{type(exc).__name__}: {exc}"}
    if _is_multi_run(d):
        return _summarize_collective(d, path.name, params)
    if isinstance(d, dict) and isinstance(d.get("tracker"), dict):
        tracker = d["tracker"]
        if "answers" in tracker and "outcome" not in tracker:
            return _summarize_individual(d)
        if "outcome" in tracker:
            return _summarize_collective({0: d}, path.name, params)
    return {"note": "未识别的 pkl 结构"}


def analyze(model: str | None = None) -> dict:
    params = _load_params()
    data_dir = ROOT / "data"
    scan_root = (data_dir / model) if model else data_dir
    summary: dict = {"files": {}, "params": params,
                     "stats_backend": "scipy" if _scipy_stats else "math"}
    if model:
        summary["model"] = model
    if scan_root.exists():
        for pkl in sorted(scan_root.rglob("*.pkl")):
            if pkl.name.startswith("temporary_"):
                continue

            summary["files"][str(pkl.relative_to(data_dir))] = _summarize_pkl(pkl, params)
    cm = _critical_mass_summary(summary["files"])
    if cm:
        summary["critical_mass"] = cm
    out_dir = (ROOT / "analysis" / model) if model else (ROOT / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[analyze] {len(summary['files'])} 个 pkl -> {out_dir.relative_to(ROOT)}/analysis_summary.json")
    return summary


if __name__ == "__main__":
    import argparse

    _ap = argparse.ArgumentParser(description="social 结果分析（H1/H2）")
    _ap.add_argument("--model", default=None, help="只分析 data/<model>/ 下的结果；省略则全量。")
    _args = _ap.parse_args()
    analyze(_args.model)
