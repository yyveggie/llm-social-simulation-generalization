#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import random
import statistics
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_IN = HERE / "metrics_long.csv"

_N_PERM = 10000
_SEED = 20260702


def _avg_ranks(vals: list[float]) -> list[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(x: list[float], y: list[float]) -> float | None:
    n = len(x)
    if n < 3:
        return None
    mx, my = statistics.mean(x), statistics.mean(y)
    sx = sum((a - mx) ** 2 for a in x) ** 0.5
    sy = sum((b - my) ** 2 for b in y) ** 0.5
    if sx == 0 or sy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def spearman_rho(x: list[float], y: list[float]) -> float | None:
    return _pearson(_avg_ranks(x), _avg_ranks(y))


def mann_kendall_s(seq: list[float]) -> int:
    s = 0
    n = len(seq)
    for i in range(n - 1):
        for j in range(i + 1, n):
            d = seq[j] - seq[i]
            s += (d > 0) - (d < 0)
    return s


def _perm_p(observed: float, stat_fn, values: list[float], rng: random.Random) -> float:
    if observed is None:
        return 1.0
    hits = 0
    pool = list(values)
    for _ in range(_N_PERM):
        rng.shuffle(pool)
        s = stat_fn(pool)
        if s is not None and abs(s) >= abs(observed) - 1e-12:
            hits += 1
    return (hits + 1) / (_N_PERM + 1)


def bh_fdr(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [1.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        running = min(running, pvals[idx] * m / rank)
        q[idx] = running
    return q


def load_long_table(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        try:
            r["_value"] = float(r["value"])
            r["_date"] = date.fromisoformat(r["release_date"]) if r.get("release_date") else None
        except (ValueError, KeyError):
            continue
        out.append(r)
    return out


def group_series(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], dict[str, dict]] = {}
    for r in rows:
        if r["_date"] is None:
            continue
        key = (r["project"], r["indicator"])
        grouped.setdefault(key, {}).setdefault(r["model"], r)
    return {
        key: sorted(by_model.values(), key=lambda r: (r["_date"], r["model"]))
        for key, by_model in grouped.items()
    }


def trend_tests(series: dict[tuple[str, str], list[dict]], min_models: int) -> list[dict]:
    rng = random.Random(_SEED)
    out: list[dict] = []
    for (project, indicator), pts in sorted(series.items()):
        if len(pts) < min_models:
            continue
        vals = [p["_value"] for p in pts]
        xs = list(range(len(pts)))
        rho = spearman_rho(xs, vals)
        p_rho = _perm_p(rho, lambda v: spearman_rho(xs, v), vals, rng) if rho is not None else ""
        s = mann_kendall_s(vals)
        n = len(vals)
        tau = s / (n * (n - 1) / 2)
        p_mk = _perm_p(float(s), lambda v: float(mann_kendall_s(v)), vals, rng)
        if rho is not None and p_rho != "" and p_rho < 0.05:
            direction = "上升" if rho > 0 else "下降"
        else:
            direction = "无显著单调趋势"
        out.append({
            "project": project, "indicator": indicator, "n_models": n,
            "spearman_rho": round(rho, 4) if rho is not None else "",
            "p_spearman_perm": round(p_rho, 5) if p_rho != "" else "",
            "mk_S": s, "kendall_tau": round(tau, 4),
            "p_mk_perm": round(p_mk, 5),
            "direction": direction,
            "first_model": pts[0]["model"], "first_date": str(pts[0]["_date"]),
            "first_value": pts[0]["_value"],
            "last_model": pts[-1]["model"], "last_date": str(pts[-1]["_date"]),
            "last_value": pts[-1]["_value"],
            "delta_last_minus_first": round(pts[-1]["_value"] - pts[0]["_value"], 6),
            "models_in_order": " -> ".join(p["model"] for p in pts),
        })

    testable = [r for r in out if r["p_spearman_perm"] != ""]
    if testable:
        qs = bh_fdr([r["p_spearman_perm"] for r in testable])
        for r, q in zip(testable, qs):
            r["q_fdr"] = round(q, 5)
    for r in out:
        r.setdefault("q_fdr", "")
    return out


def family_diffs(series: dict[tuple[str, str], list[dict]]) -> list[dict]:
    out: list[dict] = []
    for (project, indicator), pts in sorted(series.items()):
        by_family: dict[str, list[dict]] = {}
        for p in pts:
            fam = p.get("family") or ""
            if fam:
                by_family.setdefault(fam, []).append(p)
        for fam, members in by_family.items():
            if len(members) < 2:
                continue
            members = sorted(members, key=lambda r: (r["_date"], r["model"]))
            for a, b in zip(members, members[1:]):
                out.append({
                    "project": project, "indicator": indicator, "family": fam,
                    "model_from": a["model"], "date_from": str(a["_date"]), "value_from": a["_value"],
                    "model_to": b["model"], "date_to": str(b["_date"]), "value_to": b["_value"],
                    "diff_to_minus_from": round(b["_value"] - a["_value"], 6),
                })
    return out


def render_plots(series: dict[tuple[str, str], list[dict]], plot_dir: Path,
                 min_models: int) -> int:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plots] 未安装 matplotlib，跳过作图。")
        return 0
    plot_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for (project, indicator), pts in sorted(series.items()):
        if len(pts) < min_models:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.2))
        families = sorted({p.get("family") or "?" for p in pts})
        cmap = plt.get_cmap("tab10")
        colors = {f: cmap(i % 10) for i, f in enumerate(families)}
        for p in pts:
            fam = p.get("family") or "?"
            ax.scatter(p["_date"], p["_value"], color=colors[fam], s=45, zorder=3)
            ax.annotate(p["model"], (p["_date"], p["_value"]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        ax.plot([p["_date"] for p in pts], [p["_value"] for p in pts],
                color="grey", alpha=0.4, lw=1, zorder=1)
        ax.set_title(f"{project} · {indicator}", fontsize=10)
        ax.set_xlabel("release date")
        ax.set_ylabel("value")
        fig.autofmt_xdate()
        fig.tight_layout()
        safe = f"{project}__{indicator}".replace("/", "-").replace("[", "(").replace("]", ")")
        fig.savefig(plot_dir / f"{safe}.png", dpi=140)
        plt.close(fig)
        count += 1
    return count


def main() -> None:
    ap = argparse.ArgumentParser(description="跨模型演化趋势检验")
    ap.add_argument("--in", dest="input", default=str(DEFAULT_IN),
                    help="collect_metrics.py 产出的长表路径")
    ap.add_argument("--min-models", type=int, default=3,
                    help="纳入趋势检验所需的最少模型数（默认 3）")
    ap.add_argument("--plot-dir", default=None, help="可选：输出散点图目录")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.is_file():
        raise SystemExit(f"未找到 {in_path}；请先运行 python trends/collect_metrics.py 生成长表。")

    rows = load_long_table(in_path)
    n_undated = len({r["model"] for r in rows if r["_date"] is None})
    series = group_series(rows)

    tests = trend_tests(series, args.min_models)
    tests_path = HERE / "trend_tests.csv"
    if tests:
        cols = list(tests[0].keys())
        with open(tests_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(tests)

    diffs = family_diffs(series)
    diffs_path = HERE / "family_diffs.csv"
    if diffs:
        with open(diffs_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(diffs[0].keys()))
            w.writeheader()
            w.writerows(diffs)

    sig = [t for t in tests if t["direction"] != "无显著单调趋势"]
    print(f"[trends] 指标序列 {len(series)} 组；≥{args.min_models} 模型可检验 {len(tests)} 组，"
          f"其中 {len(sig)} 组呈显著单调趋势（逐条 p<0.05，q_fdr 列为 FDR 校正后值）")
    if n_undated:
        print(f"[trends] 有 {n_undated} 个模型缺 release_date（registry 待补全），未进入检验。")
    for t in sorted(sig, key=lambda r: r["p_spearman_perm"])[:15]:
        print(f"  {t['project']:22s} {t['indicator']:42s} "
              f"ρ={t['spearman_rho']:+.2f} p={t['p_spearman_perm']:.4f} q={t['q_fdr']} "
              f"{t['direction']} n={t['n_models']} Δ={t['delta_last_minus_first']:+g}")
    if tests:
        print(f"[trends] 明细 -> {tests_path}")
    if diffs:
        print(f"[trends] 家族内代际差分 {len(diffs)} 对 -> {diffs_path}")

    if args.plot_dir:
        n_fig = render_plots(series, Path(args.plot_dir), args.min_models)
        if n_fig:
            print(f"[trends] 已输出 {n_fig} 张散点图 -> {args.plot_dir}")


if __name__ == "__main__":
    main()
