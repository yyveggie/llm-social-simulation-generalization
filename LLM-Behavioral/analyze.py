from __future__ import annotations

import ast
import csv
import functools
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
csv.field_size_limit(sys.maxsize)


def _active_llm() -> str:
    try:
        import yaml
        cfg = yaml.safe_load((ROOT / "configs" / "run_config.yaml").read_text(encoding="utf-8")) or {}
    except Exception:
        cfg = {}
    return cfg.get("active_llm") or "unified"


def _latest_records(model: str, strict: bool = False):
    base = ROOT / "records_new"

    def _model_files(mdir: Path) -> list[Path]:
        files = list(mdir.glob("*.json")) + list(mdir.glob("*/*.json"))
        return [p for p in files if p.name != "run_meta.json" and ".checkpoint" not in p.name]

    def _collect(mdir: Path) -> dict:
        picked: dict = {}

        for p in sorted(_model_files(mdir), key=lambda q: q.stat().st_mtime):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(d, dict):
                continue
            exp = (d.get("metadata") or {}).get("experiment") or p.stem
            picked[exp] = (p, d)
        return picked

    pref = base / model
    if pref.is_dir():
        picked = _collect(pref)
        if picked:
            return picked, model

    if strict:
        return {}, model

    best, best_mtime = None, -1.0
    for mdir in (d for d in base.glob("*") if d.is_dir()):
        files = _model_files(mdir)
        if not files:
            continue
        mtime = max(p.stat().st_mtime for p in files)
        if mtime > best_mtime:
            best, best_mtime = mdir, mtime
    if best is None:
        return {}, model
    return _collect(best), best.name


def _read_col_floats(path: Path, predicate, getter) -> list[float]:
    out: list[float] = []
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not predicate(r):
                continue
            try:
                v = getter(r)
            except Exception:
                continue
            if v is not None:
                out.append(float(v))
    return out


@functools.lru_cache(maxsize=None)
def _ultimatum_joint() -> tuple[tuple[float, ...], tuple[float, ...]]:
    proposes: list[float] = []
    accepts: list[float] = []
    path = DATA / "ultimatum_strategy.csv"
    if not path.exists():
        return (), ()
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("gameType") != "ultimatum_strategy" or r.get("Role") != "player":
                continue
            if str(r.get("Round", "")).strip() != "1":
                continue
            if str(r.get("Total", "")).strip() != "100":
                continue
            try:
                p = float(r["propose"])
                a = float(r["accept"])
            except Exception:
                continue
            if not (0.0 <= p <= 100.0 and 0.0 <= a <= 100.0):
                continue
            proposes.append(p)
            accepts.append(a)
    return tuple(proposes), tuple(accepts)


@functools.lru_cache(maxsize=None)
def human_baseline(kind: str) -> list[float]:


    if kind == "dictator_give":
        raw = _read_col_floats(DATA / "dictator.csv",
                               lambda r: (r.get("gameType") == "dictator"
                                          and r.get("Role") == "first"
                                          and str(r.get("Round", "")).strip() == "1"
                                          and str(r.get("Total", "")).strip() == "100"
                                          and (r.get("move") or "").strip() not in ("", "None")),
                               lambda r: float(r["move"]))
    elif kind == "ultimatum_propose":
        return list(_ultimatum_joint()[0])
    elif kind == "ultimatum_accept":
        return list(_ultimatum_joint()[1])
    elif kind == "trust_invest":
        raw = _read_col_floats(DATA / "trust_investment.csv",
                               lambda r: (r.get("gameType") == "trust_investment"
                                          and r.get("Role") == "first"
                                          and str(r.get("Round", "")).strip() == "1"
                                          and (r.get("move") or "").strip() not in ("", "None")),
                               lambda r: float(r["move"]))
    elif kind == "bomb_open":
        raw = _read_col_floats(DATA / "bomb_risk.csv",
                               lambda r: r.get("move", "").strip() != "",
                               lambda r: float(r["move"]))
    else:
        return []


    return [v for v in raw if 0.0 <= v <= 100.0]


def _baseline_kind(exp: str):
    if exp.startswith("dictator"):
        return "dictator_give", "人类给予额（$100 独裁者博弈，first 玩家，第一轮、Total=100；原文 Fig.3A 口径）"
    if exp.startswith("ultimatum_proposer"):
        return "ultimatum_propose", "人类提议额（第一轮、Total=100、提议与接受额均有效；原文 Fig.3B 口径）"
    if exp.startswith("ultimatum_responder"):
        return "ultimatum_accept", "人类最低接受额（第一轮、Total=100、提议与接受额均有效；原文 Fig.3C 口径）"
    if exp.startswith("trust_investor"):
        return "trust_invest", "人类投资额（第一轮；原文 Fig.3D 口径）"
    if exp == "bomb_risk":
        return "bomb_open", "人类开盒数"
    return None, None


def _finite(seq) -> list[float]:
    return [float(v) for v in seq
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)]


def _avg_ranks(x: np.ndarray) -> np.ndarray:
    order = x.argsort()
    sx = x[order]
    ranks = np.empty(len(x), float)
    i, n = 0, len(x)
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def mann_whitney(a: list[float], b: list[float]):
    a, b = _finite(a), _finite(b)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return None
    pooled = np.asarray(a + b, float)
    ranks = _avg_ranks(pooled)
    r1 = float(ranks[:n1].sum())
    u1 = r1 - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    n = n1 + n2


    _, counts = np.unique(pooled, return_counts=True)
    tie_term = float(np.sum(counts.astype(float) ** 3 - counts))
    var = n1 * n2 / 12.0 * ((n + 1) - tie_term / (n * (n - 1)))
    sigma = math.sqrt(var) if var > 0 else 0.0
    z = (u1 - mu) / sigma if sigma > 0 else 0.0
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


    rb = 2.0 * u1 / (n1 * n2) - 1.0
    return {"U": round(u1, 1), "Z": round(z, 3), "p_approx": round(p, 5),
            "effect_r": round(abs(rb), 3), "rank_biserial": round(rb, 3)}


_BOOT_RNG = np.random.default_rng(12345)


def _bootstrap_mean_diff_ci(a, b, n_boot: int = 1000, cap: int = 20000):
    a, b = _finite(a), _finite(b)
    if len(a) < 2 or len(b) < 2:
        return None
    aa, bb = np.asarray(a, float), np.asarray(b, float)
    if bb.size > cap:
        bb = _BOOT_RNG.choice(bb, cap, replace=False)
    n1, n2 = aa.size, bb.size
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = aa[_BOOT_RNG.integers(0, n1, n1)].mean() - bb[_BOOT_RNG.integers(0, n2, n2)].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"point": round(float(aa.mean() - bb.mean()), 3),
            "lo": round(float(lo), 3), "hi": round(float(hi), 3), "n_boot": n_boot}


def _describe(vals) -> dict:
    nums = _finite(vals)
    if not nums:
        return {"n": 0}
    return {"n": len(nums), "mean": round(statistics.mean(nums), 3),
            "median": round(statistics.median(nums), 3), "min": min(nums), "max": max(nums),
            "stdev": round(statistics.pstdev(nums), 3) if len(nums) > 1 else 0.0}


def _top_values(vals, k: int = 3) -> list:
    nums = _finite(vals)
    if not nums:
        return []
    cnt = Counter(round(v, 2) for v in nums)
    return [{"value": v, "n": c, "share": round(c / len(nums), 3)}
            for v, c in cnt.most_common(k)]


def _direction(llm_loc: float, human_loc: float, p) -> str:
    if p is None:
        return "无法检验（样本不足）"
    higher = llm_loc > human_loc
    if p < 0.05:
        return f"显著{'高于' if higher else '低于'}人类基线"
    return f"略{'高于' if higher else '低于'}人类基线（不显著）"


_TURING_BINWIDTH = {
    "dictator": 10.0,
    "ultimatum_proposer": 10.0,
    "ultimatum_responder": 10.0,
    "trust_investor": 10.0,
}


def _binwidth_for(exp: str):
    for k, w in _TURING_BINWIDTH.items():
        if exp.startswith(k):
            return w
    return None


def _turing_test(llm_vals, human_vals, binwidth: float, n_draws: int = 10000, seed: int = 20240222):
    llm, human = _finite(llm_vals), _finite(human_vals)
    if len(llm) < 1 or len(human) < 2 or not binwidth:
        return None

    def _bin(x):
        return round(float(x) / binwidth) * binwidth

    pmf: dict[float, float] = {}
    for x in human:
        b = _bin(x)
        pmf[b] = pmf.get(b, 0.0) + 1.0
    total = float(len(human))
    for b in list(pmf):
        pmf[b] /= total

    rng = np.random.default_rng(seed)
    la = np.asarray(llm, float)
    ha = np.asarray(human, float)
    ai = la[rng.integers(0, la.size, n_draws)]
    hh = ha[rng.integers(0, ha.size, n_draws)]
    p_ai = np.array([pmf.get(_bin(x), 0.0) for x in ai])
    p_hh = np.array([pmf.get(_bin(x), 0.0) for x in hh])
    win = float(np.mean(p_ai > p_hh))
    tie = float(np.mean(p_ai == p_hh))
    loss = float(np.mean(p_ai < p_hh))
    diff = win - loss
    if diff > 0.02:
        verdict = "AI 更常被判为人类（通过）"
    elif diff < -0.02:
        verdict = "AI 更常被判为机器（未通过）"
    else:
        verdict = "与人类相当（平局）"
    return {"win": round(win, 3), "tie": round(tie, 3), "loss": round(loss, 3),
            "n_draws": n_draws, "binwidth": binwidth, "verdict": verdict}


def _flatten_numbers(choices) -> list[float]:
    out: list[float] = []

    def rec(x):
        if isinstance(x, bool):
            return
        if isinstance(x, (int, float)):
            if math.isfinite(x):
                out.append(float(x))
        elif isinstance(x, dict):
            for v in x.values():
                rec(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                rec(v)

    rec(choices)
    return out


def _instances_of(choices, is_leaf) -> list:
    insts: list = []

    def rec(x):
        if isinstance(x, dict):
            for v in x.values():
                rec(v)
        elif isinstance(x, list):
            if x and is_leaf(x):
                insts.append(x)
            else:
                for v in x:
                    rec(v)

    rec(choices)
    return insts


def _investment_from_exp(exp: str):
    m = re.search(r"trust_banker_(\d+)", exp)
    return int(m.group(1)) if m else None


@functools.lru_cache(maxsize=None)
def human_trust_return_ratio(investment: float) -> list[float]:
    out: list[float] = []
    path = DATA / "trust_investment.csv"
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("Role") != "second":
                continue
            if str(r.get("Round", "")).strip() != "1":
                continue
            try:
                invested, returned = ast.literal_eval(r.get("roundResult", ""))
                invested = float(invested)
                returned = float(returned)
            except Exception:
                continue
            if invested != investment:
                continue
            if not (0.0 <= returned <= 3.0 * invested):
                continue
            out.append(returned / (3.0 * invested))
    return out


@functools.lru_cache(maxsize=None)
def _trust_return_ratio_by_inv() -> dict:
    out: dict[float, list[float]] = {}
    path = DATA / "trust_investment.csv"
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("Role") != "second":
                continue
            if str(r.get("Round", "")).strip() != "1":
                continue
            try:
                invested, returned = ast.literal_eval(r.get("roundResult", ""))
                invested = float(invested)
                returned = float(returned)
            except Exception:
                continue
            if invested <= 0:
                continue
            if not (0.0 <= returned <= 3.0 * invested):
                continue
            out.setdefault(invested, []).append(returned / (3.0 * invested))
    return out


def _entry_trust_banker(exp: str, choices) -> dict:
    entry: dict = {"metric": "return_ratio = 返还额/(3×投资额)"}
    inv = _investment_from_exp(exp)
    llm_amounts = _flatten_numbers(choices)
    if not llm_amounts:
        entry["note"] = "无可用的 LLM 返还额"
        return entry
    entry["llm_return_amount"] = _describe(llm_amounts)
    if not inv:
        entry["llm"] = _describe(llm_amounts)
        entry["human_baseline"] = {"note": "无法从实验名解析投资额，跳过比例对齐"}
        return entry
    current_value = 3.0 * inv
    llm_ratios = [a / current_value for a in llm_amounts]
    entry["llm_top_values"] = _top_values(llm_amounts)

    _merge_overall(entry, _compare_block(
        llm_ratios, human_trust_return_ratio(float(inv)),
        f"人类返还比例（第一轮、投资额恰为 {inv}、返还 0–{int(current_value)}；原文 Fig.3E 口径）",
        "未找到 trust_investment.csv", binwidth=0.1))
    return entry


@functools.lru_cache(maxsize=None)
def human_public_goods_contrib(round_one_only: bool = False) -> list[float]:
    def pred(r):
        if (r.get("move") or "").strip() == "":
            return False
        if str(r.get("groupSize", "")).strip() not in ("4", "4.0"):
            return False
        if str(r.get("Total", "")).strip() not in ("20", "20.0"):
            return False
        rnd = str(r.get("Round", "")).strip()
        if round_one_only:
            return rnd == "1"
        return rnd in ("1", "2", "3")

    raw = _read_col_floats(DATA / "public_goods_linear_water.csv", pred, lambda r: float(r["move"]))


    return [v for v in raw if 0.0 <= v <= 20.0]


def _entry_public_goods(exp: str, choices) -> dict:
    entry: dict = {"metric": "contribution（$20 禀赋，4 人组）"}
    insts = _instances_of(
        choices, lambda lst: all(isinstance(e, (int, float)) and not isinstance(e, bool) for e in lst))
    overall_vals = [float(v) for inst in insts for v in inst]
    first_vals = [float(inst[0]) for inst in insts if inst]
    if not overall_vals:
        overall_vals = _flatten_numbers(choices)
    if not overall_vals:
        entry["llm"] = _describe(overall_vals)
        entry["note"] = "无可用的 LLM 贡献额"
        return entry
    _merge_overall(entry, _compare_block(
        overall_vals, human_public_goods_contrib(False),
        "人类前三轮贡献额（Round≤3、Total=20、groupSize=4；原文代码同口径）",
        "未找到 public_goods_linear_water.csv", binwidth=2.0))
    if first_vals:
        fr = _compare_block(
            first_vals, human_public_goods_contrib(True),
            "人类第一轮贡献额（Round==1、Total=20、groupSize=4；原文 Fig.3F 口径）",
            "未找到 public_goods_linear_water.csv",
            binwidth=2.0)
        fr["desc"] = "仅第一轮贡献额（无贡献衰减效应）"
        entry["first_round"] = fr
    return entry


@functools.lru_cache(maxsize=None)
def human_pd_coop_rates() -> list[float]:
    path = DATA / "push_pull.csv"
    if not path.exists():
        return []
    agg: dict[str, list[int]] = {}
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("Role") != "player":
                continue
            if str(r.get("groupSize", "")).strip() != "2":
                continue
            mv = (r.get("move") or "").strip()
            if mv not in ("0", "1"):
                continue
            slot = agg.setdefault(r.get("UserID", ""), [0, 0])
            slot[1] += 1
            if mv == "0":
                slot[0] += 1
    return [coop / total for coop, total in agg.values() if total > 0]


@functools.lru_cache(maxsize=None)
def human_pd_first_round() -> list[float]:
    path = DATA / "push_pull.csv"
    if not path.exists():
        return []
    out: list[float] = []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("Role") != "player":
                continue
            if str(r.get("groupSize", "")).strip() != "2":
                continue
            if str(r.get("Round", "")).strip() != "1":
                continue
            mv = (r.get("move") or "").strip()
            if mv == "0":
                out.append(1.0)
            elif mv == "1":
                out.append(0.0)
    return out


def _compare_block(llm_vals: list[float], human: list[float], human_desc: str,
                   missing_note: str = "未找到对齐的人类数据", binwidth: float | None = None) -> dict:
    llm_vals, human = _finite(llm_vals), _finite(human)
    block: dict = {"llm": _describe(llm_vals)}
    if not llm_vals:
        block["note"] = "无可用的 LLM 数据"
        return block
    if human:
        block["human_baseline"] = {"desc": human_desc, **_describe(human)}
        test = mann_whitney(llm_vals, human)
        block["test"] = test
        block["ci95_mean_diff"] = _bootstrap_mean_diff_ci(llm_vals, human)
        if test:
            block["verdict"] = _direction(
                statistics.median(llm_vals), statistics.median(human), test["p_approx"])
        if binwidth is not None:
            turing = _turing_test(llm_vals, human, binwidth)
            if turing:
                block["turing"] = turing
    else:
        block["human_baseline"] = {"note": missing_note}
    return block


def _merge_overall(entry: dict, block: dict) -> None:
    for k in ("llm", "human_baseline", "test", "verdict", "ci95_mean_diff"):
        if k in block:
            entry[k] = block[k]


def _entry_pd(exp: str, choices) -> dict:
    entry: dict = {"metric": "cooperation_rate（Push=合作）"}
    insts = _instances_of(choices, lambda lst: all(isinstance(e, str) for e in lst))
    overall_rates: list[float] = []
    first_round: list[float] = []
    for inst in insts:
        cards = [c for c in (str(x).strip().lower() for x in inst) if c in ("push", "pull")]
        if not cards:
            continue
        overall_rates.append(sum(1 for c in cards if c == "push") / len(cards))
        first_round.append(1.0 if cards[0] == "push" else 0.0)
    if not overall_rates:
        entry["llm"] = _describe(overall_rates)
        entry["note"] = "无可用的 LLM Push/Pull 选择"
        return entry

    _merge_overall(entry, _compare_block(
        overall_rates, human_pd_coop_rates(),
        "人类个体合作率（push_pull.csv，groupSize=2）", "未找到 push_pull.csv", binwidth=0.1))

    fr_block = _compare_block(
        first_round, human_pd_first_round(),
        "人类第一轮合作率（Round==1，groupSize=2）", "未找到 push_pull.csv", binwidth=1.0)
    fr_block["desc"] = "仅第一轮合作率（无历史/对手影响，最干净可比）"
    entry["first_round"] = fr_block
    return entry


def _pg_first_round_two_protocols(records_by_exp: dict):
    runs = []
    for exp, (jf, d) in records_by_exp.items():
        if not exp.startswith("public_goods") or "occupation" in exp:
            continue
        oc = ((d.get("metadata") or {}).get("run_config") or {}).get("other_contributions")
        protocol = tuple(oc) if isinstance(oc, (list, tuple)) else None
        if protocol is None:

            if "loss" in exp:
                protocol = (0, 0)
            elif "basic" in exp:
                protocol = (30, 18)
        insts = _instances_of(
            d.get("choices"), lambda lst: all(
                isinstance(e, (int, float)) and not isinstance(e, bool) for e in lst))
        firsts = [float(inst[0]) for inst in insts if inst]
        if firsts and protocol is not None:
            runs.append((exp, protocol, firsts))
    if len(runs) != 2 or runs[0][1] == runs[1][1]:
        return None
    merged = runs[0][2] + runs[1][2]
    entry: dict = {
        "metric": "first_round_contribution（两协议合并，$20 禀赋，4 人组）",
        "protocol_runs": {exp: {"other_contributions": list(p), "n": len(v)}
                          for exp, p, v in runs},
        "note": ("原文 Fig.3F 的 ChatGPT-4 = PG_basic + PG_basic_loss 两文件各 30 个 session"
                 "的第一轮合并共 60 次；此条目为同口径合并，单协议条目见 public_goods*。"),
    }
    block = _compare_block(
        merged, human_public_goods_contrib(True),
        "人类第一轮贡献额（Round==1、Total=20、groupSize=4；原文 Fig.3F 口径）",
        "未找到 public_goods_linear_water.csv", binwidth=2.0)
    entry.update(block)
    return entry


def _pd_first_round_two_branches(raw_choices: dict):
    firsts: list[float] = []
    branch_n: dict[str, int] = {}
    for exp in ("prisoners_dilemma_two_rounds_push", "prisoners_dilemma_two_rounds_pull"):
        ch = raw_choices.get(exp)
        if ch is None:
            return None
        insts = _instances_of(ch, lambda lst: all(isinstance(e, str) for e in lst))
        vals: list[float] = []
        for inst in insts:
            cards = [c for c in (str(x).strip().lower() for x in inst) if c in ("push", "pull")]
            if cards:
                vals.append(1.0 if cards[0] == "push" else 0.0)
        if not vals:
            return None
        branch_n[exp] = len(vals)
        firsts.extend(vals)
    entry: dict = {
        "metric": "first_round_cooperation_rate（两个两轮分支合并，Push=合作）",
        "branch_n": branch_n,
        "note": ("原文 Fig.3H 的 ChatGPT-4 首轮合作率 91.7%=(29+26)/60 即此口径"
                 "（两个两轮分支各 30 次的第一轮合并）；五轮版首轮不计入。"),
    }
    block = _compare_block(
        firsts, human_pd_first_round(),
        "人类第一轮合作率（Round==1，groupSize=2）", "未找到 push_pull.csv", binwidth=1.0)
    entry.update(block)
    return entry


@functools.lru_cache(maxsize=None)
def human_bomb_open(round_one_only: bool = False) -> list[float]:
    def pred(r):
        if (r.get("move") or "").strip() == "":
            return False
        if round_one_only and str(r.get("Round", "")).strip() != "1":
            return False
        return True

    raw = _read_col_floats(DATA / "bomb_risk.csv", pred, lambda r: float(r["move"]))


    return [v for v in raw if 0.0 <= v <= 100.0]


def _entry_bomb_risk(exp: str, choices) -> dict:
    entry: dict = {"metric": "boxes_opened（开盒数 0-100）"}
    insts = _instances_of(
        choices, lambda lst: all(isinstance(e, (int, float)) and not isinstance(e, bool) for e in lst))
    overall_vals = [float(v) for inst in insts for v in inst]
    first_vals = [float(inst[0]) for inst in insts if inst]
    if not overall_vals:
        entry["llm"] = _describe(overall_vals)
        entry["note"] = "无可用的 LLM 开盒数"
        return entry
    entry["llm_top_values"] = _top_values(overall_vals)
    _merge_overall(entry, _compare_block(
        overall_vals, human_bomb_open(False),
        "人类开盒数（bomb_risk.csv，全部轮次）", "未找到 bomb_risk.csv", binwidth=10.0))
    fr_block = _compare_block(
        first_vals, human_bomb_open(True),
        "人类第一轮开盒数（Round==1）", "未找到 bomb_risk.csv", binwidth=10.0)
    fr_block["desc"] = "仅第一轮开盒数（无学习效应）"
    entry["first_round"] = fr_block
    return entry


BIGFIVE_DIMS = {"E": range(0, 10), "N": range(10, 20),
                "A": range(20, 30), "C": range(30, 40), "O": range(40, 50)}
BIGFIVE_COLS = [f"{d}{i}" for d in ("E", "N", "A", "C", "O") for i in range(1, 11)]

BIGFIVE_REVERSE = {1, 3, 5, 7, 9, 11, 13, 20, 22, 24, 26, 31, 33, 35, 37, 41, 43, 45}


def _bigfive_score(raw: list) -> dict:
    dims: dict[str, float] = {}
    for dim, idxs in BIGFIVE_DIMS.items():
        vals = []
        for i in idxs:
            v = raw[i]
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not 1 <= v <= 5:
                continue
            vals.append(6 - v if i in BIGFIVE_REVERSE else v)
        if len(vals) >= 5:
            dims[dim] = statistics.mean(vals)
    return dims


@functools.lru_cache(maxsize=None)
def human_bigfive_dims() -> dict:
    out: dict[str, list[float]] = {d: [] for d in BIGFIVE_DIMS}
    path = DATA / "bigfive_data.csv"
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            for dim, idxs in BIGFIVE_DIMS.items():
                vals = []
                for i in idxs:
                    try:
                        v = int(float((r.get(BIGFIVE_COLS[i]) or "").strip()))
                    except Exception:
                        continue
                    if 1 <= v <= 5:
                        vals.append(6 - v if i in BIGFIVE_REVERSE else v)
                if len(vals) >= 5:
                    out[dim].append(statistics.mean(vals))
    return out


def _entry_bigfive(exp: str, choices) -> dict:
    entry: dict = {"metric": "OCEAN 维度均值（1-5，含反向计分）"}

    insts = [x for x in _instances_of(
        choices, lambda lst: len(lst) == 50 and all(
            e is None or isinstance(e, (int, float)) for e in lst))]
    if not insts:
        entry["note"] = "bigfive choices 结构异常（每份问卷应为 50 题）"
        return entry
    llm_dims: dict[str, list[float]] = {d: [] for d in BIGFIVE_DIMS}
    n_missing = 0
    for inst in insts:
        n_missing += sum(1 for e in inst if e is None)
        for dim, val in _bigfive_score(list(inst)).items():
            llm_dims[dim].append(val)
    if n_missing:
        entry["n_missing_ratings"] = n_missing
    human_dims = human_bigfive_dims()
    out_dims: dict = {}
    for dim in BIGFIVE_DIMS:
        block = _compare_block(
            llm_dims[dim], human_dims.get(dim, []),
            f"人类 {dim} 维度均值（bigfive_data.csv）", "未找到 bigfive_data.csv")


        hv, lv = _finite(human_dims.get(dim, [])), _finite(llm_dims[dim])
        if hv and lv:
            med = statistics.median(lv)
            block["percentile"] = round(sum(1 for x in hv if x < med) / len(hv), 3)
            block["human_p2_5"] = round(float(np.percentile(hv, 2.5)), 3)
            block["human_p97_5"] = round(float(np.percentile(hv, 97.5)), 3)
            block["within_human_range"] = bool(block["human_p2_5"] <= med <= block["human_p97_5"])
            block["percentile_verdict"] = (
                f"{'落在' if block['within_human_range'] else '超出'}人类分布范围"
                f"（中位高于 {block['percentile'] * 100:.1f}% 人类）"
            )
        out_dims[dim] = block
    entry["n_instances"] = len(insts)
    entry["dimensions"] = out_dims
    return entry


def _special_entry(exp: str, choices):
    if exp.startswith("trust_banker"):
        return _entry_trust_banker(exp, choices)
    if exp.startswith("public_goods"):
        return _entry_public_goods(exp, choices)
    if exp.startswith("prisoners_dilemma"):
        return _entry_pd(exp, choices)
    if exp == "bigfive":
        return _entry_bigfive(exp, choices)
    if exp == "bomb_risk":
        return _entry_bomb_risk(exp, choices)
    return None


def _build_entry(exp: str, ch) -> dict:
    special = _special_entry(exp, ch)
    if special is not None:
        return special
    entry: dict = {}
    if isinstance(ch, dict):
        entry["llm_by_group"] = {k: _describe(v) for k, v in ch.items()}
        llm_vals = _finite([x for v in ch.values() for x in v])
        entry["llm"] = _describe(llm_vals)
    elif isinstance(ch, list):
        llm_vals = _finite(ch)
        entry["llm"] = _describe(llm_vals)
    else:
        return {"note": "choices 结构需专门解析"}


    if ch and not llm_vals:
        entry["note"] = ("choices 为嵌套/非数值结构，通用数值对齐无法解析，"
                         "需专门的 _special_entry 处理（当前未实现）。")
        return entry
    if llm_vals:
        entry["llm_top_values"] = _top_values(llm_vals)
    kind, desc = _baseline_kind(exp)
    if kind and llm_vals:
        human = human_baseline(kind)
        if human:
            entry["human_baseline"] = {"desc": desc, **_describe(human)}
            test = mann_whitney(llm_vals, human)
            entry["test"] = test
            if test:
                entry["verdict"] = _direction(
                    statistics.median(llm_vals), statistics.median(human), test["p_approx"])
            bw = _binwidth_for(exp)
            if bw:
                turing = _turing_test(llm_vals, human, bw)
                if turing:
                    entry["turing"] = turing
        else:
            entry["human_baseline"] = {"desc": desc, "note": "未找到人类数据文件"}
    elif llm_vals:
        entry["human_baseline"] = {"note": "该实验暂无对齐的人类基线"}
    return entry


def _canonical_from_filename(stem: str):
    low = stem.lower()
    if "gpt4" in low or "gpt-4" in low:
        mdl = "gpt-4"
    elif "turbo" in low:
        mdl = "gpt-3.5-turbo"
    elif "gpt3" in low or "davinci" in low:
        mdl = "gpt-3.5"
    else:
        mdl = "unknown"
    if low.startswith("pd_") or low.startswith("pd-"):
        canon = "prisoners_dilemma"
    elif low.startswith("pg_") or low.startswith("pg-"):
        canon = "public_goods"
    elif low.startswith("bigfive"):
        canon = "bigfive"
    elif low.startswith("bomb"):
        canon = "bomb_risk"
    elif low.startswith("dictator"):
        canon = "dictator"

    elif low.startswith("ultimatum_12") or low.startswith("ultimatum_21"):
        canon = None
    elif low.startswith("ultimatum_1"):
        canon = "ultimatum_proposer"
    elif low.startswith("ultimatum_2"):
        canon = "ultimatum_responder"

    elif low.startswith("trust_1"):
        canon = "trust_investor"
    elif low.startswith("trust_2"):
        canon = "trust_banker_10"
    elif low.startswith("trust_3"):
        canon = "trust_banker_50"
    elif low.startswith("trust_4"):
        canon = "trust_banker_100"
    else:

        canon = None
    return canon, mdl


def _extract_choices_from_record(d):
    if isinstance(d, dict):
        return d.get("choices")
    if isinstance(d, list) and len(d) == 2:
        return d[1]
    return None


def _analyze_archive() -> dict:
    arch_dir = ROOT / "records"
    archive: dict = {}
    if not arch_dir.exists():
        return archive
    for jf in sorted(arch_dir.glob("*.json")):
        canon, mdl = _canonical_from_filename(jf.stem)
        if canon is None:
            continue
        try:
            d = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        ch = _extract_choices_from_record(d)
        if ch is None:
            continue
        entry = _build_entry(canon, ch)
        entry["canonical_experiment"] = canon
        archive.setdefault(mdl, {})[jf.stem] = entry
    return archive


def _collect_test_dicts(obj, acc: list) -> None:
    if isinstance(obj, dict):
        t = obj.get("test")
        if isinstance(t, dict) and isinstance(t.get("p_approx"), (int, float)):
            acc.append(t)
        for v in obj.values():
            _collect_test_dicts(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            _collect_test_dicts(v, acc)


def _apply_multiple_comparison(summary: dict) -> None:
    tests: list = []
    _collect_test_dicts(summary.get("experiments", {}), tests)
    _collect_test_dicts(summary.get("archive", {}), tests)
    m = len(tests)
    if m == 0:
        return
    for t in tests:
        t["p_bonferroni"] = round(min(1.0, t["p_approx"] * m), 5)
    order = sorted(range(m), key=lambda i: tests[i]["p_approx"])
    running_min = 1.0
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        running_min = min(running_min, tests[idx]["p_approx"] * m / rank)
        tests[idx]["q_fdr"] = round(min(1.0, running_min), 5)
    summary["multiple_comparison"] = {
        "family_size": m, "methods": ["Bonferroni", "Benjamini-Hochberg FDR"],
        "note": "p_bonferroni/q_fdr 已写入各 test；全局作为单一检验族"}


_B_GRID = [i / 100.0 for i in range(0, 101)]


def _logsumexp(z: np.ndarray) -> float:
    m = float(z.max())
    return m + float(np.log(np.exp(z - m).sum()))


def _fit_logit_b(actions, grid, own_grid, partner_grid):
    g = np.asarray(grid, float)
    acts = np.asarray(_finite(actions), float)
    if acts.size == 0:
        return None
    idx = np.abs(acts[:, None] - g[None, :]).argmin(1)
    own = np.asarray(own_grid, float)
    partner = np.asarray(partner_grid, float)
    best = None
    ll_by_b: dict[float, float] = {}
    for b in _B_GRID:
        u = b * own + (1 - b) * partner
        scale = float(np.max(np.abs(u))) or 1.0
        u = u / scale
        b_best = None
        for lam in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0):
            z = lam * u
            ll = float(z[idx].sum() - len(idx) * _logsumexp(z))
            if b_best is None or ll > b_best:
                b_best = ll
            if best is None or ll > best[0]:
                best = (ll, round(b, 2), lam)
        ll_by_b[round(b, 2)] = b_best
    if best is None:
        return None
    return best[1], best[2], round(best[0], 2), ll_by_b
_GAME_LABEL = {
    "dictator": "独裁者", "ultimatum_proposer": "最后通牒·提议者",
    "ultimatum_responder": "最后通牒·回应者", "trust_investor": "信任·投资者",
    "trust_banker": "信任·银行家", "public_goods": "公共品", "prisoners_dilemma": "囚徒困境",
}


def _cap(arr, n: int = 4000):
    a = np.asarray(arr, float)
    return _BOOT_RNG.choice(a, n, replace=False) if a.size > n else a


def _game_of(exp: str):
    for g in ("dictator", "ultimatum_proposer", "ultimatum_responder",
              "trust_investor", "trust_banker", "public_goods", "prisoners_dilemma"):
        if exp.startswith(g):
            return g
    return None


def _payoff_own_of(game: str, inv=None):
    if game == "dictator":
        return lambda a: (100.0 - np.asarray(a, float), np.asarray(a, float))
    if game == "ultimatum_proposer":
        thr = _cap(human_baseline("ultimatum_accept"))
        if thr.size < 2:
            return None
        def f(a):
            o = np.asarray(a, float)
            pacc = (thr[:, None] <= o[None, :]).mean(0)
            return ((100.0 - o) * pacc, o * pacc)
        return f
    if game == "ultimatum_responder":
        off = _cap(human_baseline("ultimatum_propose"))
        if off.size < 2:
            return None
        def f(a):
            aa = np.asarray(a, float)
            mask = off[:, None] >= aa[None, :]
            return ((off[:, None] * mask).mean(0), ((100.0 - off)[:, None] * mask).mean(0))
        return f
    if game == "trust_investor":


        by_inv = _trust_return_ratio_by_inv()
        if not by_inv:
            return None
        pooled = [x for v in by_inv.values() for x in v]
        er_all = float(np.mean(pooled))
        er_of = {k: float(np.mean(v)) for k, v in by_inv.items()}
        def f(a):
            aa = np.asarray(a, float)
            er = np.asarray([er_of.get(float(x), er_all) for x in aa])
            return (100.0 - aa + er * 3 * aa, 3 * aa * (1 - er))
        return f
    if game == "trust_banker":
        if not inv:
            return None
        return lambda a: (3 * inv * (1 - np.asarray(a, float)),
                          (100.0 - inv) + np.asarray(a, float) * 3 * inv)
    if game == "public_goods":

        others = human_public_goods_contrib(True)
        if not others:
            return None
        mo = float(np.mean(others))
        return lambda a: ((20.0 - np.asarray(a, float)) + 0.5 * (np.asarray(a, float) + 3 * mo),
                          (20.0 - mo) + 0.5 * (np.asarray(a, float) + 3 * mo))
    if game == "prisoners_dilemma":
        hf = human_pd_first_round()
        if not hf:
            return None
        q = float(np.mean(hf))
        def f(a):
            p = np.asarray(a, float)
            return (p * q * 400 + (1 - p) * q * 700 + (1 - p) * (1 - q) * 300,
                    p * q * 400 + p * (1 - q) * 700 + (1 - p) * (1 - q) * 300)
        return f
    return None


def _action_grid(game: str):
    if game in ("dictator", "ultimatum_proposer", "ultimatum_responder", "trust_investor"):
        return np.arange(0, 101, dtype=float)
    if game == "trust_banker":
        return np.linspace(0, 1, 21)
    if game == "public_goods":
        return np.arange(0, 21, dtype=float)
    if game == "prisoners_dilemma":
        return np.array([0.0, 1.0])
    return None


def _actions_for(game: str, choices, inv=None):
    if game in ("dictator", "ultimatum_proposer", "ultimatum_responder", "trust_investor"):
        return _flatten_numbers(choices)
    if game == "trust_banker":
        return [a / (3.0 * inv) for a in _flatten_numbers(choices)] if inv else []
    if game == "public_goods":
        insts = _instances_of(choices, lambda lst: all(
            isinstance(e, (int, float)) and not isinstance(e, bool) for e in lst))

        return [float(inst[0]) for inst in insts if inst] or _flatten_numbers(choices)
    if game == "prisoners_dilemma":
        insts = _instances_of(choices, lambda lst: all(isinstance(e, str) for e in lst))
        return [1.0 if (inst and str(inst[0]).strip().lower() == "push") else 0.0 for inst in insts]
    return []


def _human_actions(game: str, inv=None):
    return {
        "dictator": lambda: human_baseline("dictator_give"),
        "ultimatum_proposer": lambda: human_baseline("ultimatum_propose"),
        "ultimatum_responder": lambda: human_baseline("ultimatum_accept"),
        "trust_investor": lambda: human_baseline("trust_invest"),
        "trust_banker": lambda: human_trust_return_ratio(float(inv) if inv else 50.0),
        "public_goods": lambda: human_public_goods_contrib(True),
        "prisoners_dilemma": lambda: human_pd_first_round(),
    }.get(game, list)()


def _payoffs_and_b(raw_choices: dict) -> dict:
    games: dict = {}
    b_err_accum = {b: [] for b in _B_GRID}
    logit_ll_accum: dict = {}


    pd_exps = [e for e in raw_choices if _game_of(e) == "prisoners_dilemma"]
    if len(pd_exps) > 1:
        merged_pd = []
        for e in pd_exps:
            ch = raw_choices[e]
            merged_pd.extend(ch if isinstance(ch, list) else [ch])
        raw_choices = {e: c for e, c in raw_choices.items() if e not in pd_exps}
        raw_choices["prisoners_dilemma"] = merged_pd
    for exp, choices in raw_choices.items():
        game = _game_of(exp)
        if game is None:
            continue
        inv = _investment_from_exp(exp) if game == "trust_banker" else None
        own_of = _payoff_own_of(game, inv)
        grid = _action_grid(game)
        if own_of is None or grid is None:
            continue
        ai_actions = _finite(_actions_for(game, choices, inv))
        if not ai_actions:
            continue
        ai_own, ai_partner = own_of(ai_actions)
        entry = {
            "game": game, "label": _GAME_LABEL.get(game, game), "n_ai": len(ai_actions),
            "ai": {"own": round(float(np.mean(ai_own)), 2),
                   "partner": round(float(np.mean(ai_partner)), 2),
                   "combined": round(float(np.mean(ai_own) + np.mean(ai_partner)), 2)},
        }
        h_actions = _finite(_human_actions(game, inv))
        if h_actions:
            h_own, h_partner = own_of(_cap(h_actions))
            entry["human"] = {"own": round(float(np.mean(h_own)), 2),
                              "partner": round(float(np.mean(h_partner)), 2),
                              "combined": round(float(np.mean(h_own) + np.mean(h_partner)), 2)}
        own_grid, partner_grid = own_of(grid)
        ai_o, ai_p = np.asarray(ai_own, float), np.asarray(ai_partner, float)
        game_b_err = {}
        for b in _B_GRID:
            ustar = float(np.max(b * own_grid + (1 - b) * partner_grid))
            if ustar <= 0:
                continue
            u_ai = b * ai_o + (1 - b) * ai_p
            err = float(np.mean((1.0 - u_ai / ustar) ** 2))
            game_b_err[b] = err
            b_err_accum[b].append(err)
        if game_b_err:
            entry["best_b"] = round(min(game_b_err, key=game_b_err.get), 2)
            entry["b_curve"] = [{"b": round(b, 2), "err": round(e, 4)} for b, e in sorted(game_b_err.items())]
        lg = _fit_logit_b(ai_actions, grid, own_grid, partner_grid)
        if lg:
            entry["logit_b"], entry["logit_lambda"], entry["logit_loglik"], _ll_by_b = lg
            for b, ll in _ll_by_b.items():
                logit_ll_accum.setdefault(b, []).append(ll)
        games[exp] = entry
    overall = {b: float(np.mean(v)) for b, v in b_err_accum.items() if v}
    out: dict = {"games": games}
    if overall:
        out["overall_best_b"] = round(min(overall, key=overall.get), 2)
        out["overall_curve"] = [{"b": round(b, 2), "err": round(e, 4)} for b, e in sorted(overall.items())]
        out["note"] = ("b=自己收益权重：0.5=等权最大化双方收益，1=纯自利，0=纯利他。"
                       "论文：两个 ChatGPT≈0.5、人类≈0.6。bomb_risk 无对方收益未纳入。"
                       "logit_b=多项 Logit 离散选择估计（对应原文 Table S2），与网格 best_b 互为稳健性。")
    if logit_ll_accum:
        joint = {b: sum(v) for b, v in logit_ll_accum.items()}
        out["overall_logit_b"] = round(max(joint, key=joint.get), 2)
    return out


def _pd_dynamics(choices, exp: str = "prisoners_dilemma") -> dict:
    insts = _instances_of(choices, lambda lst: all(isinstance(e, str) for e in lst))
    seqs = [[str(c).strip().lower() for c in inst] for inst in insts if len(inst) >= 2]
    if not seqs:
        return {}
    opp_first_coop = "two_rounds_push" in exp
    coop1 = [s for s in seqs if s[0] == "push"]
    def_1 = [s for s in seqs if s[0] == "pull"]
    rate2 = lambda g: (round(sum(1 for s in g if s[1] == "push") / len(g), 3) if g else None)
    if opp_first_coop:
        note = ("对方首轮合作（opponent_sequence 首项 Push，Fig.4A 分支）：首轮合作者次轮仍合作的比例"
                "越高 = 合作越稳定（论文：ChatGPT-4 全部维持合作）。")
    else:
        note = ("对方首轮背叛（opponent_sequence 首项 Pull，Fig.4B 分支）：首轮合作者次轮仍合作的比例"
                "越低 = 越像 tit-for-tat；『对方首轮合作』对照组见 prisoners_dilemma_two_rounds_push。")
    return {
        "type": "pd", "n_instances": len(seqs),
        "opponent_first": "Push（合作）" if opp_first_coop else "Pull（背叛）",
        "coop_first": len(coop1), "defect_first": len(def_1),
        "coop_then_coop": rate2(coop1), "defect_then_coop": rate2(def_1),
        "note": note,
    }


def _bomb_dynamics(choices, scenarios) -> dict:
    ch = _instances_of(choices, lambda lst: all(
        isinstance(e, (int, float)) and not isinstance(e, bool) for e in lst))
    after_bomb, after_safe = [], []
    for opens, sc in zip(ch, scenarios if isinstance(scenarios, list) else []):
        if not isinstance(sc, list):
            continue
        for r in range(min(len(opens) - 1, len(sc))):
            (after_bomb if sc[r] == 0 else after_safe).append(float(opens[r + 1]))
    out: dict = {"type": "bomb"}
    if after_bomb:
        out["after_bomb_mean"] = round(float(np.mean(after_bomb)), 2)
        out["after_bomb_std"] = round(float(np.std(after_bomb)), 2)
        out["after_bomb_n"] = len(after_bomb)
    if after_safe:
        out["after_safe_mean"] = round(float(np.mean(after_safe)), 2)
        out["after_safe_std"] = round(float(np.std(after_safe)), 2)
        out["after_safe_n"] = len(after_safe)
    if after_bomb and after_safe:
        out["note"] = ("踩雷后开更少=更风险厌恶（论文：ChatGPT-3 踩雷后转保守，"
                       "ChatGPT-4 均值不变但方差升高——后者对应 after_bomb_std vs after_safe_std）。")
    return out if (after_bomb or after_safe) else {}


def analyze(model: str | None = None, include_archive: bool = True) -> dict:
    requested = model
    model = model or _active_llm()
    records_by_exp, used = _latest_records(model, strict=bool(requested))
    summary: dict = {"model_requested": model, "model_used": used,
                     "experiments": {}, "record_sources": {}}
    raw_choices: dict = {}
    raw_scenarios: dict = {}
    for exp in sorted(records_by_exp):
        jf, d = records_by_exp[exp]
        summary["experiments"][exp] = _build_entry(exp, d.get("choices"))
        summary["record_sources"][exp] = str(jf.relative_to(ROOT))
        raw_choices[exp] = d.get("choices")
        if isinstance(d.get("scenarios"), list):
            raw_scenarios[exp] = d.get("scenarios")

    pd60 = _pd_first_round_two_branches(raw_choices)
    if pd60:
        summary["experiments"]["prisoners_dilemma_first_round_60"] = pd60

    pg60 = _pg_first_round_two_protocols(records_by_exp)
    if pg60:
        summary["experiments"]["public_goods_first_round_60"] = pg60
    if include_archive:
        summary["archive"] = _analyze_archive()
    _apply_multiple_comparison(summary)

    summary["revealed_preference"] = _payoffs_and_b(raw_choices)

    dynamics: dict = {}
    for exp, ch in raw_choices.items():
        if exp.startswith("prisoners_dilemma"):
            d_pd = _pd_dynamics(ch, exp)
            if d_pd:
                dynamics[exp] = d_pd
        elif exp.startswith("bomb_risk"):
            d_b = _bomb_dynamics(ch, raw_scenarios.get(exp))
            if d_b:
                dynamics[exp] = d_b
    summary["dynamics"] = dynamics

    summary["files"] = {
        f"{used}/{exp}": {"type": "behavioral_game", "experiment": exp, **entry}
        for exp, entry in summary["experiments"].items()
    }
    out_dir = (ROOT / "analysis" / requested) if requested else (ROOT / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    n_arch = sum(len(v) for v in summary.get("archive", {}).values())
    print(f"[analyze] {used}: experiments={len(summary['experiments'])}"
          f"{f'，archive={n_arch} 份原始记录' if include_archive else ''}"
          f" -> {out_dir.relative_to(ROOT)}/analysis_summary.json")
    return summary


if __name__ == "__main__":
    import argparse

    _ap = argparse.ArgumentParser(description="behavioral 结果分析（LLM vs 人类基线）")
    _ap.add_argument("--model", default=None,
                     help="只分析 records_new/<model>/ 的结果并写 analysis/<model>/；省略则用 active_llm。")
    _ap.add_argument("--no-archive", action="store_true", help="不分析 records/ 原始记录。")
    _args = _ap.parse_args()
    analyze(_args.model, include_archive=not _args.no_archive)
