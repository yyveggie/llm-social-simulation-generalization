from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "analysis"

MODELS = [
    "claude-opus-4-8", "claude-sonnet-4-6", "gpt-5.5", "gpt-5.2",
    "gemini-3.5-flash", "gemini-3-flash-preview",
    "deepseek-v4-pro", "deepseek-v3.2", "doubao-seed-2-0-pro-260215",
    "glm-5.2", "glm-5.1", "kimi-k2.5", "qwen3.7-max", "qwen3.6-flash",
    "MiniMax-M2.5",
]

EXPERIMENTS = [
    "bigfive", "dictator", "ultimatum_proposer", "ultimatum_responder",
    "trust_investor", "trust_banker_50", "public_goods",
    "public_goods_basic", "public_goods_loss", "public_goods_first_round_60", "bomb_risk",
    "prisoners_dilemma", "prisoners_dilemma_two_rounds_push",
    "prisoners_dilemma_two_rounds_pull", "prisoners_dilemma_first_round_60",
]


def load(model):
    p = ANALYSIS / model / "analysis_summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def g(d, *path, default=None):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def econ_row(exp, s):
    e = g(s, "experiments", exp)
    if not e or "llm" not in e:
        return None
    llm = e["llm"]
    row = {
        "n": llm.get("n"), "mean": llm.get("mean"), "median": llm.get("median"),
        "min": llm.get("min"), "max": llm.get("max"), "stdev": llm.get("stdev"),
        "human_mean": g(e, "human_baseline", "mean"),
        "human_median": g(e, "human_baseline", "median"),
        "verdict": e.get("verdict"),
        "p": g(e, "test", "p_approx"), "q_fdr": g(e, "test", "q_fdr"),
        "effect_r": g(e, "test", "effect_r"),
        "turing_win": g(e, "turing", "win"), "turing_tie": g(e, "turing", "tie"),
        "turing_loss": g(e, "turing", "loss"), "turing_verdict": g(e, "turing", "verdict"),
    }
    if exp == "trust_banker_50":
        row["llm_return_amount_mean"] = g(e, "llm_return_amount", "mean")
        row["llm_return_amount_median"] = g(e, "llm_return_amount", "median")
    if exp == "public_goods":
        row["first_round_mean"] = g(e, "first_round", "llm", "mean")
        row["first_round_human_mean"] = g(e, "first_round", "human_baseline", "mean")
    if exp == "prisoners_dilemma":
        row["first_round_coop"] = g(e, "first_round", "llm", "mean")
        row["first_round_human_coop"] = g(e, "first_round", "human_baseline", "mean")

    row["revealed_b"] = g(s, "revealed_preference", "games", exp, "best_b")
    row["payoff_own"] = g(s, "revealed_preference", "games", exp, "ai", "own")
    row["payoff_partner"] = g(s, "revealed_preference", "games", exp, "ai", "partner")

    if exp.startswith("prisoners_dilemma"):
        row["coop_then_coop"] = g(s, "dynamics", exp, "coop_then_coop")
    if exp == "bomb_risk":
        row["after_bomb_mean"] = g(s, "dynamics", exp, "after_bomb_mean")
        row["after_safe_mean"] = g(s, "dynamics", exp, "after_safe_mean")
    return row


def bigfive_row(s):
    bf = g(s, "experiments", "bigfive")
    if not bf or "dimensions" not in bf:
        return None
    row = {"n_instances": bf.get("n_instances"), "n_missing_ratings": bf.get("n_missing_ratings", 0),
           "dims": {}}
    for dim, blk in bf["dimensions"].items():
        row["dims"][dim] = {
            "llm_median": g(blk, "llm", "median"),
            "percentile": blk.get("percentile"),
            "within_human_range": blk.get("within_human_range"),
            "verdict": blk.get("percentile_verdict"),
        }
    return row


def main():
    matrix = {"models": MODELS, "experiments": {}}
    summaries = {m: load(m) for m in MODELS}
    matrix["experiments"]["bigfive"] = {
        m: bigfive_row(summaries[m]) for m in MODELS if summaries[m]}
    for exp in EXPERIMENTS:
        if exp == "bigfive":
            continue
        matrix["experiments"][exp] = {
            m: econ_row(exp, summaries[m]) for m in MODELS
            if summaries[m] and econ_row(exp, summaries[m])}

    baselines = {}
    for exp in EXPERIMENTS:
        for m in MODELS:
            e = g(summaries.get(m) or {}, "experiments", exp)
            if e and "human_baseline" in e:
                baselines[exp] = e["human_baseline"]
                break
    matrix["human_baselines"] = baselines
    out = ANALYSIS / "cross_model" / "cross_model_matrix.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(matrix, ensure_ascii=False, indent=1), encoding="utf-8")


    for exp in EXPERIMENTS:
        if exp == "bigfive":
            continue
        print(f"\n=== {exp} ===  human_median={g(baselines, exp, 'median')}")
        print(f"{'model':<28}{'median':>8}{'mean':>8}{'verdict':<22}{'turing':>8}{'b':>6}")
        for m in MODELS:
            r = matrix["experiments"][exp].get(m)
            if not r:
                continue
            print(f"{m:<28}{str(r['median']):>8}{str(round(r['mean'],1) if r['mean'] is not None else '—'):>8}"
                  f"{(r['verdict'] or '')[:20]:<22}{str(r['turing_win']):>8}{str(r['revealed_b']):>6}")
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
