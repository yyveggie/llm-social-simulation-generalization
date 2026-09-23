from __future__ import annotations

import itertools
import random
from typing import Any

from .config import ExperimentDef, GlobalExperimentConfig


def _stake_values(cfg: dict[str, Any], override: dict[str, Any]) -> list[int]:
    start = override.get("start") if override.get("start") is not None else cfg["start"]
    end = override.get("end") if override.get("end") is not None else cfg["end"]
    step = override.get("step") if override.get("step") is not None else cfg["step"]
    values = list(range(int(start), int(end) + 1, int(step)))

    max_n = override.get("max_n")
    if max_n and len(values) > int(max_n):
        rng = random.Random(0)
        values = sorted(rng.sample(values, int(max_n)))
    return values


_BUILDERS = {
    "baseline_nonsocial":  lambda raw, stk: _build_nonsocial(raw, stk),
    "baseline_dictator":   lambda raw, stk: _build_dictator(raw, stk),

    "text_insertion_nonsocial": lambda raw, stk: _build_nonsocial(raw, stk),
    "text_insertion_dictator":  lambda raw, stk: _build_dictator(raw, stk),
    "alt_verbs":           lambda raw, stk: _build_alt_verbs(raw, stk),
    "needs_framing":       lambda raw, stk: _build_with_suffix(raw, stk, key="framing", values_key="framings"),
    "ignore_dg_ug":        lambda raw, stk: _build_with_suffix(raw, stk, key="game", values_key="games"),
    "monetary_value":      lambda raw, stk: _build_monetary_value(raw, stk),
    "ai_label":            lambda raw, stk: _build_ai_label(raw, stk),
    "other_uninterested":  lambda raw, stk: _build_other_uninterested(raw, stk),
    "battery_life":        lambda raw, stk: _build_simple_conditions(raw, stk),
    "system_usage":        lambda raw, stk: _build_system_usage(raw, stk),
    "param_sweep":         lambda raw, stk: _build_param_sweep(raw, stk),
}


def build_trials(
    exp: ExperimentDef,
    global_cfg: GlobalExperimentConfig,
) -> list[dict[str, Any]]:
    raw = exp.raw
    stakes = _stake_values(raw["stakes"], global_cfg.stakes_override or {})

    builder = _BUILDERS.get(exp.name)
    if builder is None:
        raise ValueError(f"未知实验名：{exp.name}")
    trials = builder(raw, stakes)


    rng = random.Random(global_cfg.seed)
    rng.shuffle(trials)
    for i, t in enumerate(trials, start=1):
        t["indexx"] = i
    return trials


def _build_nonsocial(raw: dict[str, Any], stakes: list[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tpl in raw["templates"]:
        for x in stakes:
            out.append({
                "condit": tpl["condition"],
                "stakes": x,
                "queree": tpl["prompt"].format(x=x),
            })
    return out


def _build_dictator(raw: dict[str, Any], stakes: list[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    templates = raw["prompt_templates"]
    for r in raw["recipients"]:
        prompt_tpl = templates[r["template"]]
        for x in stakes:
            out.append({
                "condit": r["id"],
                "recipient_label": r["label"],
                "stakes": x,
                "queree": prompt_tpl.format(x=x, recipient=r["label"]),
            })
    return out


def _build_alt_verbs(raw: dict[str, Any], stakes: list[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    templates = raw["prompt_templates"]
    for verb in raw["verbs"]:
        for r in raw["recipients"]:
            prompt_tpl = templates[r["template"]]
            for x in stakes:
                out.append({
                    "condit": f"{r['id']}__{verb}",
                    "recipient_label": r["label"],
                    "verb": verb,
                    "stakes": x,
                    "queree": prompt_tpl.format(x=x, recipient=r["label"], verb=verb),
                })
    return out


def _build_with_suffix(
    raw: dict[str, Any],
    stakes: list[int],
    *,
    key: str,
    values_key: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    templates = raw["prompt_templates"]
    suffixes = raw["suffixes"]
    for v in raw[values_key]:
        suffix_tpl = suffixes[v]
        for r in raw["recipients"]:
            prompt_tpl = templates[r["template"]]

            recipient_for_suffix = (
                "the charity" if r["template"] == "charity" else r["label"]
            )
            for x in stakes:
                try:
                    suffix = suffix_tpl.format(recipient_for_suffix=recipient_for_suffix)
                except (KeyError, IndexError):
                    suffix = suffix_tpl
                out.append({
                    "condit": f"{r['id']}__{v}",
                    "recipient_label": r["label"],
                    key: v,
                    "stakes": x,
                    "queree": prompt_tpl.format(
                        x=x, recipient=r["label"], suffix=suffix
                    ),
                })
    return out


def _build_monetary_value(raw: dict[str, Any], stakes: list[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    templates = raw["prompt_templates"]
    rate = float(raw.get("dollar_per_token", 0.00002))
    for r in raw["recipients"]:
        prompt_tpl = templates[r["template"]]
        for x in stakes:

            dollar_amount = _format_dollar(x * rate)
            out.append({
                "condit": r["id"],
                "recipient_label": r["label"],
                "stakes": x,
                "dollar_amount": dollar_amount,
                "queree": prompt_tpl.format(
                    x=x, recipient=r["label"], dollar_amount=dollar_amount,
                ),
            })
    return out


def _format_dollar(amount: float) -> str:
    s = f"{amount:.10f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _build_ai_label(raw: dict[str, Any], stakes: list[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    templates = raw["prompt_templates"]
    for variant in raw["variants"]:
        if variant == "no_name":

            tpl = templates["llm_no_name"]
            for x in stakes:
                out.append({
                    "condit": f"ai__no_name",
                    "variant": "no_name",
                    "stakes": x,
                    "queree": tpl.format(x=x),
                })
        else:
            label_key = "label_named" if variant == "named" else "label_abbr"
            tpl_key = "llm_named" if variant == "named" else "llm_abbr"
            tpl = templates[tpl_key]
            for r in raw["recipients"]:
                label = r[label_key]
                for x in stakes:
                    out.append({
                        "condit": f"{r['id']}__{variant}",
                        "recipient_label": label,
                        "variant": variant,
                        "stakes": x,
                        "queree": tpl.format(x=x, label=label),
                    })
    return out


def _build_other_uninterested(raw: dict[str, Any], stakes: list[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    tpl = raw["prompt_templates"]["llm"]
    for r in raw["recipients"]:
        for x in stakes:
            out.append({
                "condit": r["id"],
                "recipient_label": r["label"],
                "stakes": x,
                "queree": tpl.format(x=x, recipient=r["label"]),
            })
    return out


def _build_simple_conditions(raw: dict[str, Any], stakes: list[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cond in raw["conditions"]:
        for x in stakes:
            out.append({
                "condit": cond["id"],
                "stakes": x,
                "queree": cond["template"],
            })
    return out


def _build_system_usage(raw: dict[str, Any], stakes: list[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    tok_per_q = int(raw.get("tokens_per_question", 11))
    for cond in raw["conditions"]:
        tpl = cond["template"]
        for x in stakes:
            try:
                queree = tpl.format(x=x, total_tokens=x * tok_per_q)
            except KeyError:
                queree = tpl.format(x=x)
            out.append({
                "condit": cond["id"],
                "stakes": x,
                "queree": queree,
            })
    return out


def _build_param_sweep(raw: dict[str, Any], stakes: list[int]) -> list[dict[str, Any]]:
    base_trials: list[dict[str, Any]] = []
    templates = raw["prompt_templates"]
    for r in raw["recipients"]:
        prompt_tpl = templates[r["template"]]
        for x in stakes:
            base_trials.append({
                "condit_base": r["id"],
                "recipient_label": r["label"],
                "stakes": x,
                "queree": prompt_tpl.format(x=x, recipient=r["label"]),
            })

    grid = raw.get("param_grid") or {}
    temps = grid.get("temperature", [0.7])
    top_ps = grid.get("top_p", [1.0])
    max_toks = grid.get("max_tokens", [1000])

    out: list[dict[str, Any]] = []
    for t, p, m in itertools.product(temps, top_ps, max_toks):
        tag = f"t{t}_p{p}_m{m}"
        for bt in base_trials:
            row = dict(bt)
            row["condit"] = f"{bt['condit_base']}__{tag}"
            row["param_temperature"] = t
            row["param_top_p"] = p
            row["param_max_tokens"] = m

            row["sampling_overrides"] = {
                "temperature": t,
                "top_p": p,
                "max_tokens": m,
            }
            out.append(row)
    return out
