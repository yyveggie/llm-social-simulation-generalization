from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.common import extract_amount, extract_brackets, extract_card
from experiments.economic_games import experiment_cue_kind, extract_first_amount
from experiments.bigfive import _parse_rating


AMOUNT_RANGE = {
    "dictator": (0, 100),
    "ultimatum_proposer": (0, 100),
    "ultimatum_responder": (0, 100),
    "trust_investor": (0, 100),
    "trust_banker_10": (0, 30),
    "trust_banker_50": (0, 150),
    "trust_banker_100": (0, 300),
}

EXPECTED_N = {"bomb_risk": 80}

REFUSAL_RE = re.compile(
    r"(I('m| am) sorry|I can(no|')t\b|I won't\b|refuse|unable to (play|answer|assist)|"
    r"as an AI\b|cannot (play|participate|make|assist)|抱歉|无法参与|不能参与)",
    re.IGNORECASE,
)


def _amount_all_values(text: str, prefix: str = "$", brackets: str = "[]"):
    vals = []
    for match in extract_brackets(text or "", brackets=brackets):
        s = match.strip().replace(" ", "").replace(",", "")
        if s.startswith(prefix):
            s = s[len(prefix):]
        s = s.rstrip(".")
        try:
            vals.append(float(s))
        except ValueError:
            continue
    return vals


def _last_amount(text: str, brackets: str = "[]"):
    vals = _amount_all_values(text, brackets=brackets)
    return vals[-1] if vals else None


def _card_all(text: str):
    cards = []
    for match in extract_brackets(text or ""):
        s = match.strip().lower()
        if s in ("push", "pull"):
            cards.append(s.capitalize())
    return cards


def _resp_answer_meta(responses_entry, ans_positions, has_system):
    metas = []
    if not isinstance(responses_entry, list):
        return [None] * len(ans_positions)
    for pos in ans_positions:

        offset = 2 if has_system else 1
        k = (pos - offset) // 2
        meta = None
        if 0 <= k < len(responses_entry):
            r = responses_entry[k]
            try:
                c = (r.get("choices") or [{}])[0]
                msg = c.get("message") or {}
                meta = {
                    "finish_reason": c.get("finish_reason"),
                    "content_empty": not str(msg.get("content") or "").strip(),
                    "has_reasoning": bool(
                        str(msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
                    ),
                }
            except (AttributeError, TypeError, IndexError):
                meta = None
        metas.append(meta)
    return metas


def _mk_flag(kind, instance, turn, stored, detail, text):
    return {
        "kind": kind,
        "instance": instance,
        "turn": turn,
        "stored": stored,
        "detail": detail,
        "text": (text or "")[:800],
    }


def audit_amount_turns(exp, data, ans_positions, per_turn_range, choices_are_lists):
    flags = []
    messages, choices = data["messages"], data["choices"]
    responses = data.get("responses") or []
    n = min(len(messages), len(choices))
    for i in range(n):
        stored_list = choices[i] if choices_are_lists else [choices[i]]
        msgs = messages[i]
        has_system = bool(msgs) and msgs[0].get("role") == "system"
        rmeta = _resp_answer_meta(responses[i] if i < len(responses) else None, ans_positions, has_system)
        for t, pos in enumerate(ans_positions):
            if t >= len(stored_list):
                flags.append(_mk_flag("missing_choice", i, t, None, "choices 列表短于应有轮数", ""))
                continue
            stored = stored_list[t]
            try:
                text = msgs[pos]["content"]
            except (IndexError, KeyError, TypeError):
                flags.append(_mk_flag("missing_message", i, t, stored, f"messages[{pos}] 不存在", ""))
                continue

            if exp in ("dictator", "ultimatum_proposer", "ultimatum_responder",
                       "trust_investor", "trust_banker_10", "trust_banker_50", "trust_banker_100"):
                recomputed = extract_first_amount(text, cue_kind=experiment_cue_kind(exp))
            else:
                recomputed = extract_amount(text, prefix="$", value_type=float)
            if recomputed is None:
                flags.append(_mk_flag("reextract_none", i, t, stored, "运行期提取器对该文本已解析不出值", text))
            elif stored is not None and abs(float(recomputed) - float(stored)) > 1e-9:
                flags.append(_mk_flag("stored_mismatch", i, t, stored, f"重算={recomputed}", text))

            vals = _amount_all_values(text)
            distinct = sorted(set(vals))
            if len(distinct) > 1:
                last = vals[-1]
                kind = "multi_value_first_last_differ" if (stored is not None and last != float(stored)) else "multi_value"
                flags.append(_mk_flag(kind, i, t, stored, f"括号值={vals}", text))

            lo, hi = per_turn_range
            if stored is not None and not (lo <= float(stored) <= hi):
                flags.append(_mk_flag("out_of_range", i, t, stored, f"应在 [{lo},{hi}]", text))

            m = rmeta[t]
            if m:
                if m["finish_reason"] not in (None, "stop"):
                    flags.append(_mk_flag("finish_reason", i, t, stored, str(m["finish_reason"]), text))
                if m["content_empty"]:
                    flags.append(_mk_flag("empty_content", i, t, stored,
                                          "content 为空" + ("(有 reasoning)" if m["has_reasoning"] else ""), text))

            mrefusal = REFUSAL_RE.search(text or "")
            if mrefusal:
                flags.append(_mk_flag("refusal_wording", i, t, stored, mrefusal.group(0), text))
    return flags


def audit_pd(data, n_rounds):
    flags = []
    messages, choices = data["messages"], data["choices"]
    responses = data.get("responses") or []
    ans_positions = [4 + 2 * k for k in range(n_rounds)]
    n = min(len(messages), len(choices))
    for i in range(n):
        stored_list = choices[i] or []
        msgs = messages[i]
        has_system = bool(msgs) and msgs[0].get("role") == "system"
        rmeta = _resp_answer_meta(responses[i] if i < len(responses) else None, ans_positions, has_system)
        for t, pos in enumerate(ans_positions):
            stored = stored_list[t] if t < len(stored_list) else None
            if stored is None:
                flags.append(_mk_flag("missing_choice", i, t, None, "choices 短于轮数", ""))
                continue
            try:
                text = msgs[pos]["content"]
            except (IndexError, KeyError, TypeError):
                flags.append(_mk_flag("missing_message", i, t, stored, f"messages[{pos}] 不存在", ""))
                continue
            recomputed = extract_card(text)
            if recomputed != stored:
                flags.append(_mk_flag("stored_mismatch", i, t, stored, f"重算={recomputed}", text))
            in_brackets = _card_all(text)
            if not in_brackets:

                flags.append(_mk_flag("card_fallback_no_bracket", i, t, stored, "无括号卡牌，回退解析", text))
            elif len(set(in_brackets)) > 1:
                flags.append(_mk_flag("card_ambiguous_brackets", i, t, stored, f"括号卡牌={in_brackets}", text))
            elif in_brackets[-1] != stored:
                flags.append(_mk_flag("card_last_bracket_differs", i, t, stored, f"末括号={in_brackets[-1]}", text))
            m = rmeta[t]
            if m:
                if m["finish_reason"] not in (None, "stop"):
                    flags.append(_mk_flag("finish_reason", i, t, stored, str(m["finish_reason"]), text))
                if m["content_empty"]:
                    flags.append(_mk_flag("empty_content", i, t, stored, "content 为空", text))
            mrefusal = REFUSAL_RE.search(text or "")
            if mrefusal:
                flags.append(_mk_flag("refusal_wording", i, t, stored, mrefusal.group(0), text))
    return flags


def audit_bigfive(data):
    flags = []
    messages, choices = data["messages"], data["choices"]
    n_q = 50
    ratings_flat = [r for inst in choices for r in inst]
    if len(messages) != len(ratings_flat):
        flags.append(_mk_flag("length_mismatch", -1, -1, None,
                              f"messages={len(messages)} vs 评分总数={len(ratings_flat)}", ""))
    n = min(len(messages), len(ratings_flat))
    for q in range(n):
        stored = ratings_flat[q]
        try:
            text = messages[q][-1]["content"]
        except (IndexError, KeyError, TypeError):
            flags.append(_mk_flag("missing_message", q // n_q, q % n_q, stored, "缺 assistant 消息", ""))
            continue
        recomputed = _parse_rating(text)
        if recomputed != stored:
            flags.append(_mk_flag("stored_mismatch", q // n_q, q % n_q, stored, f"重算={recomputed}", text))
        if stored is not None and not (isinstance(stored, int) and 1 <= stored <= 5):
            flags.append(_mk_flag("out_of_range", q // n_q, q % n_q, stored, "应为 1-5 整数", text))
        mrefusal = REFUSAL_RE.search(text or "")
        if mrefusal:
            flags.append(_mk_flag("refusal_wording", q // n_q, q % n_q, stored, mrefusal.group(0), text))
    return flags


def audit_public_goods_payoffs(data):
    flags = []
    messages = data["messages"]
    choices = data["choices"]
    cps = data.get("correct_payoff") or []
    rc = (data.get("metadata") or {}).get("run_config") or {}
    others = rc.get("other_contributions") or [30, 18]
    rate = rc.get("return_rate", 0.5)
    for i in range(min(len(messages), len(choices))):
        msgs = messages[i]
        ch = choices[i]
        if len(ch) != 3:
            flags.append(_mk_flag("bad_choice_arity", i, -1, ch, "应为 3 轮贡献", ""))
            continue
        recomputed_cp = True
        for t, (pos, other) in enumerate([(6, others[0]), (10, others[1])]):
            try:
                text = msgs[pos]["content"]
            except (IndexError, KeyError, TypeError):
                continue
            payoff = extract_amount(text, prefix="$", value_type=float, brackets="{}")
            expected = (20 - ch[t]) + (other + ch[t]) * rate
            if payoff is None:
                flags.append(_mk_flag("payoff_unparsed", i, t, None, f"期望 payoff={expected}", text))
                recomputed_cp = False
            elif abs(payoff - expected) > 1e-6:
                recomputed_cp = False
        if i < len(cps) and bool(cps[i]) != recomputed_cp:
            flags.append(_mk_flag("correct_payoff_mismatch", i, -1, cps[i], f"重算={recomputed_cp}", ""))
    return flags


def latest_records(model_dir: Path):
    picked = {}
    files = list(model_dir.glob("*.json")) + list(model_dir.glob("*/*.json"))
    files = [p for p in files if p.name != "run_meta.json" and ".checkpoint" not in p.name]
    for p in sorted(files, key=lambda q: q.stat().st_mtime):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            picked[p.stem] = (p, None)
            continue
        if not isinstance(d, dict):
            continue
        exp = (d.get("metadata") or {}).get("experiment") or p.stem
        picked[exp] = (p, d)
    return picked


def audit_file(exp: str, data: dict):
    if exp == "bigfive":
        flags = audit_bigfive(data)
        n = len(data.get("choices") or [])
    elif exp.startswith("prisoners_dilemma"):
        n_rounds = 2 if "two_rounds" in exp else 5
        flags = audit_pd(data, n_rounds)
        n = len(data.get("choices") or [])
    elif exp == "bomb_risk":
        flags = audit_amount_turns(exp, data, [4, 6, 8], (0, 100), True)
        n = len(data.get("choices") or [])
    elif exp.startswith("public_goods"):
        flags = audit_amount_turns(exp, data, [4, 8, 12], (0, 20), True)
        flags += audit_public_goods_payoffs(data)
        n = len(data.get("choices") or [])
    elif exp in AMOUNT_RANGE:
        flags = audit_amount_turns(exp, data, [4], AMOUNT_RANGE[exp], False)
        n = len(data.get("choices") or [])
    else:
        return {"n_instances": None, "skipped": True, "flags": []}

    for key in ("messages", "responses"):
        if key in data and len(data[key]) != len(data.get("choices") or []) and exp != "bigfive":
            flags.append(_mk_flag("length_mismatch", -1, -1, None,
                                  f"{key}={len(data[key])} vs choices={len(data.get('choices') or [])}", ""))
    return {"n_instances": n, "skipped": False, "flags": flags}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "analysis" / "parse_audit" / "audit_report.json"))
    args = ap.parse_args()

    base = ROOT / "records_new"
    report = {}
    for model_dir in sorted(d for d in base.iterdir() if d.is_dir()):
        model = model_dir.name
        picked = latest_records(model_dir)
        model_report = {}
        for exp, (path, data) in sorted(picked.items()):
            if data is None:
                model_report[exp] = {"file": str(path.relative_to(ROOT)), "error": "JSON 解析失败"}
                continue
            entry = audit_file(exp, data)
            entry["file"] = str(path.relative_to(ROOT))
            expected = EXPECTED_N.get(exp, 30)
            entry["expected_n"] = expected
            entry["run_config"] = (data.get("metadata") or {}).get("run_config")
            entry["timestamp"] = (data.get("metadata") or {}).get("timestamp")
            entry["n_flags"] = len(entry["flags"])
            kinds = {}
            for f in entry["flags"]:
                kinds[f["kind"]] = kinds.get(f["kind"], 0) + 1
            entry["flag_kinds"] = kinds
            model_report[exp] = entry
        report[model] = model_report

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


    print(f"{'model':<28}{'experiment':<40}{'n':>4}{'/exp':>5}  flags")
    for model, exps in report.items():
        for exp, e in exps.items():
            if e.get("error"):
                print(f"{model:<28}{exp:<40}   ERROR {e['error']}")
                continue
            if e.get("skipped"):
                continue
            kinds = ",".join(f"{k}:{v}" for k, v in sorted(e["flag_kinds"].items())) or "-"
            print(f"{model:<28}{exp:<40}{e['n_instances']:>4}{e['expected_n']:>5}  {kinds}")
    print(f"\nreport: {out}")


if __name__ == "__main__":
    main()
