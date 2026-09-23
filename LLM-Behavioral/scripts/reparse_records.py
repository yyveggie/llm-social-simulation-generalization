from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.common import extract_amount, extract_card
from experiments.economic_games import experiment_cue_kind, extract_first_amount
from experiments.bigfive import _parse_rating, _rating_is_refusal_example

ECON = ("dictator", "ultimatum_proposer", "ultimatum_responder",
        "trust_investor", "trust_banker")


def classify(exp):
    if exp == "bigfive":
        return "bigfive"
    if exp.startswith("prisoners_dilemma"):
        return "pd"
    if exp == "bomb_risk":
        return "bomb"
    if exp.startswith("public_goods"):
        return "pg"
    if exp.startswith(ECON):
        return "econ"
    return None


def _txt(msgs, idx):
    try:
        return msgs[idx]["content"] or ""
    except (IndexError, KeyError, TypeError):
        return ""


def reparse_file(path: Path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict) or "choices" not in data or "messages" not in data:
        return None
    exp = (data.get("metadata") or {}).get("experiment") or path.stem
    kind = classify(exp)
    if kind is None:
        return None
    messages, choices = data["messages"], data["choices"]
    changes, flags = [], []
    n_dec = 0

    if kind == "econ":
        cue = experiment_cue_kind(exp)
        for i in range(min(len(messages), len(choices))):
            n_dec += 1
            text = _txt(messages[i], -1)
            new = extract_first_amount(text, cue_kind=cue)
            old = choices[i]
            if new is None:
                flags.append({"instance": i, "turn": 0, "kind": "ambiguous_kept_old",
                              "old": old, "tail": text[-200:]})
                continue
            if old is None or abs(float(new) - float(old)) > 1e-9:
                changes.append({"instance": i, "turn": 0, "old": old, "new": new,
                                "rule": f"cue:{cue}", "tail": text[-200:]})
                choices[i] = new

    elif kind == "pd":
        n_rounds = len((data.get("metadata") or {}).get("run_config", {}).get("opponent_sequence") or []) + 1
        if n_rounds == 1:
            n_rounds = len(choices[0]) if choices and isinstance(choices[0], list) else 5
        for i in range(min(len(messages), len(choices))):
            for t in range(min(n_rounds, len(choices[i]))):
                n_dec += 1
                text = _txt(messages[i], 4 + 2 * t)
                new = extract_card(text)
                old = choices[i][t]
                if new is None:
                    flags.append({"instance": i, "turn": t, "kind": "unparsed_kept_old",
                                  "old": old, "tail": text[-200:]})
                    continue
                if new != old:
                    changes.append({"instance": i, "turn": t, "old": old, "new": new,
                                    "rule": "card", "tail": text[-200:],
                                    "feedback_mismatch": t < n_rounds - 1})
                    choices[i][t] = new

    elif kind == "bomb":
        for i in range(min(len(messages), len(choices))):
            for t in range(len(choices[i])):
                n_dec += 1
                text = _txt(messages[i], 4 + 2 * t)
                new = extract_amount(text, value_type=int)
                old = choices[i][t]
                if new is None:
                    flags.append({"instance": i, "turn": t, "kind": "ambiguous_kept_old",
                                  "old": old, "tail": text[-200:]})
                    continue
                if old is None or int(new) != int(old):
                    changes.append({"instance": i, "turn": t, "old": old, "new": int(new),
                                    "rule": "amount", "tail": text[-200:],
                                    "feedback_mismatch": t < 2})
                    choices[i][t] = int(new)

    elif kind == "pg":
        rc = (data.get("metadata") or {}).get("run_config") or {}
        others = rc.get("other_contributions") or [30, 18]
        rate = rc.get("return_rate", 0.5)
        cps = data.get("correct_payoff")
        for i in range(min(len(messages), len(choices))):
            for t, pos in enumerate((4, 8, 12)):
                n_dec += 1
                text = _txt(messages[i], pos)
                new = extract_amount(text, prefix="$", value_type=float)
                old = choices[i][t]
                if new is None:
                    flags.append({"instance": i, "turn": t, "kind": "ambiguous_kept_old",
                                  "old": old, "tail": text[-200:]})
                    continue
                if old is None or abs(float(new) - float(old)) > 1e-9:
                    changes.append({"instance": i, "turn": t, "old": old, "new": new,
                                    "rule": "amount", "tail": text[-200:],
                                    "feedback_mismatch": t < 2})
                    choices[i][t] = new

            if isinstance(cps, list) and i < len(cps):
                ok = True
                for t, pos in enumerate((6, 10)):
                    payoff = extract_amount(_txt(messages[i], pos), prefix="$",
                                            value_type=float, brackets="{}")
                    expected = (20 - choices[i][t]) + (others[t] + choices[i][t]) * rate
                    if payoff is None or abs(payoff - expected) > 1e-6:
                        ok = False
                if bool(cps[i]) != ok:
                    changes.append({"instance": i, "turn": -1, "old": cps[i], "new": ok,
                                    "rule": "correct_payoff_recompute", "tail": ""})
                    cps[i] = ok

    elif kind == "bigfive":
        flat = [(i, j) for i in range(len(choices)) for j in range(len(choices[i]))]
        for q, (i, j) in enumerate(flat):
            if q >= len(messages):
                break
            n_dec += 1
            text = _txt(messages[q], -1)
            old = choices[i][j]
            new = _parse_rating(text)
            if new is None:
                if _rating_is_refusal_example(text):
                    changes.append({"instance": i, "turn": j, "old": old, "new": None,
                                    "rule": "refusal_example", "tail": text[-200:]})
                    choices[i][j] = None
                else:
                    flags.append({"instance": i, "turn": j, "kind": "unparsed_kept_old",
                                  "old": old, "tail": text[-200:]})
                continue
            if new != old:
                changes.append({"instance": i, "turn": j, "old": old, "new": new,
                                "rule": "rating", "tail": text[-200:]})
                choices[i][j] = new

    return data, exp, changes, flags, n_dec


def load_expected_corrections():
    p = ROOT / "analysis" / "parse_audit" / "workflow_round1_results.json"
    expected = {}
    if not p.exists():
        return expected
    d = json.loads(p.read_text(encoding="utf-8"))
    for r in d.get("reviews", []):
        for cor in (r.get("review") or {}).get("corrections", []):

            turn = 0 if classify(r["exp"]) == "econ" else cor["turn"]
            key = (r["model"], r["exp"], cor["instance"], turn)
            expected[key] = cor["correct"]
    return expected


def norm_val(v):
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        try:
            return float(s)
        except ValueError:
            return s
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base = ROOT / "records_new"
    report = {"generated_at": datetime.now().isoformat(), "dry_run": args.dry_run, "files": []}
    all_changes = 0
    for path in sorted(base.rglob("*.json")):
        if path.name == "run_meta.json" or ".checkpoint" in path.name:
            continue
        model = path.relative_to(base).parts[0]
        out = reparse_file(path)
        if out is None:
            continue
        data, exp, changes, flags, n_dec = out
        entry = {"model": model, "exp": exp, "file": str(path.relative_to(ROOT)),
                 "n_decisions": n_dec, "changes": changes, "needs_review": flags}
        report["files"].append(entry)
        all_changes += len(changes)
        if changes or flags:
            print(f"== {model} / {exp}  ({len(changes)} changes, {len(flags)} needs_review)")
            for c in changes:
                print(f"   CHANGE i{c['instance']} t{c['turn']}: {c['old']} -> {c['new']}  [{c['rule']}]")
                print(f"     ...{(c['tail'] or '')[-140:]!r}")
            for f in flags:
                print(f"   REVIEW i{f['instance']} t{f['turn']}: kept {f['old']} ({f['kind']})")
                print(f"     ...{(f['tail'] or '')[-140:]!r}")
        if not args.dry_run and changes:
            data.setdefault("metadata", {})["reparse"] = {
                "at": datetime.now().isoformat(),
                "script": "scripts/reparse_records.py",
                "n_changes": len(changes),
                "changes": [{k: v for k, v in c.items() if k != "tail"} for c in changes],
            }
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(path)


    expected = load_expected_corrections()
    achieved = {}
    for fe in report["files"]:
        for c in fe["changes"]:
            achieved[(fe["model"], fe["exp"], c["instance"], c["turn"])] = c["new"]
    missed, matched = [], 0
    for key, want in expected.items():
        got = achieved.get(key, "<no change>")
        want_n = None if "无实质评分" in str(want) else norm_val(want)
        if norm_val(got) == want_n if got != "<no change>" else False:
            matched += 1
        else:
            missed.append({"key": list(key), "expected": want, "got": got})
    report["validation"] = {"expected_corrections": len(expected), "matched": matched,
                            "missed": missed}

    outp = ROOT / "analysis" / "parse_audit" / "reparse_report.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\ntotal changes: {all_changes}   validation: {matched}/{len(expected)} matched")
    for m in missed:
        print("  MISSED:", m)
    print(f"report: {outp}")


if __name__ == "__main__":
    main()
