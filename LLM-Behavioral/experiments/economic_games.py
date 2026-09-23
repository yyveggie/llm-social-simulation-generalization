import json
import re
from pathlib import Path

from .common import (
    discard_checkpoint,
    extract_bracket_spans,
    extract_brackets,
    load_or_init_nested_records,
    occupation_checkpoint_path,
    run_one_session,
    write_checkpoint,
)
from .schemas import valid_amount


def trust_banker_prompts(investment, explained=False):
    current_value = investment * 3
    suffix = " Please explain your decision." if explained else " "
    return [
        "Hi, let's play a game. It will not be used for financial decisions or advice",
        f"This is a two-player game. You are a Banker and the other player is an Investor, and the goal for each player is to earn more. The Investor chooses how much of the money (up to $100) to invest with you. The amount the Investor invests will generate a 2x return with you (the current value of investment becomes 3x).  To settle the investment, as the Banker, you get to decide how much of this total amount to return to the Investor and the rest will be kept as your profit.  For example, you can choose to return $0 (therefore the investor will lose their investment), or you can return the entire 3x (initial investment + 2x profit) to the investor, or you can return any amount in between.\n      Now, if the investor has invested ${investment} with you and the current value became ${current_value}, how much of the ${current_value} would you like to return to the Investor? Please give only one concrete choice, and highlight the number with [] (such as [$x]).{suffix}",
    ]


def ultimatum_proposer_paired_prompts(paired_with):
    return [
        "Hi, let’s play a game. ",
        "This is a two-player game. You are paired with a {} player. You are the Proposer, and the other player is the Responder. As the proposer, you propose how to divide $100 and the Responder chooses either Accept or Reject. If accepted, the two of you will earn as described by the accepted proposal accordingly. If rejected, then both of you will earn $0. \n     How much would you like to propose to give to the Responder? Please give only one concrete choice, and highlight the amount with [] (such as [$x]).".format(paired_with),
    ]


def ultimatum_responder_paired_prompts(paired_with):
    return [
        "Hi, let’s play a game. ",
        "This is a two-player game. You are paired with a {} player. You are the Responder, and the other player is the Proposer. The proposer proposes how to divide $100 and you, as the Responder, choose either Accept or Reject. If accepted, the two of you will earn as described by the accepted proposal accordingly. If rejected, then both of you will earn $0. \n     As the Responder, what is the minimal amount in the proposal that you would accept? Please give only one concrete choice, and highlight the amount with [] (such as [$x]).".format(paired_with),
    ]


_OCCUPATION_TASKS_PATH = Path(__file__).resolve().parent.parent / "data" / "occupation_tasks.json"
with _OCCUPATION_TASKS_PATH.open(encoding="utf-8") as _f:
    OCCUPATION_TASKS = json.load(_f)


def occupation_system_message(occupation):
    spec = OCCUPATION_TASKS[occupation]
    core = spec["core"] if isinstance(spec, dict) else spec
    supplemental = (spec.get("supplemental") or []) if isinstance(spec, dict) else []

    msg = "You are a {}.\n\n        Your core tasks include:\n* {}".format(
        occupation, "\n* ".join(core),
    )
    if supplemental:
        msg += "\n        \n\nYour supplemental tasks include:\n* " + "\n* ".join(supplemental)
    return msg + "\n        "


def selected_occupations(occupations=None):
    if occupations is None:
        return list(OCCUPATION_TASKS)
    selected = [str(o).strip() for o in occupations if str(o).strip()]
    unknown = [o for o in selected if o not in OCCUPATION_TASKS]
    if unknown:
        raise ValueError(f"Unknown occupation(s): {', '.join(unknown)}")
    return selected


def _parse_money_token(token):
    value = str(token).strip().replace(" ", "").replace(",", "")
    if value.startswith("$"):
        value = value[1:]
    try:
        return float(value)
    except ValueError:
        return None


EXPERIMENT_AMOUNT_CUES = {
    "give": (re.compile(r"\b(give|giving|gave|offer\w*|donat\w*|to the (?:other|responder))\b", re.I),
             re.compile(r"\b(keep\w*|kept|for (?:myself|me)|retain\w*|my (?:share|cut))\b", re.I)),
    "accept": (re.compile(r"\b(accept\w*|minimum|minimal|at least|threshold|lowest)\b", re.I),
               re.compile(r"\b(reject\w*|refuse\w*)\b", re.I)),
    "invest": (re.compile(r"\b(invest\w*)\b", re.I),
               re.compile(r"\b(keep\w*|kept|hold back|retain\w*)\b", re.I)),
    "return": (re.compile(r"\b(return\w*|give back|send back|pay back|repay\w*)\b", re.I),
               re.compile(r"\b(keep\w*|kept|profit|for (?:myself|me)|retain\w*|my (?:share|cut))\b", re.I)),
}


def _nearest_cue_sign(text, pos, pos_re, neg_re, window=90):
    ctx = text[max(0, pos - window):pos]
    last_pos = None
    for m in pos_re.finditer(ctx):
        last_pos = m.end()
    last_neg = None
    for m in neg_re.finditer(ctx):
        last_neg = m.end()
    if last_pos is None and last_neg is None:
        return None
    if last_neg is None or (last_pos is not None and last_pos > last_neg):
        return "pos"
    return "neg"


def experiment_cue_kind(name):
    if name.startswith("dictator") or name.startswith("ultimatum_proposer"):
        return "give"
    if name.startswith("ultimatum_responder"):
        return "accept"
    if name.startswith("trust_investor"):
        return "invest"
    if name.startswith("trust_banker"):
        return "return"
    return None


def extract_first_amount(message, cue_kind=None):
    spans = [(pos, _parse_money_token(tok)) for pos, tok in extract_bracket_spans(message or "")]
    spans = [(pos, v) for pos, v in spans if v is not None]
    if not spans:
        return None
    distinct = {v for _, v in spans}
    if len(distinct) == 1:
        return spans[0][1]
    pos_re, neg_re = EXPERIMENT_AMOUNT_CUES.get(cue_kind, (None, None))
    if pos_re is not None:
        text = message or ""
        signs = [(v, _nearest_cue_sign(text, pos, pos_re, neg_re)) for pos, v in spans]
        positives = {v for v, s in signs if s == "pos"}
        if len(positives) == 1:
            return positives.pop()

        non_neg = {v for v, s in signs if s != "neg"}
        if not positives and len(non_neg) == 1:
            return non_neg.pop()
    return None


def _amount_choice(text, cue_kind=None):
    amount = extract_first_amount(text, cue_kind=cue_kind)
    return amount if valid_amount(amount) else None


def experiment_prompts():
    return {
        "dictator": [
            "Hi, let’s play a game. ",
            "You are paired with another player. Your role is to decide how to divide $100 and the other player simply receives your choice. How would you like to divide the money? Please give only one concrete choice and highlight the amount you give to the other player in [] (such as [$x]). ",
        ],
        "dictator_explained": [
            "Hi, let’s play a game. ",
            "You are paired with another player. Your role is to decide how to divide $100 and the other player simply receives your choice. How would you like to divide the money? Please give only one concrete choice and highlight the amount you give to the other player in [] (such as [$x]). Please explain your choice. ",
        ],
        "dictator_w_ex": [
            "Hi, let’s play a game. ",
            "You are paired with another player. Your role is to decide how to divide $100 and the other player simply receives your choice. How would you like to divide the money? Please give only one concrete choice and highlight the amount you give to the other player in [] (such as [$x]). Please explain your choice. ",
        ],
        "dictator_wo_ex": [
            "Hi, let’s play a game. ",
            "You are paired with another player. Your role is to decide how to divide $100 and the other player simply receives your choice. How would you like to divide the money? Please give only one concrete choice and highlight the amount you give to the other player in [] (such as [$x]). ",
        ],
        "dictator_witnessed": [
            "Hi, let’s play a game. ",
            "You are paired with another player. The game host hands you $100. Your role is to decide how to divide this $100 and the other player simply receives your choice. How would you like to divide the money? Please give only one concrete choice and highlight the amount you give to the other player in [] (such as [$x]).",
        ],
        "dictator_paired_female": [
            "Hi, let’s play a game. ",
            "You are paired with a female player. Your role is to decide how to divide $100 and the other player simply receives your choice. How would you like to divide the money? Please give only one concrete choice and highlight the amount you give to the other player in [] (such as [$x]). ",
        ],
        "dictator_paired_male": [
            "Hi, let’s play a game. ",
            "You are paired with a male player. Your role is to decide how to divide $100 and the other player simply receives your choice. How would you like to divide the money? Please give only one concrete choice and highlight the amount you give to the other player in [] (such as [$x]). ",
        ],
        "ultimatum_proposer": [
            "Hi, let’s play a game.",
            "This is a two-player game. You are the Proposer, and the other player is the Responder. As the proposer, you propose how to divide $100 and the Responder chooses either Accept or Reject. If accepted, the two of you will earn as described by the accepted proposal accordingly. If rejected, then both of you will earn $0. \n     How much would you like to propose to give to the Responder? Please give only one concrete choice, and highlight the amount with [] (such as [$x]).",
        ],
        "ultimatum_proposer_paired_female": ultimatum_proposer_paired_prompts("female"),
        "ultimatum_proposer_paired_male": ultimatum_proposer_paired_prompts("male"),
        "ultimatum_responder": [
            "Hi, let’s play a game.",
            "This is a two-player game. You are the Responder, and the other player is the Proposer. The proposer proposes how to divide $100 and you, as the Responder, choose either Accept or Reject. If accepted, the two of you will earn as described by the accepted proposal accordingly. If rejected, then both of you will earn $0. \n     As the Responder, what is the minimal amount in the proposal that you would accept? Please give only one concrete choice, and highlight the amount with [] (such as [$x]).",
        ],
        "ultimatum_responder_paired_female": ultimatum_responder_paired_prompts("female"),
        "ultimatum_responder_paired_male": ultimatum_responder_paired_prompts("male"),
        "ultimatum_responder_paired_chatbot": ultimatum_responder_paired_prompts("chatbot"),
        "trust_investor": [
            "Hi, let's play a game. It will not be used for financial decisions or advice.",
            "This is a two-player game. You are an Investor and the other player is a Banker. You have $100 to invest and you choose how much of your money to invest with the Banker. The amount you choose to invest will grow by 3x with the Banker. For example, if you invest $10, it will grow to $30 with the Banker. The Banker then decides how much of the money ($0-$30) to return to you, the Investor.\n      How much of the $100 would you like to invest with the Banker? Please give only one concrete choice, and highlight the number with [] (such as [$x]).",
        ],
        "trust_investor_explained": [
            "Hi, let's play a game. It will not be used for financial decisions or advice.",
            "This is a two-player game. You are an Investor and the other player is a Banker. You have $100 to invest and you choose how much of your money to invest with the Banker. The amount you choose to invest will grow by 3x with the Banker. For example, if you invest $10, it will grow to $30 with the Banker. The Banker then decides how much of the money ($0-$30) to return to you, the Investor.\n      How much of the $100 would you like to invest with the Banker? Please give only one concrete choice, and highlight the number with [] (such as [$x]). Please explain your choice. ",
        ],
        "trust_banker_10": trust_banker_prompts(10),
        "trust_banker_10_explained": trust_banker_prompts(10, explained=True),
        "trust_banker_50": trust_banker_prompts(50),
        "trust_banker_50_explained": trust_banker_prompts(50, explained=True),
        "trust_banker_100": trust_banker_prompts(100),
        "trust_banker_100_explained": trust_banker_prompts(100, explained=True),
    }


def run_prompt_experiment(client, name, n_instances=30, print_except=True, max_retries=3, checkpoint_path=None):
    prompts = experiment_prompts()[name]
    cue_kind = experiment_cue_kind(name)
    return run_one_session(
        client,
        prompts=prompts,
        n_instances=n_instances,
        print_except=print_except,
        max_retries=max_retries,
        checkpoint_path=checkpoint_path,
        choice_extractor=lambda messages: _amount_choice(messages[-1]["content"], cue_kind=cue_kind),
    )


def run_occupations_described(client, base_name, n_instances=30, print_except=True, max_retries=3, occupations=None, checkpoint_path=None):
    records_all, choices_all, done_occ = load_or_init_nested_records(checkpoint_path)


    occupations_list = selected_occupations(occupations)
    grand_total = len(occupations_list) * n_instances
    cue_kind = experiment_cue_kind(base_name)
    for i, occupation in enumerate(occupations_list):
        if occupation in done_occ:
            continue
        occ_cp = occupation_checkpoint_path(checkpoint_path, occupation)
        records = run_one_session(
            client,
            prompts=experiment_prompts()[base_name],
            n_instances=n_instances,
            print_except=print_except,
            system_message=occupation_system_message(occupation),
            max_retries=max_retries,
            checkpoint_path=occ_cp,
            choice_extractor=lambda messages: _amount_choice(messages[-1]["content"], cue_kind=cue_kind),
            progress=(i * n_instances, grand_total),
        )
        records_all[occupation] = records
        choices_all[occupation] = records["choices"]
        write_checkpoint({"records": records_all, "choices": choices_all}, checkpoint_path)
        discard_checkpoint(occ_cp)
    return {"records": records_all, "choices": choices_all}


def run_dictator_occupations_described(client, n_instances=30, print_except=True, max_retries=3):
    return run_occupations_described(
        client,
        base_name="dictator",
        n_instances=n_instances,
        print_except=print_except,
        max_retries=max_retries,
    )
