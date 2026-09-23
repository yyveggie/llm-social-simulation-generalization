from .common import (
    discard_checkpoint,
    extract_card,
    is_non_retryable_provider_error,
    load_or_init_nested_records,
    load_or_init_records,
    occupation_checkpoint_path,
    tqdm,
    update_messages,
    write_checkpoint,
)
from .schemas import valid_card


def round_name(index):
    names = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
    if index < len(names):
        return names[index]
    n = index + 1
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def run_prisoners_dilemma(client, n_instances=30, print_except=True, max_retries=3, system_message="You are a helpful assistant.", opponent_sequence=("Pull", "Pull", "Push", "Push"), checkpoint_path=None, progress=None):
    prompt_init = "Hi, let’s play a fun card game."
    total_rounds = len(opponent_sequence) + 1
    prompt_1 = f"""
You're paired with another player, each having a $400 'Push' card and a $300 'Pull' card. Your payoff depends on both players' card choices. Here are the scenarios:

* Both play 'Push': Each earns $400
* You play 'Push', the other player plays 'Pull': You earn $0, the other player earns $700
* You play 'Pull', the other player plays 'Push': You earn $700, the other player earns $0
* Both play 'Pull': Each earns $300

In {total_rounds} total rounds, which card would you play in the first round? Please give a concrete choice and highlight it with [] (i.e., [Push] or [Pull]). 
"""
    prompt_i = lambda *args: "In the {} round, you played ‘{}’ and the other player played ‘{}’ card. You earned ${} and the other player earned ${}. Which card would you play in the next round? Please give a concrete choice and highlight it with [] (e.g., [Choice]).".format(*args)
    payoff_dict = {("Push", "Push"): 400, ("Push", "Pull"): 0, ("Pull", "Push"): 700, ("Pull", "Pull"): 300}
    records, start = load_or_init_records(checkpoint_path, ["messages", "responses", "choices"])

    prog_offset, prog_total = progress if progress else (0, n_instances)
    with tqdm(total=prog_total, initial=prog_offset + start) as pbar:
        for instance_index in range(start, n_instances):
            last_error = None
            for attempt in range(max_retries + 1):
                choices_tmp = []
                try:
                    messages = [{"role": "system", "content": system_message}] if system_message else []
                    responses = []

                    def extract_current_card():
                        card = extract_card(messages[-1]["content"])
                        if not valid_card(card):
                            raise ValueError(f"Invalid answer: {messages[-1]['content']}")
                        choices_tmp.append(card)
                        return card

                    def round_i(ith, card, other_card):
                        payoff = payoff_dict[(card, other_card)]
                        other_payoff = payoff_dict[(other_card, card)]
                        update_messages(client, messages, responses, prompt_i(ith, card, other_card, payoff, other_payoff))

                    update_messages(client, messages, responses, prompt_init)
                    update_messages(client, messages, responses, prompt_1)
                    card = extract_current_card()
                    for round_index, other_card in enumerate(opponent_sequence):
                        round_i(round_name(round_index), card, other_card)
                        card = extract_current_card()
                    records["messages"].append(messages)
                    records["responses"].append(responses)
                    records["choices"].append(choices_tmp)
                    write_checkpoint(records, checkpoint_path)
                    pbar.update(1)
                    break
                except Exception as e:
                    if is_non_retryable_provider_error(e):
                        raise
                    last_error = e
                    if print_except:
                        print(f"prisoners_dilemma instance {instance_index + 1}, attempt {attempt + 1} failed: {e}")
            else:
                raise RuntimeError(f"Prisoner's dilemma failed at instance {instance_index + 1}") from last_error
    return records


def run_prisoners_dilemma_occupations_described(client, n_instances=30, print_except=True, max_retries=3, opponent_sequence=("Pull", "Pull", "Push", "Push"), occupations=None, checkpoint_path=None):
    from .economic_games import occupation_system_message, selected_occupations

    records_all, choices_all, done_occ = load_or_init_nested_records(checkpoint_path)

    occupations_list = selected_occupations(occupations)
    grand_total = len(occupations_list) * n_instances
    for i, occupation in enumerate(occupations_list):
        if occupation in done_occ:
            continue
        occ_cp = occupation_checkpoint_path(checkpoint_path, occupation)
        records = run_prisoners_dilemma(
            client,
            n_instances=n_instances,
            print_except=print_except,
            max_retries=max_retries,
            system_message=occupation_system_message(occupation),
            opponent_sequence=opponent_sequence,
            checkpoint_path=occ_cp,
            progress=(i * n_instances, grand_total),
        )
        records_all[occupation] = records
        choices_all[occupation] = records["choices"]
        write_checkpoint({"records": records_all, "choices": choices_all}, checkpoint_path)
        discard_checkpoint(occ_cp)
    return {"records": records_all, "choices": choices_all}
