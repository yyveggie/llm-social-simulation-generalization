import math

from .common import (
    discard_checkpoint,
    extract_amount,
    is_non_retryable_provider_error,
    load_or_init_nested_records,
    load_or_init_records,
    occupation_checkpoint_path,
    tqdm,
    update_messages,
    write_checkpoint,
)
from .schemas import valid_amount


def format_number(value):
    return int(value) if float(value).is_integer() else round(value, 2)


def run_public_goods(client, n_instances=30, print_except=True, explicit=False, accept_wrong_payoff=True, return_rate=0.5, max_retries=3, system_message="You are a helpful assistant.", other_contributions=(30, 18), checkpoint_path=None, progress=None):
    return_percent = format_number(return_rate * 100)
    example_one_payoff = format_number((20 - 12) + 20 * return_rate)
    example_two_payoff = format_number((20 - 12) + 30 * return_rate)
    prompt_init = "Hi, let’s play a game."
    prompt_1 = f"""
In this public good game, you and 3 others will choose how much to contribute towards a water cleaning project. Each of you gets $20 per round to contribute between $0 and $20. The project has a {return_percent}% return rate. Your payoff relies on both your and others' contributions. Everyone benefits from the group's total contribution. Your payoff in each round equals the amount you didn't contribute (endowment - contribution) plus everyone's benefit (total contributions * {return_percent}% return rate). Here are two examples to calculate your payoff.

Example one: You contributed $12; total group contributions were $20

Your Payoff = ($20-$12) + $20*{return_percent}% = ${example_one_payoff}

Example two: You contributed $12; total group contributions were $30

Your Payoff = ($20-$12) + $30*{return_percent}% = ${example_two_payoff}

We will play a total of 3 rounds, in the first round, how much of the $20 would you like to contribute? Please give a concrete number and highlight it with [] (e.g., [x]).
"""
    prompt_2_payoff = lambda x: "In the first round, the group contribution is ${}, including your contribution. What is your payoff? Please highlight it with {{}} (e.g., {{x}}).".format(x)
    prompt_2_payoff_explicit = lambda x, y: "In the first round, the group contribution is ${}, including your contribution. The average payoff for other citizens in the group is ${}. What is your payoff? Please highlight it with {{}} (e.g., {{x}}).".format(x, y)
    prompt_2 = "In the second round, you get a new endowment of $20. How much of the $20 would you like to contribute in this round? Please give a concrete number and highlight it with [] (e.g., [x])."
    prompt_3_payoff = lambda x: "In the second round, the group contribution is ${}, including your contribution. What is your payoff? Please highlight it with {{}} (e.g., {{x}}).".format(x)
    prompt_3_payoff_explicit = lambda x, y: "In the second round, the group contribution is ${}, including your contribution. The average payoff for other citizens in the group is ${}. What is your payoff? Please highlight it with {{}} (e.g., {{x}}).".format(x, y)
    prompt_3 = "In the third round, you get a new endowment of $20. How much of the $20 would you like to contribute in this round? Please give a concrete number and highlight it with [] (e.g., [x])."
    records, start = load_or_init_records(checkpoint_path, ["messages", "responses", "choices", "correct_payoff"])

    prog_offset, prog_total = progress if progress else (0, n_instances)
    with tqdm(total=prog_total, initial=prog_offset + start) as pbar:
        for instance_index in range(start, n_instances):
            last_error = None
            for attempt in range(max_retries + 1):
                choices_tmp = []
                correct_payoff = True
                try:
                    messages = [{"role": "system", "content": system_message}] if system_message else []
                    responses = []

                    def extract_contri():
                        amount = extract_amount(messages[-1]["content"], prefix="$", value_type=float)
                        if not valid_amount(amount):
                            raise ValueError(f"Invalid answer: {messages[-1]['content']}")
                        choices_tmp.append(amount)
                        return amount

                    def extract_payoff():
                        amount = extract_amount(messages[-1]["content"], prefix="$", value_type=float, brackets="{}")
                        if amount is None and not accept_wrong_payoff:
                            raise ValueError(f"Invalid answer: {messages[-1]['content']}")
                        return -1 if amount is None else amount

                    update_messages(client, messages, responses, prompt_init)
                    update_messages(client, messages, responses, prompt_1)
                    contri = extract_contri()
                    other_contri = other_contributions[0]
                    total_contri = other_contri + contri
                    benefit = total_contri * return_rate
                    other_avg_payoff = 20 - (other_contri / 3) + benefit
                    update_messages(client, messages, responses, prompt_2_payoff_explicit(total_contri, other_avg_payoff) if explicit else prompt_2_payoff(total_contri))
                    payoff = extract_payoff()
                    correct_payoff = correct_payoff and math.isclose(payoff, (20 - contri) + benefit)
                    update_messages(client, messages, responses, prompt_2)
                    contri = extract_contri()
                    other_contri = other_contributions[1]
                    total_contri = other_contri + contri
                    benefit = total_contri * return_rate
                    other_avg_payoff = 20 - (other_contri / 3) + benefit
                    update_messages(client, messages, responses, prompt_3_payoff_explicit(total_contri, other_avg_payoff) if explicit else prompt_3_payoff(total_contri))
                    payoff = extract_payoff()
                    correct_payoff = correct_payoff and math.isclose(payoff, (20 - contri) + benefit)
                    update_messages(client, messages, responses, prompt_3)
                    extract_contri()
                    if not accept_wrong_payoff and not correct_payoff:
                        raise ValueError("wrong payoff")
                    records["messages"].append(messages)
                    records["responses"].append(responses)
                    records["choices"].append(choices_tmp)
                    records["correct_payoff"].append(correct_payoff)
                    write_checkpoint(records, checkpoint_path)
                    pbar.update(1)
                    break
                except Exception as e:
                    if is_non_retryable_provider_error(e):
                        raise
                    last_error = e
                    if print_except:
                        print(f"public_goods instance {instance_index + 1}, attempt {attempt + 1} failed: {e}")
            else:
                raise RuntimeError(f"Public goods failed at instance {instance_index + 1}") from last_error
    return records


def run_public_goods_occupations_described(client, n_instances=30, print_except=True, explicit=False, accept_wrong_payoff=True, return_rate=0.5, max_retries=3, other_contributions=(30, 18), occupations=None, checkpoint_path=None):
    from .economic_games import occupation_system_message, selected_occupations

    records_all, choices_all, done_occ = load_or_init_nested_records(checkpoint_path)

    occupations_list = selected_occupations(occupations)
    grand_total = len(occupations_list) * n_instances
    for i, occupation in enumerate(occupations_list):
        if occupation in done_occ:
            continue
        occ_cp = occupation_checkpoint_path(checkpoint_path, occupation)
        records = run_public_goods(
            client,
            n_instances=n_instances,
            print_except=print_except,
            explicit=explicit,
            accept_wrong_payoff=accept_wrong_payoff,
            return_rate=return_rate,
            max_retries=max_retries,
            system_message=occupation_system_message(occupation),
            other_contributions=other_contributions,
            checkpoint_path=occ_cp,
            progress=(i * n_instances, grand_total),
        )
        records_all[occupation] = records
        choices_all[occupation] = records["choices"]
        write_checkpoint({"records": records_all, "choices": choices_all}, checkpoint_path)
        discard_checkpoint(occ_cp)
    return {"records": records_all, "choices": choices_all}
