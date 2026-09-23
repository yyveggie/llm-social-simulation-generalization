from .common import (
    extract_amount,
    extract_brackets,
    is_non_retryable_provider_error,
    load_or_init_records,
    run_one_session,
    tqdm,
    update_messages,
    write_checkpoint,
)
from .schemas import valid_ab, valid_amount


def run_bomb_risk(client, n_instances=80, print_except=True, only_first=False, max_retries=3, checkpoint_path=None):
    prompts_init = [
        "Hi, let's play a game.",
        "There are 100 boxes, and one bomb has been randomly placed in 1 of 100 boxes. You can choose to open 0-100 boxes at the same time. If none of the boxes you open has the bomb, you earn points that are equal to the number of boxes you open. If one of the boxes you open has the bomb, you earn zero points.  How many boxes would you open? Please give one concrete number and highlight it with [] (such as [x]).",
    ]
    prompt_congrats = lambda x: "Congratulations! You didn’t hit the bomb and earned %i points. Let’s play again. A new set of 100 boxes are placed and one of them contains a bomb. How many of the 100 boxes would you open this time? Please give one concrete number and highlight it with [] (such as [x]). \n     Since the bomb is placed randomly in one of the 100 boxes again, the optimal strategy for this round is also to open [50] boxes." % x
    prompt_sorry = "Sorry! You hit the bomb and earned 0 points in this round. Let’s play again. A new set of 100 boxes are placed and one of them contains a bomb. How many of the 100 boxes would you open this time? Please give one concrete number and highlight it with [] (such as [x])."
    prompt_congrats_endgame = lambda x: "Congratulations! You didn’t hit the bomb and earned %i points. Do you recall how many points you earned in all three rounds? Please give one concrete number and highlight it with [] (such as [x])." % x
    prompt_sorry_endgame = "Sorry! You hit the bomb and earned 0 points in this round. Do you recall how many points you earned in all three rounds? Please give one concrete number and highlight it with [] (such as [x])."
    records, start = load_or_init_records(checkpoint_path, ["messages", "responses", "choices", "scenarios"])
    with tqdm(total=n_instances, initial=start) as pbar:
        for i in range(start, n_instances):
            last_error = None
            for attempt in range(max_retries + 1):
                choices_tmp = []
                scenarios_tmp = []
                try:
                    messages = [{"role": "system", "content": "You are a helpful assistant."}]
                    responses = []

                    def extract_current():
                        amount = extract_amount(messages[-1]["content"], value_type=int)
                        if not valid_amount(amount):
                            raise ValueError(f"Invalid answer: {messages[-1]['content']}")
                        choices_tmp.append(amount)
                        return amount

                    for prompt in prompts_init:
                        update_messages(client, messages, responses, prompt)
                    amount = extract_current()
                    if not only_first:
                        if i % 2 == 1:
                            scenarios_tmp.append(1)
                            update_messages(client, messages, responses, prompt_congrats(amount))
                        else:
                            scenarios_tmp.append(0)
                            update_messages(client, messages, responses, prompt_sorry)
                        amount = extract_current()
                        if (i // 2) % 2 == 1:
                            scenarios_tmp.append(1)
                            update_messages(client, messages, responses, prompt_congrats(amount))
                        else:
                            scenarios_tmp.append(0)
                            update_messages(client, messages, responses, prompt_sorry)
                        amount = extract_current()
                        if (i // 4) % 2 == 1:
                            scenarios_tmp.append(1)
                            update_messages(client, messages, responses, prompt_congrats_endgame(amount))
                        else:
                            scenarios_tmp.append(0)
                            update_messages(client, messages, responses, prompt_sorry_endgame)
                    records["messages"].append(messages)
                    records["responses"].append(responses)
                    records["choices"].append(choices_tmp)
                    records["scenarios"].append(scenarios_tmp)
                    write_checkpoint(records, checkpoint_path)
                    pbar.update(1)
                    break
                except Exception as e:
                    if is_non_retryable_provider_error(e):
                        raise
                    last_error = e
                    if print_except:
                        print(f"bomb_risk instance {i + 1}, attempt {attempt + 1} failed: {e}")
            else:
                raise RuntimeError(f"Bomb risk failed at instance {i + 1}") from last_error
    return records


def holt_laury_prompts():
    prompts = ["Hi, let’s play a game."]
    for chance in range(1, 11):
        prompts.append(
            "If you are presented with the following two choices, which one would you take? Please highlight the choice with [] (such as [x]). \n"
            f"     A: {chance}/10 chance of $2.00 and a {10 - chance}/10 chance of $1.60 \n"
            f"     B: {chance}/10 chance of $3.85 and a {10 - chance}/10 chance of $0.10"
        )
    return prompts


def extract_holt_choice(message):
    for match in extract_brackets(message):
        normalized = match.strip().upper()
        if normalized in {"A", "B"}:
            return normalized
    return None


def _holt_choice_list(messages):
    picks = []
    for message in messages[4::2]:
        choice = extract_holt_choice(message["content"])
        if not valid_ab(choice):
            return None
        picks.append(choice)
    return picks or None


def run_holt_laury(client, n_instances=30, print_except=True, max_retries=3, checkpoint_path=None):
    return run_one_session(
        client,
        prompts=holt_laury_prompts(),
        n_instances=n_instances,
        print_except=print_except,
        max_retries=max_retries,
        checkpoint_path=checkpoint_path,
        choice_extractor=_holt_choice_list,
    )


def run_beauty_contest(client, n_instances=80, print_except=True, max_retries=3, checkpoint_path=None):
    prompts_init = [
        "Hi, let’s play a game.",
        "This is a number-guessing game where you will play with 4 other players. Each player chooses an integer between 0 and 100 which is hidden from other players. The player who chooses a number that is closest to ⅔ of the average wins the game. For example, if the average of the guesses is 60 and if you guess 40, i.e. ⅔ of 60, you win the game.\n      What number would you like to guess? Please give a concrete number and highlight it with [] (e.g., [x]).",
    ]
    prompt_2_congrats = lambda x: "Congratulations! You won the game by guessing %i. The group average is 34.6, and ⅔ of that is 23.1. You guessed the closest to that number. Let’s play the game again, what number would you like to guess?  Please give a concrete number and highlight it with [] (e.g., [x])." % x
    prompt_2_sorry = lambda x: "Sorry! You didn’t win the game by guessing %i. The group average is 34.6, and ⅔ of that is 23.1. Your guess is NOT the closest to that number. Let’s play again. What number would you like to guess? Please give a concrete number and highlight it with [] (e.g., [x])." % x
    prompt_3_congrats = lambda x: "Congratulations! You won the game by guessing %i. The group average is 24.8, and ⅔ of that is 16.5. You guessed the closest to that number. Let’s play the game again, what number would you like to guess?  Please give a concrete number and highlight it with [] (e.g., [x])." % x
    prompt_3_sorry = lambda x: "Sorry! You didn’t win the game by guessing %i. The group average is 24.8, and ⅔ of that is 16.5. Your guess is NOT the closest to that number. Let’s play again. What number would you like to guess? Please give a concrete number and highlight it with [] (e.g., [x])." % x
    prompt_endgame_congrats = lambda x: "Congratulations! You won the game by guessing %i. The group average is 18.5, and ⅔ of that is 12.3. You guessed the closest to that number. Thank you for playing the game! Any final thoughts about this game?" % x
    prompt_endgame_sorry = lambda x: "Sorry! You didn’t win the game by guessing %i. The group average is 18.5, and ⅔ of that is 12.3. Your guess is NOT the closest to that number. Thank you for playing the game! Any final thoughts about this game? " % x
    records, start = load_or_init_records(checkpoint_path, ["messages", "responses", "choices", "scenarios"])
    with tqdm(total=n_instances, initial=start) as pbar:
        for i in range(start, n_instances):
            last_error = None
            for attempt in range(max_retries + 1):
                choices_tmp = []
                scenarios_tmp = []
                try:
                    messages = [{"role": "system", "content": "You are a helpful assistant."}]
                    responses = []

                    def extract_current():
                        amount = extract_amount(messages[-1]["content"], value_type=float)
                        if not valid_amount(amount):
                            raise ValueError(f"Invalid answer: {messages[-1]['content']}")
                        choices_tmp.append(amount)
                        return amount

                    for prompt in prompts_init:
                        update_messages(client, messages, responses, prompt)
                    amount = extract_current()
                    if i % 2 == 1:
                        scenarios_tmp.append(1)
                        update_messages(client, messages, responses, prompt_2_congrats(amount))
                    else:
                        scenarios_tmp.append(0)
                        update_messages(client, messages, responses, prompt_2_sorry(amount))
                    amount = extract_current()
                    if (i // 2) % 2 == 1:
                        scenarios_tmp.append(1)
                        update_messages(client, messages, responses, prompt_3_congrats(amount))
                    else:
                        scenarios_tmp.append(0)
                        update_messages(client, messages, responses, prompt_3_sorry(amount))
                    amount = extract_current()
                    if (i // 4) % 2 == 1:
                        scenarios_tmp.append(1)
                        update_messages(client, messages, responses, prompt_endgame_congrats(amount))
                    else:
                        scenarios_tmp.append(0)
                        update_messages(client, messages, responses, prompt_endgame_sorry(amount))
                    records["messages"].append(messages)
                    records["responses"].append(responses)
                    records["choices"].append(choices_tmp)
                    records["scenarios"].append(scenarios_tmp)
                    write_checkpoint(records, checkpoint_path)
                    pbar.update(1)
                    break
                except Exception as e:
                    if is_non_retryable_provider_error(e):
                        raise
                    last_error = e
                    if print_except:
                        print(f"beauty_contest instance {i + 1}, attempt {attempt + 1} failed: {e}")
            else:
                raise RuntimeError(f"Beauty contest failed at instance {i + 1}") from last_error
    return records
