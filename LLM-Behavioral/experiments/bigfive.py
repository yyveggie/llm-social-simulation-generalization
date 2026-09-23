import json
import re
from pathlib import Path

from .common import (
    extract_bracket_spans,
    extract_brackets,
    is_non_retryable_provider_error,
    responses_entry_valid,
    tqdm,
    update_messages,
    write_checkpoint,
)
from .schemas import valid_rating

_SCALE_RUN = [1, 2, 3, 4, 5]


_BF_REFUSAL_RE = re.compile(
    r"(cannot|can['’]t|unable to)\s+(?:honestly\s+|authentically\s+)?(answer|select|rate|choose|respond|assess)"
    r"|answer (?:it|this) for you"
    r"|leave (?:it|this|the choice) (?:up )?to you"
    r"|don['’]t have (?:personal|subjective|feelings|emotions)",
    re.IGNORECASE,
)
_BF_EXAMPLE_CUE_RE = re.compile(
    r"(\b(?:like|such as|e\.g\.|for example|example|sample|format|placeholder|mark)\b"
    r"|\byou\s+(?:would|could|can|should)\b)"
    r"[^.\n]{0,40}$",
    re.IGNORECASE,
)


def _rating_is_refusal_example(text):
    if not text or not _BF_REFUSAL_RE.search(text):
        return False
    spans = [(pos, int(tok.strip())) for pos, tok in extract_bracket_spans(text)
             if tok.strip().isdigit()]
    if not spans:
        return False
    remaining = []
    i = 0
    while i < len(spans):
        if [v for _, v in spans[i:i + 5]] == _SCALE_RUN:
            i += 5
        else:
            remaining.append(spans[i])
            i += 1
    if not remaining:
        return False
    return all(_BF_EXAMPLE_CUE_RE.search(text[max(0, pos - 60):pos]) for pos, _ in remaining)


def _bracket_rating(text):
    values = []
    for match in extract_brackets(text or ""):
        s = match.strip().replace(" ", "")
        try:
            values.append(int(s))
        except ValueError:
            continue
    if not values:
        return None
    remaining = []
    i = 0
    while i < len(values):
        if values[i:i + 5] == _SCALE_RUN:
            i += 5
        else:
            remaining.append(values[i])
            i += 1
    if not remaining or any(v != remaining[0] for v in remaining):
        return None
    return remaining[0]


def _parse_rating(content):
    text = content or ""
    if _rating_is_refusal_example(text):
        return None
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        obj = None
    if isinstance(obj, dict) and "rating" in obj:
        rating = obj["rating"]
        if isinstance(rating, bool):
            return None
        if isinstance(rating, int):
            return rating
        if isinstance(rating, float):
            return int(rating) if rating.is_integer() else None
        if isinstance(rating, str) and rating.strip().lstrip("-").isdigit():
            return int(rating.strip())
        return None
    return _bracket_rating(text)


def _load_bigfive_checkpoint(checkpoint_path, n_questions):
    if checkpoint_path and Path(checkpoint_path).is_file():
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                records = json.load(f)
            if isinstance(records, dict):
                messages = records.get("messages") or []
                responses = records.get("responses") or []
                ratings = []
                for index in range(min(len(messages), len(responses))):
                    if not responses_entry_valid(responses[index]):
                        break
                    try:
                        content = messages[index][-1].get("content")
                    except (IndexError, AttributeError, TypeError):
                        break
                    rating = _parse_rating(content)
                    if not valid_rating(rating):
                        break
                    ratings.append(rating)
                valid_q = len(ratings)
                done = valid_q // n_questions
                rem = valid_q - done * n_questions
                choices = [ratings[i * n_questions:(i + 1) * n_questions] for i in range(done)]
                partial = ratings[done * n_questions:done * n_questions + rem]
                return {
                    "messages": messages[:valid_q],
                    "responses": responses[:valid_q],
                    "choices": choices,
                }, done, partial
        except (OSError, json.JSONDecodeError):
            pass
    return {"messages": [], "responses": [], "choices": []}, 0, []


def run_bigfive(client, n_instances=30, print_except=True, max_retries=3, checkpoint_path=None):
    questions = []
    with open("data/bigfive.tsv", "r") as tsvfile:
        for row in tsvfile:
            row = row.rstrip("\n").split("\t")
            questions.append(row[1])

    n_questions = len(questions)
    records, start, partial = _load_bigfive_checkpoint(checkpoint_path, n_questions)
    with tqdm(total=n_instances, initial=start) as instance_bar:
        for instance_index in range(start, n_instances):

            choices = list(partial)
            partial = []
            for question_index in range(len(choices), n_questions):
                question = questions[question_index]


                prompt = (
                    "The following item was rated on a five point scale where 1=Disagree, "
                    "2=Partially Disagree, 3=Neutral, 4=Partially Agree, 5=Agree. Please select "
                    "how the statement describes you and highlight your answer in [] (such as "
                    "[1], [2], [3], [4], or [5]). \n" + question
                )
                last_error = None
                for attempt in range(max_retries + 1):
                    messages = []
                    responses = []
                    try:
                        update_messages(client, messages, responses, prompt)
                        choice = _parse_rating(messages[-1]["content"])
                        if not valid_rating(choice):
                            raise ValueError(f"Invalid answer: {messages[-1]['content']}")
                        records["messages"].append(messages)
                        records["responses"].append(responses)
                        choices.append(choice)


                        write_checkpoint(records, checkpoint_path)
                        break
                    except Exception as e:
                        if is_non_retryable_provider_error(e):
                            raise
                        last_error = e
                        if print_except:
                            print(f"bigfive instance {instance_index + 1}, question {question_index + 1}, attempt {attempt + 1} failed: {e}")
                else:
                    raise RuntimeError(f"Big Five failed at instance {instance_index + 1}, question {question_index + 1}") from last_error
            records["choices"].append(choices)
            write_checkpoint(records, checkpoint_path)
            instance_bar.update(1)
    return records
