import json
import math
import os
import re
import signal
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import logging

import openai

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable=None, total=None, disable=False, initial=0):
        return iterable if iterable is not None else range(initial, total or 0)


class TimeoutException(Exception):
    pass


class NonRetryableProviderError(Exception):
    pass


_NON_RETRYABLE_PROVIDER_MARKERS = (
    "用户额度不足",
    "额度不足",
    "余额不足",
    "insufficient_quota",
    "insufficient quota",
    "quota exceeded",
    "billing",
    "credit balance",
)


def is_non_retryable_provider_error(exc):
    if isinstance(exc, NonRetryableProviderError):
        return True
    text = str(exc).lower()
    return any(marker.lower() in text for marker in _NON_RETRYABLE_PROVIDER_MARKERS)


@contextmanager
def time_limit(seconds):


    if threading.current_thread() is not threading.main_thread():
        yield
        return
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    previous = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def strip_inline_comment(value):
    quote = None
    escaped = False
    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char in {"'", '"'}:
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
            continue
        if char == "#" and quote is None:
            return value[:index].rstrip()
    return value.strip()


def parse_scalar(value):
    value = strip_inline_comment(value).strip()
    if value == "":
        return ""
    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        return value[1:-1]
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def simple_yaml_load(path):
    root = {}
    stack = [(-1, root)]
    with open(path, "r") as f:
        for raw_line in f:
            if not raw_line.strip() or raw_line.lstrip().startswith("#"):
                continue
            indent = len(raw_line) - len(raw_line.lstrip(" "))
            text = strip_inline_comment(raw_line.strip())
            if not text:
                continue
            while stack and indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]
            if text.startswith("- "):
                item_text = text[2:].strip()
                item = {}
                parent.append(item)
                if item_text and ":" in item_text:
                    key, value = item_text.split(":", 1)
                    item[key.strip()] = parse_scalar(value)
                stack.append((indent, item))
                continue
            key, value = text.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value == "":
                child = [] if key == "llms" else {}
                parent[key] = child
                stack.append((indent, child))
            else:
                parent[key] = parse_scalar(value)
    return root


def load_yaml(path):
    try:
        import yaml
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except ModuleNotFoundError:
        return simple_yaml_load(path)


def resolve_env(value):
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        return os.environ.get(value[2:-1])
    return value


def load_env_file(path=".env"):
    env_path = Path(path)
    if not env_path.exists():
        return
    with open(env_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            os.environ.setdefault(key, value)


def merge_llm_configs(base_config, local_config):
    merged = dict(base_config or {})
    llms_by_name = {
        llm.get("name"): dict(llm)
        for llm in (merged.get("llms") or [])
        if llm.get("name")
    }
    for llm in (local_config or {}).get("llms", []) or []:
        name = llm.get("name")
        if name:
            llms_by_name[name] = dict(llm)
    merged["llms"] = list(llms_by_name.values())
    return merged


def load_llm_config(path, active_llm):
    config = load_yaml(path)
    local_path = Path(path).with_name(f"{Path(path).stem}.local{Path(path).suffix}")
    if local_path.exists():
        config = merge_llm_configs(config, load_yaml(local_path))
    llms = config.get("llms", [])
    for llm in llms:
        for key, value in list(llm.items()):
            llm[key] = resolve_env(value)
    matched = [llm for llm in llms if llm.get("name") == active_llm]
    if not matched:
        raise ValueError(f"LLM config not found: {active_llm}")
    llm = matched[0]
    if not llm.get("api_key"):
        hint = (
            " 请通过 orchestrator 启动实验，或使用 run.py --config 指向 orchestrator 生成的 run_config.yaml。"
            if active_llm == "unified"
            else ""
        )
        raise ValueError(f"Missing API key for {active_llm}.{hint}")
    return llm


def response_to_dict(response):
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "to_dict_recursive"):
        return response.to_dict_recursive()
    if hasattr(response, "to_dict"):
        return response.to_dict()
    return response


def message_to_dict(message):
    if isinstance(message, dict):
        return dict(message)
    if hasattr(message, "model_dump"):
        return message.model_dump()
    if hasattr(message, "to_dict"):
        return message.to_dict()
    return {"role": getattr(message, "role", "assistant"), "content": getattr(message, "content", "")}


logger = logging.getLogger(__name__)


_PARAM_CONSTRAINT_RE = re.compile(
    r"invalid\s+([A-Za-z_]+)\s*:\s*only\s+([0-9.]+)\s+is\s+allowed", re.IGNORECASE
)


def _parse_param_constraint(text):
    m = _PARAM_CONSTRAINT_RE.search(text or "")
    if not m:
        return None
    name = m.group(1)
    try:
        val = float(m.group(2))
    except ValueError:
        return None
    if name in ("max_tokens", "n", "top_k", "best_of"):
        val = int(val)
    return name, val


_THINKING_OFF_DEEPSEEK = {
    "thinking": {"type": "disabled"},
    "enable_thinking": False,
    "chat_template_kwargs": {"thinking": False},
}
_THINKING_OFF_TYPE = {"thinking": {"type": "disabled"}}
_THINKING_OFF_FLAG = {"enable_thinking": False}
_THINKING_OFF_GEMINI = {"extra_body": {"google": {"thinking_config": {"thinking_budget": 0}}}}


def _thinking_off_overrides(model, base_url):
    if os.environ.get("LLM_THINKING", "").lower() == "on":
        return {}
    text = f"{model or ''} {base_url or ''}".lower()

    if "127.0.0.1" in text or "localhost" in text:
        return {}
    if "deepseek" in text:
        return _THINKING_OFF_DEEPSEEK
    if "gemini" in text and ("gemini-3" in text or "2.5" in text):
        return _THINKING_OFF_GEMINI
    if any(k in text for k in ("kimi", "moonshot", "glm", "zhipu", "bigmodel", "doubao", "volces")):
        return _THINKING_OFF_TYPE
    if any(k in text for k in ("qwen", "dashscope", "aliyuncs", "bailian")):
        return _THINKING_OFF_FLAG
    if any(k in text for k in ("claude", "anthropic", "minimax")):
        return {}
    return {"reasoning_effort": "none"}


class ChatClient:
    def __init__(self, llm_config):
        self.name = llm_config["name"]
        self.model = llm_config["model"]
        self.base_url = llm_config.get("base_url")
        self.api_key = llm_config["api_key"]
        self.timeout = int(llm_config.get("timeout", 60))
        self.temperature = llm_config.get("temperature", 1)
        self.max_tokens = llm_config.get("max_tokens")
        self.n_choices = int(llm_config.get("n_choices", llm_config.get("n", 1)))
        self.top_p = llm_config.get("top_p")
        self.seed = llm_config.get("seed")
        self._forced_params = {}
        self.client = None
        if hasattr(openai, "OpenAI"):
            try:
                self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
            except TypeError:
                self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            openai.api_key = self.api_key
            openai.api_base = self.base_url

    def complete(self, messages, **kwargs):
        request_kwargs = {"temperature": self.temperature}
        if self.max_tokens not in (None, ""):
            request_kwargs["max_tokens"] = int(self.max_tokens)
        if self.n_choices != 1:
            request_kwargs["n"] = self.n_choices
        if self.top_p not in (None, ""):
            request_kwargs["top_p"] = float(self.top_p)
        if self.seed not in (None, ""):
            request_kwargs["seed"] = int(self.seed)
        request_kwargs.update(kwargs)
        request_kwargs.update(self._forced_params)
        thinking_off = _thinking_off_overrides(self.model, self.base_url)
        if thinking_off:
            extra_body = dict(request_kwargs.get("extra_body") or {})
            extra_body.update(thinking_off)
            request_kwargs["extra_body"] = extra_body


        applied = set()
        for _ in range(6):
            try:
                return response_to_dict(self._raw_create(messages, request_kwargs))
            except Exception as exc:
                if is_non_retryable_provider_error(exc):
                    raise NonRetryableProviderError(str(exc)) from exc
                pc = _parse_param_constraint(str(exc))
                if not pc or pc[0] in applied:
                    raise
                applied.add(pc[0])
                request_kwargs[pc[0]] = pc[1]
                if self._forced_params.get(pc[0]) != pc[1]:
                    self._forced_params[pc[0]] = pc[1]
                    logger.warning(
                        "模型 %s 不接受 %s 的设定，已自动改用该模型唯一允许值 %s=%s"
                        "（模型强制；本次在该参数上偏离原文，请在结论中注明）。",
                        self.model, pc[0], pc[0], pc[1],
                    )
        return response_to_dict(self._raw_create(messages, request_kwargs))

    def _raw_create(self, messages, request_kwargs):
        with time_limit(self.timeout):
            if self.client is not None:
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **request_kwargs,
                )
            return openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                **request_kwargs,
            )


def assistant_message(response):
    message = response["choices"][0].get("message")
    if message is not None:
        return message_to_dict(message)
    return {"role": "assistant", "content": response["choices"][0].get("text", "")}


def update_messages(client, messages, responses, prompt, **kwargs):
    messages.append({"role": "user", "content": prompt})
    response = client.complete(messages, **kwargs)
    responses.append(response)
    messages.append(assistant_message(response))


def run_one_session(client, prompts, n_instances=30, orders=None, print_except=True, system_message="You are a helpful assistant.", max_retries=3, checkpoint_path=None, choice_extractor=None, progress=None):


    fields = ["messages", "responses"] + (["choices"] if choice_extractor is not None else [])
    records, start = load_or_init_records(checkpoint_path, fields)
    if orders is None:
        orders = [list(range(len(prompts)))] * n_instances
    prog_offset, prog_total = progress if progress else (0, n_instances)
    with tqdm(total=prog_total, initial=prog_offset + start) as pbar:
        for i in range(start, n_instances):
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    messages = [{"role": "system", "content": system_message}] if system_message else []
                    responses = []
                    for prompt_id in orders[i]:
                        update_messages(client, messages, responses, prompts[prompt_id])
                    choice = None
                    if choice_extractor is not None:
                        choice = choice_extractor(messages)
                        if not choice_entry_valid(choice):
                            raise ValueError(f"Invalid/unparseable answer: {messages[-1]['content']!r}")
                    records["messages"].append(messages)
                    records["responses"].append(responses)
                    if choice_extractor is not None:
                        records["choices"].append(choice)
                    write_checkpoint(records, checkpoint_path)
                    pbar.update(1)
                    break
                except Exception as e:
                    if is_non_retryable_provider_error(e):
                        raise
                    last_error = e
                    if print_except:
                        print(f"instance {i + 1}, attempt {attempt + 1} failed: {e}")
            else:
                raise RuntimeError(f"Failed instance {i + 1} after {max_retries + 1} attempts") from last_error
    return records


def clean_bracket_token(token):
    s = str(token).strip()
    s = s.strip("[]")
    s = s.replace("\\", "")
    s = s.strip("*_`~\"'“”‘’ \t\n")
    return s.strip()


def extract_brackets(text, brackets="[]"):
    pattern = re.escape(brackets[0]) + r"(.*?)" + re.escape(brackets[1])
    return [clean_bracket_token(m) for m in re.findall(pattern, text)]


def extract_bracket_spans(text, brackets="[]"):
    pattern = re.escape(brackets[0]) + r"(.*?)" + re.escape(brackets[1])
    return [(m.start(), clean_bracket_token(m.group(1))) for m in re.finditer(pattern, text)]


def extract_amount(message, prefix="", value_type=float, brackets="[]"):
    matches = extract_brackets(message, brackets=brackets)
    matches = [s.replace(" ", "") for s in matches]
    matches = [s[len(prefix):] if s.startswith(prefix) else s for s in matches]
    if not matches:
        return None
    if any(match != matches[0] for match in matches):


        values = []
        for match in matches:
            try:
                values.append(value_type(match))
            except Exception:
                continue
        if len(set(values)) == 1:
            return values[0]
        return None
    try:
        return value_type(matches[0])
    except Exception:
        return None


_CARD_DECISION_CUE_RE = re.compile(
    r"(i(?:['’]ll| will| would| shall| am going to)?\s+(?:play|choose|pick|select|go\s+with"
    r"|going\s+with|respond\s+with)\b"
    r"|my\s+(?:final\s+|first(?:[- ]round)?\s+|next\s+)?(?:choice|move|decision|answer|play)\b"
    r"|final\s+(?:answer|decision|choice)\b)",
    re.IGNORECASE,
)


_CARD_PROSE_FIRST_PERSON_RE = re.compile(
    r"\bI(?:['’]ll| will| would| shall| should| must|['’]m going to| am going to)?\s+"
    r"(?:(?:play|choose|pick|select|go\s+with|respond\s+with)\b\W{0,12})?\[?\**(push|pull)\b",
    re.IGNORECASE,
)


_CARD_PROSE_DECISION_RE = re.compile(
    r"(?:play|choose|pick|select|go(?:ing)?\s+with|respond\s+with|decision\s+is|choice\s+is"
    r"|answer\s+is|answer:)\b\W{0,16}(push|pull)\b",
    re.IGNORECASE,
)

_CARD_IF_GUARD_RE = re.compile(r"\b(if|when|whether|suppose|unless)\W{0,12}$", re.IGNORECASE)


def extract_card(message):
    text = message or ""
    pattern = re.compile(r"\[(.*?)\]")
    cards = []
    for m in pattern.finditer(text):
        tok = clean_bracket_token(m.group(1))
        if tok.strip().lower() in ("push", "pull"):
            cards.append((m.start(), m.end(), tok.lower().capitalize()))
    if cards:
        distinct = {card for _, _, card in cards}
        if len(distinct) == 1:
            return cards[0][2]


        standalone = []
        for line in text.splitlines():
            s = line.strip().strip("*_`~>#").strip()
            if re.fullmatch(r"\[+[^\[\]]*\]+", s):
                tok = clean_bracket_token(s)
                if tok.strip().lower() in ("push", "pull"):
                    standalone.append(tok.lower().capitalize())
        if len(set(standalone)) == 1:
            return standalone[0]

        last_start, last_end, last_card = cards[-1]
        if len(text[last_end:].strip()) <= 30:
            return last_card
        pool = [(pos, card) for pos, _, card in cards
                if _CARD_DECISION_CUE_RE.search(text[max(0, pos - 100):pos])]
        for m in _CARD_PROSE_FIRST_PERSON_RE.finditer(text):
            if not _CARD_IF_GUARD_RE.search(text[max(0, m.start() - 20):m.start()]):
                pool.append((m.start(), m.group(1).lower().capitalize()))
        if pool:
            return max(pool)[1]
        return last_card
    prose = list(_CARD_PROSE_DECISION_RE.finditer(text))
    if prose:
        return prose[-1].group(1).lower().capitalize()
    words = list(re.finditer(r"\b(Push|Pull)\b", text, flags=re.IGNORECASE))
    if not words:
        return None
    return words[-1].group(1).lower().capitalize()


def add_metadata(records, run_config, llm_config, experiment_name):
    records["metadata"] = {
        "experiment": experiment_name,
        "llm_name": llm_config["name"],
        "model": llm_config["model"],
        "base_url": llm_config.get("base_url"),
        "timestamp": datetime.now().isoformat(),
        "sampling": {
            "temperature": llm_config.get("temperature", 1),
            "top_p": llm_config.get("top_p"),
            "max_tokens": llm_config.get("max_tokens"),
            "seed": llm_config.get("seed"),
            "n_choices": llm_config.get("n_choices", llm_config.get("n", 1)),
        },
        "run_config": run_config,
    }
    return records


def response_text(response):
    if not isinstance(response, dict):
        return ""
    if isinstance(response.get("message"), str):
        return response.get("message", "").strip()
    if isinstance(response.get("content"), str):
        return response.get("content", "").strip()
    choices = response.get("choices") or []
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0] or {}
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if isinstance(message, dict):
        return str(message.get("content") or "").strip()
    return str(first.get("text") or "").strip()


def response_is_successful(response):
    return bool(response_text(response))


def responses_entry_valid(entry):
    if isinstance(entry, dict):
        return response_is_successful(entry)
    if isinstance(entry, list):
        return bool(entry) and all(responses_entry_valid(item) for item in entry)
    return False


def choice_entry_valid(entry):
    if entry is None:
        return False
    if isinstance(entry, bool):
        return True
    if isinstance(entry, (int, float)):
        return math.isfinite(entry)
    if isinstance(entry, str):
        return bool(entry.strip())
    if isinstance(entry, list):
        return bool(entry) and all(choice_entry_valid(item) for item in entry)
    if isinstance(entry, dict):
        return bool(entry) and all(choice_entry_valid(item) for item in entry.values())
    return True


def record_is_successful(records):
    if isinstance(records, list) and len(records) == 2:
        records = {"records": records[0], "choices": records[1]}
    if not isinstance(records, dict):
        return False
    nested_records = records.get("records")
    if isinstance(nested_records, dict):
        if not nested_records:
            return False
        choices_all = records.get("choices")
        if not isinstance(choices_all, dict):
            return False
        for key, inner in nested_records.items():
            if not record_is_successful(inner):
                return False
            if isinstance(choices_all, dict):
                if key not in choices_all or not choice_entry_valid(choices_all[key]):
                    return False
        return True
    if "responses" not in records:
        return False
    if not responses_entry_valid(records.get("responses")):
        return False
    if "choices" in records and not choice_entry_valid(records.get("choices")):
        return False
    return True


def _instance_is_successful(records, index, fields):
    for field in fields:
        value = records[field][index]
        if field == "responses" and not responses_entry_valid(value):
            return False
        if field == "choices" and not choice_entry_valid(value):
            return False
        if field == "messages" and not isinstance(value, list):
            return False
    return True


def compact_record_lists(records, fields):
    if not fields:
        return records, 0
    n = min(len(records[k]) for k in fields)
    keep = [i for i in range(n) if _instance_is_successful(records, i, fields)]
    for field in fields:
        kept = [records[field][i] for i in keep]
        records[field][:] = kept
    return records, len(keep)


def save_records(records, results_dir, game_name, output_path=None):
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    if output_path is None:
        timestamp = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        output_path = Path(results_dir) / f"{game_name}_{timestamp}.json"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    tmp_path.replace(output_path)
    print(f"saved: {output_path}")
    return output_path


def load_or_init_records(checkpoint_path, fields):
    fields = list(fields)
    if checkpoint_path and Path(checkpoint_path).is_file():
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                records = json.load(f)
            if isinstance(records, dict) and all(isinstance(records.get(k), list) for k in fields):
                records, start = compact_record_lists(records, fields)
                return records, start
        except (OSError, json.JSONDecodeError):
            pass
    return {k: [] for k in fields}, 0


def write_checkpoint(records, checkpoint_path):
    if not checkpoint_path:
        return
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
    tmp.replace(path)


def discard_checkpoint(checkpoint_path):
    if checkpoint_path:
        Path(checkpoint_path).unlink(missing_ok=True)


def occupation_checkpoint_path(checkpoint_path, occupation):
    if not checkpoint_path:
        return None
    p = Path(checkpoint_path)
    slug = re.sub(r"[^0-9A-Za-z._-]+", "-", str(occupation)).strip("-") or "occ"
    return str(p.with_name(f"{p.stem}.{slug}{p.suffix}"))


def load_or_init_nested_records(checkpoint_path):
    if checkpoint_path and Path(checkpoint_path).is_file():
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            records_all = data.get("records") or {}
            choices_all = data.get("choices") or {}
            if isinstance(records_all, dict) and isinstance(choices_all, dict):
                clean_records = {}
                clean_choices = {}
                for occupation, records in records_all.items():
                    if not record_is_successful(records):
                        continue
                    choices = choices_all.get(occupation, records.get("choices") if isinstance(records, dict) else None)
                    if not choice_entry_valid(choices):
                        continue
                    clean_records[occupation] = records
                    clean_choices[occupation] = choices
                return clean_records, clean_choices, set(clean_records.keys())
        except (OSError, json.JSONDecodeError):
            pass
    return {}, {}, set()
