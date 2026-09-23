import argparse
import concurrent.futures
import datetime as dt
import json
import logging
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SHARED_DIR = os.path.join(PROJECT_ROOT, "scripts", "shared")
for path in (PROJECT_ROOT, SHARED_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import llm_prompts as oip


PROVIDERS = {
    "openai",
    "openai-compatible",
    "anthropic",
    "gemini",
    "huggingface",
    "ollama",
}

EXPERIMENT_PRESETS = {
    "full": "gpt-4",
}

DEFAULT_ARGS = {
    "name": None,
    "provider": None,
    "model": None,
    "output": "modern_llm_generations.jsonl",
    "base_url": None,
    "api_key": None,
    "experiment_preset": None,
    "reference_config": "gpt-4",
    "identity_axes": None,
    "tasks": None,
    "allow_unconfigured_tasks": False,
    "identity_indices": None,
    "identity_values": None,
    "name_identities": False,
    "max_identities": None,
    "samples": 100,
    "samples_other": 33,
    "system_prompt_id": 2,
    "temperature": 1.0,
    "max_tokens": 512,
    "top_p": None,
    "timeout": 120,
    "sleep": 0.0,
    "retries": 3,
    "retry_backoff": 2.0,
    "concurrency": 4,
    "paper_length_instruction": True,
    "resume": True,
    "dry_run": False,
    "limit_cases": None,
}

logger = logging.getLogger(__name__)


_PARAM_CONSTRAINT_RE = re.compile(
    r"invalid\s+([A-Za-z_]+)\s*:\s*only\s+([0-9.]+)\s+is\s+allowed", re.IGNORECASE
)


def _parse_param_constraint(text: str):
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


class ProviderError(RuntimeError):
    pass


class Provider:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.api_key = resolve_api_key(args.api_key)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class OpenAIChatProvider(Provider):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self._forced_params = {}
        try:
            from openai import OpenAI
        except ImportError:
            self.client = None
        else:
            kwargs = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if args.base_url:
                kwargs["base_url"] = args.base_url
            self.client = OpenAI(**kwargs)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        messages = build_messages(system_prompt, user_prompt)
        if self.client is None:
            if not self.api_key:
                raise ProviderError(
                    "OpenAI-compatible API requires api_key. "
                    "请通过 orchestrator 启动，或使用 run.py --config 指向 runs/.../identity_config.yaml。"
                )
            base_url = (self.args.base_url or "https://api.openai.com/v1").rstrip("/")
            payload = {
                "model": self.args.model,
                "messages": messages,
                "temperature": self.args.temperature,
                "max_tokens": self.args.max_tokens,
            }
            if self.args.top_p is not None:
                payload["top_p"] = self.args.top_p
            data = post_json(
                f"{base_url}/chat/completions",
                payload,
                {"Authorization": f"Bearer {self.api_key}"},
                self.args.timeout,
            )
            choices = data.get("choices", []) if isinstance(data, dict) else []
            if choices:
                message = choices[0].get("message", {})
                return message.get("content") or choices[0].get("text", "") or ""
            raise ProviderError(f"Unexpected OpenAI-compatible response: {data}")
        create_kwargs = {
            "model": self.args.model,
            "messages": messages,
            "temperature": self.args.temperature,
            "max_tokens": self.args.max_tokens,
        }
        if self.args.top_p is not None:
            create_kwargs["top_p"] = self.args.top_p
        create_kwargs.update(self._forced_params)

        applied = set()
        for _ in range(6):
            try:
                response = self.client.chat.completions.create(**create_kwargs)
                return response.choices[0].message.content or ""
            except Exception as exc:
                pc = _parse_param_constraint(str(exc))
                if not pc or pc[0] in applied:
                    raise
                applied.add(pc[0])
                create_kwargs[pc[0]] = pc[1]
                if self._forced_params.get(pc[0]) != pc[1]:
                    self._forced_params[pc[0]] = pc[1]
                    logger.warning(
                        "模型 %s 不接受 %s 的设定，已自动改用该模型唯一允许值 %s=%s"
                        "（模型强制；本次在该参数上偏离原文，请在结论中注明）。",
                        self.args.model, pc[0], pc[0], pc[1],
                    )
        response = self.client.chat.completions.create(**create_kwargs)
        return response.choices[0].message.content or ""


class AnthropicProvider(Provider):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        try:
            import anthropic
        except ImportError as exc:
            raise ProviderError("Install the Anthropic SDK with: pip install anthropic") from exc
        kwargs = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        self.client = anthropic.Anthropic(**kwargs)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        kwargs = {
            "model": self.args.model,
            "max_tokens": self.args.max_tokens,
            "temperature": self.args.temperature,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if self.args.top_p is not None:
            kwargs["top_p"] = self.args.top_p
        if system_prompt:
            kwargs["system"] = system_prompt
        response = self.client.messages.create(**kwargs)
        parts = []
        for item in response.content:
            text = getattr(item, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts)


class GeminiProvider(Provider):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ProviderError("Install the Gemini SDK with: pip install google-generativeai") from exc
        if not self.api_key:
            raise ProviderError("Gemini requires api_key in config.yaml")
        genai.configure(api_key=self.api_key)
        self.genai = genai

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        generation_config = {
            "temperature": self.args.temperature,
            "max_output_tokens": self.args.max_tokens,
        }
        if self.args.top_p is not None:
            generation_config["top_p"] = self.args.top_p
        kwargs = {"model_name": self.args.model}
        if system_prompt:
            kwargs["system_instruction"] = system_prompt
        model = self.genai.GenerativeModel(**kwargs)
        response = model.generate_content(user_prompt, generation_config=generation_config)
        text = getattr(response, "text", None)
        if text is not None:
            return text
        candidates = getattr(response, "candidates", None) or []
        parts = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", []) or []:
                value = getattr(part, "text", None)
                if value:
                    parts.append(value)
        return "\n".join(parts)


class HuggingFaceProvider(Provider):
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if not self.api_key:
            raise ProviderError("Hugging Face Inference API requires api_key in config.yaml")
        api_url = self.args.base_url or f"https://api-inference.huggingface.co/models/{self.args.model}"
        prompt = format_plain_prompt(system_prompt, user_prompt)
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": self.args.temperature,
                "max_new_tokens": self.args.max_tokens,
                "return_full_text": False,
                **({"top_p": self.args.top_p} if self.args.top_p is not None else {}),
            },
        }
        data = post_json(api_url, payload, {"Authorization": f"Bearer {self.api_key}"}, self.args.timeout)
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict) and "generated_text" in first:
                return first["generated_text"]
        if isinstance(data, dict):
            if "generated_text" in data:
                return data["generated_text"]
            if "error" in data:
                raise ProviderError(str(data["error"]))
        raise ProviderError(f"Unexpected Hugging Face response: {data}")


class OllamaProvider(Provider):
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        api_url = self.args.base_url or "http://localhost:11434/api/chat"
        payload = {
            "model": self.args.model,
            "messages": build_messages(system_prompt, user_prompt),
            "stream": False,
            "options": {
                "temperature": self.args.temperature,
                "num_predict": self.args.max_tokens,
                **({"top_p": self.args.top_p} if self.args.top_p is not None else {}),
            },
        }
        data = post_json(api_url, payload, {}, self.args.timeout)
        if isinstance(data, dict) and "message" in data:
            return data["message"].get("content", "")
        raise ProviderError(f"Unexpected Ollama response: {data}")


def resolve_api_key(api_key: Optional[str]) -> Optional[str]:
    return api_key or None


def build_provider(args: argparse.Namespace) -> Provider:
    if args.provider in {"openai", "openai-compatible"}:
        return OpenAIChatProvider(args)
    if args.provider == "anthropic":
        return AnthropicProvider(args)
    if args.provider == "gemini":
        return GeminiProvider(args)
    if args.provider == "huggingface":
        return HuggingFaceProvider(args)
    if args.provider == "ollama":
        return OllamaProvider(args)
    raise ProviderError(f"Unsupported provider: {args.provider}")


def post_json(url: str, payload: Dict, headers: Dict[str, str], timeout: int):
    body = json.dumps(payload).encode("utf-8")
    merged_headers = {"Content-Type": "application/json", **headers}
    request = urllib.request.Request(url, data=body, headers=merged_headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ProviderError(f"HTTP {exc.code}: {detail}") from exc


def build_messages(system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def format_plain_prompt(system_prompt: str, user_prompt: str) -> str:
    if system_prompt:
        return f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
    return f"User: {user_prompt}\n\nAssistant:"


def parse_int_list(value: Optional[str]) -> Optional[List[int]]:
    if value is None or value == "":
        return None
    if isinstance(value, (list, tuple)):
        return [int(part) for part in value]
    normalized = str(value).replace("，", ",").replace(";", ",").replace("\n", ",")
    return [int(part.strip()) for part in normalized.split(",") if part.strip()]


def parse_str_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None or value == "":
        return None
    if isinstance(value, (list, tuple)):
        return [str(part).strip() for part in value if str(part).strip()]
    text = str(value)
    delimiter = "\n" if "\n" in text else (";" if ";" in text else ",")
    return [part.strip() for part in text.replace("，", ",").split(delimiter) if part.strip()]


def select_axes(args: argparse.Namespace) -> List[str]:
    reference_config = resolve_reference_config(args)
    if args.identity_axes:
        axes = args.identity_axes
    else:
        axes = list(oip.configs[reference_config].keys())
    invalid = [axis for axis in axes if axis not in oip.identities]
    if invalid:
        raise ValueError(f"Unknown identity axes: {invalid}")
    return axes


def select_tasks(axis: str, args: argparse.Namespace) -> List[str]:
    reference_config = resolve_reference_config(args)
    if args.tasks:
        tasks = args.tasks
    else:
        tasks = oip.configs[reference_config].get(axis, [])
    invalid = [task for task in tasks if task not in oip.user_prompts]
    if invalid:
        raise ValueError(f"Unknown task prompt keys: {invalid}")
    if args.allow_unconfigured_tasks:
        return list(tasks)
    configured = set(oip.configs[reference_config].get(axis, []))
    return [task for task in tasks if task in configured]


def resolve_reference_config(args: argparse.Namespace) -> str:
    return EXPERIMENT_PRESETS.get(args.experiment_preset, args.reference_config)


def name_identity_list() -> List[str]:
    flat: List[str] = []
    for idx in range(len(oip.identities["intersection"])):
        flat.extend(oip.names.get(f"intersection-{idx}", []))
    return flat


def select_identities(axis: str, args: argparse.Namespace) -> List[Tuple[int, str]]:
    if getattr(args, "name_identities", False):
        if axis != "intersection":
            raise ValueError("name_identities=True 仅支持 intersection 轴（原文 Fig.3 只在该轴做姓名实验）")
        identities = name_identity_list()
    else:
        identities = list(oip.identities[axis])
    indices = parse_int_list(args.identity_indices)
    values = parse_str_list(args.identity_values)
    selected = [(index, identity) for index, identity in enumerate(identities)]
    if indices is not None:
        invalid = [i for i in indices if i < 0 or i >= len(identities)]
        if invalid:
            raise ValueError(
                f"identity_indices 超出 '{axis}' 轴身份范围（有效 0..{len(identities) - 1}）：{invalid}"
            )
        selected = [(index, identities[index]) for index in indices]
    if values is not None:
        value_set = set(values)
        selected = [(index, identity) for index, identity in selected if identity in value_set]
    if args.max_identities is not None:
        selected = selected[: args.max_identities]
    return selected


def sample_count(axis: str, identity: str, args: argparse.Namespace) -> int:
    if axis == "other" and args.samples_other is not None:
        return args.samples_other
    return args.samples


def render_prompts(axis: str, identity: str, task: str, args: argparse.Namespace) -> Tuple[str, str]:
    user_prompt = oip.user_prompts[task].replace("[identity]", identity)
    if identity == "generic":
        system_prompt = ""
    elif getattr(args, "name_identities", False):

        system_prompt = oip.system_prompts[6].replace("[name]", identity)
    else:
        prompt_id = 7 if axis == "other" else args.system_prompt_id
        system_prompt = oip.system_prompts[prompt_id].replace("[identity]", identity)
    if args.paper_length_instruction:
        user_prompt = add_paper_length_instruction(task, user_prompt)
    return system_prompt, user_prompt


def add_paper_length_instruction(task: str, user_prompt: str) -> str:
    if task.startswith("R2b-race"):
        instruction = "Please answer each with 2-3 sentences."
    elif task.startswith("R2b-gender"):
        instruction = "Please answer the question."
    else:
        instruction = "Please provide a relevant answer to the question with one paragraph of 4-5 sentences."
    return f"{user_prompt}\n\n{instruction}"


def iter_cases(args: argparse.Namespace) -> Iterable[Dict]:
    axes = select_axes(args)
    reference_config = resolve_reference_config(args)
    for axis in axes:
        tasks = select_tasks(axis, args)
        identities = select_identities(axis, args)
        for task in tasks:
            for identity_index, identity in identities:
                system_prompt, user_prompt = render_prompts(axis, identity, task, args)
                n_samples = sample_count(axis, identity, args)
                for sample_index in range(n_samples):
                    yield {
                        "provider": args.provider,
                        "run_name": args.name,
                        "model": args.model,
                        "reference_config": reference_config,
                        "task_key": task,
                        "identity_axis": axis,
                        "identity_index": identity_index,
                        "identity": identity,
                        "system_prompt_id": None if identity == "generic" else (
                            6 if getattr(args, "name_identities", False)
                            else (7 if axis == "other" else args.system_prompt_id)),
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "temperature": args.temperature,
                        "max_tokens": args.max_tokens,
                        "sample_index": sample_index,
                    }


def load_completed(output_path: str, args: argparse.Namespace) -> set:
    completed = set()
    if not args.resume or not os.path.exists(output_path):
        return completed
    with open(output_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not str(item.get("response") or "").strip():
                continue
            key = completion_key(item)
            completed.add(key)
    return completed


def compact_output(output_path: str, args: argparse.Namespace) -> None:
    if not args.resume or not os.path.exists(output_path):
        return
    by_key: Dict[Tuple, Dict] = {}
    with open(output_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not str(item.get("response") or "").strip():
                continue
            by_key[completion_key(item)] = item
    tmp = output_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        for item in by_key.values():
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    os.replace(tmp, output_path)


def completion_key(item: Dict) -> Tuple:
    return (
        item.get("provider"),
        item.get("model"),
        item.get("task_key"),
        item.get("identity_axis"),
        item.get("identity_index"),
        item.get("sample_index"),
        item.get("temperature"),
    )


def call_with_retries(provider: Provider, system_prompt: str, user_prompt: str, args: argparse.Namespace) -> str:
    last_error = None
    for attempt in range(args.retries + 1):
        try:
            return provider.generate(system_prompt, user_prompt)
        except Exception as exc:
            last_error = exc
            if attempt >= args.retries:
                break
            wait = args.retry_backoff * (2 ** attempt)
            print(f"Request failed: {exc}. Retrying in {wait:.1f}s", file=sys.stderr)
            time.sleep(wait)
    raise ProviderError(str(last_error))


def write_jsonl(path: str, item: Dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def parse_scalar(value: str):
    value = value.strip()
    if value in {"", "null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_simple_yaml(path: str) -> Dict:
    data = {}
    current_key = None
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            stripped = line.strip()
            if stripped.startswith("- "):
                if current_key is None:
                    raise ProviderError(f"YAML list item without a key at {path}:{line_number}")
                data.setdefault(current_key, []).append(parse_scalar(stripped[2:]))
                continue
            if ":" not in stripped:
                raise ProviderError(f"Unsupported YAML line at {path}:{line_number}: {raw_line.rstrip()}")
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value:
                data[key] = parse_scalar(value)
                current_key = None
            else:
                data[key] = []
                current_key = key
    return data


def load_config(path: Optional[str]) -> Dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        try:
            import yaml
        except ImportError as exc:
            data = load_simple_yaml(path)
        else:
            data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ProviderError(f"YAML config must be a mapping/object: {path}")
    unknown = sorted(set(data) - set(DEFAULT_ARGS))
    if unknown:
        raise ProviderError(f"Unknown config keys in {path}: {unknown}")
    return data


def config_defaults(config: Dict) -> Dict:
    defaults = dict(DEFAULT_ARGS)
    for key, value in config.items():
        if value is not None:
            defaults[key] = value
    return defaults


def add_argument(parser: argparse.ArgumentParser, defaults: Dict, name: str, **kwargs) -> None:
    kwargs.setdefault("default", defaults[name])
    parser.add_argument("--" + name.replace("_", "-"), **kwargs)


def build_parser(defaults: Optional[Dict] = None) -> argparse.ArgumentParser:
    defaults = config_defaults(defaults or {})
    parser = argparse.ArgumentParser(description="Generate modern LLM responses for the identity-prompting paper prompts.")
    parser.add_argument("--config", default=None)
    add_argument(parser, defaults, "name")
    add_argument(parser, defaults, "provider", choices=sorted(PROVIDERS))
    add_argument(parser, defaults, "model")
    add_argument(parser, defaults, "output")
    add_argument(parser, defaults, "base_url")
    add_argument(parser, defaults, "api_key")
    add_argument(parser, defaults, "experiment_preset", choices=sorted(EXPERIMENT_PRESETS.keys()))
    add_argument(parser, defaults, "reference_config", choices=sorted(oip.configs.keys()))
    add_argument(parser, defaults, "identity_axes", nargs="+", choices=sorted(oip.identities.keys()))
    add_argument(parser, defaults, "tasks", nargs="+")
    add_argument(parser, defaults, "allow_unconfigured_tasks", action="store_true")
    add_argument(parser, defaults, "identity_indices")
    add_argument(parser, defaults, "identity_values")
    add_argument(parser, defaults, "name_identities", action="store_true")
    add_argument(parser, defaults, "max_identities", type=int)
    add_argument(parser, defaults, "samples", type=int)
    add_argument(parser, defaults, "samples_other", type=int)
    add_argument(parser, defaults, "system_prompt_id", type=int, choices=sorted(oip.system_prompts.keys()))
    add_argument(parser, defaults, "temperature", type=float)
    add_argument(parser, defaults, "max_tokens", type=int)
    add_argument(parser, defaults, "top_p", type=float)
    add_argument(parser, defaults, "timeout", type=int)
    add_argument(parser, defaults, "sleep", type=float)
    add_argument(parser, defaults, "retries", type=int)
    add_argument(parser, defaults, "retry_backoff", type=float)
    add_argument(parser, defaults, "concurrency", type=int)
    add_argument(parser, defaults, "paper_length_instruction", action="store_true")
    add_argument(parser, defaults, "resume", action="store_true")
    add_argument(parser, defaults, "dry_run", action="store_true")
    add_argument(parser, defaults, "limit_cases", type=int)
    return parser


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None)
    config_args, remaining = config_parser.parse_known_args()
    config = load_config(config_args.config)
    args = build_parser(config).parse_args(remaining)
    args.config = config_args.config
    if not args.provider:
        raise ProviderError("Missing required setting: provider. Set it in YAML or pass --provider.")
    if not args.model:
        raise ProviderError("Missing required setting: model. Set it in YAML or pass --model.")
    return args


def main() -> None:
    args = parse_args()
    cases = list(iter_cases(args))
    if args.limit_cases is not None:
        cases = cases[: args.limit_cases]
    print(f"Prepared {len(cases)} generation cases")
    if args.dry_run:
        for item in cases[:10]:
            preview = {key: item[key] for key in ["provider", "model", "task_key", "identity_axis", "identity", "sample_index"]}
            print(json.dumps(preview, ensure_ascii=False))
        return
    provider = build_provider(args)
    compact_output(args.output, args)
    completed = load_completed(args.output, args)
    total = len(cases)
    pending = [item for item in cases if completion_key(item) not in completed]
    skipped = total - len(pending)
    written = 0
    failed = 0
    done = skipped
    concurrency = max(1, int(getattr(args, "concurrency", 1) or 1))

    def _process(item: Dict) -> Tuple[Dict, bool]:
        try:
            item["response"] = call_with_retries(provider, item["system_prompt"], item["user_prompt"], args)
            item["error"] = ""
            ok = True
        except Exception as exc:
            item["response"] = ""
            item["error"] = f"{type(exc).__name__}: {exc}"
            ok = False
        item["created_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        if args.sleep > 0:
            time.sleep(args.sleep)
        return item, ok

    def _commit(item: Dict, ok: bool) -> None:

        nonlocal written, failed, done
        write_jsonl(args.output, item)
        done += 1
        written += 1
        if not ok:
            failed += 1
            print(f"  -> 样本失败，已记录待下次重试：{item['error']}", file=sys.stderr)
        print(f"[{done}/{total}] {item['model']} {item['task_key']} {item['identity_axis']}:{item['identity_index']} sample={item['sample_index']}")

    if concurrency <= 1:
        for item in pending:
            _commit(*_process(item))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            for fut in concurrent.futures.as_completed([pool.submit(_process, item) for item in pending]):
                _commit(*fut.result())
    print(f"Done. written={written} (failed={failed}), skipped={skipped}, output={args.output}, concurrency={concurrency}")


if __name__ == "__main__":
    main()
