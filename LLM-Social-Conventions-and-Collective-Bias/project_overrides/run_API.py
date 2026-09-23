import json
import re
import requests
from requests.adapters import HTTPAdapter
import threading
import time
import yaml
from datetime import datetime, timezone
from pathlib import Path
from munch import munchify
try:
    from .schemas import valid_choice
except ImportError:
    from project_overrides.schemas import valid_choice

ROOT = Path(__file__).resolve().parents[1]
import os
with open(os.environ.get("SOCIAL_CONFIG", ROOT / "config.yaml"), "r") as f:
    doc = yaml.safe_load(f)
config = munchify(doc)
if "api" not in doc:
    raise ValueError("Missing api configuration in config.yaml.")
api_doc = doc["api"]
api_config = munchify(api_doc)

temperature = config.params.temperature
top_p = getattr(config.params, "top_p", None)


strip_cot_instruction = bool(getattr(config.params, "strip_cot_instruction", False))
if strip_cot_instruction:
    print("已按配置屏蔽原文 CoT 指令（think step by step / examining history）——偏离原文提示词，请在结论中注明。")
_forced_params = {}
request_config = getattr(api_config, "request", munchify({}))
provider_name = api_config.active_provider
provider = api_config.providers[provider_name]
provider_type = provider.type
model_name = provider.model
timeout_seconds = getattr(provider, "timeout_seconds", getattr(request_config, "timeout_seconds", 120))
retry_sleep_seconds = getattr(request_config, "retry_sleep_seconds", 2.5)

rate_limit_sleep_seconds = getattr(request_config, "rate_limit_sleep_seconds", 30)
max_retries = getattr(request_config, "max_retries", 6)
min_request_interval_seconds = float(getattr(request_config, "min_request_interval_seconds", 0) or 0)


max_invalid_responses = int(getattr(request_config, "max_invalid_responses", 10))


max_answer_tokens = int(getattr(request_config, "max_answer_tokens", 15))


max_answer_tokens_cap = int(getattr(request_config, "max_answer_tokens_cap", max(max_answer_tokens, 1024)))
api_key = getattr(provider, "api_key", "")
if api_key in ["", "<YOUR_API_KEY>", "<YOUR_TOKEN_HERE>"]:
    orch_hint = (
        " 请通过 orchestrator 启动实验（会自动注入 runs/.../social_config.yaml），"
        "或使用 run.py --config 指向该文件。"
        if provider_name == "unified"
        else ""
    )
    raise ValueError(
        f"Missing API key for provider '{provider_name}'. "
        f"Set api.providers.{provider_name}.api_key in config.yaml.{orch_hint}"
    )
base_url = getattr(provider, "base_url", "")
if base_url.endswith("/"):
    base_url = base_url[:-1]
log_config = getattr(config, "logging", munchify({}))
log_enabled = getattr(log_config, "enabled", False)


_social_outdir = os.environ.get("SOCIAL_OUTDIR")
if _social_outdir:
    log_dir = Path(_social_outdir)
    if not log_dir.is_absolute():
        log_dir = ROOT / log_dir
else:
    log_dir = ROOT / getattr(log_config, "dir", "logs")
save_prompts = getattr(log_config, "save_prompts", True)
save_raw_responses = getattr(log_config, "save_raw_responses", True)
log_file = log_dir / f"{provider_name}_api_calls.jsonl"
if temperature == 0:
    llm_params = {"do_sample": False,
            "max_new_tokens": 12,
            "return_full_text": False, 
            }
else:
    llm_params = {"do_sample": True,
            "temperature": temperature,
            "top_k": getattr(provider, "top_k", 10),
            "max_new_tokens": 15,
            "return_full_text": False, 
            }  

_LOG_LOCK = threading.Lock()
_BUDGET_LOCK = threading.Lock()
_REQUEST_PACE_LOCK = threading.Lock()
_LAST_REQUEST_TS = 0.0


_effective_budget = max_answer_tokens


def _session_pool_size():
    try:
        mc = int(getattr(provider, "max_concurrency", 0) or 0)
    except Exception:
        mc = 0
    return max(mc, 32)

_pool_size = _session_pool_size()
_SESSION = requests.Session()
_adapter = HTTPAdapter(pool_connections=_pool_size, pool_maxsize=_pool_size, max_retries=0)
_SESSION.mount("http://", _adapter)
_SESSION.mount("https://", _adapter)


def _pace_request():
    global _LAST_REQUEST_TS
    if min_request_interval_seconds <= 0:
        return
    with _REQUEST_PACE_LOCK:
        now = time.time()
        wait = min_request_interval_seconds - (now - _LAST_REQUEST_TS)
        if wait > 0:
            time.sleep(wait)
        _LAST_REQUEST_TS = time.time()


def _write_log(record):
    if not log_enabled:
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    record["timestamp"] = datetime.now(timezone.utc).isoformat()
    record["provider"] = provider_name
    record["model"] = model_name
    line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
    with _LOG_LOCK:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line)

def _headers():
    if provider_type == "anthropic":
        return {
            "x-api-key": api_key,
            "anthropic-version": getattr(provider, "anthropic_version", "2023-06-01"),
            "content-type": "application/json",
        }
    if provider_type == "gemini":
        return {"content-type": "application/json"}
    return {"Authorization": f"Bearer {api_key}", "content-type": "application/json"}

_HEADER_RE = re.compile(
    r"<\|start_header_id\|>\s*(\w+)\s*<\|end_header_id\|>(.*?)(?=<\|eot_id\|>|<\|start_header_id\|>|$)",
    re.DOTALL,
)


_COT_SENTENCES = (
    "Please think step by step before making a decision. ",
    "Remember, examining history explicitly is important. ",
)


def _strip_cot(chat):
    for sentence in _COT_SENTENCES:
        chat = chat.replace(sentence, "")
    return chat


def _split_chat_messages(chat):
    msgs = []
    for role, content in _HEADER_RE.findall(chat):
        content = content.strip()
        if not content:
            continue
        if role not in ("system", "user", "assistant"):
            role = "user"
        msgs.append({"role": role, "content": content})
    if not msgs:
        cleaned = re.sub(r"<\|[^|]*\|>", "", chat).strip()
        msgs = [{"role": "user", "content": cleaned}]
    return msgs


def _payload(chat, max_tokens):

    if provider_type == "huggingface":
        params = llm_params.copy()
        params["max_new_tokens"] = max_tokens
        if top_p is not None:
            params["top_p"] = top_p
        return {"inputs": chat, "parameters": params, "options": {"use_cache": False}}

    messages = _split_chat_messages(chat)
    if provider_type == "anthropic":
        system_text = "\n".join(m["content"] for m in messages if m["role"] == "system")
        chat_messages = [m for m in messages if m["role"] != "system"] or [{"role": "user", "content": ""}]
        payload = {
            "model": model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": chat_messages,
        }
        if system_text:
            payload["system"] = system_text
        if top_p is not None:
            payload["top_p"] = top_p
        return payload
    if provider_type == "gemini":
        system_text = "\n".join(m["content"] for m in messages if m["role"] == "system")
        user_parts = [{"text": m["content"]} for m in messages if m["role"] != "system"] or [{"text": ""}]
        gen_config = {"temperature": temperature, "maxOutputTokens": max_tokens}
        if top_p is not None:
            gen_config["topP"] = top_p
        body = {
            "contents": [{"role": "user", "parts": user_parts}],
            "generationConfig": gen_config,
        }
        if system_text:
            body["systemInstruction"] = {"parts": [{"text": system_text}]}
        return body
    payload = {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if top_p is not None:
        payload["top_p"] = top_p
    payload.update(_forced_params)
    payload.update(_thinking_off_overrides(model_name, base_url))
    return payload


def _thinking_off_overrides(model, burl):
    if os.environ.get("LLM_THINKING", "").lower() == "on":
        return {}
    text = f"{model or ''} {burl or ''}".lower()
    if "127.0.0.1" in text or "localhost" in text:
        return {}
    if "deepseek" in text:
        return {"thinking": {"type": "disabled"}, "enable_thinking": False,
                "chat_template_kwargs": {"thinking": False}}
    if "gemini" in text and ("gemini-3" in text or "2.5" in text):
        return {"extra_body": {"google": {"thinking_config": {"thinking_budget": 0}}}}
    if any(k in text for k in ("kimi", "moonshot", "glm", "zhipu", "bigmodel", "doubao", "volces")):
        return {"thinking": {"type": "disabled"}}
    if any(k in text for k in ("qwen", "dashscope", "aliyuncs", "bailian")):
        return {"enable_thinking": False}
    if any(k in text for k in ("claude", "anthropic", "minimax")):
        return {}
    return {"reasoning_effort": "none"}

def _url():
    if provider_type == "huggingface":
        return f"{base_url}/{model_name}"
    if provider_type == "anthropic":
        return f"{base_url}/messages"
    if provider_type == "gemini":
        return f"{base_url}/models/{model_name}:generateContent?key={api_key}"
    return f"{base_url}/chat/completions"

def _extract_text(response):
    if response is None:
        return None
    if provider_type == "huggingface":
        if isinstance(response, list) and len(response) > 0:
            return response[0].get("generated_text")
        return None
    if provider_type == "anthropic":
        content = response.get("content", [])
        return "".join([block.get("text", "") for block in content])
    if provider_type == "gemini":
        candidates = response.get("candidates", [])
        if len(candidates) == 0:
            return None
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join([part.get("text", "") for part in parts])
    choices = response.get("choices", [])
    if len(choices) == 0:
        return None
    return choices[0].get("message", {}).get("content", "")

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


def _is_rate_limited(response):
    if not isinstance(response, dict):
        return False
    status = response.get("status_code")
    text = str(response.get("error", response))
    return status == 429 or "rate limit" in text.lower() or "Inference Endpoints" in text

def query(chat, max_tokens=15):
    if strip_cot_instruction:
        chat = _strip_cot(chat)
    for _ in range(max_retries):
        try:
            url = _url()
            payload = _payload(chat, max_tokens)
            _pace_request()
            start_time = time.time()
            response = _SESSION.post(url, headers=_headers(), json=payload, timeout=timeout_seconds)
            elapsed_seconds = time.time() - start_time
            try:
                body = response.json()
            except ValueError:
                print('CAUGHT JSON ERROR')
                log_record = {
                    "event": "api_response",
                    "ok": False,
                    "status_code": response.status_code,
                    "elapsed_seconds": elapsed_seconds,
                    "error": "JSON decode error",
                }
                if save_prompts:
                    log_record["prompt"] = chat
                if save_raw_responses:
                    log_record["raw_response"] = response.text
                _write_log(log_record)
                time.sleep(retry_sleep_seconds)
                continue
            log_record = {
                "event": "api_response",
                "ok": response.status_code < 400,
                "status_code": response.status_code,
                "elapsed_seconds": elapsed_seconds,
                "extracted_text": _extract_text(body),
            }
            if save_prompts:
                log_record["prompt"] = chat
            if save_raw_responses:
                log_record["raw_response"] = body
            _write_log(log_record)
            if response.status_code >= 400:
                if isinstance(body, dict):
                    body["status_code"] = response.status_code
                print("AN EXCEPTION: ", body)
                if response.status_code == 400:
                    pc = _parse_param_constraint(str(body))
                    if pc and _forced_params.get(pc[0]) != pc[1]:
                        _forced_params[pc[0]] = pc[1]
                        print(f"模型 {model_name} 不接受 {pc[0]} 的设定，已自动改用该模型唯一允许值 {pc[0]}={pc[1]}（模型强制；本次在该参数上偏离原文，请在结论中注明）。")
                        continue
                if _is_rate_limited(body):
                    print("RATE LIMIT REACHED")
                    time.sleep(rate_limit_sleep_seconds)
                else:
                    time.sleep(retry_sleep_seconds)
                continue
            return body
        except requests.RequestException as exc:
            print('CAUGHT REQUEST ERROR')
            log_record = {
                "event": "api_response",
                "ok": False,
                "error": str(exc),
            }
            if save_prompts:
                log_record["prompt"] = chat
            _write_log(log_record)
            time.sleep(retry_sleep_seconds)
    return None

def _extract_choice(text, options):
    best, best_pos = None, -1
    for opt in options:
        pattern = r"value['\"]?\s*:\s*['\"]?" + re.escape(opt) + r"(?![A-Za-z0-9_])"
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            if m.start() > best_pos:
                best, best_pos = opt, m.start()
    if best is not None:
        return best, True
    best, best_pos = None, -1
    for opt in options:
        pattern = r"(?<![A-Za-z0-9_])" + re.escape(opt) + r"(?![A-Za-z0-9_])"
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            if m.start() > best_pos:
                best, best_pos = opt, m.start()
    return best, False


def _looks_incomplete_value_object(text):
    s = (text or "").strip()
    if not re.search(r"\bvalue['\"]?\s*:", s, flags=re.IGNORECASE):
        return False
    if s.count("{") > s.count("}"):
        return True
    if re.search(r"\breason['\"]?\s*:", s, flags=re.IGNORECASE) and not s.endswith("}"):
        return True
    if s.endswith(("'", '"', ":", ";", ",")):
        return True
    return False

_ENDS_CLEAN_RE = re.compile(r"""[.!?}\]'"`)。！？）」”]\s*$""")


def _looks_cut_midsentence(text):
    s = (text or "").strip()
    return bool(s) and not _ENDS_CLEAN_RE.search(s)


def _completion_tokens(response):
    if not isinstance(response, dict):
        return None
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None
    return usage.get("completion_tokens")


def _looks_truncated(response, budget):
    ctoks = _completion_tokens(response)
    return ctoks is not None and ctoks >= budget


def _raise_budget(budget):
    global _effective_budget
    if budget >= max_answer_tokens_cap:
        return False
    with _BUDGET_LOCK:
        if _effective_budget <= budget:
            _effective_budget = min(budget * 8, max_answer_tokens_cap)
            print(
                f"取答在 {budget} token 预算内未获有效答案，已自动将取答预算抬升至 {_effective_budget}"
                "（兼容先推理后给值 / 思考型模型；本次起在生成长度上偏离原文小预算控制，请在结论中注明）。"
            )
    return True


_INVALID_MSG = (
    "连续 {attempts} 次未能得到可解析的有效答案（空响应或无法解析为 {options} 之一）。"
    "已放弃以避免无限循环；请检查 API/模型配置后重启续跑（已完成进度不会重跑）。"
)


def get_response(chat, options):

    attempts = 0
    while True:
        budget = _effective_budget
        response = query(chat, max_tokens=budget)
        text = _extract_text(response)
        if text is None:
            print('CAUGHT EMPTY RESPONSE')
        else:
            answer, confident = _extract_choice(text, options)
            if answer is not None and valid_choice(answer, options):


                if confident and _looks_truncated(response, budget) and _looks_incomplete_value_object(text):
                    _write_log({
                        "event": "rejected_answer",
                        "reason": "truncated_value_object",
                        "options": options,
                        "extracted_text": text,
                        "parsed_answer": answer,
                    })
                    if _raise_budget(budget):
                        continue


                    attempts += 1
                    if attempts >= max_invalid_responses:
                        raise RuntimeError(_INVALID_MSG.format(attempts=attempts, options=options))
                    continue

                if confident:
                    _write_log({
                        "event": "parsed_answer",
                        "options": options,
                        "extracted_text": text,
                        "parsed_answer": answer,
                    })
                    print(answer)
                    return answer


                if _looks_truncated(response, budget) or _looks_cut_midsentence(text):
                    if _raise_budget(budget):
                        continue
                    _write_log({
                        "event": "rejected_answer",
                        "reason": "tier2_no_decision_at_cap",
                        "options": options,
                        "extracted_text": text,
                        "parsed_answer": answer,
                    })
                    attempts += 1
                    if attempts >= max_invalid_responses:
                        raise RuntimeError(_INVALID_MSG.format(attempts=attempts, options=options))
                    continue
                _write_log({
                    "event": "parsed_answer",
                    "options": options,
                    "extracted_text": text,
                    "parsed_answer": answer,
                })
                print(answer)
                return answer


            if _raise_budget(budget):
                continue
        attempts += 1
        if attempts >= max_invalid_responses:
            raise RuntimeError(_INVALID_MSG.format(attempts=attempts, options=options))

def get_meta_response(chat):

    attempts = 0
    while True:
        budget = _effective_budget
        response = query(chat, max_tokens=budget)
        text = _extract_text(response)
        if text is None:
            print('CAUGHT EMPTY RESPONSE')
        else:
            if 'value' in text:
                response_split = text.split(";")
                response_split = response_split[0].split(": ")
                if len(response_split) >= 2:
                    _write_log({
                        "event": "parsed_meta_answer",
                        "extracted_text": text,
                        "parsed_answer": response_split[1],
                    })
                    print(response_split[1])
                    return response_split[1]

            if _raise_budget(budget):
                continue
        attempts += 1
        if attempts >= max_invalid_responses:
            raise RuntimeError(
                f"连续 {attempts} 次未能得到可解析的 meta 答案。"
                "已放弃以避免无限循环；请检查 API/模型配置后重启续跑（已完成进度不会重跑）。"
            )
