export const KIND_LABELS = {
  openai_chat: "OpenAI 兼容（openai_chat）",
  openai_responses: "OpenAI Responses（openai_responses）",
  anthropic: "Claude 官方（anthropic）",
  gemini: "Google Gemini（gemini）",
  huggingface: "HuggingFace（huggingface）",
  ollama: "Ollama 本地（ollama）",
};

export async function api(path, opts) {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  if (!r.ok) {
    let detail = r.statusText;
    try {
      detail = (await r.json()).detail || detail;
    } catch (e) {

    }
    const err = new Error(detail);
    err.status = r.status;
    throw err;
  }
  return r.status === 204 ? null : r.json();
}
