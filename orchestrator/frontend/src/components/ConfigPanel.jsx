import { memo } from "react";
import { KIND_LABELS } from "../api.js";

const CONFIG_META_KEYS = new Set([
  "api_key_configured",
  "effective_upstream_concurrency",
  "model_catalog",
]);

export function splitConfigPayload(payload) {
  if (!payload) return { config: null, meta: {} };
  const config = {};
  const meta = {};
  for (const [k, v] of Object.entries(payload)) {
    if (CONFIG_META_KEYS.has(k)) meta[k] = v;
    else config[k] = v;
  }
  return { config, meta };
}

function ConfigPanel({ config, setConfig, configMeta, onSave }) {
  if (!config) return <div className="muted">加载配置中…</div>;
  const llm = config.llm || {};
  const proxy = config.proxy || {};
  const catalog = configMeta?.model_catalog || [];
  const apiKeyConfigured = Boolean(configMeta?.api_key_configured);
  const pythonPath = config.python_executable || "";

  const setLLM = (k, v) => setConfig({ ...config, llm: { ...llm, [k]: v } });
  const setProxy = (k, v) => setConfig({ ...config, proxy: { ...proxy, [k]: v } });
  const num = (k, v) => setLLM(k, v === "" ? "" : Number(v));
  const numProxy = (k, v) => setProxy(k, v === "" ? "" : Number(v));

  return (
    <div className="page-stack settings-page">
      <div className="config-grid">
        <section className="config-sec config-sec-full">
          <h3>模型连接</h3>
          <div className="config-fields config-fields-4">
            <Field label="调用协议">
              <select value={llm.provider_kind || "openai_chat"} onChange={(e) => setLLM("provider_kind", e.target.value)}>
                {Object.keys(KIND_LABELS).map((k) => (
                  <option key={k} value={k}>{KIND_LABELS[k]}</option>
                ))}
              </select>
            </Field>
            <Field label="API 地址">
              <input value={llm.base_url || ""} onChange={(e) => setLLM("base_url", e.target.value)} placeholder="https://api.moonshot.cn/v1" />
            </Field>
            <Field label="API Key">
              <input
                type="password"
                value={llm.api_key || ""}
                onChange={(e) => setLLM("api_key", e.target.value)}
                placeholder={apiKeyConfigured ? "已设置（留空表示不修改）" : "${MOONSHOT_API_KEY}"}
                autoComplete="new-password"
              />
            </Field>
            <Field label="模型">
              <input
                list="orch-model-catalog"
                value={llm.model || ""}
                onChange={(e) => setLLM("model", e.target.value)}
                placeholder="kimi-k2-turbo-preview"
              />
              <datalist id="orch-model-catalog">
                {catalog.map((m) => (
                  <option key={m.name} value={m.name} />
                ))}
              </datalist>
            </Field>
          </div>
        </section>

        <section className="config-sec config-sec-full">
          <h3>系统提示词</h3>
          <textarea
            className="settings-textarea"
            rows="4"
            aria-label="系统提示词"
            value={llm.system || ""}
            onChange={(e) => setLLM("system", e.target.value)}
            placeholder="留空=不发送 system prompt，最接近原论文 raw completion。"
          />
        </section>

        <section className="config-sec config-sec-full">
          <h3>运行参数</h3>
          <div className="config-fields config-fields-6">
            <Field label="采样温度">
              <input type="number" step="0.1" value={llm.temperature ?? 0.7} onChange={(e) => num("temperature", e.target.value)} />
            </Field>
            <Field label="最大输出 token">
              <input type="number" value={llm.max_tokens ?? 1500} onChange={(e) => num("max_tokens", e.target.value)} />
            </Field>
            <Field label="子项目并发">
              <input type="number" value={llm.concurrency ?? 8} onChange={(e) => num("concurrency", e.target.value)} />
            </Field>
            <Field label="Proxy 上游并发">
              <input type="number" value={proxy.max_upstream_concurrency ?? 4} onChange={(e) => numProxy("max_upstream_concurrency", e.target.value)} />
            </Field>
            <Field label="超时（秒）">
              <input type="number" value={llm.timeout ?? 120} onChange={(e) => num("timeout", e.target.value)} />
            </Field>
            <Field label="重试次数">
              <input type="number" value={llm.max_retries ?? 5} onChange={(e) => num("max_retries", e.target.value)} />
            </Field>
          </div>
        </section>

        <section className="config-sec config-sec-full">
          <h3>运行时</h3>
          <div className="config-fields config-fields-5">
            <Field label="Python 解释器" className="span-2">
              <input
                value={pythonPath}
                onChange={(e) => setConfig({ ...config, python_executable: e.target.value || null })}
                placeholder="/usr/bin/python3"
              />
            </Field>
          </div>
        </section>
      </div>

      <div className="settings-foot">
        <button type="button" className="primary fixed-md" onClick={onSave}>保存配置</button>
      </div>
    </div>
  );
}

export default memo(ConfigPanel);

function Field({ label, className = "", children }) {
  return (
    <div className={"config-field " + className}>
      <label>{label}</label>
      {children}
    </div>
  );
}
