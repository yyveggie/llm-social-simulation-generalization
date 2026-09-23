

export const EMPTY_SEL = Object.freeze({});

export function buildDefaultParams(proj) {
  const defs = {};
  (proj.param_schema || []).forEach((f) => {
    defs[f.name] = f.default;
  });
  (proj.experiments || []).forEach((e) => {
    (e.extra_params || []).forEach((f) => {
      defs[`${e.id}__${f.name}`] = f.default;
    });
  });
  return defs;
}

export function normalizeParams(pvals, proj) {
  const out = {};
  const fields = new Map();
  (proj.param_schema || []).forEach((f) => fields.set(f.name, f));
  (proj.experiments || []).forEach((e) => {
    (e.extra_params || []).forEach((f) => fields.set(`${e.id}__${f.name}`, f));
  });

  for (const [name, field] of fields) {
    const raw = pvals[name];
    if (raw === undefined || raw === null || raw === "") {
      if (field.default !== undefined && field.default !== null) out[name] = field.default;
      continue;
    }
    if (field.type === "int") {
      const n = parseInt(String(raw), 10);
      if (!Number.isNaN(n)) out[name] = n;
      continue;
    }
    if (field.type === "float") {
      const n = parseFloat(String(raw));
      if (!Number.isNaN(n)) out[name] = n;
      continue;
    }
    if (field.type === "bool") {
      out[name] = Boolean(raw);
      continue;
    }
    out[name] = raw;
  }
  return out;
}

export function apiKeyReady(config, configMeta) {
  if (configMeta?.api_key_configured) return true;
  const key = (config?.llm?.api_key || "").trim();
  if (!key) return false;
  if (key.startsWith("${") && key.endsWith("}")) return true;
  return true;
}


export function formatOrchestratorError(raw) {
  const message = String(raw || "未知错误").trim();
  const rules = [
    {
      test: () => /config_path 为空|未生成 per-job 配置|回落子项目/.test(message),
      title: "配置注入失败",
      body: "编排器未能为本次任务生成独立配置文件，已阻止启动，以免误读子项目里的旧模板。",
      hint: "请刷新页面后重新点击「启动实验」。若反复出现，请确认 orchestrator 后端正在运行。",
    },
    {
      test: () => /llm_proxy_token|LLM proxy token/.test(message),
      title: "API 代理未就绪",
      body: "任务未能获取本地 LLM 代理令牌，无法把前端配置转发给子项目。",
      hint: "请先在「统一配置」页填写 API Key 与 base_url 并保存，再重新发起实验。",
    },
    {
      test: () => /per-job 配置文件不存在|models 配置不存在/.test(message),
      title: "任务配置已丢失",
      body: "本次任务在 runs/ 下的临时配置文件不存在，可能目录被手动删除。",
      hint: "请发起新实验；不要对损坏的旧任务点「继续」或「重跑」。",
    },
    {
      test: () => /已有任务正在运行|已有其它任务/.test(message),
      title: "同项目已有任务在跑",
      body: "该项目同一时间只允许一组任务运行，避免互相覆盖配置。",
      hint: "请到「监控总览」等待当前任务结束，或先停止后再启动新任务。",
    },
    {
      test: () => /统一配置解析失败|api_key 为空|未解析/.test(message),
      title: "统一 API 未配置完整",
      body: "编排器读不到可用的 API Key 或 base_url。",
      hint: "请打开「统一配置」，填写 API Key 与接口地址后保存。",
    },
    {
      test: () => /缺少 job_state\.json|启动单元信息/.test(message),
      title: "任务状态文件不完整",
      body: "runs/ 下的 job_state.json 缺少启动单元信息，常见于仓库目录被移动/重命名后旧状态未迁移。",
      hint: "重启 orchestrator 后端（新版本会自动改写旧路径并补全状态），再点「续跑」。",
    },
    {
      test: () => /不能续跑|模型与任务记录不一致/.test(message),
      title: "无法续跑该任务",
      body: "任务保存的模型/密钥快照与当前环境不一致，或密钥已失效。",
      hint: "建议发起新实验；若必须续跑，请在「统一配置」恢复对应 endpoint 的 key 后重试。",
    },
    {
      test: () => /Missing API key for provider 'kimi'/.test(message),
      title: "子进程误读了旧模板配置",
      body: "实验子项目回退到了根目录 config.yaml（默认 kimi），未使用 orchestrator 注入的配置。",
      hint: "请对该任务点「重跑」或重新发起实验；当前版本已在启动前强制校验注入配置。",
    },
    {
      test: () => /Missing API key for provider 'unified'|请通过 orchestrator 启动/.test(message),
      title: "未走 orchestrator 配置注入",
      body: "子项目未拿到 runs/ 下的临时配置或 proxy token。",
      hint: "请从「发起实验」重新启动，不要直接在子项目目录运行 run.py。",
    },
    {
      test: () => /无效或过期的 LLM proxy token|缺少 LLM proxy token/.test(message),
      title: "代理令牌已过期",
      body: "orchestrator 重启后，旧任务里的本地 proxy token 会失效。",
      hint: "请对任务点「继续」或「重跑」，系统会重新注入新 token。",
    },
    {
      test: () => /成本熔断保护|cost circuit|circuit breaker/.test(message),
      title: "成本熔断保护",
      body: "该任务的历史结果里空响应或 API 请求失败已超过阈值，系统默认禁止直接续跑以避免继续消耗费用。",
      hint: "请在任务表点击「续跑」，在确认框里输入新的 max_empty_responses / max_request_failures；0 表示关闭对应熔断。",
    },
  ];

  for (const rule of rules) {
    if (rule.test(message)) {
      const lines = [rule.title];
      if (rule.body) lines.push(rule.body);
      if (rule.hint) lines.push("建议：" + rule.hint);
      return lines.join("\n");
    }
  }
  return `操作失败\n${message}`;
}
