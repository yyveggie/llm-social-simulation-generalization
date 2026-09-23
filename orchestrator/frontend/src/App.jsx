import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import { api } from "./api.js";
import Sidebar from "./components/Sidebar.jsx";
import MonitorView from "./components/MonitorView.jsx";
import ConfigPanel, { splitConfigPayload } from "./components/ConfigPanel.jsx";
import ProjectPanel from "./components/ProjectPanel.jsx";
import ConclusionsPanel from "./components/ConclusionsPanel.jsx";
import AnalysisPanel from "./components/AnalysisPanel.jsx";
import TrendsPanel from "./components/TrendsPanel.jsx";
import ConfirmDialog from "./components/ConfirmDialog.jsx";
import { formatOrchestratorError } from "./launchUtils.js";

const TITLES = { monitor: "监控总览", launch: "发起实验", analysis: "结果分析", trends: "演化趋势", conclusions: "原作者结论", settings: "统一配置" };

function stableKey(value) {
  let hash = 2166136261;
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return `api_${(hash >>> 0).toString(36)}`;
}

function apiCallKey(job) {
  return stableKey(job.model || "");
}

function apiCallLabel(job) {
  return job.model || "未命名模型";
}

function numeric(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : 0;
}

export default function App() {
  const [view, setView] = useState("monitor");
  const [config, setConfig] = useState(null);
  const [configMeta, setConfigMeta] = useState({});
  const [configSavedJson, setConfigSavedJson] = useState("");
  const [projects, setProjects] = useState([]);
  const [jobs, setJobs] = useState([]);
  const [series, setSeries] = useState([]);
  const [tokenUsage, setTokenUsage] = useState(null);
  const [selectedJobId, setSelectedJobId] = useState(null);
  const [toastMsg, setToastMsg] = useState(null);
  const [confirmReq, setConfirmReq] = useState(null);

  const askConfirm = useCallback(
    (opts) => new Promise((resolve) => setConfirmReq({ ...opts, resolve })),
    [],
  );

  const toastTimer = useRef(null);
  const toast = useCallback((msg, ms = 2600, type = null) => {

    const t = type || (/(失败|错误|出错)/.test(String(msg)) ? "error" : "info");
    if (toastTimer.current) clearTimeout(toastTimer.current);

    setToastMsg({ text: String(msg), type: t, ms, ts: Date.now() });
    toastTimer.current = setTimeout(() => setToastMsg(null), ms);
  }, []);

  const refreshJobs = useCallback(async () => {
    try {

      const [j, tu] = await Promise.all([
        api("/api/jobs"),
        api("/api/token-usage").catch(() => null),
      ]);
      setJobs(j);
      if (tu) setTokenUsage(tu);
      const running = j.filter((x) => x.status === "running").length;
      const ok = j.filter((x) => x.status === "succeeded").length;
      const bad = j.filter((x) => x.status === "failed" || x.status === "stopped").length;
      const apiCalls = {};
      const apiCallLabels = {};
      let apiCallsTotal = 0;

      j.forEach((job) => {
        const key = apiCallKey(job);
        const observed = numeric(job.api_calls);
        const progressDone = numeric(job.progress?.current);
        const done = observed || progressDone;
        apiCalls[key] = (apiCalls[key] || 0) + done;
        apiCallLabels[key] = apiCallLabel(job);
        apiCallsTotal += done;
      });
      const t = new Date().toLocaleTimeString("zh-CN", { hour12: false });
      setSeries((s) => {
        const prev = s[s.length - 1];
        const point = { t, running, ok, bad, total: j.length, apiCalls, apiCallLabels, apiCallsTotal };
        if (prev) {

          point.ok = Math.max(point.ok, prev.ok);
          point.bad = Math.max(point.bad, prev.bad);
          point.apiCallsTotal = Math.max(point.apiCallsTotal, prev.apiCallsTotal);
          point.apiCalls = { ...prev.apiCalls };
          for (const [k, v] of Object.entries(apiCalls)) {
            point.apiCalls[k] = Math.max(numeric(point.apiCalls[k]), numeric(v));
          }
          point.apiCallLabels = { ...prev.apiCallLabels, ...apiCallLabels };
        }
        return [...s.slice(-59), point];
      });
      return running;
    } catch (e) {

      return 0;
    }
  }, []);

  useEffect(() => {
    api("/api/config")
      .then((payload) => {
        const { config: cfg, meta } = splitConfigPayload(payload);
        setConfig(cfg);
        setConfigMeta(meta);
        setConfigSavedJson(JSON.stringify(cfg));
      })
      .catch((e) => toast("配置加载失败：" + e.message));
    api("/api/projects").then(setProjects).catch((e) => toast("项目加载失败：" + e.message));

    let timer = null;
    let stopped = false;
    let busy = false;
    const schedule = (delay) => {
      if (stopped) return;
      if (timer) clearTimeout(timer);
      timer = setTimeout(tick, delay);
    };
    const tick = async () => {
      if (stopped || busy) return;
      busy = true;
      let running = 0;
      try {
        running = document.hidden ? 0 : await refreshJobs();
      } finally {
        busy = false;
      }
      if (stopped) return;
      schedule(!document.hidden && running > 0 ? 2000 : 8000);
    };
    const onVisible = () => {
      if (!document.hidden && !busy) schedule(0);
    };
    document.addEventListener("visibilitychange", onVisible);
    tick();
    return () => {
      stopped = true;
      if (timer) clearTimeout(timer);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, []);

  const configDirty = config != null && JSON.stringify(config) !== configSavedJson;

  const applyConfigPayload = useCallback((payload) => {
    const { config: cfg, meta } = splitConfigPayload(payload);
    setConfig(cfg);
    setConfigMeta(meta);
    setConfigSavedJson(JSON.stringify(cfg));
    return cfg;
  }, []);

  const saveConfig = useCallback(async () => {
    const numOr = (v, d) => {
      const n = Number(v);
      return v === "" || v == null || Number.isNaN(n) ? d : n;
    };
    const llm = config.llm || {};
    const proxy = config.proxy || {};
    const clean = {
      ...config,
      llm: {
        ...llm,
        system: llm.system ?? "",
        temperature: numOr(llm.temperature, 0.7),
        max_tokens: numOr(llm.max_tokens, 1500),
        concurrency: numOr(llm.concurrency, 8),
        timeout: numOr(llm.timeout, 120),
        max_retries: numOr(llm.max_retries, 5),
      },
      proxy: {
        ...proxy,
        max_upstream_concurrency: numOr(proxy.max_upstream_concurrency, 4),
      },
    };
    try {
      const saved = await api("/api/config", { method: "PUT", body: JSON.stringify(clean) });
      applyConfigPayload(saved);
      toast("配置已保存");
    } catch (e) {
      toast("保存失败：" + e.message);
    }
  }, [config, applyConfigPayload, toast]);

  const jobAction = useCallback(async (id, act, payload = null) => {
    try {
      if (act === "delete") {
        const res = await api(`/api/jobs/${id}`, { method: "DELETE" });
        setSelectedJobId((prev) => (prev === id ? null : prev));
        await refreshJobs();
        const n = res?.deleted?.length || 0;
        const skipped = res?.skipped?.length || 0;
        toast(skipped ? `已删除 ${n} 个目录，跳过 ${skipped} 项` : `已删除 ${n} 个目录`);
        return;
      }
      await api(`/api/jobs/${id}/${act}`, {
        method: "POST",
        ...(payload ? { body: JSON.stringify(payload) } : {}),
      });
      refreshJobs();
    } catch (e) {
      toast(formatOrchestratorError(e.message), 5200);
    }
  }, [refreshJobs, toast]);

  const goto = useCallback(async (v) => {
    if (view === "settings" && v !== "settings" && configDirty) {
      const ok = await askConfirm({
        title: "离开配置页",
        body: "配置页有未保存的修改，确定离开？",
        confirmText: "离开",
        danger: true,
      });
      if (!ok) return;
    }
    setView(v);
    setSelectedJobId(null);
  }, [view, configDirty, askConfirm]);

  const handleLaunched = useCallback((jobIds) => {
    setView("monitor");
    if (jobIds?.length) setSelectedJobId(jobIds[jobIds.length - 1]);
  }, []);

  const handleSelectJob = useCallback((id) => {
    setSelectedJobId((prev) => (prev === id ? null : id));
  }, []);

  const running = jobs.filter((j) => j.status === "running").length;
  const model = config?.llm?.model;

  const tokenSeries = useMemo(() => {
    const tl = tokenUsage?.timeline;
    return Array.isArray(tl)
      ? tl.map((b) => ({ ts: b.ts, label: b.label, tokens: b.by_model || {} }))
      : [];
  }, [tokenUsage]);

  return (
    <>
      <div className="app-shell">
        <Sidebar view={view} setView={goto} runningCount={running} />

        <div className="main">
          <header className="topbar">
            <div className="page-title">{TITLES[view]}</div>
            <div className="spacer"></div>
            {model && <div className="chip">模型 <code>{model}</code></div>}
            {config?.llm?.concurrency != null && <div className="chip">并发 <b>{config.llm.concurrency}</b></div>}

            <div className="chip" title="有运行中任务时每 2 秒刷新，空闲时降为 8 秒">
              <span className={"dot" + (running > 0 ? " live" : "")}></span> 实时 · {running > 0 ? "2s" : "8s"}
            </div>
          </header>

          <div className="content">
            {view === "monitor" && (
              <MonitorView
                jobs={jobs}
                series={series}
                tokenUsage={tokenUsage}
                tokenSeries={tokenSeries}
                projects={projects}
                projectCount={projects.length}
                selectedJobId={selectedJobId}
                onSelect={handleSelectJob}
                onAction={jobAction}
                toast={toast}
                askConfirm={askConfirm}
                refreshJobs={refreshJobs}
              />
            )}
            {view === "launch" && (
              <ProjectPanel
                projects={projects}
                config={config}
                configMeta={configMeta}
                toast={toast}
                refreshJobs={refreshJobs}
                onLaunched={handleLaunched}
              />
            )}
            {view === "analysis" && (
              <AnalysisPanel projects={projects} toast={toast} />
            )}
            {view === "trends" && (
              <TrendsPanel toast={toast} />
            )}
            {view === "conclusions" && (
              <ConclusionsPanel projects={projects} />
            )}
            {view === "settings" && (
              <ConfigPanel
                config={config}
                setConfig={setConfig}
                configMeta={configMeta}
                onSave={saveConfig}
              />
            )}
          </div>
        </div>
      </div>

      {confirmReq && (
        <ConfirmDialog
          title={confirmReq.title}
          body={confirmReq.body}
          confirmText={confirmReq.confirmText}
          danger={confirmReq.danger}
          fields={confirmReq.fields}
          onResolve={(val) => { setConfirmReq(null); confirmReq.resolve(val); }}
        />
      )}

      {toastMsg && (
        <div
          key={toastMsg.ts}
          className={"toast toast-" + toastMsg.type + (toastMsg.text.includes("\n") ? " toast-multiline" : "")}
          style={{ "--toast-out-delay": Math.max(200, toastMsg.ms - 260) + "ms" }}
        >
          {toastMsg.text}
        </div>
      )}
    </>
  );
}
