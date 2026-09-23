import { useState, useEffect, useRef } from "react";
import { api } from "../api.js";
import { formatOrchestratorError } from "../launchUtils.js";

const fmtTs = (ts) => (ts ? new Date(ts * 1000).toLocaleString("zh-CN", { hour12: false }) : "—");

const TABS = [
  { id: "logs", label: "实时日志" },
  { id: "files", label: "结果文件" },
  { id: "transcript", label: "对话" },
  { id: "failed", label: "失败/不可用" },
  { id: "meta", label: "元数据" },
];

const MAX_LOG_LINES = 4000;
const capLines = (lines) => (lines.length > MAX_LOG_LINES ? lines.slice(lines.length - MAX_LOG_LINES) : lines);

export default function JobDrawer({ job, onClose, toast }) {
  const [tab, setTab] = useState("logs");
  const [logs, setLogs] = useState([]);
  const [expanded, setExpanded] = useState(false);
  const [results, setResults] = useState(null);
  const [transcript, setTranscript] = useState(null);
  const esRef = useRef(null);
  const logRef = useRef(null);
  const modalLogRef = useRef(null);
  const jobId = job.id;

  const onlyFailed = tab === "failed";

  useEffect(() => {
    if (esRef.current) { esRef.current.close(); esRef.current = null; }
    setLogs([]); setResults(null); setTranscript(null); setTab("logs");

    let closed = false, retries = 0, lastProg = false, timer = null, es = null;
    const connect = (isReconnect) => {
      if (closed) return;
      if (isReconnect) { setLogs([]); lastProg = false; }
      es = new EventSource(`/api/jobs/${jobId}/logs/stream`);
      esRef.current = es;
      es.onmessage = (ev) => { retries = 0; lastProg = false; setLogs((l) => capLines([...l, ev.data])); };
      es.addEventListener("progress", (ev) => {
        retries = 0;
        setLogs((l) => {
          if (lastProg && l.length) { const c = [...l]; c[c.length - 1] = ev.data; return c; }
          return capLines([...l, ev.data]);
        });
        lastProg = true;
      });
      es.addEventListener("end", () => { closed = true; es.close(); });
      es.onerror = () => {
        es.close();
        if (closed || retries >= 5) return;
        retries += 1;
        timer = setTimeout(() => connect(true), 1500);
      };
    };
    connect(false);
    return () => { closed = true; if (timer) clearTimeout(timer); if (es) es.close(); };
  }, [jobId]);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
    if (modalLogRef.current) modalLogRef.current.scrollTop = modalLogRef.current.scrollHeight;
  }, [logs]);

  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") { if (expanded) setExpanded(false); else onClose(); } };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [expanded, onClose]);

  const loadResults = async () => {
    try { setResults(await api(`/api/jobs/${jobId}/results`)); }
    catch (e) { toast("读取结果失败：" + e.message); }
  };
  const loadTranscript = async (offset = 0, failed = onlyFailed) => {
    try { setTranscript(await api(`/api/jobs/${jobId}/transcript?offset=${offset}&limit=50&only_failed=${failed ? 1 : 0}`)); }
    catch (e) { toast("读取对话失败：" + e.message); }
  };

  useEffect(() => {
    if (tab === "files" && !results) loadResults();
  }, [tab]);

  useEffect(() => {
    if (tab === "transcript" || tab === "failed") loadTranscript(0, tab === "failed");
  }, [tab]);

  const paramEntries = Object.entries(job.params || {}).filter(([, v]) => v !== "" && v !== null && v !== undefined);

  return (
    <div className="job-detail">
        <div className="drawer-tabs">
          {TABS.map((t) => (
            <div key={t.id} className={"tab " + (tab === t.id ? "active" : "")} onClick={() => setTab(t.id)}>{t.label}</div>
          ))}
        </div>

        <div className="drawer-body">
          {tab === "logs" && (
            <>
              {job.status === "failed" && job.error && (
                <div className="errbox">{formatOrchestratorError(job.error)}</div>
              )}
              <div className="log-head">
                <span className="t-title">
                  实时日志 · {logs.length} 行
                  {logs.length >= MAX_LOG_LINES && <span className="muted">（超上限，仅保留最近 {MAX_LOG_LINES} 行，全量见磁盘日志）</span>}
                </span>
                <button className="log-zoom" onClick={() => setExpanded(true)} title="全屏放大">⤢ 放大</button>
              </div>
              <div className="log tall" ref={logRef}>{logs.join("\n")}</div>
            </>
          )}

          {tab === "files" && (
            <>
              <div className="actions"><button className="sm" onClick={loadResults}>刷新</button></div>
              {results ? (
                <div className="files">
                  <div className="muted text-xs files-head">结果目录：{results.results_dir}（{results.files.length} 个文件）</div>
                  {results.files.length === 0 && <div className="muted">暂无文件。</div>}
                  {results.files.slice(0, 80).map((f) => (
                    <a key={f.path} href={`/api/jobs/${jobId}/results/download?path=${encodeURIComponent(f.path)}`} target="_blank" rel="noreferrer">
                      {f.path} · {(f.size / 1024).toFixed(1)} KB
                    </a>
                  ))}
                </div>
              ) : <div className="muted">加载中…</div>}
            </>
          )}

          {(tab === "transcript" || tab === "failed") && (
            <>
              {transcript ? (
                <>
                  <div className="actions" style={{ marginBottom: "var(--sp-3)" }}>
                    <span className="muted text-xs">
                      {onlyFailed ? "失败/不可用 " : "共 "}{transcript.total} 条
                      {transcript.total > 0 && `（${transcript.offset + 1}–${Math.min(transcript.offset + transcript.limit, transcript.total)}）`}
                    </span>
                    <div className="spacer" style={{ flex: 1 }}></div>
                    <button className="sm" disabled={transcript.offset <= 0} onClick={() => loadTranscript(Math.max(0, transcript.offset - transcript.limit))}>上一页</button>
                    <button className="sm" disabled={transcript.offset + transcript.limit >= transcript.total} onClick={() => loadTranscript(transcript.offset + transcript.limit)}>下一页</button>
                  </div>
                  {transcript.items.length === 0 && <div className="muted">{onlyFailed ? "没有失败/不可用样本。" : "暂无数据。"}</div>}
                  {transcript.items.map((it, i) => {

                    const cat = it.category || (it.valid === false ? "fail" : "ok");
                    const isFail = cat === "fail";
                    const isUnusable = cat === "unusable";
                    const reason = {
                      error: "API 错误", api_error: "API 错误", empty: "空响应",
                      retry_capped_empty: "空响应 · 已达重试上限",
                      retry_capped_api_error: "API 错误 · 已达重试上限",
                      refusal: "模型拒答", no_number: "未给出数字", out_of_range: "超出范围",
                      bad_stakes: "stakes 异常", unparsable: "无法解析",
                    }[it.fail_reason];
                    const color = isFail ? "var(--bad)" : isUnusable ? "var(--warn)" : undefined;
                    const tagText = isFail ? "失败" : "不可用";
                    return (
                      <div key={transcript.offset + i} className="tr-item" style={color ? { borderLeft: `2px solid ${color}`, paddingLeft: "8px" } : undefined}>
                        <div className="tr-label">
                          #{transcript.offset + i + 1} · {it.label}
                          {(isFail || isUnusable) && <span style={{ marginLeft: 8, color, fontWeight: 600, fontSize: 11 }}>{tagText}{reason ? " · " + reason : ""}</span>}
                        </div>
                        <div className="tr-block"><span className="tr-tag">PROMPT</span><pre>{it.prompt}</pre></div>
                        <div className="tr-block"><span className="tr-tag resp">RESPONSE</span><pre>{it.response || (isFail ? "（空响应）" : "")}</pre></div>
                      </div>
                    );
                  })}
                </>
              ) : <div className="muted">加载中…</div>}
            </>
          )}

          {tab === "meta" && (
            <div className="meta-card">
              <div className="meta-title">运行元数据 · run_meta.json（schema run_meta/v1）</div>
              <div className="meta-row"><span>论文</span><b>{job.paper || "—"}</b></div>
              <div className="meta-row"><span>项目</span><b>{job.project_name}</b></div>
              <div className="meta-row"><span>实验</span><b>{job.label}（{job.experiment_id}）</b></div>
              <div className="meta-row"><span>模型</span><b>{job.model || "—"}</b></div>
              <div className="meta-row"><span>状态</span><b>{job.status}{job.exit_code != null ? ` · exit ${job.exit_code}` : ""}</b></div>
              <div className="meta-row"><span>开始</span><b>{fmtTs(job.started_at)}</b></div>
              <div className="meta-row"><span>结束</span><b>{fmtTs(job.finished_at)}</b></div>
              <div className="meta-row"><span>结果目录</span><b className="mono">{job.results_dir || "—"}</b></div>
              <div className="meta-row"><span>日志</span><b className="mono">{job.log_path || "—"}</b></div>
              <div className="meta-row"><span>实验参数</span><b>{paramEntries.length ? `${paramEntries.length} 项` : "（无）"}</b></div>
              {paramEntries.length > 0 && (
                <div className="meta-params">
                  {paramEntries.map(([k, v]) => (<div key={k} className="meta-kv"><code>{k}</code><span>{String(v)}</span></div>))}
                </div>
              )}
              <div className="muted text-xs meta-note">已写入 run_meta.json（任务目录 + 结果目录副本）、runs_index.jsonl、LAST_RUN.txt</div>
            </div>
          )}
        </div>

      {expanded && (
        <div className="log-modal-backdrop" onClick={() => setExpanded(false)}>
          <div className="log-modal" onClick={(e) => e.stopPropagation()}>
            <div className="log-modal-head">
              <span className="t-title">
                {job.project_name} · {job.label} — 实时日志 · {logs.length} 行
                {logs.length >= MAX_LOG_LINES && <span className="muted">（超上限，仅保留最近 {MAX_LOG_LINES} 行）</span>}
              </span>
              <button onClick={() => setExpanded(false)}>关闭 ✕（Esc）</button>
            </div>
            <div className="log-modal-body" ref={modalLogRef}>{logs.join("\n")}</div>
          </div>
        </div>
      )}
    </div>
  );
}
