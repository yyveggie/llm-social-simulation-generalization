import { Fragment, useMemo, useState } from "react";
import StatusPill from "./StatusPill.jsx";
import JobDrawer from "./JobDrawer.jsx";

const fmtClock = (ts) => (ts ? new Date(ts * 1000).toLocaleTimeString("zh-CN", { hour12: false }) : "—");

function durationSecs(job) {
  const start = Number(job.started_at || 0);
  if (!start) return 0;
  const pausedTotal = Math.max(0, Number(job.paused_total || 0));
  const end = job.finished_at
    || (job.status === "paused" ? (job.paused_at || start + pausedTotal) : Date.now() / 1000);
  return Math.max(0, Math.round(Number(end) - start - pausedTotal));
}

function fmtDur(job) {
  if (!Number(job.started_at || 0)) return "—";
  const secs = durationSecs(job);
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = secs % 60;
  const pad = (n) => String(n).padStart(2, "0");
  return h ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
}

const SORT_ACCESSORS = {
  status: (j) => j.status || "",
  project_name: (j) => j.project_name || "",
  label: (j) => j.label || "",
  progress: (j) => Number(j.progress?.percent || 0),
  sample: (j) => Number(j.sample_stats?.ok || 0),
  calls: (j) => Number(j.api_calls) || Number(j.progress?.current) || 0,
  model: (j) => j.model || "",
  started_at: (j) => Number(j.started_at || 0),
  duration: (j) => durationSecs(j),
};

const COST_CIRCUIT_RE = /成本熔断保护|circuit breaker|aborting to avoid runaway API cost/;

function isCostCircuitJob(job) {
  return (job.status === "failed" || job.status === "stopped") && COST_CIRCUIT_RE.test(String(job.error || ""));
}

function ProgressCell({ job }) {
  const p = job.progress;
  if (p && p.total > 0) {
    const cls = job.status === "succeeded" ? "ok" : job.status === "failed" || job.status === "stopped" ? "bad" : job.status === "paused" ? "warn" : "";
    return (
      <span className="bar">
        <span className="bar-track"><span className={"bar-fill " + cls} style={{ width: p.percent + "%" }} /></span>
        <span className="bar-pct">{p.percent}%</span>
      </span>
    );
  }
  if (p && p.current > 0) return <span className="c-num">已处理 {p.current}</span>;
  return <span className="muted">—</span>;
}

function SampleCell({ stats }) {
  if (!stats) return <span className="muted sample-pending" title="后台正在同步该任务的样本统计">同步中</span>;
  const ok = Number(stats.ok || 0);
  const unusable = Number(stats.unusable || 0);
  const fail = Number(stats.fail || 0);

  const retried = Number(stats.failed_calls || 0);
  return (
    <span
      className="sample-cell"
      title={`有效 ${ok}${unusable ? ` · 不可用/拒答 ${unusable}（合法数据，不重试）` : ""} · 失败/待重试 ${fail}${retried ? ` · 另有 ${retried} 次调用失败后已自动重试（不缺样本，详见「失败/不可用」页）` : ""}`}
    >
      <span className="s-ok">✓ {ok}</span>
      {unusable > 0 && <><span className="s-sep">·</span><span className="s-warn">⊘ {unusable}</span></>}
      <span className="s-sep">/</span>
      <span className={fail > 0 ? "s-bad" : "muted"}>✗ {fail}</span>
      {retried > 0 && <><span className="s-sep">·</span><span className="s-warn">↻ {retried}</span></>}
    </span>
  );
}

export default function RunsTable({ jobs, selectedId, onSelect, onAction, toast, askConfirm }) {

  const [sort, setSort] = useState({ key: null, dir: "asc" });
  const sortedJobs = useMemo(() => {
    const accessor = SORT_ACCESSORS[sort.key];
    if (!accessor) return jobs;
    return [...jobs].sort((a, b) => {
      const av = accessor(a);
      const bv = accessor(b);
      const cmp = typeof av === "number" && typeof bv === "number"
        ? av - bv
        : av.toString().localeCompare(bv.toString(), "zh-CN", { numeric: true });
      return sort.dir === "asc" ? cmp : -cmp;
    });
  }, [jobs, sort]);
  const toggleSort = (key) =>
    setSort((s) =>
      s.key !== key ? { key, dir: "asc" }
        : s.dir === "asc" ? { key, dir: "desc" }
        : { key: null, dir: "asc" },
    );

  const sortArrow = (key) => (
    <span className="sort-arrow">{sort.key === key ? (sort.dir === "asc" ? "▲" : "▼") : ""}</span>
  );
  if (!jobs.length) {
    return <div className="empty-block">没有匹配的任务。在「发起实验」中选择实验并启动。</div>;
  }
  const jobName = (job) => `「${job.project_name} / ${job.label} / ${job.model || "未命名模型"}」`;
  const confirmDelete = async (job) => {
    const ok = await askConfirm({
      title: "删除任务",
      body: `确认删除${jobName(job)}吗？\n\n这会从磁盘物理删除该任务目录和该实验的结果目录，无法撤销。`,
      confirmText: "删除",
      danger: true,
    });
    if (ok) onAction(job.id, "delete");
  };
  const confirmRerun = async (job) => {
    const ok = await askConfirm({
      title: "从 0 完全重跑",
      body: `确认从 0 完全重跑${jobName(job)}吗？\n\n这会先删除该实验已有的结果与断点文件，再从头开始运行；已完成的进度将丢失，无法撤销。`,
      confirmText: "重跑",
      danger: true,
    });
    if (ok) onAction(job.id, "rerun");
  };
  const confirmStop = async (job) => {
    const ok = await askConfirm({
      title: "停止任务",
      body: `确认停止${jobName(job)}吗？\n\n将终止运行进程并释放资源；已产出的结果与断点会保留，之后可点「续跑」从断点继续（最多损失最近一次落盘之后的少量进度）。`,
      confirmText: "停止",
    });
    if (ok) onAction(job.id, "stop");
  };
  const confirmCostCircuitResume = async (job) => {
    const currentEmpty = job.params?.max_empty_responses ?? 5;
    const currentFailures = job.params?.max_request_failures ?? 25;
    const res = await askConfirm({
      title: "确认续跑熔断任务",
      body: `该任务因空响应或 API 请求失败超过阈值而停止。\n\n当前保存阈值：max_empty_responses=${currentEmpty}，max_request_failures=${currentFailures}。\n\n请为本次断点续跑输入新的非负整数阈值；0 表示关闭对应熔断。`,
      confirmText: "确认续跑",
      danger: true,
      fields: [
        {
          name: "max_empty_responses",
          label: "空响应熔断阈值",
          type: "int",
          min: 0,
          default: 0,
          help: "0 = 本次续跑关闭空响应熔断",
        },
        {
          name: "max_request_failures",
          label: "请求失败熔断阈值",
          type: "int",
          min: 0,
          default: 0,
          help: "0 = 本次续跑关闭请求失败熔断",
        },
      ],
    });
    if (res?.ok) {
      onAction(job.id, "resume", {
        force_cost_circuit_resume: true,
        params_override: res.values,
      });
    }
  };
  return (
    <div className="table-wrap">
      <table className="runs">
        <thead>
          <tr>
            <th className="sortable" onClick={() => toggleSort("status")} title="点击按状态排序">状态{sortArrow("status")}</th>
            <th className="sortable" onClick={() => toggleSort("project_name")} title="点击按项目排序">项目{sortArrow("project_name")}</th>
            <th className="sortable" onClick={() => toggleSort("label")} title="点击按实验排序">实验{sortArrow("label")}</th>
            <th className="sortable" onClick={() => toggleSort("progress")} title="点击按进度排序">进度{sortArrow("progress")}</th>
            <th className="sortable" onClick={() => toggleSort("sample")} title="点击按有效样本数排序。✓有效 · ⊘不可用/拒答（合法数据，不重试） / ✗失败（待重试） · ↻调用失败后已自动重试（不缺样本，仅内部重试类项目如 Social 上报）">样本（有效/不可用/失败）{sortArrow("sample")}</th>
            <th className="sortable" onClick={() => toggleSort("calls")} title="点击按调用次数排序">调用次数{sortArrow("calls")}</th>
            <th className="sortable" onClick={() => toggleSort("model")} title="点击按模型排序">模型{sortArrow("model")}</th>
            <th className="sortable" onClick={() => toggleSort("started_at")} title="点击按开始时间排序">开始{sortArrow("started_at")}</th>
            <th className="sortable" onClick={() => toggleSort("duration")} title="点击按时长排序">时长{sortArrow("duration")}</th>
            <th className="c-actions">操作</th>
          </tr>
        </thead>
        <tbody>
          {sortedJobs.map((j) => {
            const isSel = j.id === selectedId;
            const terminal = j.status === "succeeded" || j.status === "failed" || j.status === "stopped";
            const hasFail = j.sample_stats && Number(j.sample_stats.fail) > 0;

            const incomplete = j.progress && Number(j.progress.total) > 0
              && Number(j.progress.current) < Number(j.progress.total);

            const canResume = hasFail || incomplete || j.status === "stopped" || j.status === "failed";
            const costCircuit = isCostCircuitJob(j);
            const calls = Number(j.api_calls) || Number(j.progress?.current) || 0;
            return (
              <Fragment key={j.id}>
              <tr className={isSel ? "sel" : ""} onClick={() => onSelect(j.id)}>
                <td><StatusPill status={j.status} /></td>
                <td className="c-proj">{j.project_name}</td>
                <td className="c-exp" title={j.label}>{j.label}</td>
                <td><ProgressCell job={j} /></td>
                <td className="c-sample"><SampleCell stats={j.sample_stats} /></td>
                <td className="c-num">{calls ? calls.toLocaleString() : <span className="muted">—</span>}</td>
                <td className="c-mono">{j.model || "—"}</td>
                <td className="c-mono">{fmtClock(j.started_at)}</td>
                <td className="c-num">{fmtDur(j)}</td>
                <td className="c-actions" onClick={(e) => e.stopPropagation()}>
                  <div className="actions">
                    {j.status === "running" && (
                      <button key="pause" className="sm fixed-sm" onClick={() => onAction(j.id, "pause")} title="暂停：进程原地冻结在内存中，立即停止消耗 CPU 与 API 调用；点「继续」瞬间恢复，零进度损失。与「停止」的区别：不杀进程、不释放资源，适合临时让路">暂停</button>
                    )}
                    {j.status === "paused" ? (
                      <button key="resume" className="primary sm fixed-sm" onClick={() => onAction(j.id, "resume")} title="恢复暂停中的进程：瞬间继续、零损失；若原进程已不在（如后端重启过），则自动重启子进程并按断点续跑">继续</button>
                    ) : terminal && canResume ? (
                      <button
                        key="retry"
                        className="primary sm fixed-sm"
                        onClick={() => (costCircuit ? confirmCostCircuitResume(j) : onAction(j.id, "resume"))}
                        title={costCircuit
                          ? "该任务已触发成本熔断；确认并输入新的熔断阈值后断点续跑"
                          : "断点续跑：补齐未完成进度与失败/空响应样本；已完成的部分不会重跑"}
                      >
                        续跑
                      </button>
                    ) : (
                      <button key="stop" className="warn sm fixed-sm" onClick={() => confirmStop(j)} disabled={j.status !== "running"} title="停止：终止运行进程、释放内存与资源；已产出结果与断点保留，之后可「续跑」从断点冷启动继续（最多损失最近一次落盘后的少量进度）。与「暂停」的区别：进程被杀掉，恢复需重启">停止</button>
                    )}
                    {(terminal || j.status === "paused") && (
                      <button key="rerun" className="ghost sm fixed-sm" onClick={() => confirmRerun(j)} title="清空该实验已有结果与断点，从 0 完全重跑">重跑</button>
                    )}
                    <button key="delete" className="danger sm fixed-sm" onClick={() => confirmDelete(j)}>删除</button>
                    <button key="detail" className="sm" onClick={() => onSelect(j.id)}>{isSel ? "收起" : "详情"}</button>
                  </div>
                </td>
              </tr>
              {isSel && (
                <tr className="detail-row">
                  <td colSpan={10} onClick={(e) => e.stopPropagation()}>
                    <JobDrawer job={j} onClose={() => onSelect(null)} toast={toast} />
                  </td>
                </tr>
              )}
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
