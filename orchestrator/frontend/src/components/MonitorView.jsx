import { useState, useMemo } from "react";
import Sparkline from "./Sparkline.jsx";
import MetricsCharts from "./MetricsCharts.jsx";
import RunsTable from "./RunsTable.jsx";
import { api } from "../api.js";
import { buildDefaultParams, normalizeParams, formatOrchestratorError } from "../launchUtils.js";

const FILTERS = [
  { id: "all", label: "全部" },
  { id: "running", label: "运行中" },
  { id: "succeeded", label: "成功" },
  { id: "failed", label: "失败/停止" },
  { id: "paused", label: "已暂停" },
];

export default function MonitorView({ jobs, series, tokenUsage, tokenSeries, projects, projectCount, selectedJobId, onSelect, onAction, toast, askConfirm, refreshJobs }) {
  const [filter, setFilter] = useState("all");
  const [q, setQ] = useState("");

  const running = jobs.filter((j) => j.status === "running").length;
  const ok = jobs.filter((j) => j.status === "succeeded").length;
  const bad = jobs.filter((j) => j.status === "failed" || j.status === "stopped").length;
  const done = ok + bad;
  const rate = done ? Math.round((ok / done) * 100) : null;

  const rateSeries = series
    .filter((s) => s.ok + s.bad > 0)
    .map((s) => (s.ok / (s.ok + s.bad)) * 100);

  const matchFilter = (j) =>
    filter === "all" ? true :
    filter === "failed" ? (j.status === "failed" || j.status === "stopped") :
    j.status === filter;
  const needle = q.trim().toLowerCase();
  const matchQ = (j) =>
    !needle || [j.project_name, j.label, j.model, j.experiment_id].some((x) => (x || "").toLowerCase().includes(needle));
  const filtered = jobs.filter((j) => matchFilter(j) && matchQ(j));

  return (
    <>
      <div className="stat-grid">
        <div className="stat run">
          <div className="s-label"><span className="dot live"></span>运行中</div>
          <div className="s-row"><div className="s-value">{running}</div><Sparkline data={series.map((s) => s.running)} /></div>
        </div>
        <div className="stat ok">
          <div className="s-label">成功</div>
          <div className="s-row"><div className="s-value">{ok}</div><Sparkline data={series.map((s) => s.ok)} /></div>
        </div>
        <div className="stat bad">
          <div className="s-label">失败 / 停止</div>
          <div className="s-row"><div className="s-value">{bad}</div><Sparkline data={series.map((s) => s.bad)} /></div>
        </div>
        <div className="stat">
          <div className="s-label">成功率</div>
          <div className="s-row"><div className="s-value">{rate == null ? "—" : rate + "%"}</div>{rateSeries.length > 1 && <Sparkline data={rateSeries} />}</div>
        </div>
        <div className="stat">
          <div className="s-label">任务总数</div>
          <div className="s-row"><div className="s-value">{jobs.length}</div><div className="s-sub">{projectCount} 个项目</div></div>
        </div>
      </div>

      <MetricsCharts series={series} tokenUsage={tokenUsage} tokenSeries={tokenSeries} />

      <ModelExperimentMatrix jobs={jobs} projects={projects} toast={toast} askConfirm={askConfirm} refreshJobs={refreshJobs} />

      <div className="card">
        <div className="card-head">
          <div className="card-title">运行任务</div>
          <div className="card-sub">{filtered.length} / {jobs.length}</div>
          <div className="spacer"></div>
          <div className="toolbar">
            <div className="seg">
              {FILTERS.map((f) => (
                <button key={f.id} className={filter === f.id ? "on" : ""} onClick={() => setFilter(f.id)}>{f.label}</button>
              ))}
            </div>
            <input className="search" placeholder="搜索 项目 / 实验 / 模型…" value={q} onChange={(e) => setQ(e.target.value)} />
          </div>
        </div>
        <RunsTable jobs={filtered} selectedId={selectedJobId} onSelect={onSelect} onAction={onAction} toast={toast} askConfirm={askConfirm} />
      </div>
    </>
  );
}

function ModelExperimentMatrix({ jobs, projects, toast, askConfirm, refreshJobs }) {
  const [launching, setLaunching] = useState(() => new Set());
  const { models, experiments, groups, doneSet, runningSet } = useMemo(() => {

    const impLookup = new Map();
    const paperByProj = new Map();
    (projects || []).forEach((p) => {
      paperByProj.set(p.id, p.paper || "");
      (p.experiments || []).forEach((e) => {
        impLookup.set(`${p.id}//${e.id}`, {
          label: e.label || e.id,
          tier: e.importance_tier || "",
          rank: e.importance_rank ?? null,
          basis: e.importance_basis || "",
        });
      });
    });

    const modelSet = new Set();
    const expMap = new Map();
    const done = new Set();
    const running = new Set();
    jobs.forEach((j) => {
      const model = j.model || "未命名模型";
      const projectId = j.project_id || "";
      const project = j.project_name || "";
      modelSet.add(model);
      const rawId = j.experiment_id || j.label || "未命名实验";
      const subIds = String(rawId).includes("+")
        ? String(rawId).split("+").filter(Boolean)
        : [rawId];
      subIds.forEach((expId) => {
        const expKey = `${projectId}//${expId}`;
        if (!expMap.has(expKey)) {
          const imp = impLookup.get(expKey) || {};

          const label = subIds.length > 1
            ? (imp.label || expId)
            : (j.label || imp.label || expId);
          expMap.set(expKey, {
            key: expKey, projectId, project, label,
            paper: paperByProj.get(projectId) || "",
            tier: imp.tier || "", rank: imp.rank ?? null, basis: imp.basis || "",
          });
        }
        const cellKey = `${model}@@${expKey}`;
        if (j.status === "succeeded") done.add(cellKey);
        else if (j.status === "running") running.add(cellKey);
      });
    });

    const sortedModels = [...modelSet].sort((a, b) => a.localeCompare(b, "zh-CN"));

    const sortedExps = [...expMap.values()].sort(
      (a, b) =>
        (a.project || "").localeCompare(b.project || "", "zh-CN") ||
        (a.rank ?? Infinity) - (b.rank ?? Infinity) ||
        a.label.localeCompare(b.label, "zh-CN"),
    );

    const grp = [];
    sortedExps.forEach((e) => {
      const last = grp[grp.length - 1];
      if (last && last.projectId === e.projectId) last.count += 1;
      else grp.push({ projectId: e.projectId, project: e.project, paper: e.paper, count: 1 });
    });

    return { models: sortedModels, experiments: sortedExps, groups: grp, doneSet: done, runningSet: running };
  }, [jobs, projects]);

  const colTitle = (e) => {
    const head = e.tier
      ? `${e.label}（${e.tier} 级${e.rank != null ? ` · 第 ${e.rank} 重要` : ""}）`
      : e.label;
    return e.basis ? `${head}\n\n排名依据：${e.basis}` : head;
  };

  const launchable = useMemo(() => {
    const map = new Map();
    (projects || []).forEach((p) => {
      (p.experiments || []).forEach((e) => map.set(`${p.id}//${e.id}`, { proj: p, exp: e }));
    });
    return map;
  }, [projects]);

  const launchByCol = useMemo(() => {
    const map = new Map();
    experiments.forEach((e) => {
      const direct = launchable.get(e.key);
      if (direct) {
        map.set(e.key, { proj: direct.proj, ids: [direct.exp.id], label: direct.exp.label });
      }
    });
    return map;
  }, [experiments, launchable]);

  const launchCell = async (model, expCol, cellKey) => {
    const hit = launchByCol.get(expCol.key);
    if (!hit) return;
    const { proj, ids, label } = hit;
    const ok = await askConfirm?.({
      title: "补跑实验",
      body: `将以默认参数启动「${proj.name} · ${label}」\n模型：${model}\nAPI Key 与接口地址沿用当前统一配置。`,
      confirmText: "启动",
    });
    if (!ok) return;
    setLaunching((s) => new Set(s).add(cellKey));
    try {

      await api("/api/launch", {
        method: "POST",
        body: JSON.stringify({
          project_id: proj.id,
          experiment_ids: ids,
          params: normalizeParams(buildDefaultParams(proj), proj),
          llm_override: { model },
        }),
      });
      toast?.(`已启动 ${model} · ${label}`);
      await refreshJobs?.();
    } catch (err) {
      toast?.(formatOrchestratorError(err.message), 5200);
    } finally {
      setLaunching((s) => { const n = new Set(s); n.delete(cellKey); return n; });
    }
  };

  const exportExcel = () => {
    if (models.length === 0 || experiments.length === 0) return;
    const xml = buildMatrixExcelXml({ models, experiments, groups, doneSet, runningSet });
    const stamp = new Date().toISOString().slice(0, 19).replace(/[-:T]/g, "");
    const blob = new Blob([xml], { type: "application/vnd.ms-excel;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `模型-实验完成矩阵-${stamp}.xls`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    toast?.("已导出 Excel");
  };

  return (
    <div className="card">
      <div className="card-head matrix-head">
        <div className="card-title">模型 × 实验 完成矩阵</div>
        <div className="spacer"></div>
        <div className="card-sub">✓ = 已成功完成 · <span className="matrix-dot" /> 进行中 · 悬停空白格可一键补跑 · {models.length} 模型 × {experiments.length} 实验 · {groups.length} 项目</div>
        <button
          type="button"
          className="ghost sm"
          onClick={exportExcel}
          disabled={models.length === 0 || experiments.length === 0}
          title="导出当前模型 × 实验完成矩阵为 Excel 文件"
        >
          导出 Excel
        </button>
      </div>
      {models.length === 0 || experiments.length === 0 ? (
        <div className="empty-block">暂无任务，发起并跑完实验后将在此显示完成矩阵。</div>
      ) : (
        <div className="table-wrap">
          <table className="matrix">
            <thead>
              <tr>
                <th className="matrix-corner" rowSpan={2}>模型 \ 实验</th>
                {groups.map((g) => (
                  <th
                    key={`g-${g.projectId}`}
                    className="matrix-group"
                    colSpan={g.count}
                    title={g.paper ? `${g.project} · ${g.paper}` : g.project}
                  >
                    <span className="matrix-group-name">{g.project || "未归类项目"}</span>
                    {g.paper && <span className="matrix-group-paper">{g.paper}</span>}
                  </th>
                ))}
              </tr>
              <tr>
                {experiments.map((e) => (
                  <th key={e.key} className="matrix-col" title={colTitle(e)}>
                    {e.tier && (
                      <span className={"tier-badge tier-" + e.tier}>
                        {e.tier}
                        {e.rank != null && <span className="tier-rank">#{e.rank}</span>}
                      </span>
                    )}
                    <span className="matrix-col-label">{e.label}</span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr key={m}>
                  <td className="matrix-row-h" title={m}>{m}</td>
                  {experiments.map((e) => {
                    const cellKey = `${m}@@${e.key}`;
                    const ok = doneSet.has(cellKey);
                    const run = !ok && runningSet.has(cellKey);
                    const busy = launching.has(cellKey);
                    const canLaunch = !ok && !run && !busy && launchByCol.has(e.key);
                    return (
                      <td key={e.key} className={"matrix-cell" + (ok ? " done" : "")}>
                        {ok ? "✓" : (run || busy) ? <span className="matrix-dot" /> : canLaunch ? (
                          <button
                            type="button"
                            className="cell-launch"
                            title={`以默认参数为 ${m} 补跑「${e.label}」`}
                            aria-label={`以默认参数为 ${m} 补跑 ${e.label}`}
                            onClick={() => launchCell(m, e, cellKey)}
                          >
                            <svg viewBox="0 0 10 10" width="9" height="9" aria-hidden="true">
                              <path d="M2.8 1.7v6.6L8.2 5Z" fill="currentColor" />
                            </svg>
                          </button>
                        ) : ""}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function matrixStatus(model, expKey, doneSet, runningSet) {
  const cellKey = `${model}@@${expKey}`;
  if (doneSet.has(cellKey)) return "完成";
  if (runningSet.has(cellKey)) return "运行中";
  return "";
}

function excelText(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function excelCell(value, attrs = "", style = "") {
  const styleAttr = style ? ` ss:StyleID="${style}"` : "";
  const attrText = attrs ? ` ${attrs}` : "";
  return `<Cell${styleAttr}${attrText}><Data ss:Type="String">${excelText(value)}</Data></Cell>`;
}

function excelRow(cells) {
  return `<Row>${cells.join("")}</Row>`;
}

function experimentExportLabel(e) {
  const tier = e.tier ? `${e.tier}${e.rank != null ? `#${e.rank}` : ""}` : "";
  return tier ? `${e.label} (${tier})` : e.label;
}

function buildMatrixExcelXml({ models, experiments, groups, doneSet, runningSet }) {
  const matrixRows = [
    excelRow([
      excelCell("项目", "", "Header"),
      ...groups.map((g) => excelCell(
        g.project || "未归类项目",
        g.count > 1 ? `ss:MergeAcross="${g.count - 1}"` : "",
        "Header",
      )),
    ]),
    excelRow([
      excelCell("模型", "", "Header"),
      ...experiments.map((e) => excelCell(experimentExportLabel(e), "", "Header")),
    ]),
    ...models.map((m) => excelRow([
      excelCell(m, "", "Model"),
      ...experiments.map((e) => excelCell(matrixStatus(m, e.key, doneSet, runningSet))),
    ])),
  ];

  const detailRows = [
    excelRow([
      excelCell("项目", "", "Header"),
      excelCell("论文", "", "Header"),
      excelCell("实验", "", "Header"),
      excelCell("重要级别", "", "Header"),
      excelCell("重要排名", "", "Header"),
      excelCell("排名依据", "", "Header"),
    ]),
    ...experiments.map((e) => excelRow([
      excelCell(e.project || ""),
      excelCell(e.paper || ""),
      excelCell(e.label || ""),
      excelCell(e.tier || ""),
      excelCell(e.rank ?? ""),
      excelCell(e.basis || ""),
    ])),
  ];

  return `<?xml version="1.0" encoding="UTF-8"?>
<?mso-application progid="Excel.Sheet"?>
<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet"
 xmlns:o="urn:schemas-microsoft-com:office:office"
 xmlns:x="urn:schemas-microsoft-com:office:excel"
 xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet">
  <Styles>
    <Style ss:ID="Header">
      <Font ss:Bold="1"/>
      <Interior ss:Color="#E8EEF7" ss:Pattern="Solid"/>
    </Style>
    <Style ss:ID="Model">
      <Font ss:Bold="1"/>
    </Style>
  </Styles>
  <Worksheet ss:Name="完成矩阵">
    <Table>
      <Column ss:Width="220"/>
      ${experiments.map(() => '<Column ss:Width="190"/>').join("\n      ")}
      ${matrixRows.join("\n      ")}
    </Table>
  </Worksheet>
  <Worksheet ss:Name="实验说明">
    <Table>
      <Column ss:Width="180"/>
      <Column ss:Width="280"/>
      <Column ss:Width="260"/>
      <Column ss:Width="80"/>
      <Column ss:Width="80"/>
      <Column ss:Width="520"/>
      ${detailRows.join("\n      ")}
    </Table>
  </Worksheet>
</Workbook>`;
}
