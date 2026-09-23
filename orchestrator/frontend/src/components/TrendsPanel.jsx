import { Fragment, useState, useEffect, useMemo, useCallback } from "react";
import {
  ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, ReferenceLine, ReferenceDot,
} from "recharts";
import { api } from "../api.js";
import Sparkline from "./Sparkline.jsx";
import { GRID, TICK, SERIES_COLORS as FAMILY_COLORS } from "./chartKit.jsx";




const PROJECT_LABELS = {
  behavioral: "行为图灵测试",
  altruism: "利他行为",
  social: "社会惯例",
  identity: "身份群体刻画",
  age_gender_distortion: "年龄-性别扭曲",
};

function num(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function fmt(v, digits = 3) {
  const n = num(v);
  if (n === null) return "—";
  const abs = Math.abs(n);
  if (abs !== 0 && (abs < 0.001 || abs >= 100000)) return n.toExponential(1);
  return Number(n.toFixed(digits)).toString();
}

function dirClass(direction) {
  if (direction === "上升") return { color: "var(--ok, #16a34a)", fontWeight: 600 };
  if (direction === "下降") return { color: "var(--bad, #dc2626)", fontWeight: 600 };
  return { color: "var(--text-3)" };
}


function dirText(direction) {
  if (direction === "上升") return "↑ 上升";
  if (direction === "下降") return "↓ 下降";
  return direction;
}



function theilSen(points) {
  if (points.length < 2) return null;
  const slopes = [];
  for (let i = 0; i < points.length; i += 1) {
    for (let j = i + 1; j < points.length; j += 1) {
      const dx = points[j].ts - points[i].ts;
      if (dx !== 0) slopes.push((points[j].value - points[i].value) / dx);
    }
  }
  if (!slopes.length) return null;
  const med = (xs) => {
    const s = [...xs].sort((a, b) => a - b);
    const m = s.length >> 1;
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  };
  const slope = med(slopes);
  const intercept = med(points.map((p) => p.value - slope * p.ts));
  return { slope, intercept };
}


function IndicatorScatter({ metrics, project, indicator }) {
  const points = useMemo(() => {
    const seen = new Set();
    const out = [];
    for (const r of metrics) {
      if (r.project !== project || r.indicator !== indicator) continue;
      if (seen.has(r.model)) continue;
      seen.add(r.model);
      const value = num(r.value);
      const ts = r.release_date ? Date.parse(r.release_date) : NaN;
      if (value === null || Number.isNaN(ts)) continue;
      out.push({ ts, value, model: r.model, family: r.family || "?", date: r.release_date, n: r.n, p: r.p });
    }
    return out.sort((a, b) => a.ts - b.ts);
  }, [metrics, project, indicator]);

  const families = useMemo(() => [...new Set(points.map((p) => p.family))].sort(), [points]);


  const { xMin, xMax } = useMemo(() => {
    if (!points.length) return { xMin: 0, xMax: 1 };
    const ts = points.map((p) => p.ts);
    const lo = Math.min(...ts);
    const hi = Math.max(...ts);
    const pad = Math.max((hi - lo) * 0.03, 86400000);
    return { xMin: lo - pad, xMax: hi + pad };
  }, [points]);

  const fit = useMemo(() => theilSen(points), [points]);
  const first = points[0];
  const last = points[points.length - 1];

  if (!points.length) {
    return <div className="chart-empty">该指标没有带发布日期的数据点（registry 可能待补全）。</div>;
  }
  const fmtDate = (ts) => new Date(ts).toISOString().slice(0, 10);
  return (
    <div style={{ height: 280 }}>
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 18, right: 24, left: 0, bottom: 4 }}>
          <CartesianGrid stroke={GRID} />
          <XAxis
            dataKey="ts" type="number" domain={[xMin, xMax]}
            tickFormatter={fmtDate} tick={TICK} axisLine={{ stroke: GRID }} tickLine={false}
          />
          <YAxis dataKey="value" type="number" domain={["auto", "auto"]}
                 tick={TICK} axisLine={false} tickLine={false} width={64} />
          <Tooltip
            cursor={{ strokeDasharray: "3 3" }}
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const p = payload[0].payload;
              return (
                <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 6, padding: "6px 10px", fontSize: 12 }}>
                  <div style={{ fontWeight: 600 }}>{p.model}</div>
                  <div>发布：{p.date}</div>
                  <div>值：{fmt(p.value, 4)}{p.n ? `（n=${p.n}）` : ""}</div>
                  {p.p !== "" && p.p != null && <div>p(vs 基线)：{fmt(p.p, 4)}</div>}
                </div>
              );
            }}
          />
          <Legend iconSize={10} wrapperStyle={{ fontSize: 11, color: "var(--text-3)" }} />

          {fit && (
            <ReferenceLine
              segment={[
                { x: first.ts, y: fit.slope * first.ts + fit.intercept },
                { x: last.ts, y: fit.slope * last.ts + fit.intercept },
              ]}
              stroke="var(--text-3)"
              strokeDasharray="6 4"
              strokeWidth={1.4}
              ifOverflow="hidden"
            />
          )}
          {families.map((f, i) => (
            <Scatter
              key={f} name={f}
              data={points.filter((p) => p.family === f)}
              fill={FAMILY_COLORS[i % FAMILY_COLORS.length]}
              isAnimationActive={false}
            />
          ))}

          {points.length >= 2 && [
            { p: first, anchor: "start" },
            { p: last, anchor: "end" },
          ].map(({ p, anchor }) => (
            <ReferenceDot
              key={`lbl-${anchor}`}
              x={p.ts}
              y={p.value}
              r={0}
              ifOverflow="visible"
              label={{
                value: p.model,
                position: "top",
                fontSize: 10,
                fill: "var(--text-2)",
                textAnchor: anchor === "start" ? "start" : "end",
              }}
            />
          ))}
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}


export default function TrendsPanel({ toast }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [projectFilter, setProjectFilter] = useState("all");
  const [onlySig, setOnlySig] = useState(false);
  const [selected, setSelected] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setData(await api("/api/trends"));
    } catch (e) {
      toast("趋势数据加载失败：" + e.message);
    } finally {
      setLoading(false);
    }
  }, [toast]);

  useEffect(() => { load(); }, [load]);

  const refresh = useCallback(async () => {
    setRefreshing(true);
    try {
      const res = await api("/api/trends/refresh", { method: "POST", body: "{}" });
      setData(res);
      toast("趋势汇总已更新");
    } catch (e) {
      toast((e.status === 409 ? "" : "趋势汇总失败：") + e.message, 5200);
    } finally {
      setRefreshing(false);
    }
  }, [toast]);

  const tests = data?.tests || [];
  const metrics = data?.metrics || [];
  const projects = useMemo(() => [...new Set(tests.map((t) => t.project))].sort(), [tests]);


  const seriesByKey = useMemo(() => {
    const map = new Map();
    for (const r of metrics) {
      const ts = r.release_date ? Date.parse(r.release_date) : NaN;
      const value = num(r.value);
      if (value === null || Number.isNaN(ts)) continue;
      const key = r.project + "|" + r.indicator;
      let entry = map.get(key);
      if (!entry) { entry = { seen: new Set(), pts: [] }; map.set(key, entry); }
      if (entry.seen.has(r.model)) continue;
      entry.seen.add(r.model);
      entry.pts.push({ ts, value });
    }
    const out = new Map();
    for (const [key, { pts }] of map) {
      out.set(key, pts.sort((a, b) => a.ts - b.ts).map((p) => p.value));
    }
    return out;
  }, [metrics]);

  const shown = useMemo(() => {
    let rows = tests;
    if (projectFilter !== "all") rows = rows.filter((t) => t.project === projectFilter);
    if (onlySig) rows = rows.filter((t) => t.direction && t.direction !== "无显著单调趋势");

    return [...rows].sort((a, b) => {
      const pa = num(a.p_spearman_perm), pb = num(b.p_spearman_perm);
      if (pa === null && pb === null) return 0;
      if (pa === null) return 1;
      if (pb === null) return -1;
      if (pa !== pb) return pa - pb;
      return Math.abs(num(b.spearman_rho) || 0) - Math.abs(num(a.spearman_rho) || 0);
    });
  }, [tests, projectFilter, onlySig]);

  const nSig = useMemo(
    () => tests.filter((t) => t.direction && t.direction !== "无显著单调趋势").length,
    [tests],
  );
  const nModels = useMemo(() => new Set(metrics.map((r) => r.model)).size, [metrics]);

  if (loading) return <div className="panel">加载中…</div>;

  const empty = !data?.available;
  return (
    <div className="stack">
      <div className="card">
        <div className="toolbar">
          <div className="chip">模型 <b>{nModels}</b></div>
          <div className="chip">指标序列 <b>{new Set(metrics.map((r) => r.project + "|" + r.indicator)).size}</b></div>
          <div className="chip">可检验 <b>{tests.length}</b></div>
          <div className="chip" title="逐条置换 p<0.05；严格结论请看表中 FDR 校正后的 q 列">显著趋势 <b>{nSig}</b></div>
          {data?.tests_generated_at && <div className="chip">检验时间 {data.tests_generated_at}</div>}
          <div className="spacer"></div>
          <button type="button" className="primary" onClick={refresh} disabled={refreshing}>
            {refreshing ? <><span className="btn-spin" aria-hidden="true" />汇总中…</> : "重新汇总"}
          </button>
        </div>
        {(data?.undated_models?.length || 0) > 0 && (
          <div style={{ marginTop: 8, fontSize: 12, color: "var(--text-3)" }}>
            以下模型缺发布日期、未进入趋势检验（用 <code>python trends/collect_metrics.py --scaffold</code> 生成
            注册表占位条目后补全）：{data.undated_models.join("、")}
          </div>
        )}
      </div>

      {empty ? (
        <div className="panel">
          <h2>还没有趋势数据</h2>
          <p style={{ fontSize: 13, color: "var(--text-2)" }}>
            趋势层从各项目的分析产物中抽取核心指标，并按模型发布时间做单调趋势检验。
            点右上角「重新汇总」即可生成（纯本地计算、不调用任何 API）。
            某项目某模型没有分析产物时会被跳过——先在「结果分析」页为各模型跑一键分析。
          </p>
        </div>
      ) : (
        <>
          <div className="card">
            <div className="toolbar" style={{ marginBottom: 8 }}>
              <div className="card-title">趋势检验（按发布时间的单调性）</div>
              <div className="spacer"></div>
              <label style={{ fontSize: 12, color: "var(--text-2)", display: "flex", alignItems: "center", gap: 4 }}>
                <input type="checkbox" checked={onlySig} onChange={(e) => setOnlySig(e.target.checked)} />
                只看显著
              </label>
              <select value={projectFilter} onChange={(e) => setProjectFilter(e.target.value)}>
                <option value="all">全部项目</option>
                {projects.map((p) => (
                  <option key={p} value={p}>{PROJECT_LABELS[p] || p}</option>
                ))}
              </select>
            </div>
            <div className="table-wrap">
              <table className="runs">
                <thead>
                  <tr>
                    <th>项目</th><th>指标</th><th>n</th>
                    <th title="Spearman 秩相关（发布顺序 × 指标值）">ρ</th>
                    <th title="置换检验 p（双侧，未校正）">p</th>
                    <th title="Benjamini-Hochberg FDR 校正后">q</th>
                    <th>方向</th>
                    <th title="按发布时间排序的指标值走向">走势</th>
                    <th title="最新模型值 − 最早模型值">Δ 首→末</th>
                  </tr>
                </thead>
                <tbody>
                  {shown.map((t) => {
                    const key = t.project + "|" + t.indicator;
                    const active = selected && selected.project === t.project && selected.indicator === t.indicator;
                    const spark = seriesByKey.get(key) || [];
                    return (
                      <Fragment key={key}>
                        <tr
                          onClick={() => setSelected(active ? null : { project: t.project, indicator: t.indicator })}
                          style={{ cursor: "pointer", background: active ? "var(--surface-hover)" : undefined }}
                          title={active ? "点击收起散点图" : "点击展开该指标的 发布时间 × 值 散点图"}
                        >
                          <td>{PROJECT_LABELS[t.project] || t.project}</td>
                          <td><code style={{ fontSize: 11 }}>{t.indicator}</code></td>
                          <td>{t.n_models}</td>
                          <td>{fmt(t.spearman_rho, 2)}</td>
                          <td>{fmt(t.p_spearman_perm, 4)}</td>
                          <td>{fmt(t.q_fdr, 4)}</td>
                          <td style={dirClass(t.direction)}>{dirText(t.direction)}</td>
                          <td>{spark.length > 1 ? <Sparkline data={spark} width={72} height={20} /> : <span style={{ color: "var(--text-3)" }}>—</span>}</td>
                          <td>{fmt(t.delta_last_minus_first, 3)}</td>
                        </tr>

                        {active && (
                          <tr className="detail-row">
                            <td colSpan={9}>
                              <div style={{ padding: "var(--sp-3) var(--sp-4)", background: "var(--bg)" }}>
                                <div className="card-sub" style={{ marginBottom: 6 }}>
                                  {(PROJECT_LABELS[t.project] || t.project)} · <code>{t.indicator}</code> — 按家族着色；横轴为模型发布日期
                                </div>
                                <IndicatorScatter metrics={metrics} project={t.project} indicator={t.indicator} />
                              </div>
                            </td>
                          </tr>
                        )}
                      </Fragment>
                    );
                  })}
                  {!shown.length && (
                    <tr><td colSpan={9} style={{ color: "var(--text-3)" }}>没有满足条件的指标（试试取消筛选，或先「重新汇总」）。</td></tr>
                  )}
                </tbody>
              </table>
            </div>
            <div style={{ marginTop: 6, fontSize: 11, color: "var(--text-3)" }}>
              模型数少时功效有限：趋势结论以效应量（ρ、Δ）为主、显著性为辅；「方向」基于未校正 p&lt;0.05，严格口径看 q 列。
            </div>
          </div>

        </>
      )}
    </div>
  );
}
