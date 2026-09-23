import { useMemo, useState } from "react";
import {
  ResponsiveContainer, AreaChart, Area,
  BarChart, Bar, LineChart, Line, Legend, LabelList, XAxis, YAxis, CartesianGrid, Tooltip,
} from "recharts";
import { GRID, TICK, SERIES_COLORS } from "./chartKit.jsx";

function fmtInt(value) {
  const n = Number(value || 0);
  return Number.isFinite(n) ? n.toLocaleString("zh-CN") : "0";
}

function fmtCompact(value) {
  const n = Number(value || 0);
  if (!Number.isFinite(n)) return "0";
  const abs = Math.abs(n);
  if (abs >= 1e9) return (n / 1e9).toFixed(2) + "B";
  if (abs >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (abs >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return String(n);
}

function ChartCard({ title, sub, extra, hasData, children }) {
  return (
    <div className="card">
      <div className="card-head">
        <div className="card-title">{title}</div>
        <div className="spacer"></div>
        {extra}
        <div className="card-sub">{sub}</div>
      </div>
      {hasData ? (
        <div className="chart-box">
          <ResponsiveContainer width="100%" height="100%">{children}</ResponsiveContainer>
        </div>
      ) : (
        <div className="chart-empty">暂无数据，启动实验后将实时绘制</div>
      )}
    </div>
  );
}

function CallBarChart({ specs, total }) {
  const hasData = specs.some((s) => s.value > 0);
  const data = specs.map((s) => ({
    ...s,
    short: s.label.length > 20 ? `${s.label.slice(0, 19)}…` : s.label,
  }));

  return (
    <ChartCard title="API 调用次数" sub={`累计 ${fmtInt(total)}`} hasData={hasData}>
      <BarChart data={data} layout="vertical" margin={{ top: 2, right: 46, left: 8, bottom: 0 }}>
        <CartesianGrid stroke={GRID} horizontal={false} />
        <XAxis type="number" allowDecimals={false} tick={TICK} axisLine={{ stroke: GRID }} tickLine={false} tickFormatter={fmtCompact} height={20} />
        <YAxis
          type="category"
          dataKey="short"
          tick={TICK}
          axisLine={false}
          tickLine={false}
          width={132}
          interval={0}
        />
        <Tooltip
          formatter={(v) => [fmtInt(v), "调用次数"]}
          labelFormatter={(_, items) => items?.[0]?.payload?.label ?? ""}
        />
        <Bar dataKey="value" name="调用次数" fill="var(--accent)" radius={[0, 3, 3, 0]} isAnimationActive={false}>
          <LabelList dataKey="value" position="right" formatter={fmtInt} fill="var(--text-2)" fontSize={10} />
        </Bar>
      </BarChart>
    </ChartCard>
  );
}

function TokenTimeChart({ buckets, usage }) {
  const [mode, setMode] = useState("cum");

  const models = useMemo(() => {
    const u = usage || {};
    if (Array.isArray(u.by_model) && u.by_model.length) {
      return u.by_model
        .filter((m) => Number(m.total_tokens || 0) > 0)
        .map((m) => m.model || "未命名模型");
    }
    const seen = new Set();
    (buckets || []).forEach((b) => Object.keys(b.tokens || {}).forEach((k) => seen.add(k)));
    return [...seen];
  }, [usage, buckets]);

  const total = useMemo(() => {
    const u = usage || {};
    return Array.isArray(u.by_model)
      ? u.by_model.reduce((s, m) => s + Number(m.total_tokens || 0), 0)
      : 0;
  }, [usage]);

  const data = useMemo(() => {
    const bs = buckets || [];
    if (mode === "cum") return bs;
    return bs.map((b, i) => {
      if (i === 0) return { ...b, tokens: {} };
      const prev = bs[i - 1].tokens || {};
      const inc = {};
      for (const [m, v] of Object.entries(b.tokens || {})) {
        const p = prev[m];
        if (p != null) inc[m] = Math.max(0, Number(v) - Number(p));
      }
      return { ...b, tokens: inc };
    });
  }, [buckets, mode]);

  const hasData = models.length > 0 && (buckets || []).length > 0;

  const modeSwitch = (
    <div className="seg">
      <button type="button" className={mode === "cum" ? "on" : ""} onClick={() => setMode("cum")}>累计</button>
      <button type="button" className={mode === "inc" ? "on" : ""} onClick={() => setMode("inc")}>增量</button>
    </div>
  );

  return (
    <ChartCard title="Token 消耗" sub={`累计 ${fmtInt(total)}`} extra={modeSwitch} hasData={hasData}>
      <LineChart data={data} margin={{ top: 8, right: 12, left: -4, bottom: 0 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis dataKey="label" tick={TICK} axisLine={{ stroke: GRID }} tickLine={false} minTickGap={36} />
        <YAxis tick={TICK} axisLine={false} tickLine={false} width={48} tickFormatter={fmtCompact} />
        <Tooltip formatter={(v, n) => [fmtInt(v), n]} />
        <Legend iconType="plainline" iconSize={12} wrapperStyle={{ fontSize: 11, color: "var(--text-3)" }} />
        {models.map((m, i) => (
          <Line
            key={`${mode}-${m}`}
            type="monotone"
            dataKey={(b) => b.tokens?.[m] ?? null}
            name={m}
            stroke={SERIES_COLORS[i % SERIES_COLORS.length]}
            strokeWidth={1.8}
            dot={false}
            activeDot={{ r: 4 }}
            connectNulls
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ChartCard>
  );
}

export default function MetricsCharts({ series, tokenUsage, tokenSeries }) {
  const hasData = series.length > 1;
  const latest = series[series.length - 1] || {};
  const callSpecs = Object.entries(latest.apiCallLabels || {})
    .map(([key, label]) => ({ key, label, value: Number(latest.apiCalls?.[key] || 0) }))
    .sort((a, b) => b.value - a.value || a.label.localeCompare(b.label));

  return (
    <div className="chart-grid">
      <ChartCard title="实时并发" sub="运行中任务数" hasData={hasData}>
        <AreaChart data={series} margin={{ top: 6, right: 8, left: -18, bottom: 0 }}>
          <defs>
            <linearGradient id="gRun" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--info)" stopOpacity={0.22} />
              <stop offset="100%" stopColor="var(--info)" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke={GRID} vertical={false} />
          <XAxis dataKey="t" tick={TICK} axisLine={{ stroke: GRID }} tickLine={false} minTickGap={48} />
          <YAxis allowDecimals={false} tick={TICK} axisLine={false} tickLine={false} width={34} />
          <Tooltip />
          <Area
            type="monotone" dataKey="running" name="运行中"
            stroke="var(--info)" strokeWidth={1.6} fill="url(#gRun)" isAnimationActive={false}
          />
        </AreaChart>
      </ChartCard>

      <TokenTimeChart buckets={tokenSeries} usage={tokenUsage} />

      <CallBarChart specs={callSpecs} total={latest.apiCallsTotal} />
    </div>
  );
}
