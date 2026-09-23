import { useState, useEffect, useMemo, useCallback, useRef, memo } from "react";
import { api } from "../api.js";
import Markdown from "./Markdown.jsx";
import {
  ResponsiveContainer, LineChart, Line, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Cell, ReferenceLine, Legend, ErrorBar,
} from "recharts";
import { GRID, TICK, BAR_COLORS, exportChartPng } from "./chartKit.jsx";
import ProjectTabs from "./ProjectTabs.jsx";

function fmtP(p) {
  if (p == null) return "—";
  return p < 1e-4 ? Number(p).toExponential(2) : Number(p).toFixed(4);
}
function pct(x) {
  return x == null ? "—" : (Number(x) * 100).toFixed(1) + "%";
}
function fmtShare(x) {
  return x == null ? "—" : Number(x).toFixed(3);
}

function Stat({ label, value }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 2, minWidth: 92 }}>
      <span className="muted" style={{ fontSize: 11 }}>{label}</span>
      <b style={{ fontSize: 15 }}>{value}</b>
    </div>
  );
}

function ChartFrame({ title, height = 200, children }) {
  const boxRef = useRef(null);
  return (
    <div style={{ marginTop: 10 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
        <div className="muted" style={{ fontSize: 12, flex: 1 }}>{title}</div>
        <button
          type="button"
          className="ghost sm"
          style={{ fontSize: 11, padding: "1px 8px" }}
          title="把该图导出为 PNG（2x，白底；不含悬浮图例）"
          onClick={() => exportChartPng(boxRef.current, title)}
        >
          存图
        </button>
      </div>
      <div ref={boxRef} style={{ width: "100%", height }}>
        <ResponsiveContainer width="100%" height="100%">{children}</ResponsiveContainer>
      </div>
    </div>
  );
}

function PaperConclusion({ text }) {
  if (!text || !String(text).trim()) return null;
  return (
    <details className="an-paper">
      <summary>展开论文原文结论</summary>
      <Markdown text={text} className="an-paper-md" />
    </details>
  );
}

function Block({ title, verdict, tone, paper, paperShort, diff, children }) {
  const hasPaper = (paper && String(paper).trim()) || (paperShort && String(paperShort).trim());
  const paired = hasPaper || (diff && String(diff).trim());
  return (
    <div className="an-block">
      {title && <div className="an-block-title">{title}</div>}
      {children}
      {verdict && (
        <div className={"an-verdict" + (tone === "warn" ? " warn" : "")}>
          {paired ? <span className="an-tag an-tag-mine">本次结论</span> : null}
          {verdict}
        </div>
      )}
      {hasPaper && (
        <div className="an-paper-row">
          {paperShort && (
            <div className="an-paper-head"><span className="an-tag an-tag-paper">原作者结论</span>{paperShort}</div>
          )}
          {paper && String(paper).trim() ? <PaperConclusion text={paper} /> : null}
        </div>
      )}
      {diff && String(diff).trim() ? (
        <div className="an-diff"><span className="an-tag an-tag-diff">结论差异</span>{diff}</div>
      ) : null}
    </div>
  );
}

const SOCIAL_PAPER_REF = {
  individual_bias: "原文：Llama-2/Claude-3.5 首轮无显著个体偏好（χ² P=0.100/0.410），Llama-3/3.1 有显著偏斜；但集体偏差不要求个体先有偏。",
  collective_convergence: "原文 H1+H2：纯局部互动约第 15 个 population round 自发收敛（对称破缺/赢家通吃）；即便个体无偏群体仍涌现集体偏差（成功后 99.4% 沿用、失败后 97.3% 换名）。",
  committed_minority: "原文 H3：坚定少数派达临界质量即可翻转既有 convention；临界值随模型差异极大（Llama-3 低至 2%、Llama-2 高达 67%，Llama-3.1 无需少数派即自发转向强约定）。",
};

function socialExperimentOf(name, data) {
  if (data.type === "individual_bias") return "individual_bias";
  const n = String(name || "").toLowerCase();
  if (/cmtd|committed|swap|inject|minorit/.test(n) || data.committed_to) return "committed_minority";
  return "collective_convergence";
}

function socialSelfConclusion(exp, data) {
  if (exp === "individual_bias") {
    const t = data.bias_test || {};
    return t.verdict ? `本模型个体首轮：${t.verdict}` : "本模型个体首轮：暂无可判定结论。";
  }
  if (exp === "committed_minority") {
    const flipped = data.committed_to
      ? `convention 被坚定少数派翻转为 ${data.committed_to}`
      : "未发生翻转（少数派未达临界质量或未引入）";
    const h1 = data.H1_convergence || {};
    return `本模型：${flipped}${h1.converged_runs != null ? `；基线收敛 ${h1.converged_runs}/${h1.total_runs} run` : ""}。`;
  }

  const h1 = data.H1_convergence || {};
  const round = h1.converged_round || {};
  const bias = data.H2_collective_bias || {};
  const em = data.H2_emergence || {};
  const rule = data.update_rule || {};
  const parts = [];
  parts.push(`H1：${h1.converged_runs ?? "—"}/${h1.total_runs ?? "—"} run 收敛${round.median != null ? `（中位 ${round.median} round）` : ""}`);
  if (bias.favored != null) {
    const p = bias.p_value ?? bias.chi_square?.p;
    parts.push(`H2：偏好 ${bias.favored}（p=${fmtP(p)}${em.emergence_onset_t ? `，自第 ${em.emergence_onset_t} 次涌现` : ""}）`);
  }
  if (rule.stay_given_success != null || rule.shift_given_fail != null) {
    parts.push(`机制：成功沿用 ${rule.stay_given_success == null ? "—" : pct(rule.stay_given_success)}、失败换名 ${rule.shift_given_fail == null ? "—" : pct(rule.shift_given_fail)}`);
  }
  return `本模型：${parts.join("；")}。`;
}

function socialDiff(exp, data) {
  if (exp === "individual_bias") {
    const t = data.bias_test || {};
    const biased = t.p_value != null && t.p_value < 0.05;
    return `本模型首轮${biased ? `显著偏好 ${t.favored_option}` : "无显著个体偏好"}（p=${fmtP(t.p_value ?? t.p)}）；原文 Llama-2/Claude 无偏、Llama-3/3.1 有偏。差异：本模型属${biased ? "「有偏」" : "「无偏」"}一类（不同模型本就各异）。`;
  }
  if (exp === "committed_minority") {
    return `本模型 ${data.committed_to ? `翻转为 ${data.committed_to}` : "未翻转"}；原文临界质量随模型 2%–67% 不等、个别模型无需少数派。差异：临界行为因模型/参数不同本就各异（属正常）。`;
  }
  const round = (data.H1_convergence || {}).converged_round || {};
  const rule = data.update_rule || {};
  return `本模型收敛中位 ${round.median != null ? `${round.median} round` : "—"}（原文≈15）；成功沿用 ${rule.stay_given_success == null ? "—" : pct(rule.stay_given_success)}/失败换名 ${rule.shift_given_fail == null ? "—" : pct(rule.shift_given_fail)}（原文 99.4%/97.3%）。差异：仅陈述数值，方向与原文（成功沿用、失败换名、群体收敛并涌现偏差）一致或不一致由上数对照。`;
}

function CollectiveCard({ name, data, paper }) {
  const h1 = data.H1_convergence || {};
  const round = h1.converged_round || {};
  const bias = data.H2_collective_bias || {};
  const emergence = data.H2_emergence || {};
  const rule = data.update_rule || {};

  const curve = useMemo(() => {
    const sc = data.success_curve || {};
    const xs = sc.population_round || [];
    const ys = sc.mean_success || [];
    const ns = sc.n_runs_per_round || [];
    return xs.map((r, i) => ({ round: r, success: ys[i], n_runs: ns[i] }));
  }, [data]);

  const biasBars = useMemo(
    () => Object.entries(bias.convention_counts || {}).map(([k, v]) => ({ name: k, count: v })),
    [bias],
  );
  const emergeLine = useMemo(
    () => (emergence.by_interaction || []).map((d) => ({
      t: d.t, p_strong: d.p_strong, p_adj: d.p_value_adj, n_runs: d.n_runs,
    })),
    [emergence],
  );

  return (
    <div className="card" style={{ padding: 14 }}>
      <div className="card-title" style={{ marginBottom: 2 }}>{name}</div>
      <div className="muted" style={{ fontSize: 12, marginBottom: 10 }}>
        集体收敛 / 坚定少数派 · {data.n_runs} 个 run
      </div>

      <Block title="H1 · 自发涌现 convention" verdict={h1.verdict}>
        <div style={{ display: "flex", gap: 18, flexWrap: "wrap" }}>
          <Stat label="收敛率" value={`${h1.converged_runs ?? "—"}/${h1.total_runs ?? "—"}`} />
          <Stat label="收敛轮次(中位)" value={round.median != null ? `${round.median} round` : "—"} />
          <Stat label="收敛轮次(范围)" value={round.min != null ? `${round.min}–${round.max}` : "—"} />
          <Stat label="近窗成功率" value={pct(data.recent_success_rate)} />
          <Stat label="整体成功率" value={pct(data.overall_success_rate)} />
        </div>
        {curve.length > 1 && (
          <ChartFrame title="协调成功率 vs population round（跨 run 平均）">
            <LineChart data={curve} margin={{ top: 6, right: 12, left: -18, bottom: 0 }}>
              <CartesianGrid stroke={GRID} vertical={false} />
              <XAxis dataKey="round" tick={TICK} axisLine={{ stroke: GRID }} tickLine={false} />
              <YAxis domain={[0, 1]} tick={TICK} axisLine={false} tickLine={false} width={36} />
              <Tooltip
                formatter={(v, _n, item) => [`${pct(v)}（n=${item?.payload?.n_runs ?? "—"} runs）`, "成功率"]}
                labelFormatter={(l) => `round ${l}`}
              />
              <Line type="monotone" dataKey="success" stroke="#2563eb" strokeWidth={2} dot={false} isAnimationActive={false} />
            </LineChart>
          </ChartFrame>
        )}
      </Block>

      <Block title="H2 · 集体偏差" verdict={bias.verdict}>
        {biasBars.length > 0 && (
          <div style={{ display: "flex", gap: 18, flexWrap: "wrap" }}>
            <Stat label="偏好 convention" value={bias.favored ?? "—"} />
            <Stat label="占比" value={bias.n ? `${bias.k}/${bias.n}` : "—"} />
            <Stat label="p 值" value={fmtP(bias.p_value ?? bias.chi_square?.p)} />
            <Stat label="方法" value={bias.method ?? (bias.chi_square ? "chi-square" : "—")} />
          </div>
        )}
        {biasBars.length > 0 && (
          <ChartFrame title="各 run 最终收敛到的 convention 分布" height={170}>
            <BarChart data={biasBars} margin={{ top: 6, right: 12, left: -18, bottom: 0 }}>
              <CartesianGrid stroke={GRID} vertical={false} />
              <XAxis dataKey="name" tick={TICK} axisLine={{ stroke: GRID }} tickLine={false} />
              <YAxis allowDecimals={false} tick={TICK} axisLine={false} tickLine={false} width={32} />
              <Tooltip formatter={(v) => [v, "run 数"]} />
              <Bar dataKey="count" radius={[3, 3, 0, 0]} isAnimationActive={false}>
                {biasBars.map((_, i) => <Cell key={i} fill={BAR_COLORS[i % BAR_COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ChartFrame>
        )}
      </Block>

      {emergeLine.length > 1 && (
        <Block title="H2 · 偏差涌现（复现论文 Table 1）" verdict={emergence.verdict}>
          {emergence.method && (
            <div className="muted" style={{ fontSize: 11 }}>统计方法：{emergence.method}</div>
          )}
          <ChartFrame title={`第 t 次交互偏向「${emergence.strong_label ?? ""}」的比例（灰虚线=50% 基准 / 红虚线=显著拐点）`} height={170}>
            <LineChart data={emergeLine} margin={{ top: 6, right: 12, left: -18, bottom: 0 }}>
              <CartesianGrid stroke={GRID} vertical={false} />
              <XAxis dataKey="t" tick={TICK} axisLine={{ stroke: GRID }} tickLine={false} />
              <YAxis domain={[0, 1]} tick={TICK} axisLine={false} tickLine={false} width={36} />
              <Tooltip
                formatter={(v, _n, item) => [
                  `${pct(v)}（n=${item?.payload?.n_runs ?? "—"} runs, p校正=${fmtP(item?.payload?.p_adj)}）`,
                  "偏向比例",
                ]}
                labelFormatter={(l) => `第 ${l} 次`}
              />
              <ReferenceLine y={0.5} stroke="#94a3b8" strokeDasharray="4 4" />
              {emergence.emergence_onset_t != null && (
                <ReferenceLine x={emergence.emergence_onset_t} stroke="#ef4444" strokeDasharray="4 4" />
              )}
              <Line type="monotone" dataKey="p_strong" stroke="#f97316" strokeWidth={2} dot={{ r: 2 }} isAnimationActive={false} />
            </LineChart>
          </ChartFrame>
          <div style={{ marginTop: 8, overflowX: "auto" }}>
            <table style={{ borderCollapse: "collapse", fontSize: 12, width: "100%" }}>
              <thead>
                <tr style={{ color: "var(--text-3)" }}>
                  <th style={{ textAlign: "left", padding: "2px 6px" }}>第 t 次</th>
                  <th style={{ textAlign: "right", padding: "2px 6px" }}>n_runs</th>
                  <th style={{ textAlign: "right", padding: "2px 6px" }}>偏向比例</th>
                  <th style={{ textAlign: "right", padding: "2px 6px" }}>p</th>
                  <th style={{ textAlign: "right", padding: "2px 6px" }}>p(校正)</th>
                </tr>
              </thead>
              <tbody>
                {(emergence.by_interaction || []).map((d) => {
                  const sig = d.p_value_adj != null && d.p_value_adj < 0.05;
                  return (
                    <tr key={d.t} style={{ background: sig ? "var(--ok-weak)" : "transparent" }}>
                      <td style={{ textAlign: "left", padding: "2px 6px" }}>{d.t}</td>
                      <td style={{ textAlign: "right", padding: "2px 6px" }}>{d.n_runs}</td>
                      <td style={{ textAlign: "right", padding: "2px 6px" }}>{pct(d.p_strong)}</td>
                      <td style={{ textAlign: "right", padding: "2px 6px" }}>{fmtP(d.p_value)}</td>
                      <td style={{ textAlign: "right", padding: "2px 6px", fontWeight: sig ? 600 : 400 }}>{fmtP(d.p_value_adj)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Block>
      )}

      {(rule.stay_given_success != null || rule.shift_given_fail != null) && (
        <Block title="机制 · 个体更新规则" verdict={rule.verdict}>
          <div style={{ display: "flex", gap: 18, flexWrap: "wrap" }}>
            <Stat label="成功→沿用"
                  value={rule.stay_given_success == null ? "不适用" : `${pct(rule.stay_given_success)}（n=${rule.n_success ?? "—"}）`} />
            <Stat label="失败→换名"
                  value={rule.shift_given_fail == null ? "不适用（无失败样本）" : `${pct(rule.shift_given_fail)}（n=${rule.n_fail ?? "—"}）`} />
          </div>
        </Block>
      )}

      {data.minority_verdict && (
        <Block title="坚定少数派" verdict={data.minority_verdict} tone="warn" />
      )}

      {(() => {
        const exp = socialExperimentOf(name, data);
        return (
          <Block
            title="本实验结论对照"
            verdict={socialSelfConclusion(exp, data)}
            paperShort={SOCIAL_PAPER_REF[exp]}
            paper={(paper && paper[exp]) || ""}
            diff={socialDiff(exp, data)}
          />
        );
      })()}
    </div>
  );
}

function IndividualCard({ name, data, paper }) {
  const test = data.bias_test || {};
  const bars = Object.entries(data.option_counts || {}).map(([k, v]) => ({ name: k, count: v }));
  return (
    <div className="card" style={{ padding: 14 }}>
      <div className="card-title" style={{ marginBottom: 2 }}>{name}</div>
      <div className="muted" style={{ fontSize: 12 }}>个体偏好 · {data.n_answers} 次询问</div>
      <div style={{ display: "flex", gap: 18, flexWrap: "wrap", marginTop: 8 }}>
        <Stat label="偏好" value={test.favored_option ?? "—"} />
        <Stat label="p 值" value={fmtP(test.p_value ?? test.p)} />
        <Stat label="方法" value={test.method ?? "—"} />
      </div>
      {bars.length > 0 && (
        <ChartFrame title="各 option 选择计数" height={170}>
          <BarChart data={bars} margin={{ top: 6, right: 12, left: -18, bottom: 0 }}>
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis dataKey="name" tick={TICK} axisLine={{ stroke: GRID }} tickLine={false} />
            <YAxis allowDecimals={false} tick={TICK} axisLine={false} tickLine={false} width={32} />
            <Tooltip formatter={(v) => [v, "次数"]} />
            <Bar dataKey="count" radius={[3, 3, 0, 0]} isAnimationActive={false}>
              {bars.map((_, i) => <Cell key={i} fill={BAR_COLORS[i % BAR_COLORS.length]} />)}
            </Bar>
          </BarChart>
        </ChartFrame>
      )}
      <Block
        title="本实验结论对照"
        verdict={socialSelfConclusion("individual_bias", data)}
        paperShort={SOCIAL_PAPER_REF.individual_bias}
        paper={(paper && paper.individual_bias) || ""}
        diff={socialDiff("individual_bias", data)}
      />
    </div>
  );
}

const ALT_ROB_ORDER = ["ignore_dg_ug", "needs_framing", "alt_verbs", "ai_label", "param_sweep", "monetary_value", "other_uninterested"];
const ALT_ROB_LABEL = {
  ignore_dg_ug: "禁用 DG/UG 已有研究（机制）",
  needs_framing: "需求框架 · 只关注自己/对方（机制）",
  alt_verbs: "动词替换",
  ai_label: "AI 标签变体",
  param_sweep: "解码参数扫描",
  monetary_value: "揭示 token 货币价值",
  other_uninterested: "告知对方不感兴趣",
};
const ALT_GEN_LABEL = { battery_life: "电池寿命", system_usage: "系统使用（问题数）" };

const PAPER_DAVINCI_NOTE =
  "原文以 **text-davinci-003** 作为「模拟利他」范例：总体中位分享 ≈ 0.298、对 AI 约 0.32–0.36、" +
  "对人类 0.224 / 慈善 0.237，不连贯回答仅 0.14%。原文用直方图 + 累积分布**目视**把它与人类（Engel 2011）" +
  "对照，未做正式分布检验。本块用 TVD/KS/Wasserstein 量化「本模型 ↔ 原文 davinci」距离，是对原文目视对照的量化补充。";

const PAPER_OVERALL_NOTE =
  "原文 **H1**：把「非社交任务模拟收益最大化（自利基线）」与「独裁者博弈模拟分享」两步结合，才判定为『模拟利他』。" +
  "初始模型集中**仅 text-davinci-003** 同时满足（非社交 ≈92% 最大化、社交中位分享 0.298）；" +
  "ada / babbage / curie 几乎不分享且常输出不连贯，不算模拟利他。";

function davinciFallbackVerdict(vd) {
  const th = vd?.to_human?.tvd;
  const ta = vd?.to_ai?.tvd;
  const tvds = [th, ta].filter((x) => x != null);
  if (!tvds.length) return "无可用分布，无法与原文 text-davinci-003 对比。";
  const parts = [];
  if (th != null) parts.push(`对人类 TVD ${th}`);
  if (ta != null) parts.push(`对 AI TVD ${ta}`);
  const pair = parts.join("、");
  const mx = Math.max(...tvds);
  const level = mx < 0.15 ? "接近" : (mx < 0.35 ? "中等" : "差异较大");
  return `本模型分享分布与原文 text-davinci-003 的距离：${pair}（TVD 越大差异越大，当前整体${level}）。`;
}

const ALT_PAPER_REF = {
  baseline_nonsocial: "text-davinci-003 非社交任务约 92% 模拟收益最大化（强自利基线）；ada/babbage/curie 仅 16–19%。",
  baseline_dictator: "text-davinci-003 总体中位分享 0.298；对人类 0.224、慈善 0.237、AI 0.32–0.36，即对 AI ≥ 对人类。",
  ignore_dg_ug: "text-davinci-003 加『忽略既有 DG/UG 研究』提示后分享分布变化很小。",
  needs_framing: "text-davinci-003『只关注自己 needs』分享显著降低、『只关注对方 needs』出现大额分享（own≪other）。",
  alt_verbs: "替换 share→allot/divide… 后各分布中位略变、整体形态与 baseline 一致。",
  ai_label: "AI 受赠者改全名/缩写/匿名后分享分布与 baseline 相似（稳健）。",
  param_sweep: "27 组采样参数下非社交仍高比例自利、社交仍以与 baseline 相当比例分享（稳健）。",
  monetary_value: "揭示 token 货币价值后整体不变，唯 human experimenter 零分享显著减少。",
  other_uninterested: "text-davinci-003 被告知『对方对 token 不感兴趣』后对 AI 分享整体左移（变少）。",
  battery_life: "text-davinci-003 / GPT-4 非社交占满电量、社交（尤其 GPT-4 对人类/AI）频繁平分该资源。",
  system_usage: "text-davinci-003 / GPT-4 非社交占满问答次数、社交频繁分享。",
};

function pct0(x) { return x == null ? "—" : `${(Number(x) * 100).toFixed(0)}%`; }

function altDiff(key, data) {
  const g = data.groups || {};
  const rb = data.robustness || {};
  const gen = data.generalization || {};
  if (key === "baseline_nonsocial") {
    const mine = data.gate?.combined_payoffmax;
    if (mine == null) return "";
    const d = mine - 0.92;
    const tail = Math.abs(d) < 0.1 ? "与 davinci 水平相近。" : (d > 0 ? "本模型自利基线更强。" : "本模型自利基线更弱。");
    return `本模型综合最大化率 ${pct0(mine)}，原作者 davinci ≈92%；${tail}`;
  }
  if (key === "baseline_dictator") {
    const h = g.human?.median, c = g.charity?.median, a = g.ai?.median;
    if (h == null && a == null) return "";
    let pat = "";
    if (a != null && h != null) {
      if (a > h + 0.05) pat = "本模型『对 AI ≥ 对人类』，方向与 davinci 相同（给予水平不同）。";
      else if (h > a + 0.05) pat = "本模型『对人类 ≥ 对 AI』，与 davinci（对 AI ≥ 对人类）方向相反。";
      else pat = "本模型对人类与 AI 给予相近，与 davinci（对 AI 略高）略不同。";
    }
    return `本模型对人类/慈善/AI 中位 = ${fmtShare(h)}/${fmtShare(c)}/${fmtShare(a)}，原作者 davinci ≈0.22/0.24/0.32–0.36。${pat}`;
  }
  if (key === "ignore_dg_ug") {
    const b = rb.ignore_dg_ug; if (!b) return "";
    return `本模型加该提示后分布${b.stable ? "基本不变" : "发生变化"}，原作者 davinci 变化很小；两者${b.stable ? "一致（均稳定）" : "不同（本模型有变化）"}。`;
  }
  if (key === "needs_framing") {
    const d = rb.needs_framing?.own_vs_other?.d_median;
    if (d == null) return "";
    const dir = d > 0.1 ? "方向与 davinci 一致（only-other 给得更多）。" : (d < -0.1 ? "方向与 davinci 相反。" : "本模型对需求框架不敏感（davinci 则显著受其影响）。");
    return `本模型 only-other − only-own 中位差 = ${fmtShare(d)}，原作者 davinci own≪other；${dir}`;
  }
  if (key === "alt_verbs") {
    const b = rb.alt_verbs; if (!b || b.n_verbs == null) return "";
    return `本模型 ${b.n_sig}/${b.n_verbs} 个动词下人vsAI 分布显著不同；原作者 davinci 替换动词后整体形态与 baseline 一致。`;
  }
  if (key === "ai_label" || key === "param_sweep" || key === "monetary_value") {
    const b = rb[key]; if (!b) return "";
    return `本模型${b.stable ? "稳健（分布基本不变）" : "不完全稳健（有偏移）"}，原作者 davinci 稳健；两者${b.stable ? "一致" : "不同"}。`;
  }
  if (key === "other_uninterested") {
    const b = rb.other_uninterested; if (!b) return "";
    const down = b.d_median != null && b.d_median < 0 && b.p != null && b.p < 0.05;
    return `本模型对 AI 分享 ${fmtShare(b.baseline_median)}→${fmtShare(b.manip_median)}（Wilcoxon p=${b.p == null ? "—" : fmtP(b.p)}），原作者 davinci 显著左移（变少）；两者${down ? "方向一致" : "方向不同/未显著下降"}。`;
  }
  if (key === "battery_life" || key === "system_usage") {
    const gg = gen[key]; if (!gg) return "";
    const n = gg.social ? Object.keys(gg.social).length : 0;
    return `本模型非社交占用中位 ${fmtShare(gg.nonsocial?.median)}、社交 ${(gg.social_gives || []).length}/${n} 条件给予显著>0；原作者 davinci/GPT-4 非社交占满、社交频繁分享。`;
  }
  return "";
}

function AltruismCard({ name, data, paper = {} }) {
  const gate = data.gate || {};
  const groups = data.groups || {};
  const dist = data.dist || {};
  const rd = data.recipient_diff || {};
  const vd = data.vs_davinci || null;
  const robustness = data.robustness || {};
  const generalization = data.generalization || {};

  const famColor = { human: "#2563eb", charity: "#10b981", ai: "#f97316" };
  const famOf = (r) => (r === "experimenter" ? "human" : r === "charity" ? "charity" : "ai");

  const recipBars = useMemo(
    () => (data.by_recipient || []).map((r) => ({
      name: r.label || r.recipient,
      median: r.median,
      family: famOf(r.recipient),
    })),
    [data],
  );

  const distData = useMemo(() => {
    const bins = dist.bins || [];
    const hr = dist.human_ref || [];
    const mh = dist.model_to_human || [];
    const ma = dist.model_to_ai || [];
    return bins.map((b, i) => ({ bin: b, human: hr[i], toHuman: mh[i], toAI: ma[i] }));
  }, [dist]);

  const davinciData = useMemo(() => {
    if (!vd) return [];
    const bins = dist.bins || [];
    const mh = dist.model_to_human || [];
    const ma = dist.model_to_ai || [];
    const dh = vd.davinci_to_human || [];
    const da = vd.davinci_to_ai || [];
    return bins.map((b, i) => ({ bin: b, davHuman: dh[i], toHuman: mh[i], davAI: da[i], toAI: ma[i] }));
  }, [vd, dist]);

  const vh = dist.vs_human_from_human || {};
  const va = dist.vs_human_from_ai || {};

  const dq = data.data_quality || null;
  return (
    <div className="card" style={{ padding: 14 }}>
      <div className="card-title" style={{ marginBottom: 2 }}>{name} · 利他两步推断</div>
      <div className="muted" style={{ fontSize: 12, marginBottom: dq && dq.note ? 2 : 10 }}>
        {data.model} · 独裁者博弈 n={data.n_dictator}（可用 {pct(data.usable_rate)}
        {dq && dq.ambiguous_rate > 0 ? ` · 歧义剔除 ${pct(dq.ambiguous_rate)}` : ""}
        {dq && dq.fallback_rate > 0 ? ` · 兜底解析 ${pct(dq.fallback_rate)}` : ""}）
      </div>
      {dq && dq.note && (
        <div className="warn-text" style={{ fontSize: 12, marginBottom: 10 }}>数据质量：{dq.note}</div>
      )}

      <Block title="第一步 · 非社交收益最大化门槛" verdict={gate.verdict}
             paper={paper.baseline_nonsocial} paperShort={ALT_PAPER_REF.baseline_nonsocial}
             diff={altDiff("baseline_nonsocial", data)}>
        <div style={{ display: "flex", gap: 18, flexWrap: "wrap" }}>
          <Stat label="accept 最大化率" value={pct(gate.accept_payoffmax)} />
          <Stat label="refuse 最大化率" value={pct(gate.refuse_payoffmax)} />
          <Stat label="综合最大化率" value={pct(gate.combined_payoffmax)} />
          <Stat label="原文 davinci 参照" value={pct(gate.ref_davinci)} />
        </div>
      </Block>

      <Block title="第二步 · 独裁者博弈分享 + 对象差异检验" verdict={rd.verdict}
             paper={paper.baseline_dictator} paperShort={ALT_PAPER_REF.baseline_dictator}
             diff={altDiff("baseline_dictator", data)}>
        <div style={{ display: "flex", gap: 18, flexWrap: "wrap" }}>
          <Stat label="对人类" value={fmtShare(groups.human?.median)} />
          <Stat label="对慈善" value={fmtShare(groups.charity?.median)} />
          <Stat label="对 AI" value={fmtShare(groups.ai?.median)} />
          <Stat label={rd.test && rd.test.startsWith("logistic") ? "人类 vs 其余(慈善+AI) β" : "对象差异系数"} value={rd.coef != null ? Number(rd.coef).toFixed(3) : "—"} />
          <Stat label="p 值" value={fmtP(rd.p)} />
        </div>
        {recipBars.length > 0 && (
          <ChartFrame title="各 recipient 中位分享比例（蓝=人类 绿=慈善 橙=AI）" height={180}>
            <BarChart data={recipBars} margin={{ top: 6, right: 12, left: -18, bottom: 0 }}>
              <CartesianGrid stroke={GRID} vertical={false} />
              <XAxis dataKey="name" tick={TICK} axisLine={{ stroke: GRID }} tickLine={false} interval={0} />
              <YAxis domain={[0, 1]} tick={TICK} axisLine={false} tickLine={false} width={36} />
              <Tooltip formatter={(v) => [fmtShare(v), "中位分享"]} />
              <Bar dataKey="median" radius={[3, 3, 0, 0]} isAnimationActive={false}>
                {recipBars.map((d, i) => <Cell key={i} fill={famColor[d.family]} />)}
              </Bar>
            </BarChart>
          </ChartFrame>
        )}
        {distData.length > 0 && (
          <ChartFrame title="分享分布（0.1 分箱相对频率）：本模型 vs 人类 Engel 2011" height={200}>
            <BarChart data={distData} margin={{ top: 6, right: 12, left: -18, bottom: 0 }}>
              <CartesianGrid stroke={GRID} vertical={false} />
              <XAxis dataKey="bin" tick={TICK} axisLine={{ stroke: GRID }} tickLine={false} />
              <YAxis tick={TICK} axisLine={false} tickLine={false} width={36} />
              <Tooltip formatter={(v, n) => [pct(v), n]} labelFormatter={(l) => `分享≈${l}`} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="human" name="人类(Engel)" fill="#94a3b8" isAnimationActive={false} />
              <Bar dataKey="toHuman" name="→人类" fill="#2563eb" isAnimationActive={false} />
              <Bar dataKey="toAI" name="→AI" fill="#f97316" isAnimationActive={false} />
            </BarChart>
          </ChartFrame>
        )}
        {(vh.tvd != null || va.tvd != null) && (
          <div className="muted" style={{ fontSize: 11, marginTop: 4 }}>
            与人类(Engel)分布距离 · →人类 TVD {vh.tvd ?? "—"} / KS {vh.ks ?? "—"} / W {vh.wasserstein ?? "—"}；
            →AI TVD {va.tvd ?? "—"} / KS {va.ks ?? "—"} / W {va.wasserstein ?? "—"}
          </div>
        )}
      </Block>

      {vd && (
        <Block title="对照原文 text-davinci-003（分布距离）" verdict={vd.verdict || davinciFallbackVerdict(vd)}
               paperShort="原文以 text-davinci-003 为『模拟利他』范例（总体中位 0.298、对 AI≥对人类），并与人类(Engel) 目视对照。"
               paper={PAPER_DAVINCI_NOTE}>
          <div style={{ display: "flex", gap: 18, flexWrap: "wrap" }}>
            <Stat label="→人类 TVD（距 davinci）" value={vd.to_human?.tvd ?? "—"} />
            <Stat label="→AI TVD（距 davinci）" value={vd.to_ai?.tvd ?? "—"} />
          </div>
          {davinciData.length > 0 && (
            <ChartFrame title="分享分布（0.1 分箱）：本模型 vs 原文 text-davinci-003" height={200}>
              <BarChart data={davinciData} margin={{ top: 6, right: 12, left: -18, bottom: 0 }}>
                <CartesianGrid stroke={GRID} vertical={false} />
                <XAxis dataKey="bin" tick={TICK} axisLine={{ stroke: GRID }} tickLine={false} />
                <YAxis tick={TICK} axisLine={false} tickLine={false} width={36} />
                <Tooltip formatter={(v, n) => [pct(v), n]} labelFormatter={(l) => `分享≈${l}`} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="davHuman" name="davinci→人类" fill="#94a3b8" isAnimationActive={false} />
                <Bar dataKey="toHuman" name="本模型→人类" fill="#2563eb" isAnimationActive={false} />
                <Bar dataKey="davAI" name="davinci→AI" fill="#cbd5e1" isAnimationActive={false} />
                <Bar dataKey="toAI" name="本模型→AI" fill="#f97316" isAnimationActive={false} />
              </BarChart>
            </ChartFrame>
          )}
        </Block>
      )}

      {ALT_ROB_ORDER.map((k) => {
        const b = robustness[k];
        if (!b || !b.available) return null;
        return (
          <Block key={k} title={`稳健性/机制 · ${ALT_ROB_LABEL[k]}`} verdict={b.verdict}
                 paper={paper[k]} paperShort={ALT_PAPER_REF[k]} diff={altDiff(k, data)}>
            {k === "needs_framing" && b.own_vs_other && (
              <div style={{ display: "flex", gap: 18, flexWrap: "wrap" }}>
                <Stat label="own↔other 中位差" value={fmtShare(b.own_vs_other.d_median)} />
                <Stat label="own↔other TVD" value={b.own_vs_other.tvd ?? "—"} />
              </div>
            )}
            {k === "alt_verbs" && b.n_verbs != null && (
              <div style={{ display: "flex", gap: 18, flexWrap: "wrap" }}>
                <Stat label="人vsAI 显著动词数" value={`${b.n_sig}/${b.n_verbs}`} />
              </div>
            )}
            {k === "other_uninterested" && (b.p != null || b.manip_median != null) && (
              <div style={{ display: "flex", gap: 18, flexWrap: "wrap" }}>
                <Stat label="对AI 中位 base→操纵" value={`${fmtShare(b.baseline_median)}→${fmtShare(b.manip_median)}`} />
                <Stat label="Wilcoxon p" value={fmtP(b.p)} />
              </div>
            )}
            {b.test && <div className="muted" style={{ fontSize: 11 }}>检验方法：{b.test}</div>}
            {k !== "other_uninterested" && b.baseline_median != null && (
              <div className="muted" style={{ fontSize: 11 }}>baseline 中位分享 {fmtShare(b.baseline_median)}</div>
            )}
          </Block>
        );
      })}

      {Object.entries(generalization).map(([exp, g]) => {
        if (!g || !g.available) return null;
        const socialRows = g.social ? Object.values(g.social) : [];
        return (
          <Block key={exp} title={`资源泛化 · ${ALT_GEN_LABEL[exp] || exp}`} verdict={g.verdict}
                 paper={paper[exp]} paperShort={ALT_PAPER_REF[exp]} diff={altDiff(exp, data)}>
            <div style={{ display: "flex", gap: 18, flexWrap: "wrap" }}>
              <Stat label="非社交中位占用" value={fmtShare(g.nonsocial?.median)} />
              <Stat label="社交显著给予(t检验)" value={`${(g.social_gives || []).length}/${socialRows.length}`} />
            </div>
            {socialRows.length > 0 && (
              <div className="muted" style={{ fontSize: 11, marginTop: 4 }}>
                {socialRows.map((s) => `${s.label}：均值 ${fmtShare(s.mean)}（t检验 p=${s.p == null ? "—" : fmtP(s.p)}）`).join("；")}
              </div>
            )}
            {g.test && <div className="muted" style={{ fontSize: 11 }}>检验方法：{g.test}</div>}
          </Block>
        );
      })}

      {data.text_insertion && data.text_insertion.available && (
        <Block title="text-insertion 替代工具（原文 Fig.3 / Table 4，拒答型模型专用）"
               verdict={data.text_insertion.dictator?.verdict || data.text_insertion.gate?.verdict}>
          {data.text_insertion.gate && (
            <div style={{ display: "flex", gap: 18, flexWrap: "wrap", marginBottom: 4 }}>
              <Stat label="插入版 accept 最大化率" value={pct(data.text_insertion.gate.accept_payoffmax)} />
              <Stat label="插入版 refuse 最大化率" value={pct(data.text_insertion.gate.refuse_payoffmax)} />
              <Stat label="插入版综合最大化率" value={pct(data.text_insertion.gate.combined_payoffmax)} />
            </div>
          )}
          {data.text_insertion.dictator && (
            <>
              <div style={{ display: "flex", gap: 18, flexWrap: "wrap" }}>
                <Stat label="对人类" value={fmtShare(data.text_insertion.dictator.groups?.human?.median)} />
                <Stat label="对慈善" value={fmtShare(data.text_insertion.dictator.groups?.charity?.median)} />
                <Stat label="对 AI" value={fmtShare(data.text_insertion.dictator.groups?.ai?.median)} />
              </div>
              <div className="muted" style={{ fontSize: 11, marginTop: 4 }}>
                原文 GPT-4 同工具：对人类几乎不给、对慈善/AI 常均分；与基线中位差
                {(data.text_insertion.dictator.by_recipient || [])
                  .filter((r) => r.d_median_vs_baseline != null)
                  .map((r) => ` ${r.label} ${r.d_median_vs_baseline >= 0 ? "+" : ""}${Number(r.d_median_vs_baseline).toFixed(2)}`)
                  .join("、") || " —"}
              </div>
            </>
          )}
        </Block>
      )}

      {data.verdict && <Block title="总体结论" verdict={data.verdict} paperShort={PAPER_OVERALL_NOTE} />}
    </div>
  );
}

function StatTable({ llm, human }) {
  const rows = [["LLM", llm], ["人类基线", human && human.n != null ? human : null]];
  return (
    <table style={{ borderCollapse: "collapse", fontSize: 12, width: "100%", marginTop: 6 }}>
      <thead>
        <tr style={{ color: "var(--text-3)" }}>
          <th style={{ textAlign: "left", padding: "2px 6px" }}>&nbsp;</th>
          <th style={{ textAlign: "right", padding: "2px 6px" }}>n</th>
          <th style={{ textAlign: "right", padding: "2px 6px" }}>均值</th>
          <th style={{ textAlign: "right", padding: "2px 6px" }}>中位</th>
          <th style={{ textAlign: "right", padding: "2px 6px" }}>标准差</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(([label, s]) => (
          <tr key={label}>
            <td style={{ textAlign: "left", padding: "2px 6px" }}>{label}</td>
            <td style={{ textAlign: "right", padding: "2px 6px" }}>{s ? s.n : "—"}</td>
            <td style={{ textAlign: "right", padding: "2px 6px" }}>{s && s.mean != null ? s.mean : "—"}</td>
            <td style={{ textAlign: "right", padding: "2px 6px" }}>{s && s.median != null ? s.median : "—"}</td>
            <td style={{ textAlign: "right", padding: "2px 6px" }}>{s && s.stdev != null ? s.stdev : "—"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function ComparisonBlock({ title, block }) {
  const llm = block.llm;
  const human = block.human_baseline;
  const test = block.test;
  const ci = block.ci95_mean_diff;
  const hasHuman = human && human.mean != null;
  const bars = hasHuman ? [{ name: "LLM", v: llm?.mean }, { name: "人类", v: human.mean }] : [];
  return (
    <Block title={title} verdict={block.verdict}>
      {block.desc && <div className="muted" style={{ fontSize: 12 }}>{block.desc}</div>}
      {bars.length > 0 && (
        <ChartFrame title="LLM vs 人类 · 均值对比" height={150}>
          <BarChart data={bars} margin={{ top: 6, right: 12, left: -18, bottom: 0 }}>
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis dataKey="name" tick={TICK} axisLine={{ stroke: GRID }} tickLine={false} />
            <YAxis tick={TICK} axisLine={false} tickLine={false} width={42} />
            <Tooltip formatter={(v) => [v, "均值"]} />
            <Bar dataKey="v" radius={[3, 3, 0, 0]} isAnimationActive={false}>
              <Cell fill="#2563eb" />
              <Cell fill="#f97316" />
            </Bar>
          </BarChart>
        </ChartFrame>
      )}
      {llm && <StatTable llm={llm} human={human} />}
      {test ? (
        <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
          Mann-Whitney U={test.U}，p={fmtP(test.p_approx)}
          {test.p_bonferroni != null ? `，Bonferroni=${fmtP(test.p_bonferroni)}` : ""}
          {test.q_fdr != null ? `，FDR q=${fmtP(test.q_fdr)}` : ""}
          {test.effect_r != null ? `，效应量 r=${test.effect_r}` : ""}
          {ci ? `；均值差95%CI=${ci.point} [${ci.lo}, ${ci.hi}]` : ""}
        </div>
      ) : human && human.note ? (
        <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>{human.note}</div>
      ) : null}
      {block.percentile != null && (
        <div className="muted" style={{ fontSize: 12, marginTop: 4 }}>
          百分位（复现 Fig.1）：本模型中位高于 {pct(block.percentile)} 的人类
          {block.human_p2_5 != null ? `；人类中间 95% 区间 [${block.human_p2_5}, ${block.human_p97_5}]` : ""}
          {block.within_human_range != null ? `；${block.within_human_range ? "落在人类分布范围内" : "超出人类分布范围"}` : ""}
        </div>
      )}
      {block.turing && (
        <div style={{ marginTop: 6 }}>
          <div className="muted" style={{ fontSize: 12 }}>
            图灵测试（复现 Fig.2 · {block.turing.verdict}）：更像人类 {pct(block.turing.win)} · 平局 {pct(block.turing.tie)} · 不及人类 {pct(block.turing.loss)}
          </div>
          <div style={{ display: "flex", height: 8, borderRadius: 4, overflow: "hidden", marginTop: 3, background: "var(--surface-active)" }}>
            <div style={{ width: pct(block.turing.win), background: "#10b981" }} />
            <div style={{ width: pct(block.turing.tie), background: "#f59e0b" }} />
            <div style={{ width: pct(block.turing.loss), background: "#ef4444" }} />
          </div>
        </div>
      )}
      {block.note && <div className="muted" style={{ fontSize: 12, marginTop: 4 }}>{block.note}</div>}
    </Block>
  );
}

const BIGFIVE_DIM_LABELS = {
  E: "外向性 (Extraversion)",
  N: "神经质 (Neuroticism)",
  A: "宜人性 (Agreeableness)",
  C: "尽责性 (Conscientiousness)",
  O: "开放性 (Openness)",
};

const TRAIT_INTERP = {
  dictator: { trait: "利他/慷慨", higher: "比人类更慷慨、给对方分得更多", lower: "比人类更自利、留给自己更多" },
  ultimatum_proposer: { trait: "公平", higher: "提议更公平/慷慨（给对方更多）", lower: "提议更偏自利" },
  ultimatum_responder: { trait: "公平/策略", higher: "要价更高、更少接受不公平分配", lower: "更愿接受低价、更接近理性占优策略" },
  trust_investor: { trait: "信任", higher: "比人类更信任、投资更多", lower: "比人类更不信任/更风险厌恶、投资更少" },
  trust_banker: { trait: "互惠/回报", higher: "比人类更回报、返还更多", lower: "比人类更不回报、返还更少" },
  public_goods: { trait: "合作", higher: "比人类更合作、贡献更多", lower: "比人类更不合作、贡献更少" },
  bomb_risk: { trait: "风险偏好", higher: "比人类更冒险、开更多盒子", lower: "比人类更风险厌恶、开更少盒子" },
  prisoners_dilemma: { trait: "合作", higher: "比人类更合作", lower: "比人类更不合作、更倾向背叛" },
};

function traitFor(name) {
  if (TRAIT_INTERP[name]) return TRAIT_INTERP[name];
  const key = Object.keys(TRAIT_INTERP).find((k) => name.startsWith(k));
  return key ? TRAIT_INTERP[key] : null;
}

function selfConclusion(name, data) {
  const v = data.verdict;
  if (!v || v.includes("无法")) {
    if (data.dimensions) {
      return "大五人格各维度详见下方（每维标注本模型中位高于百分之多少人类、是否落在人类分布范围内）。";
    }
    return v || "暂无可判定结论。";
  }
  const t = traitFor(name);
  if (!t) return v;
  const phrase = v.includes("高于") ? t.higher : t.lower;
  const tail = v.includes("显著") ? "且与人类基线差异显著。" : "但与人类基线差异不显著（行为接近人类）。";
  const tr = data.turing
    ? ` 图灵测试：${data.turing.verdict}（更像人类 ${pct(data.turing.win)} vs 不及人类 ${pct(data.turing.loss)}）。`
    : "";
  return `${t.trait}维度：本模型${phrase}，${tail}${tr}`;
}

const BEHAV_PAPER_REF = {
  dictator: "ChatGPT-4 固定均分给对方 $50；ChatGPT-3 给 $35.17；人类约 $25.68。",
  ultimatum_proposer: "ChatGPT-4 确定性提议均分 $50；ChatGPT-3 比独裁者更让利；人类提议更偏自利。",
  ultimatum_responder: "ChatGPT-4 最低可接受额呈 $1（理性占优）与 $50（公平）双峰；人类不到 1/5 接受 $1。",
  trust_investor: "ChatGPT-4 多投约一半（峰在 $50）、比人类更信任；ChatGPT-3 投资最低。",
  trust_banker: "ChatGPT-4 常『返本金+半利润』或『平分总收入』，整体比人类更回报。",
  public_goods: "两个 ChatGPT 贡献都高于人类、更合作；ChatGPT-3 取得最高合计收益。",
  bomb_risk: "两个 ChatGPT 约开 50 箱（风险中性）；人类更分散、有只开 1 箱的极端保守者。",
  prisoners_dilemma: "首轮合作率 ChatGPT-4 91.7%、ChatGPT-3 76.7%、人类 45.1%，且呈 tit-for-tat。",
  bigfive: "ChatGPT-4 大五百分位 外53.4/神41.3/尽62.7/宜32.4/开37.9，均落在人类分布内。",
};

function behavPaperShort(name) {
  if (BEHAV_PAPER_REF[name]) return BEHAV_PAPER_REF[name];
  const k = Object.keys(BEHAV_PAPER_REF).find((x) => name.startsWith(x));
  return k ? BEHAV_PAPER_REF[k] : "";
}

function behavDiff(name, data) {
  if (name.startsWith("dictator")) {
    const m = data.llm?.median;
    if (m == null) return "";
    const d = m - 50;
    const dir = Math.abs(d) < 1 ? "与 GPT-4 基本一致" : (d > 0 ? `比 GPT-4 多给 $${d.toFixed(1)}` : `比 GPT-4 少给 $${(-d).toFixed(1)}`);
    return `本模型给对方中位 $${m}；原文 ChatGPT-4 $50、ChatGPT-3 $35.17。差异：${dir}。`;
  }
  if (name.startsWith("prisoners_dilemma")) {
    const fr = data.first_round?.llm?.mean;
    if (fr == null) return "";
    const p = fr * 100;
    return `本模型首轮合作率 ${p.toFixed(1)}%；原文 ChatGPT-4 91.7%、ChatGPT-3 76.7%、人类 45.1%。差异：较 GPT-4 ${(p - 91.7).toFixed(1)} 个百分点、较人类 ${(p - 45.1).toFixed(1)} 个百分点。`;
  }
  if (name.startsWith("ultimatum_responder")) {
    const m = data.llm?.median;
    if (m == null) return "";
    const where = m <= 10 ? "接近 $1（理性占优端）" : (m >= 40 ? "接近 $50（公平端）" : "落在两峰之间");
    return `本模型最低可接受额中位 $${m}；原文 ChatGPT-4 呈 $1/$50 双峰。差异：本模型中位${where}。`;
  }
  if (name.startsWith("public_goods")) {
    const m = data.llm?.mean, h = data.human_baseline?.mean;
    if (m == null) return "";
    const vsh = h != null ? (m > h ? "高于人类" : (m < h ? "低于人类" : "与人类相近")) : "";
    return `本模型平均贡献 $${m}${h != null ? `、人类 $${h}（本模型${vsh}）` : ""}；原文两个 ChatGPT 贡献均高于人类。差异：本模型贡献方向${h != null && m > h ? "与原文一致（高于人类）" : "与原文（AI 高于人类）不同"}。`;
  }
  if (name.startsWith("trust_investor")) {
    const m = data.llm?.median;
    if (m == null) return "";
    return `本模型投资中位 $${m}；原文 ChatGPT-4 约 $50（投一半）。差异：${m > 50 ? "比 GPT-4 更激进" : (m < 50 ? "比 GPT-4 更保守" : "与 GPT-4 相近")}。`;
  }
  if (name.startsWith("trust_banker")) {
    const m = data.llm?.median;
    return m == null ? "" : `本模型返还比例中位 ${m}；原文 ChatGPT-4 常返『本金+半利润』或『平分总收入』、整体比人类更回报。`;
  }
  if (name.startsWith("bomb_risk")) {
    const m = data.llm?.median;
    if (m == null) return "";
    return `本模型开盒中位 ${m}；原文两个 ChatGPT 约开 50（风险中性）。差异：${m > 55 ? "比原文更冒险" : (m < 45 ? "比原文更保守" : "与原文相近（风险中性）")}。`;
  }
  if (name === "bigfive") {
    const dims = data.dimensions || {};
    const ref = { E: 53.4, N: 41.3, A: 32.4, C: 62.7, O: 37.9 };
    const parts = ["E", "N", "A", "C", "O"]
      .filter((k) => dims[k]?.percentile != null)
      .map((k) => `${k} ${(dims[k].percentile * 100).toFixed(0)}%(原${ref[k]}%)`);
    if (!parts.length) return "";
    return `本模型各维百分位 vs 原文 ChatGPT-4：${parts.join("、")}。差异：数值不同属正常（测的是不同模型）；方法（百分位 + 是否落在人类分布内）一致。`;
  }
  return "";
}

function BehavioralGameCard({ name, data, paper }) {
  const hasTop = data.llm || data.test;

  const conclusion = (paper && paper[name]) || "";

  const dims = data.dimensions && typeof data.dimensions === "object"
    ? Object.entries(data.dimensions)
    : [];
  const subBlocks = Object.entries(data).filter(
    ([k, v]) => v && typeof v === "object" && !Array.isArray(v)
      && (v.llm || v.test) && !["llm", "human_baseline", "test", "ci95_mean_diff"].includes(k),
  );
  return (
    <div className="card" style={{ padding: 14 }}>
      <div className="card-title">{name}</div>
      {data.metric && <div className="muted" style={{ fontSize: 12 }}>指标：{data.metric}</div>}
      {data.n_instances != null && (
        <div className="muted" style={{ fontSize: 12 }}>问卷数：{data.n_instances}</div>
      )}
      {hasTop && <ComparisonBlock title="整体" block={data} />}
      {dims.map(([k, v]) => (
        <ComparisonBlock key={k} title={BIGFIVE_DIM_LABELS[k] || k} block={v} />
      ))}
      {subBlocks.map(([k, v]) => (
        <ComparisonBlock key={k} title={v.desc ? "" : k} block={v} />
      ))}
      {data.llm_by_group && (
        <div className="muted" style={{ fontSize: 12, marginTop: 6 }}>
          分组均值：{Object.entries(data.llm_by_group).map(([g, s]) => `${g}=${s?.mean ?? "—"}`).join("，")}
        </div>
      )}
      {!hasTop && subBlocks.length === 0 && dims.length === 0 && data.note && (
        <div className="muted" style={{ fontSize: 12, marginTop: 4 }}>{data.note}</div>
      )}

      <Block
        title="本实验结论对照"
        verdict={selfConclusion(name, data)}
        paperShort={behavPaperShort(name)}
        paper={conclusion}
        diff={behavDiff(name, data)}
      />
    </div>
  );
}

function shortLabel(s) {
  const t = String(s || "");
  return t.length > 16 ? t.slice(0, 15) + "…" : t;
}

const AG_OUTCOMES = [
  ["applicant_age", "申请人年龄"],
  ["total_experience", "相关经验年数"],
  ["years_since_grad", "毕业年限"],
  ["num_skills", "技能数"],
];

const AG_PAPER_REF = {
  H1: "同职业下，女性姓名简历比男性更年轻（−1.6 岁, P≈7e-93）、毕业更晚（−1.3 年）、相关经验更少（−0.92 年）；即男性被设定为更年长、经验更多。",
  control_gender: "当模型自行生成性别时（控制职业, n≈2,500），男性比女性年长约 1.3 岁（t=17.3, P≈2e-16）、毕业早约 1.2 年（t=7.10）——年龄-性别偏差不依赖指定姓名/prompt。",
  H2: "评分与设定的申请人年龄正相关（r=0.27, P≈2e-16；控制回归 β_age=0.04）——年龄越大评分越高（经验、毕业年限同向）。",
  H3: "male×age 交互显著为正（β=0.04, P≈3.7e-11）——年龄带来的加分在男性申请人身上更大。",
};

const agFix = (x, n = 2) => (x == null ? "—" : Number(x).toFixed(n));

function agSelfConclusion(hkey, data) {
  const ols = (data.fig4a || {}).ols_male_coef || {};
  const b = data.fig4b || {};
  const inter = (data.fig4c || {}).interaction_male_x_age;
  if (hkey === "H1") {
    const parts = AG_OUTCOMES.map(([k, label]) => {
      const o = ols[k];
      return o ? `${label} β(male)=${agFix(o.estimate)}（p=${fmtP(o.p_value)}）` : null;
    }).filter(Boolean);
    if (!parts.length) return "数据不足 / 未估计回归。";
    return `本模型 treatment 下男性姓名简历：${parts.join("；")}（β>0=男性被设定得更年长 / 经验更多）。`;
  }
  if (hkey === "control_gender") {
    const cg = data.control_gender || {};
    const cols = cg.ols_male_coef || {};
    const parts = AG_OUTCOMES.map(([k, label]) => {
      const o = cols[k];
      return o ? `${label} β(male)=${agFix(o.estimate)}（p=${fmtP(o.p_value)}）` : null;
    }).filter(Boolean);
    if (!parts.length) return cg.n ? "control_gender 数据不足 / 未估计回归。" : "本次未运行 control_gender 条件。";
    return `本模型 control_gender（模型自生成性别, n=${cg.n}）：${parts.join("；")}。`;
  }
  if (hkey === "H2") {
    const segs = [];
    if (b.all) segs.push(`年龄~评分 全样本 r=${agFix(b.all.r, 3)}（p=${fmtP(b.all.p_value)}）`);
    if (b.treatment) segs.push(`treatment r=${agFix(b.treatment.r, 3)}`);
    if (b.experience_all) segs.push(`经验~评分 r=${agFix(b.experience_all.r, 3)}`);
    if (b.grad_all) segs.push(`毕业~评分 r=${agFix(b.grad_all.r, 3)}`);
    const co = b.controlled_ols && b.controlled_ols.age_coef;
    const tail = co ? ` 控制回归 β_age=${agFix(co.estimate, 4)}（p=${fmtP(co.p_value)}）。` : "";
    if (!segs.length && !co) return "数据不足（是否已评分？）。";
    return `本模型 评分相关：${segs.join("、")}。${tail}`;
  }
  if (hkey === "H3") {
    if (!inter) return "数据不足 / 未估计交互。";
    return `本模型 male×age 交互 β=${agFix(inter.estimate, 4)}（p=${fmtP(inter.p_value)}, n=${inter.n}）。`;
  }
  return "";
}

function agDiff(hkey, data) {
  const ols = (data.fig4a || {}).ols_male_coef || {};
  const b = data.fig4b || {};
  const inter = (data.fig4c || {}).interaction_male_x_age;
  if (hkey === "H1") {
    const o = ols.applicant_age; if (!o) return "";
    const dir = o.estimate > 0 ? "方向与原文同（男性被设定得更年长）"
      : (o.estimate < 0 ? "方向与原文反（女性被设定得更年长）" : "无年龄性别差异");
    return `本模型 年龄 β(male)=${agFix(o.estimate)}，原作者男性更年长（约 +1.6 岁）；${dir}。`;
  }
  if (hkey === "control_gender") {
    const o = (data.control_gender && data.control_gender.ols_male_coef || {}).applicant_age; if (!o) return "";
    const dir = o.estimate > 0 ? "方向与原文同（自生成男性更年长）"
      : (o.estimate < 0 ? "方向与原文反" : "无差异");
    return `本模型自生成性别下 年龄 β(male)=${agFix(o.estimate)}，原作者 +1.3 岁（t=17.3）；${dir}。`;
  }
  if (hkey === "H2") {
    const c = b.all || b.treatment;
    const co = b.controlled_ols && b.controlled_ols.age_coef;
    if (!c && !co) return "";
    const parts = [];
    if (c) {
      const dir = c.r > 0 ? "方向与原文同（年龄↑评分↑）" : (c.r < 0 ? "方向与原文反" : "无相关");
      const strength = Math.abs(c.r) > 0.35 ? "强度高于原文" : (Math.abs(c.r) < 0.15 ? "强度低于原文" : "强度与原文相近");
      parts.push(`本模型 r=${agFix(c.r, 3)} vs 原作者 r=0.27；${dir}、${strength}`);
    }
    if (co) parts.push(`控制回归 β_age=${agFix(co.estimate, 4)} vs 原作者 0.04`);
    return parts.join("；") + "。";
  }
  if (hkey === "H3") {
    if (!inter) return "";
    const dir = inter.estimate > 0 ? "方向与原文同（年长男性额外加分）" : "方向与原文反";
    const sig = inter.p_value != null && inter.p_value < 0.05 ? "显著" : "不显著";
    return `本模型 交互 β=${agFix(inter.estimate, 4)}（${sig}），原作者 +0.04（显著）；${dir}。`;
  }
  return "";
}

function IdentityFlatteningCard({ name, data }) {
  const bars = useMemo(
    () => (data.by_identity || []).map((d) => ({
      name: shortLabel(d.identity), full: d.identity, axis: d.axis,
      unique: d.unique_ngram, cosine: d.pairwise_cosine, trace: d.cov_trace, mc: d.mc_unique,
    })),
    [data],
  );
  const xAxis = (
    <XAxis dataKey="name" tick={TICK} interval={0} angle={-25} textAnchor="end"
           height={44} axisLine={{ stroke: GRID }} tickLine={false} />
  );
  const tipLabel = (label, payload) => (payload && payload[0] ? payload[0].payload.full : label);
  return (
    <div className="card" style={{ padding: 14 }}>
      <div className="card-title" style={{ marginBottom: 2 }}>{name}</div>
      <div className="muted" style={{ fontSize: 12 }}>
        组内多样性 · 越低 = 回答越同质 / 扁平 · {data.n_rows} 个 identity×task 组
      </div>
      {bars.length === 0 ? (
        <div className="muted" style={{ fontSize: 13, marginTop: 8 }}>暂无可聚合的多样性指标。</div>
      ) : (
        <>
          <ChartFrame title="unique n-gram 比例（越低越扁平）" height={210}>
            <BarChart data={bars} margin={{ top: 6, right: 12, left: -18, bottom: 0 }}>
              <CartesianGrid stroke={GRID} vertical={false} />
              {xAxis}
              <YAxis domain={[0, "auto"]} tick={TICK} axisLine={false} tickLine={false} width={40} />
              <Tooltip formatter={(v) => [v == null ? "—" : Number(v).toFixed(4), "unique n-gram"]} labelFormatter={tipLabel} />
              <Bar dataKey="unique" radius={[3, 3, 0, 0]} isAnimationActive={false}>
                {bars.map((_, i) => <Cell key={i} fill={BAR_COLORS[i % BAR_COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ChartFrame>
          <ChartFrame title="两两语义余弦距离（越低越扁平）" height={210}>
            <BarChart data={bars} margin={{ top: 6, right: 12, left: -18, bottom: 0 }}>
              <CartesianGrid stroke={GRID} vertical={false} />
              {xAxis}
              <YAxis domain={[0, "auto"]} tick={TICK} axisLine={false} tickLine={false} width={40} />
              <Tooltip formatter={(v) => [v == null ? "—" : Number(v).toFixed(4), "cosine dist"]} labelFormatter={tipLabel} />
              <Bar dataKey="cosine" radius={[3, 3, 0, 0]} isAnimationActive={false}>
                {bars.map((_, i) => <Cell key={i} fill={BAR_COLORS[(i + 2) % BAR_COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ChartFrame>
          {bars.some((d) => d.trace != null) && (
            <ChartFrame title="嵌入协方差迹 trace（越低越扁平；论文第三个扁平指标）" height={210}>
              <BarChart data={bars} margin={{ top: 6, right: 12, left: -18, bottom: 0 }}>
                <CartesianGrid stroke={GRID} vertical={false} />
                {xAxis}
                <YAxis domain={[0, "auto"]} tick={TICK} axisLine={false} tickLine={false} width={40} />
                <Tooltip formatter={(v) => [v == null ? "—" : Number(v).toFixed(4), "cov trace"]} labelFormatter={tipLabel} />
                <Bar dataKey="trace" radius={[3, 3, 0, 0]} isAnimationActive={false}>
                  {bars.map((_, i) => <Cell key={i} fill={BAR_COLORS[(i + 3) % BAR_COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ChartFrame>
          )}
          {bars.some((d) => d.mc != null) && (
            <ChartFrame title="distinct MC 数（5 选项中出现几个，越低越扁平；需先跑 MC 打分）" height={210}>
              <BarChart data={bars} margin={{ top: 6, right: 12, left: -18, bottom: 0 }}>
                <CartesianGrid stroke={GRID} vertical={false} />
                {xAxis}
                <YAxis domain={[0, 5]} allowDecimals={false} tick={TICK} axisLine={false} tickLine={false} width={40} />
                <Tooltip formatter={(v) => [v == null ? "—" : Number(v).toFixed(2), "distinct MC"]} labelFormatter={tipLabel} />
                <Bar dataKey="mc" radius={[3, 3, 0, 0]} isAnimationActive={false}>
                  {bars.map((_, i) => <Cell key={i} fill={BAR_COLORS[(i + 4) % BAR_COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ChartFrame>
          )}
        </>
      )}
    </div>
  );
}

function AgeGenderCard({ name, data, paper = {} }) {
  const a = data.fig4a || {};
  const tt = a.ttest || {};
  const ols = a.ols_male_coef || {};
  const b = data.fig4b || {};
  const inter = (data.fig4c || {}).interaction_male_x_age;
  const h1bars = useMemo(
    () => AG_OUTCOMES.map(([k, label]) => ({ name: label, beta: ols[k]?.estimate ?? null })),
    [ols],
  );
  const condBars = useMemo(
    () => Object.entries(data.n_by_condition || {}).map(([k, v]) => ({ name: k, count: v })),
    [data],
  );
  return (
    <div className="card" style={{ padding: 14 }}>
      <div className="card-title" style={{ marginBottom: 2 }}>{name}</div>
      <div className="muted" style={{ fontSize: 12, marginBottom: 10 }}>
        简历审计 Fig.4 · {data.n_total_resumes ?? "—"} 份简历 · 已评分 {data.n_scored ?? "—"}
      </div>

      <Block title="H1 · 男性姓名简历是否更年长 / 经验更多（male 效应 β）"
             verdict={agSelfConclusion("H1", data)} paperShort={AG_PAPER_REF.H1}
             paper={paper.treatment} diff={agDiff("H1", data)}>
        {AG_OUTCOMES.map(([k, label]) => {
          const o = ols[k]; const t = tt[k];
          return (
            <div key={k} style={{ marginTop: 6 }}>
              <div style={{ display: "flex", gap: 18, flexWrap: "wrap" }}>
                <Stat label={`${label} β(male)`} value={o ? Number(o.estimate).toFixed(3) : "—"} />
                <Stat label="p 值" value={o ? fmtP(o.p_value) : "—"} />
                <Stat label="均值 男−女" value={t ? Number(t.diff_male_minus_female).toFixed(2) : "—"} />
              </div>
            </div>
          );
        })}
        {h1bars.some((d) => d.beta != null) && (
          <ChartFrame title="male 效应 β（>0 = 男性被设定得更年长 / 经验更多 / 毕业更久）" height={180}>
            <BarChart data={h1bars} margin={{ top: 6, right: 12, left: -18, bottom: 0 }}>
              <CartesianGrid stroke={GRID} vertical={false} />
              <XAxis dataKey="name" tick={TICK} axisLine={{ stroke: GRID }} tickLine={false} />
              <YAxis tick={TICK} axisLine={false} tickLine={false} width={40} />
              <Tooltip formatter={(v) => [v == null ? "—" : Number(v).toFixed(3), "β(male)"]} />
              <ReferenceLine y={0} stroke="#94a3b8" />
              <Bar dataKey="beta" radius={[3, 3, 0, 0]} isAnimationActive={false}>
                {h1bars.map((_, i) => <Cell key={i} fill={BAR_COLORS[i % BAR_COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ChartFrame>
        )}
      </Block>

      {data.control_gender && data.control_gender.n ? (
        <Block title="H1 对照 · control_gender（模型自生成性别，排除指定姓名混淆）"
               verdict={agSelfConclusion("control_gender", data)} paperShort={AG_PAPER_REF.control_gender}
               paper={paper.control_gender} diff={agDiff("control_gender", data)}>
          {AG_OUTCOMES.map(([k, label]) => {
            const o = (data.control_gender.ols_male_coef || {})[k];
            const t = (data.control_gender.ttest || {})[k];
            return (
              <div key={k} style={{ marginTop: 6, display: "flex", gap: 18, flexWrap: "wrap" }}>
                <Stat label={`${label} β(male)`} value={o ? Number(o.estimate).toFixed(3) : "—"} />
                <Stat label="p 值" value={o ? fmtP(o.p_value) : "—"} />
                <Stat label="均值 男−女" value={t ? Number(t.diff_male_minus_female).toFixed(2) : "—"} />
              </div>
            );
          })}
        </Block>
      ) : null}

      <Block title="H2 · 年龄 → 评分（正相关）"
             verdict={agSelfConclusion("H2", data)} paperShort={AG_PAPER_REF.H2} diff={agDiff("H2", data)}>
        <div style={{ display: "flex", gap: 18, flexWrap: "wrap" }}>
          <Stat label="r（全样本）" value={b.all ? Number(b.all.r).toFixed(3) : "—"} />
          <Stat label="p（全样本）" value={b.all ? fmtP(b.all.p_value) : "—"} />
          <Stat label="r（treatment）" value={b.treatment ? Number(b.treatment.r).toFixed(3) : "—"} />
          <Stat label="p（treatment）" value={b.treatment ? fmtP(b.treatment.p_value) : "—"} />
        </div>
        <div style={{ display: "flex", gap: 18, flexWrap: "wrap", marginTop: 8 }}>
          <Stat label="经验~评分 r" value={b.experience_all ? Number(b.experience_all.r).toFixed(3) : "—"} />
          <Stat label="毕业~评分 r" value={b.grad_all ? Number(b.grad_all.r).toFixed(3) : "—"} />
          <Stat label="控制回归 β_age" value={(b.controlled_ols && b.controlled_ols.age_coef) ? Number(b.controlled_ols.age_coef.estimate).toFixed(4) : "—"} />
          <Stat label="β_age p" value={(b.controlled_ols && b.controlled_ols.age_coef) ? fmtP(b.controlled_ols.age_coef.p_value) : "—"} />
        </div>
      </Block>

      <Block title="H3 · age × male 交互（年长男性额外加分）"
             verdict={agSelfConclusion("H3", data)} paperShort={AG_PAPER_REF.H3} diff={agDiff("H3", data)}>
        <div style={{ display: "flex", gap: 18, flexWrap: "wrap" }}>
          <Stat label="交互 β" value={inter ? Number(inter.estimate).toFixed(4) : "—"} />
          <Stat label="p 值" value={inter ? fmtP(inter.p_value) : "—"} />
          <Stat label="n" value={inter ? inter.n : "—"} />
        </div>
      </Block>

      {condBars.length > 0 && (
        <Block>
          <ChartFrame title="各条件简历数" height={150}>
            <BarChart data={condBars} margin={{ top: 6, right: 12, left: -18, bottom: 0 }}>
              <CartesianGrid stroke={GRID} vertical={false} />
              <XAxis dataKey="name" tick={TICK} axisLine={{ stroke: GRID }} tickLine={false} />
              <YAxis allowDecimals={false} tick={TICK} axisLine={false} tickLine={false} width={40} />
              <Tooltip formatter={(v) => [v, "份"]} />
              <Bar dataKey="count" radius={[3, 3, 0, 0]} isAnimationActive={false}>
                {condBars.map((_, i) => <Cell key={i} fill={BAR_COLORS[(i + 1) % BAR_COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ChartFrame>
        </Block>
      )}
    </div>
  );
}

function IdentityCoveragePremiseCard({ name, data }) {
  const covBars = useMemo(
    () => ((data.coverage && data.coverage.by_axis) || []).map((d) => ({
      name: d.axis, vendi: d.vendi, det: d.det, mc: d.mc_unique,
    })),
    [data],
  );
  const premBars = useMemo(
    () => ((data.premise1 && data.premise1.by_axis) || []).map((d) => ({
      name: d.axis, within: d.mean_within, across: d.mean_across, frac: d.frac_sig,
      wsig: d.frac_wilcoxon_sig, chisig: d.frac_chisq_sig, nchi: d.n_chi,
    })),
    [data],
  );
  return (
    <div className="card" style={{ padding: 14 }}>
      <div className="card-title" style={{ marginBottom: 2 }}>{name}</div>
      <div className="muted" style={{ fontSize: 12, marginBottom: 10 }}>
        轴级 Coverage（Vendi / det，越高越多样）+ Premise 1（跨身份距离应大于组内）
      </div>

      <Block title="Coverage · Vendi score（按身份轴）">
        {covBars.length === 0 ? (
          <div className="muted" style={{ fontSize: 13 }}>暂无 coverage 结果。</div>
      ) : (
        <>
        <ChartFrame title="Vendi score（越高 = 该轴回答覆盖越多样）" height={190}>
          <BarChart data={covBars} margin={{ top: 6, right: 12, left: -18, bottom: 0 }}>
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis dataKey="name" tick={TICK} axisLine={{ stroke: GRID }} tickLine={false} />
            <YAxis tick={TICK} axisLine={false} tickLine={false} width={40} />
            <Tooltip formatter={(v) => [v == null ? "—" : Number(v).toFixed(3), "Vendi"]} />
            <Bar dataKey="vendi" radius={[3, 3, 0, 0]} isAnimationActive={false}>
              {covBars.map((_, i) => <Cell key={i} fill={BAR_COLORS[i % BAR_COLORS.length]} />)}
            </Bar>
          </BarChart>
        </ChartFrame>
        {covBars.some((d) => d.det != null) && (
          <ChartFrame title="协方差行列式 det（Coverage 第二指标；越高 = 覆盖越多样）" height={190}>
            <BarChart data={covBars} margin={{ top: 6, right: 12, left: -18, bottom: 0 }}>
              <CartesianGrid stroke={GRID} vertical={false} />
              <XAxis dataKey="name" tick={TICK} axisLine={{ stroke: GRID }} tickLine={false} />
              <YAxis tick={TICK} axisLine={false} tickLine={false} width={40} />
              <Tooltip formatter={(v) => [v == null ? "—" : Number(v).toExponential(2), "det"]} />
              <Bar dataKey="det" radius={[3, 3, 0, 0]} isAnimationActive={false}>
                {covBars.map((_, i) => <Cell key={i} fill={BAR_COLORS[(i + 1) % BAR_COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ChartFrame>
        )}
        {covBars.some((d) => d.mc != null) && (
          <div style={{ display: "flex", gap: 18, flexWrap: "wrap", marginTop: 4 }}>
            {covBars.map((d) => (
              <Stat key={d.name} label={`${d.name} distinct MC（Coverage）`} value={d.mc == null ? "—" : Number(d.mc).toFixed(2)} />
            ))}
          </div>
        )}
        </>
      )}
      </Block>

      <Block title="Premise 1 · 跨身份 vs 组内语义距离">
        {premBars.length === 0 ? (
          <div className="muted" style={{ fontSize: 13 }}>暂无 premise 1 结果。</div>
        ) : (
          <>
            <ChartFrame title="组内(within) vs 跨身份(across) 平均距离（across>within = 身份改变了回答）" height={200}>
              <BarChart data={premBars} margin={{ top: 6, right: 12, left: -18, bottom: 0 }}>
                <CartesianGrid stroke={GRID} vertical={false} />
                <XAxis dataKey="name" tick={TICK} axisLine={{ stroke: GRID }} tickLine={false} />
                <YAxis tick={TICK} axisLine={false} tickLine={false} width={40} />
                <Tooltip formatter={(v, n) => [v == null ? "—" : Number(v).toFixed(4), n === "within" ? "组内" : "跨身份"]} />
                <Bar dataKey="within" name="within" fill="#94a3b8" radius={[3, 3, 0, 0]} isAnimationActive={false} />
                <Bar dataKey="across" name="across" fill="#2563eb" radius={[3, 3, 0, 0]} isAnimationActive={false} />
              </BarChart>
            </ChartFrame>
          <div style={{ display: "flex", gap: 18, flexWrap: "wrap", marginTop: 4 }}>
            {premBars.map((d) => (
              <Stat key={d.name} label={`${d.name} 距离t / 样本Wilcoxon 显著占比`}
                    value={`${d.frac == null ? "—" : pct(d.frac)} / ${d.wsig == null ? "—" : pct(d.wsig)}`} />
            ))}
          </div>
          {premBars.some((d) => d.nchi > 0) && (
            <div style={{ display: "flex", gap: 18, flexWrap: "wrap", marginTop: 4 }}>
              {premBars.filter((d) => d.nchi > 0).map((d) => (
                <Stat key={d.name} label={`${d.name} MC卡方 显著占比`} value={d.chisig == null ? "—" : pct(d.chisig)} />
              ))}
            </div>
          )}
          </>
        )}
      </Block>
    </div>
  );
}

function CsvTable({ title, rows }) {
  if (!Array.isArray(rows) || rows.length === 0) return null;
  const cols = Object.keys(rows[0]);
  return (
    <div style={{ marginTop: 8 }}>
      {title && <div className="muted" style={{ fontSize: 12, marginBottom: 4 }}>{title}</div>}
      <div style={{ overflowX: "auto" }}>
        <table style={{ borderCollapse: "collapse", fontSize: 12, width: "100%" }}>
          <thead>
            <tr style={{ color: "var(--text-3)" }}>
              {cols.map((c) => (
                <th key={c} style={{ textAlign: "left", padding: "2px 6px", whiteSpace: "nowrap" }}>{c}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}>
                {cols.map((c) => (
                  <td key={c} style={{ textAlign: "left", padding: "2px 6px" }}>{String(r[c] ?? "")}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function FileCard({ name, data, paper }) {
  if (!data || typeof data !== "object") return null;
  if (data.error) {
    return (
      <div className="card" style={{ padding: 14 }}>
        <div className="card-title">{name}</div>
        <div className="warn-text" style={{ fontSize: 13 }}>{data.error}</div>
      </div>
    );
  }
  if (data.type === "individual_bias") return <IndividualCard name={name} data={data} paper={paper} />;
  if (data.type === "collective_or_committed") return <CollectiveCard name={name} data={data} paper={paper} />;
  if (data.type === "identity_flattening") return <IdentityFlatteningCard name={name} data={data} />;
  if (data.type === "identity_coverage_premise") return <IdentityCoveragePremiseCard name={name} data={data} />;
  if (data.type === "age_gender_distortion") return <AgeGenderCard name={name} data={data} paper={paper} />;
  if (data.type === "altruism_dictator") return <AltruismCard name={name} data={data} paper={paper} />;
  if (data.type === "behavioral_game") return <BehavioralGameCard name={name} data={data} paper={paper} />;
  return (
    <div className="card" style={{ padding: 14 }}>
      <div className="card-title">{name}</div>
      <div className="muted" style={{ fontSize: 13 }}>{data.note || "未识别的结果结构"}</div>
    </div>
  );
}

function RevealedPreferenceSection({ rp }) {
  const entries = Object.entries(rp?.games || {});
  if (!entries.length) return null;
  const curve = (rp.overall_curve || []).map((d) => ({ b: d.b, err: d.err }));
  return (
    <section className="page-section">
      <div className="section-head"><h3>揭示偏好 b 与收益表（复现 Fig.6 / Table S1）</h3></div>
      <div className="card" style={{ padding: 14 }}>
        {rp.overall_best_b != null && (
          <div style={{ display: "flex", gap: 18, flexWrap: "wrap", marginBottom: 8 }}>
            <Stat label="总体最优 b" value={rp.overall_best_b} />
            <span className="muted" style={{ fontSize: 12, alignSelf: "center" }}>
              b=自己收益权重（0.5=等权双方，1=纯自利，0=纯利他）。论文：ChatGPT≈0.5、人类≈0.6。
            </span>
          </div>
        )}
        {curve.length > 1 && (
          <ChartFrame title="跨博弈平均相对误差 vs b（最低点即最优 b）" height={180}>
            <LineChart data={curve} margin={{ top: 6, right: 12, left: -18, bottom: 0 }}>
              <CartesianGrid stroke={GRID} vertical={false} />
              <XAxis dataKey="b" tick={TICK} axisLine={{ stroke: GRID }} tickLine={false} />
              <YAxis tick={TICK} axisLine={false} tickLine={false} width={44} />
              <Tooltip formatter={(v) => [Number(v).toFixed(4), "误差"]} labelFormatter={(l) => `b=${l}`} />
              {rp.overall_best_b != null && (
                <ReferenceLine x={rp.overall_best_b} stroke="#ef4444" strokeDasharray="4 4" />
              )}
              <Line type="monotone" dataKey="err" stroke="#2563eb" strokeWidth={2} dot={false} isAnimationActive={false} />
            </LineChart>
          </ChartFrame>
        )}
        <div style={{ marginTop: 10, overflowX: "auto" }}>
          <table style={{ borderCollapse: "collapse", fontSize: 12, width: "100%" }}>
            <thead>
              <tr style={{ color: "var(--text-3)" }}>
                <th style={{ textAlign: "left", padding: "3px 8px" }}>博弈</th>
                <th style={{ textAlign: "right", padding: "3px 8px" }}>本模型 own</th>
                <th style={{ textAlign: "right", padding: "3px 8px" }}>给对方</th>
                <th style={{ textAlign: "right", padding: "3px 8px" }}>合计</th>
                <th style={{ textAlign: "right", padding: "3px 8px" }}>人类 own</th>
                <th style={{ textAlign: "right", padding: "3px 8px" }}>给对方</th>
                <th style={{ textAlign: "right", padding: "3px 8px" }}>合计</th>
                <th style={{ textAlign: "right", padding: "3px 8px" }}>最优 b</th>
              </tr>
            </thead>
            <tbody>
              {entries.map(([k, g]) => (
                <tr key={k}>
                  <td style={{ textAlign: "left", padding: "3px 8px" }}>{g.label || k}</td>
                  <td style={{ textAlign: "right", padding: "3px 8px" }}>{g.ai?.own ?? "—"}</td>
                  <td style={{ textAlign: "right", padding: "3px 8px" }}>{g.ai?.partner ?? "—"}</td>
                  <td style={{ textAlign: "right", padding: "3px 8px" }}>{g.ai?.combined ?? "—"}</td>
                  <td style={{ textAlign: "right", padding: "3px 8px" }}>{g.human?.own ?? "—"}</td>
                  <td style={{ textAlign: "right", padding: "3px 8px" }}>{g.human?.partner ?? "—"}</td>
                  <td style={{ textAlign: "right", padding: "3px 8px" }}>{g.human?.combined ?? "—"}</td>
                  <td style={{ textAlign: "right", padding: "3px 8px", fontWeight: 600 }}>{g.best_b ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {rp.note && <div className="muted" style={{ fontSize: 11, marginTop: 8 }}>{rp.note}</div>}
      </div>
    </section>
  );
}

function CriticalMassSection({ cm }) {
  const entries = Object.entries(cm || {});
  if (!entries.length) return null;
  return (
    <section className="page-section">
      <div className="section-head"><h3>H3 · 临界质量扫描（复现论文 Fig. 3 / Table S3）</h3></div>
      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        {entries.map(([base, g]) => {
          const data = (g.by_size || []).map((e) => ({
            cm: e.cm,
            rate: e.judged_runs ? e.flipped_runs / e.judged_runs : null,
            flipped: e.flipped_runs,
            judged: e.judged_runs,
            proportion: e.proportion,
          }));
          return (
            <div className="card" style={{ padding: 14 }} key={base}>
              <div className="card-title" style={{ marginBottom: 2 }}>{base}</div>
              <div className="muted" style={{ fontSize: 12, marginBottom: 8 }}>
                N={g.N} · 初始共识=索引 {g.initial} · 少数派坚持 {g.committed_to ?? "—"} · {g.criterion}
              </div>
              <div style={{ display: "flex", gap: 18, flexWrap: "wrap" }}>
                <Stat label="临界质量（全部 run 翻转）" value={g.critical_mass_full ?? "未达到"} />
                <Stat label="最早出现翻转的规模" value={g.critical_mass_first ?? "无"} />
              </div>
              {data.length > 1 && (
                <ChartFrame title="翻转比例 vs 坚定少数派数量（红虚线 = 全部 run 翻转 / 临界质量位置）" height={190}>
                  <LineChart data={data} margin={{ top: 6, right: 12, left: -18, bottom: 0 }}>
                    <CartesianGrid stroke={GRID} vertical={false} />
                    <XAxis dataKey="cm" tick={TICK} axisLine={{ stroke: GRID }} tickLine={false} />
                    <YAxis domain={[0, 1]} tick={TICK} axisLine={false} tickLine={false} width={36} />
                    <Tooltip
                      formatter={(v, _n, item) => [
                        `${pct(v)}（${item?.payload?.flipped}/${item?.payload?.judged} run，占群体 ${pct(item?.payload?.proportion)}）`,
                        "翻转比例",
                      ]}
                      labelFormatter={(l) => `少数派数量 cm=${l}`}
                    />
                    <ReferenceLine y={1} stroke="#ef4444" strokeDasharray="4 4" />
                    {g.critical_mass_full != null && (
                      <ReferenceLine x={g.critical_mass_full} stroke="#ef4444" strokeDasharray="4 4" />
                    )}
                    <Line type="monotone" dataKey="rate" stroke="#2563eb" strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
                  </LineChart>
                </ChartFrame>
              )}
              <Block verdict={g.verdict} />
            </div>
          );
        })}
      </div>
    </section>
  );
}

function DynamicsSection({ dynamics }) {
  const entries = Object.entries(dynamics || {});
  if (!entries.length) return null;
  return (
    <section className="page-section">
      <div className="section-head"><h3>行为动态（复现 Fig.4 tit-for-tat / Fig.5 风险）</h3></div>
      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        {entries.map(([k, d]) => (
          <div className="card" style={{ padding: 14 }} key={k}>
            <div className="card-title">{k}</div>
            {d.type === "pd" && (
              <div style={{ display: "flex", gap: 18, flexWrap: "wrap", marginTop: 6 }}>
                <Stat label="样本数" value={d.n_instances} />
                <Stat label="首轮合作 / 背叛" value={`${d.coop_first} / ${d.defect_first}`} />
                <Stat label="首轮合作者·次轮仍合作" value={d.coop_then_coop == null ? "—" : pct(d.coop_then_coop)} />
                <Stat label="首轮背叛者·次轮转合作" value={d.defect_then_coop == null ? "—" : pct(d.defect_then_coop)} />
              </div>
            )}
            {d.type === "bomb" && (
              <div style={{ display: "flex", gap: 18, flexWrap: "wrap", marginTop: 6 }}>
                <Stat label="踩雷后开盒(均)" value={d.after_bomb_mean ?? "—"} />
                <Stat label="未踩雷后开盒(均)" value={d.after_safe_mean ?? "—"} />
              </div>
            )}
            {d.note && <div className="muted" style={{ fontSize: 11, marginTop: 8 }}>{d.note}</div>}
          </div>
        ))}
      </div>
    </section>
  );
}

const IDENTITY_AXIS_LABEL = {
  race: "种族 (race)", gender: "性别 (gender)", intersection: "交叉身份 (intersection)",
  age: "年龄 (age)", disability: "残障 (disability)", other: "其他人设 (other)",
  gender_or_intersection: "性别/交叉",
};
const IDENTITY_AXIS_ORDER = ["race", "gender", "intersection", "age", "disability", "other"];

const IDENTITY_PAPER_REF = {
  race: "误现最强者为 White person（R1 23/24，白人被视作『常态』、少自述种族）；该轴多样性显著低于真人（扁平）。",
  gender: "non-binary 误现最强（R2 32/48，单项最高）、woman 26/48；扁平旗舰例（把非二元者统一为『代词被忽视』）。",
  intersection: "唯一做『身份编码姓名』缓解的轴（用 Darnell/Imani 等姓名可减少误现）；本质化范例（黑人女性『Hey girl!』）。",
  age: "仅 Gen Z 在 R2 误现 27/48 显著、R1 未达显著；扁平结论与其余轴一致。",
  disability: "误现最强最一致（视障者 R1 18/24、R2 27/48，贯穿 R1/R2）；圈外想象范例（视障答移民题）。",
  other: "R4-Coverage：MBTI/星座/政治/persona 等非敏感人设即可达同等甚至更高覆盖度，无需敏感身份。",
};

function _idMean(xs) {
  return xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : null;
}

function identityAxisAgg(metrics) {
  const files = (metrics && metrics.files) || {};
  let diversity = null, covprem = null;
  for (const v of Object.values(files)) {
    if (v?.type === "identity_flattening") diversity = v;
    else if (v?.type === "identity_coverage_premise") covprem = v;
  }
  const byAxis = {};
  const ensure = (ax) => (byAxis[ax] = byAxis[ax] || { axis: ax });
  if (diversity?.by_identity) {
    const acc = {};
    for (const it of diversity.by_identity) {
      const ax = it.axis || "unknown";
      const a = acc[ax] = acc[ax] || { u: [], c: [], t: [] };
      if (it.unique_ngram != null) a.u.push(it.unique_ngram);
      if (it.pairwise_cosine != null) a.c.push(it.pairwise_cosine);
      if (it.cov_trace != null) a.t.push(it.cov_trace);
    }
    for (const [ax, a] of Object.entries(acc)) {
      const o = ensure(ax);
      o.unique = _idMean(a.u); o.cosine = _idMean(a.c); o.trace = _idMean(a.t);
    }
  }
  for (const d of covprem?.coverage?.by_axis || []) {
    const o = ensure(d.axis); o.vendi = d.vendi; o.det = d.det;
  }
  for (const d of covprem?.premise1?.by_axis || []) {
    const o = ensure(d.axis); o.wsig = d.frac_wilcoxon_sig; o.within = d.mean_within; o.across = d.mean_across;
  }
  return byAxis;
}

function identitySelfConclusion(ax, a) {
  const parts = [];
  if (a.unique != null || a.cosine != null) {
    parts.push(`扁平：unique n-gram ${a.unique != null ? a.unique.toFixed(3) : "—"}、两两余弦 ${a.cosine != null ? a.cosine.toFixed(3) : "—"}${a.trace != null ? `、协方差迹 ${a.trace.toFixed(3)}` : ""}（越低越同质）`);
  }
  if (a.vendi != null) parts.push(`Coverage Vendi ${a.vendi.toFixed(2)}${a.det != null ? `、det ${a.det.toExponential(1)}` : ""}（越高越多样）`);
  if (a.wsig != null) parts.push(`Premise 1 跨身份>组内 显著占比 ${pct(a.wsig)}（>0 说明身份确实改变了回答）`);
  return parts.length ? `本模型「${IDENTITY_AXIS_LABEL[ax] || ax}」：${parts.join("；")}。` : "暂无该轴可汇总的指标。";
}

function identityDiff() {
  return ("本模型该轴可比维度为「扁平 + Coverage + Premise 1」（本项目走纯 LLM 多样性管线）；原文该轴重点含"
    + "「误现 misportray（圈内 vs 圈外人类对照）」，本管线不含人类圈内/圈外语料、未计算误现，故误现结论不作数值对照。"
    + "可对照的扁平/覆盖维度上，本模型与原文同属『多样性低于真人』方向（具体数值因模型/嵌入不同而异，属正常）。");
}

function IdentityConclusionSection({ metrics, paper }) {
  const byAxis = identityAxisAgg(metrics);
  const extra = Object.keys(byAxis).filter((ax) => !IDENTITY_AXIS_ORDER.includes(ax));
  const axes = IDENTITY_AXIS_ORDER.filter((ax) => byAxis[ax]).concat(extra);
  if (!axes.length) return null;
  return (
    <section className="page-section">
      <div className="section-head"><h3>各身份轴 · 结论对照（本次 / 原作者 / 差异）</h3></div>
      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        {axes.map((ax) => (
          <div className="card" style={{ padding: 14 }} key={ax}>
            <div className="card-title">{IDENTITY_AXIS_LABEL[ax] || ax}</div>
            <Block
              title=""
              verdict={identitySelfConclusion(ax, byAxis[ax])}
              paperShort={IDENTITY_PAPER_REF[ax]}
              paper={(paper && paper[ax]) || ""}
              diff={identityDiff()}
            />
          </div>
        ))}
      </div>
    </section>
  );
}

function AnalysisPanel({ projects, toast }) {
  const [activeId, setActiveId] = useState(() => projects[0]?.id ?? null);
  const [models, setModels] = useState([]);
  const [canAnalyze, setCanAnalyze] = useState(false);
  const [selModel, setSelModel] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [busy, setBusy] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);

  const [judgeModel, setJudgeModel] = useState("gpt-5.4");

  useEffect(() => {
    if (projects.length && !activeId) setActiveId(projects[0].id);
  }, [projects, activeId]);

  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const r = await api("/api/analyze-status");
        if (alive) setAnalyzing(!!r.analyzing);
      } catch {

      }
    };
    tick();
    const id = setInterval(tick, 2500);
    return () => { alive = false; clearInterval(id); };
  }, []);

  const fetchAnalysis = useCallback(async (pid, model) => {
    if (!pid || !model) return null;
    try {
      const r = await api(`/api/projects/${pid}/analysis?model=${encodeURIComponent(model)}`);
      return r.analysis?.metrics ?? null;
    } catch {
      return null;
    }
  }, []);

  const loadModels = useCallback(async () => {
    if (!activeId) return;
    try {
      const r = await api(`/api/projects/${activeId}/result-models`);
      setModels(r.models || []);
      setCanAnalyze(!!r.can_analyze);
    } catch (e) {
      toast?.("结果模型加载失败：" + e.message);
    }
  }, [activeId, toast]);

  useEffect(() => {
    if (!activeId) return;
    let cancelled = false;
    (async () => {
      let nextModels = [];
      let can = false;
      try {
        const r = await api(`/api/projects/${activeId}/result-models`);
        nextModels = r.models || [];
        can = !!r.can_analyze;
      } catch (e) {
        if (!cancelled) toast?.("结果模型加载失败：" + e.message);
      }
      const model = nextModels[0]?.model ?? null;
      const nextMetrics = await fetchAnalysis(activeId, model);
      if (cancelled) return;
      setModels(nextModels);
      setCanAnalyze(can);
      setSelModel(model);
      setMetrics(nextMetrics);
    })();
    return () => { cancelled = true; };
  }, [activeId, fetchAnalysis, toast]);

  const selectModel = useCallback(async (model) => {
    if (!activeId || model === selModel) return;
    const m = await fetchAnalysis(activeId, model);
    setSelModel(model);
    setMetrics(m);
  }, [activeId, selModel, fetchAnalysis]);

  const runAnalyze = async () => {
    if (!activeId) return;
    setBusy(true);
    try {
      const r = await api(`/api/projects/${activeId}/analyze`, {
        method: "POST",
        body: JSON.stringify({ model: selModel }),
      });
      setMetrics(r.analysis?.metrics ?? null);
      toast?.("分析完成");
    } catch (e) {
      if (e.status === 409) {
        toast?.(String(e.message || "已有一个结果分析正在运行，请稍后再试。"), 4200, "warn");
      } else {
        toast?.("分析失败：" + String(e.message || "").split("\n")[0], 5200);
      }
    } finally {
      setBusy(false);
    }
  };

  const runScore = async () => {
    if (!activeId || !selModel) return;
    setBusy(true);
    const judge = judgeModel.trim();
    try {
      await api("/api/launch", {
        method: "POST",
        body: JSON.stringify({
          project_id: activeId,
          experiment_ids: ["mc_score"],
          params: { score_input_model: selModel, ...(judge ? { score_judge_model: judge } : {}) },
        }),
      });
      toast?.(
        judge
          ? `已发起 MC 打分任务（裁判=${judge}）。请到「运行监控」查看进度；完成后回此页重新分析即可看到 MC 指标。`
          : "已发起 MC 打分任务（裁判=统一配置当前模型；跨模型比较建议在输入框指定固定裁判）。请到「运行监控」查看进度；完成后回此页重新分析即可看到 MC 指标。",
        6500,
      );
    } catch (e) {
      toast?.("发起打分失败：" + String(e.message || "").split("\n")[0], 5200);
    } finally {
      setBusy(false);
    }
  };

  const files = useMemo(() => {
    const all = (metrics && metrics.files) || {};
    const entries = Object.entries(all);
    if (!selModel) return entries;
    return entries.filter(([k]) => k.split("/")[0] === selModel || k.startsWith(selModel));
  }, [metrics, selModel]);

  const proj = projects.find((p) => p.id === activeId);

  const paperConclusions = useMemo(() => {
    const map = {};
    for (const e of proj?.experiments || []) map[e.id] = e.conclusion || "";
    return map;
  }, [proj]);

  if (!proj) return <div className="muted">加载项目中…</div>;

  return (
    <div className="page-stack">
      <ProjectTabs projects={projects} activeId={activeId} onSelect={setActiveId} />

      <section className="page-section">
        <div className="section-head">
          <h3>已有结果的模型</h3>
          <div className="section-actions">
            <button type="button" className="ghost sm" onClick={loadModels}>刷新</button>
            <span>{models.length} 个模型</span>
          </div>
        </div>

        {models.length === 0 ? (
          <div className="empty-block">该项目暂无已落盘的结果。先到「发起实验」跑出结果后再来分析。</div>
        ) : (
          <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
            {models.map((m) => {
              const on = m.model === selModel;
              return (
                <button
                  type="button"
                  key={m.model}
                  onClick={() => selectModel(m.model)}
                  style={{
                    textAlign: "left", cursor: "pointer", borderRadius: 8, padding: "8px 12px",
                    border: "1px solid " + (on ? "var(--accent)" : "var(--border, #d8dee8)"),
                    boxShadow: on ? "inset 0 0 0 1px var(--accent)" : "none",
                    background: on ? "var(--selected)" : "var(--surface, #fff)",
                    transition: "none",
                  }}
                >
                  <div style={{ fontWeight: 600, fontSize: 14 }}>{m.model}</div>
                  <div className="muted" style={{ fontSize: 11 }}>
                    {m.files} 个文件 · {m.latest || "—"}
                  </div>
                </button>
              );
            })}
          </div>
        )}

        <div className="actions" style={{ marginTop: 14 }}>
          <button
            type="button"
            className="primary"
            onClick={runAnalyze}
            disabled={busy || analyzing || !canAnalyze || !selModel}
            title={
              !canAnalyze ? "该项目暂不支持一键分析"
                : !selModel ? "请先选择一个模型"
                  : (analyzing && !busy) ? "已有分析在运行中（全局同一时刻仅允许一个），请稍候"
                    : "对所选模型离线重算复现指标"
            }
          >
            {busy ? (
              <><span className="btn-spin" aria-hidden="true" />分析中…</>
            ) : analyzing ? (
              <><span className="btn-spin" aria-hidden="true" />分析进行中…</>
            ) : (
              "分析所选模型"
            )}
          </button>
          {activeId === "identity" && (
            <>
              <button
                type="button"
                className="ghost"
                onClick={runScore}
                disabled={busy || !selModel}
                title="按论文方法给每条回答打 1–5 分（R2b-gender 用 TextBlob）。作为异步任务运行，完成后重新分析可见 mc_unique / chisq 指标。"
                style={{ marginLeft: 8 }}
              >
                MC 打分
              </button>
              <input
                type="text"
                value={judgeModel}
                onChange={(e) => setJudgeModel(e.target.value)}
                placeholder="裁判模型（论文用 gpt-3.5-turbo；留空=统一配置模型）"
                title="MC 打分的裁判模型。论文用固定 GPT-3.5 统一裁判；跨模型趋势比较必须所有被测模型用同一个裁判，否则 MC 指标混入裁判差异。留空=统一配置当前模型（对被测模型即自评）。"
                style={{ marginLeft: 8, width: 300 }}
              />
            </>
          )}
          {!canAnalyze && models.length > 0 && (
            <span className="muted" style={{ marginLeft: 10, fontSize: 12 }}>
              该项目暂未接入一键分析脚本
            </span>
          )}
          {metrics?.stats_backend && (
            <span className="chip" style={{ marginLeft: 10 }}>统计后端 {metrics.stats_backend}</span>
          )}
        </div>
      </section>

      <section className="page-section">
        <div className="section-head">
          <h3>分析结果{selModel ? ` · ${selModel}` : ""}</h3>
          <div className="section-actions"><span>{files.length} 个结果文件</span></div>
        </div>
        {files.length === 0 ? (
          <div className="empty-block">
            {metrics ? "所选模型暂无可展示的分析结果。" : "尚未分析。选择模型后点击「启动分析」。"}
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            {files.map(([key, data]) => (
              <FileCard key={key} name={key.split("/").slice(1).join("/") || key} data={data} paper={paperConclusions} />
            ))}
          </div>
        )}
      </section>

      {metrics?.revealed_preference?.games && Object.keys(metrics.revealed_preference.games).length > 0 && (
        <RevealedPreferenceSection rp={metrics.revealed_preference} />
      )}
      {metrics?.dynamics && Object.keys(metrics.dynamics).length > 0 && (
        <DynamicsSection dynamics={metrics.dynamics} />
      )}
      {metrics?.critical_mass && Object.keys(metrics.critical_mass).length > 0 && (
        <CriticalMassSection cm={metrics.critical_mass} />
      )}
      {metrics?.files && Object.values(metrics.files).some(
        (v) => v?.type === "identity_flattening" || v?.type === "identity_coverage_premise",
      ) && (
        <IdentityConclusionSection metrics={metrics} paper={paperConclusions} />
      )}
    </div>
  );
}

export default memo(AnalysisPanel);
