import { useState, useEffect, useMemo, memo } from "react";
import { api } from "../api.js";
import ParamInput from "./ParamInput.jsx";
import Markdown from "./Markdown.jsx";
import ProjectTabs from "./ProjectTabs.jsx";
import { buildDefaultParams, normalizeParams, apiKeyReady, formatOrchestratorError, EMPTY_SEL } from "../launchUtils.js";

function ProjectPanel({
  projects,
  config,
  configMeta,
  toast,
  refreshJobs,
  onLaunched,
}) {
  const [activeId, setActiveId] = useState(() => projects[0]?.id ?? null);
  const [selections, setSelections] = useState({});
  const [paramVals, setParamVals] = useState({});
  const [busy, setBusy] = useState(false);
  const [launchError, setLaunchError] = useState(null);
  const [paramsCollapsed, setParamsCollapsed] = useState(false);

  useEffect(() => {
    if (projects.length && !activeId) setActiveId(projects[0].id);
  }, [projects, activeId]);

  useEffect(() => {
    setLaunchError(null);
  }, [activeId]);

  const proj = projects.find((p) => p.id === activeId);

  useEffect(() => {
    if (!proj) return;
    setParamVals((s) => {
      if (s[proj.id]) return s;
      return { ...s, [proj.id]: buildDefaultParams(proj) };
    });
  }, [proj]);

  const sel = proj ? (selections[proj.id] ?? EMPTY_SEL) : EMPTY_SEL;
  const pvals = proj ? (paramVals[proj.id] ?? EMPTY_SEL) : EMPTY_SEL;
  const normalizedParams = useMemo(
    () => (proj ? normalizeParams(pvals, proj) : {}),
    [proj, pvals],
  );

  const selectedIdList = useMemo(
    () => Object.keys(sel).filter((k) => sel[k]),
    [sel],
  );

  const sortedExperiments = useMemo(() => {
    const list = proj?.experiments || [];
    return [...list].sort(
      (a, b) => (a.importance_rank ?? Infinity) - (b.importance_rank ?? Infinity),
    );
  }, [proj]);

  const kindOk = proj
    ? (proj.supported_kinds || []).includes(config?.llm?.provider_kind)
    : true;
  const keyOk = apiKeyReady(config, configMeta);
  const isSingle = proj?.selection_mode === "single";

  const toggleExp = (id) => {
    if (!proj) return;
    if (isSingle) {
      setSelections((s) => ({ ...s, [proj.id]: { [id]: true } }));
      return;
    }
    setSelections((s) => ({ ...s, [proj.id]: { ...sel, [id]: !sel[id] } }));
  };

  const selectAll = () => {
    if (!proj || isSingle) return;
    const all = {};
    (proj.experiments || []).forEach((e) => { all[e.id] = true; });
    setSelections((s) => ({ ...s, [proj.id]: all }));
  };

  const clearAll = () => {
    if (!proj) return;
    setSelections((s) => ({ ...s, [proj.id]: {} }));
  };

  const setParam = (name, v) => {
    if (!proj) return;
    setParamVals((s) => ({ ...s, [proj.id]: { ...pvals, [name]: v } }));
  };

  const afterLaunch = async (infos) => {
    toast(`已启动 ${infos.length} 个任务`);
    await refreshJobs();
    const ids = (infos || []).map((j) => j.id).filter(Boolean);
    onLaunched?.(ids);
  };

  const doLaunch = async () => {
    if (!selectedIdList.length) return toast("请先选择至少一个实验");
    if (!kindOk) return toast("当前调用协议不被该项目支持，请先在「配置」页修改");
    if (!keyOk) return toast("API Key 未配置，请先在「配置」页设置");
    setBusy(true);
    setLaunchError(null);
    try {
      const infos = await api("/api/launch", {
        method: "POST",
        body: JSON.stringify({
          project_id: proj.id,
          experiment_ids: selectedIdList,
          params: normalizedParams,
        }),
      });
      await afterLaunch(infos);
    } catch (e) {
      const msg = formatOrchestratorError(e.message);
      setLaunchError(msg);
      toast(msg.split("\n")[0]);
    } finally {
      setBusy(false);
    }
  };

  if (!proj) return <div className="muted">加载项目中…</div>;

  const paramFields = proj.param_schema || [];
  const choiceParams = paramFields.filter((f) => f.type === "select" || f.type === "bool");
  const inputParams = paramFields.filter((f) => f.type !== "select" && f.type !== "bool");
  const hasRunParams = paramFields.length > 0;

  return (
    <div className="page-stack launch-page">
      <ProjectTabs projects={projects} activeId={activeId} onSelect={setActiveId} />

      <section className="project-summary">
        {proj.intro && <div className="intro">{proj.intro.replace(/\n{2,}/g, "\n")}</div>}
      </section>

      <section className="launch-config-banner">
        <span>将使用模型 <code>{config?.llm?.model || "—"}</code></span>
        <span>协议 <code>{config?.llm?.provider_kind || "—"}</code></span>
        <span className={keyOk ? "ok-text" : "warn-text"}>
          API Key {keyOk ? "已就绪" : "未配置"}
        </span>
        {config?.llm?.concurrency != null && (
          <span>子项目并发 <b>{config.llm.concurrency}</b></span>
        )}
      </section>

      {!kindOk && (
        <div className="config-banner warn">
          当前 provider_kind「{config?.llm?.provider_kind}」不在该项目支持列表内，启动会失败。
        </div>
      )}

      <section className="page-section">
        <div className="section-head">
          <h3>实验{isSingle ? "（单选）" : "（可多选）"}</h3>
          <div className="section-actions">
            {!isSingle && (
              <>
                <button type="button" className="ghost sm" onClick={selectAll}>全选</button>
                <button type="button" className="ghost sm" onClick={clearAll}>清空</button>
              </>
            )}
            <span>{selectedIdList.length} / {(proj.experiments || []).length} 已选</span>
          </div>
        </div>
        <div className="experiment-list">
          {sortedExperiments.map((e) => (
            <div
              key={e.id}
              className={"exp-card" + (sel[e.id] ? " on" : "")}
              role="button"
              tabIndex={0}
              onClick={() => toggleExp(e.id)}
              onKeyDown={(ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); toggleExp(e.id); } }}
            >
              <input
                type={isSingle ? "radio" : "checkbox"}
                name={isSingle ? `exp-${proj.id}` : undefined}
                checked={!!sel[e.id]}
                onChange={() => toggleExp(e.id)}
                onClick={(ev) => ev.stopPropagation()}
              />
              <div className="exp-main">
                <div className="exp-name">
                  {e.importance_tier && (
                    <span className={"tier-badge tier-" + e.importance_tier}>
                      {e.importance_tier}
                      {e.importance_rank != null && <span className="tier-rank">#{e.importance_rank}</span>}
                    </span>
                  )}
                  {e.label}
                </div>
                <div className="exp-desc">{e.description || "—"}</div>
              </div>
              {sel[e.id] && (e.extra_params || []).length > 0 && (
                <div className="exp-params" onClick={(ev) => ev.stopPropagation()}>
                  <div className="exp-params-fields">
                    {e.extra_params.map((f) => (
                      <ParamInput
                        key={f.name}
                        field={f}
                        value={pvals[`${e.id}__${f.name}`]}
                        onChange={(v) => setParam(`${e.id}__${f.name}`, v)}
                      />
                    ))}
                  </div>
                </div>
              )}
              {((e.importance_basis || "").trim() || (e.conclusion || "").trim()) && (
                <div className="exp-hovers" onClick={(ev) => ev.stopPropagation()}>
                  {(e.importance_basis || "").trim() && (
                    <div className="concl-hover">
                      <span className="concl-trigger">
                        排名依据{e.importance_tier ? `（${e.importance_tier} 级）` : ""}
                      </span>
                      <div className="concl-pop">
                        <Markdown text={e.importance_basis} className="concl-text" />
                      </div>
                    </div>
                  )}
                  {(e.conclusion || "").trim() && (
                    <div className="concl-hover">
                      <span className="concl-trigger">原实验结论</span>
                      <div className="concl-pop">
                        <Markdown text={e.conclusion} className="concl-text" />
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      </section>

      {hasRunParams && (
        <section className="page-section">
          <div className="section-head">
            <h3>参数</h3>
            <button type="button" className="ghost sm" onClick={() => setParamsCollapsed((c) => !c)}>
              {paramsCollapsed ? "展开" : "折叠"}
            </button>
          </div>
          {!paramsCollapsed && (
            <>
              {choiceParams.length > 0 && (
                <div className="param-group">
                  {choiceParams.map((f) => (
                    <ParamInput key={f.name} field={f} value={pvals[f.name]} onChange={(v) => setParam(f.name, v)} />
                  ))}
                </div>
              )}
              {inputParams.length > 0 && (
                <div className="grid2">
                  {inputParams.map((f) => (
                    <ParamInput key={f.name} field={f} value={pvals[f.name]} onChange={(v) => setParam(f.name, v)} />
                  ))}
                </div>
              )}
            </>
          )}
        </section>
      )}

      <div className="action-bar">
        {launchError && (
          <div className="errbox launch-error" role="alert">
            {launchError}
          </div>
        )}
        <div className="actions launch-actions">
          <button
            type="button"
            className="primary"
            onClick={doLaunch}
            disabled={busy || !kindOk || !keyOk}
            title={!kindOk ? "调用协议不兼容" : !keyOk ? "请先配置 API Key" : ""}
          >
            启动实验
          </button>
        </div>
      </div>
    </div>
  );
}

export default memo(ProjectPanel);
