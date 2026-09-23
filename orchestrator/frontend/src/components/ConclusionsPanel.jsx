import { useState, useEffect, memo } from "react";
import Markdown from "./Markdown.jsx";
import ProjectTabs from "./ProjectTabs.jsx";

function ConclusionsPanel({ projects }) {
  const [activeId, setActiveId] = useState(() => projects[0]?.id ?? null);

  useEffect(() => {
    if (projects.length && !activeId) setActiveId(projects[0].id);
  }, [projects]);

  const proj = projects.find((p) => p.id === activeId);
  if (!proj) return <div className="muted">加载项目中…</div>;

  const exps = proj.experiments || [];

  return (
    <div className="page-stack conclusions-page">
      <ProjectTabs projects={projects} activeId={activeId} onSelect={setActiveId} />

      <section className="project-summary">
        {proj.intro && <div className="intro">{proj.intro.replace(/\n{2,}/g, "\n")}</div>}
      </section>

      <section className="page-section">
        <div className="section-head">
          <h3>各实验 · 原作者结论</h3>
        </div>
        {exps.length === 0 ? (
          <div className="empty-block">该项目暂无实验。</div>
        ) : (
          <div className="concl-list">
            {exps.map((e) => {
              const text = (e.conclusion || "").trim();
              return (
                <div className="concl-item" key={e.id}>
                  <div className="concl-head">
                    <span className="concl-exp">{e.label}</span>
                    <code className="concl-id">{e.id}</code>
                  </div>
                  {e.description && <div className="concl-desc muted text-xs">{e.description}</div>}
                  {text ? (
                    <Markdown text={text} className="concl-text" />
                  ) : (
                    <div className="concl-text empty">（结论待补充）</div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </section>
    </div>
  );
}

export default memo(ConclusionsPanel);
