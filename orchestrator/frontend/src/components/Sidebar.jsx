import { useState } from "react";

const ICONS = {
  monitor: (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M1.5 9.5 5 6l2.5 2.5L11 4l3.5 3.5" />
      <path d="M1.5 13.5h13" opacity="0.45" />
    </svg>
  ),
  launch: (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M5 2.5h6l3 3v8a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V3.5a1 1 0 0 1 1-1Z" opacity="0.5" />
      <path d="M8 6.5v4M6 8.5l2-2 2 2" />
    </svg>
  ),
  settings: (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="8" cy="8" r="2" />
      <path d="M8 1.5v2M8 12.5v2M14.5 8h-2M3.5 8h-2M12.6 3.4l-1.4 1.4M4.8 11.2l-1.4 1.4M12.6 12.6l-1.4-1.4M4.8 4.8 3.4 3.4" opacity="0.55" />
    </svg>
  ),
  conclusions: (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3.5" y="2" width="9" height="12" rx="1" opacity="0.5" />
      <path d="M5.5 6h5M5.5 8.5h5M5.5 11h3" />
    </svg>
  ),
  analysis: (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M2 14h12" opacity="0.45" />
      <rect x="3" y="8" width="2.4" height="4" rx="0.5" />
      <rect x="6.8" y="5" width="2.4" height="7" rx="0.5" />
      <rect x="10.6" y="2.5" width="2.4" height="9.5" rx="0.5" />
    </svg>
  ),
  trends: (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M2 13 6 8.5l2.5 2L14 4" />
      <path d="M10.5 4H14v3.5" />
      <path d="M2 14.5h12" opacity="0.45" />
    </svg>
  ),
  collapse: (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="2.5" y="3" width="11" height="10" rx="2.5" />
      <path d="M6.5 3.5v9" />
    </svg>
  ),
  sun: (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="8" cy="8" r="2.6" />
      <path d="M8 1.5v1.6M8 12.9v1.6M14.5 8h-1.6M3.1 8H1.5M12.6 3.4l-1.2 1.2M4.6 11.4l-1.2 1.2M12.6 12.6l-1.2-1.2M4.6 4.6 3.4 3.4" opacity="0.7" />
    </svg>
  ),
  moon: (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M13.2 9.9A5.6 5.6 0 0 1 6.1 2.8a5.6 5.6 0 1 0 7.1 7.1Z" />
    </svg>
  ),
};

const NAV = [
  { id: "monitor", label: "监控", icon: ICONS.monitor },
  { id: "settings", label: "配置", icon: ICONS.settings },
  { id: "launch", label: "发起实验", icon: ICONS.launch },
  { id: "analysis", label: "结果分析", icon: ICONS.analysis },
  { id: "trends", label: "演化趋势", icon: ICONS.trends },
  { id: "conclusions", label: "原作者结论", icon: ICONS.conclusions },
];

export default function Sidebar({ view, setView, runningCount }) {
  const [collapsed, setCollapsed] = useState(false);

  const [theme, setTheme] = useState(() => (document.documentElement.dataset.theme === "dark" ? "dark" : "light"));
  const toggleTheme = () => {
    const next = theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch {  }
    setTheme(next);
  };
  return (
    <aside className={"sidebar" + (collapsed ? " collapsed" : "")}>
      <div className="brand">
        <div className="brand-mark" aria-hidden="true">

          <svg viewBox="0 0 16 16" fill="none" stroke="#fff" strokeWidth="1.3" strokeLinecap="round">
            <path d="M8 8 12.4 4.6M8 8 3.6 5.2M8 8l.4 4.6" />
            <circle cx="8" cy="8" r="1.7" fill="#fff" stroke="none" />
            <circle cx="12.4" cy="4.6" r="1.15" fill="#fff" stroke="none" />
            <circle cx="3.6" cy="5.2" r="1.15" fill="#fff" stroke="none" />
            <circle cx="8.4" cy="12.6" r="1.15" fill="#fff" stroke="none" />
          </svg>
        </div>
        <div className="title">LLM 认知实验平台</div>
        <button
          className="collapse-btn"
          onClick={() => setCollapsed((c) => !c)}
          title={collapsed ? "展开侧边栏" : "折叠侧边栏"}
          aria-label={collapsed ? "展开侧边栏" : "折叠侧边栏"}
        >
          {ICONS.collapse}
        </button>
      </div>

      <nav className="nav">
        {NAV.map((n) => (
          <button
            key={n.id}
            className={"nav-item" + (view === n.id ? " active" : "")}
            onClick={() => setView(n.id)}
            title={n.label}
          >
            {n.icon}
            <span>{n.label}</span>
            {n.id === "monitor" && runningCount > 0 && <span className="badge">{runningCount}</span>}
          </button>
        ))}
      </nav>

      <div className="side-foot">
        <button
          className="nav-item"
          onClick={toggleTheme}
          title={theme === "dark" ? "切换到浅色模式" : "切换到深色模式"}
        >
          {theme === "dark" ? ICONS.sun : ICONS.moon}
          <span>{theme === "dark" ? "浅色模式" : "深色模式"}</span>
        </button>
      </div>
    </aside>
  );
}
