import { useState, useEffect, useRef, useCallback } from "react";




export default function ProjectTabs({ projects, activeId, onSelect }) {
  const ref = useRef(null);
  const [fade, setFade] = useState({ left: false, right: false });

  const update = useCallback(() => {
    const el = ref.current;
    if (!el) return;
    const max = el.scrollWidth - el.clientWidth;
    const next = { left: el.scrollLeft > 2, right: el.scrollLeft < max - 2 };
    setFade((prev) => (prev.left === next.left && prev.right === next.right ? prev : next));
  }, []);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    update();
    el.addEventListener("scroll", update, { passive: true });
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => {
      el.removeEventListener("scroll", update);
      ro.disconnect();
    };
  }, [update, projects]);

  return (
    <div
      ref={ref}
      className={"tabs project-tabs" + (fade.left ? " fade-l" : "") + (fade.right ? " fade-r" : "")}
    >
      {projects.map((p) => (
        <button
          type="button"
          key={p.id}
          className={"tab " + (p.id === activeId ? "active" : "")}
          onClick={() => onSelect(p.id)}
        >
          {p.name}
        </button>
      ))}
    </div>
  );
}
