



export const GRID = "var(--border-2)";
export const TICK = { fill: "var(--text-3)", fontSize: 11 };


export const SERIES_COLORS = ["#0f5b45", "#6366f1", "#d97706", "#dc2626", "#0891b2", "#7c3aed", "#65a30d", "#db2777"];


export const BAR_COLORS = ["#2563eb", "#f97316", "#10b981", "#a855f7", "#ef4444", "#14b8a6"];

function sanitizeFilename(name) {
  return String(name || "").replace(/[\\/:*?"<>|\n]+/g, " ").trim().slice(0, 120) || "chart";
}





export async function exportChartPng(container, filename = "chart") {
  const svg = container?.querySelector("svg");
  if (!svg) return;

  const clone = svg.cloneNode(true);
  const srcTexts = svg.querySelectorAll("text, tspan");
  const dstTexts = clone.querySelectorAll("text, tspan");
  srcTexts.forEach((el, i) => {
    const d = dstTexts[i];
    if (!d) return;
    const cs = getComputedStyle(el);
    d.setAttribute("fill", cs.fill);
    d.setAttribute("font-size", cs.fontSize);
    d.setAttribute("font-family", cs.fontFamily);
  });

  const rootStyle = getComputedStyle(container);
  const raw = new XMLSerializer().serializeToString(clone);
  const resolved = raw.replace(
    /var\((--[\w-]+)(?:\s*,\s*([^)]+))?\)/g,
    (_, name, fallback) => rootStyle.getPropertyValue(name).trim() || (fallback || "#000").trim(),
  );

  const rect = svg.getBoundingClientRect();
  const w = Math.max(1, Math.round(rect.width));
  const h = Math.max(1, Math.round(rect.height));
  const svgUrl = URL.createObjectURL(new Blob([resolved], { type: "image/svg+xml;charset=utf-8" }));
  try {
    const img = new Image();
    await new Promise((res, rej) => {
      img.onload = res;
      img.onerror = rej;
      img.src = svgUrl;
    });
    const canvas = document.createElement("canvas");
    canvas.width = w * 2;
    canvas.height = h * 2;
    const ctx = canvas.getContext("2d");

    ctx.fillStyle = rootStyle.getPropertyValue("--surface").trim() || "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise((res) => canvas.toBlob(res, "image/png"));
    if (!blob) return;
    const dlUrl = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = dlUrl;
    a.download = `${sanitizeFilename(filename)}.png`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(dlUrl);
  } finally {
    URL.revokeObjectURL(svgUrl);
  }
}
