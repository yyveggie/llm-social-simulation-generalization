export default function Sparkline({ data = [], width = 90, height = 30 }) {
  const pts = data.length ? data : [0, 0];
  const max = Math.max(...pts);
  const min = Math.min(...pts);
  const span = max - min;
  const n = pts.length;
  const stepX = n > 1 ? width / (n - 1) : width;
  const y = (v) => (span === 0 ? height / 2 : height - 3 - ((v - min) / span) * (height - 6));
  const round = (x) => Math.round(x * 100) / 100;
  const coords = pts.map((v, i) => [round(i * stepX), round(y(v))]);
  const line = coords.map((p) => p.join(",")).join(" ");
  const area = `0,${height} ${line} ${width},${height}`;

  return (
    <svg className="spark" width={width} height={height} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
      <polygon className="spark-area" points={area} />
      <polyline className="spark-line" points={line} />
    </svg>
  );
}
