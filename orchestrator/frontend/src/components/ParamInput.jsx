export default function ParamInput({ field, value, onChange }) {
  const v = value === undefined || value === null ? (field.default ?? "") : value;
  const shellClass = "param-input param-" + field.type;

  if (field.type === "bool") {
    return (
      <div className={shellClass}>
        <label className="check">
          <input
            type="checkbox"
            checked={!!(value ?? field.default)}
            onChange={(e) => onChange(e.target.checked)}
          />
          {field.label}
        </label>
        {field.help && <div className="hint">{field.help}</div>}
      </div>
    );
  }

  if (field.type === "select") {
    return (
      <div className={shellClass}>
        <label>{field.label}</label>
        <select value={v} onChange={(e) => onChange(e.target.value)}>
          {(field.options || []).map((o) => (
            <option key={o} value={o}>{o}</option>
          ))}
        </select>
        {field.help && <div className="hint">{field.help}</div>}
      </div>
    );
  }

  const type = field.type === "int" || field.type === "float" ? "number" : "text";
  const paperVal = field.paper_value;
  const hasPaper = paperVal !== undefined && paperVal !== null;
  const deviates = hasPaper && String(v ?? "") !== String(paperVal);
  return (
    <div className={shellClass}>
      <label>
        {field.label}
        {deviates && <span className="pill"> ⚠ 偏离原文（{String(paperVal)}）</span>}
      </label>
      <input
        type={type}
        step={field.type === "float" ? "0.1" : "1"}
        value={v}
        placeholder={field.default == null ? "(留空)" : ""}
        onChange={(e) => onChange(e.target.value)}
      />
      {deviates && (
        <button className="warn" onClick={() => onChange(paperVal)}>恢复原文默认</button>
      )}
      {field.help && <div className="hint">{field.help}</div>}
    </div>
  );
}
