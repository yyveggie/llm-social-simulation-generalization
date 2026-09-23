import { useEffect, useRef, useState } from "react";



export default function ConfirmDialog({ title, body, confirmText = "确认", danger = false, fields = [], onResolve }) {
  const cancelRef = useRef(null);
  const hasFields = Array.isArray(fields) && fields.length > 0;
  const [values, setValues] = useState(() => {
    const initial = {};
    fields.forEach((field) => {
      initial[field.name] = field.default ?? "";
    });
    return initial;
  });
  const [error, setError] = useState("");


  useEffect(() => { cancelRef.current?.focus(); }, []);



  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "Escape") {
        e.preventDefault();
        e.stopPropagation();
        onResolve(false);
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [onResolve]);

  const updateValue = (name, value) => {
    setError("");
    setValues((prev) => ({ ...prev, [name]: value }));
  };

  const confirm = () => {
    if (!hasFields) {
      onResolve(true);
      return;
    }
    const cleaned = {};
    for (const field of fields) {
      const raw = values[field.name];
      if (raw === "" || raw == null) {
        setError(`${field.label || field.name} 不能为空。`);
        return;
      }
      if (field.type === "int") {
        const n = Number(raw);
        if (!Number.isInteger(n) || n < Number(field.min ?? 0)) {
          setError(`${field.label || field.name} 必须是非负整数。`);
          return;
        }
        cleaned[field.name] = n;
      } else {
        cleaned[field.name] = raw;
      }
    }
    onResolve({ ok: true, values: cleaned });
  };

  return (
    <div className="confirm-backdrop" onClick={() => onResolve(false)}>
      <div className="confirm-modal" role="alertdialog" aria-modal="true" onClick={(e) => e.stopPropagation()}>
        <div className="confirm-title">{title}</div>
        <div className="confirm-body">{body}</div>
        {hasFields && (
          <div className="confirm-fields">
            {fields.map((field) => (
              <label key={field.name} className="confirm-field">
                <span>{field.label}</span>
                <input
                  type={field.type === "int" ? "number" : "text"}
                  min={field.min ?? undefined}
                  step={field.type === "int" ? 1 : undefined}
                  value={values[field.name] ?? ""}
                  onChange={(e) => updateValue(field.name, e.target.value)}
                />
                {field.help && <small>{field.help}</small>}
              </label>
            ))}
            {error && <div className="confirm-error">{error}</div>}
          </div>
        )}
        <div className="confirm-actions">
          <button ref={cancelRef} onClick={() => onResolve(false)}>取消</button>
          <button className={danger ? "danger-solid" : "primary"} onClick={confirm}>{confirmText}</button>
        </div>
      </div>
    </div>
  );
}
