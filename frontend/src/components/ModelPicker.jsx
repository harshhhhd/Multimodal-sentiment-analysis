export default function ModelPicker({ models, value, onChange }) {
  if (!models.length) {
    return <p className="muted">Loading models...</p>;
  }
  return (
    <div className="models" role="radiogroup" aria-label="Model variant">
      {models.map((m) => (
        <label
          key={m.id}
          className={`model ${value === m.id ? "selected" : ""} ${m.available ? "" : "disabled"}`}
        >
          <input
            type="radio"
            name="model"
            value={m.id}
            checked={value === m.id}
            disabled={!m.available}
            onChange={() => onChange(m.id)}
          />
          <span className="model-name">{m.label}</span>
          <span className="model-meta">
            {m.available
              ? [m.needs_text && "text", m.needs_image && "image"].filter(Boolean).join(" + ")
              : "not trained yet"}
          </span>
        </label>
      ))}
    </div>
  );
}
