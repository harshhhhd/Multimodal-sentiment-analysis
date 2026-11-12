export default function TextField({
  id,
  label,
  value,
  onChange,
  onBlur,
  type = "text",
  autoComplete,
  autoFocus = false,
  error,
  hint,
  placeholder,
}) {
  const describedBy = error ? `${id}-error` : hint ? `${id}-hint` : undefined;
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      <input
        id={id}
        name={id}
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onBlur={onBlur}
        autoComplete={autoComplete}
        autoFocus={autoFocus}
        placeholder={placeholder}
        aria-invalid={error ? "true" : "false"}
        aria-describedby={describedBy}
      />
      {error ? (
        <p className="field-error" id={`${id}-error`}>
          {error}
        </p>
      ) : hint ? (
        <p className="field-hint" id={`${id}-hint`}>
          {hint}
        </p>
      ) : null}
    </div>
  );
}
