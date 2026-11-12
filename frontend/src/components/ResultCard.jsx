const SOURCE_NOTE = {
  user: "typed text",
  ocr: "text read from the image by OCR",
  none: "no text",
};

export default function ResultCard({ result }) {
  const entries = Object.entries(result.probabilities).sort((a, b) => b[1] - a[1]);

  return (
    <section className="card">
      <h2>Result &middot; {result.model_label}</h2>

      <div className="verdict">
        <span className={`pill sent-${result.label}`}>{result.label}</span>
        <span className="confidence">{(result.confidence * 100).toFixed(1)}% confidence</span>
      </div>

      <div className="bars">
        {entries.map(([label, p]) => (
          <div className="bar-row" key={label}>
            <span className="bar-label">{label}</span>
            <div className="bar-track">
              <div className={`bar-fill sent-${label}`} style={{ width: `${(p * 100).toFixed(1)}%` }} />
            </div>
            <span className="bar-value">{(p * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>

      {result.text_used ? (
        <p className="muted small">
          Text used ({SOURCE_NOTE[result.text_source]}): &ldquo;{result.text_used}&rdquo;
        </p>
      ) : (
        <p className="muted small">No text was used for this prediction.</p>
      )}
    </section>
  );
}
