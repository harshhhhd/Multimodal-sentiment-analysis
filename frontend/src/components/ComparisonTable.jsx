export default function ComparisonTable({ comparison }) {
  const { results, skipped, text_used: textUsed, text_source: source } = comparison;

  return (
    <section className="card">
      <h2>Model comparison</h2>

      {results.length > 0 ? (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Prediction</th>
                <th>Confidence</th>
                <th>Negative</th>
                <th>Neutral</th>
                <th>Positive</th>
              </tr>
            </thead>
            <tbody>
              {results.map((r) => (
                <tr key={r.model}>
                  <td>{r.model_label}</td>
                  <td>
                    <span className={`pill small sent-${r.label}`}>{r.label}</span>
                  </td>
                  <td>{(r.confidence * 100).toFixed(1)}%</td>
                  <td>{((r.probabilities.negative ?? 0) * 100).toFixed(1)}%</td>
                  <td>{((r.probabilities.neutral ?? 0) * 100).toFixed(1)}%</td>
                  <td>{((r.probabilities.positive ?? 0) * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="muted">No model could run on this input.</p>
      )}

      {skipped.length > 0 && (
        <ul className="skipped">
          {skipped.map((s) => (
            <li key={s.model}>
              <strong>{s.model_label}</strong> skipped: {s.reason}
            </li>
          ))}
        </ul>
      )}

      {textUsed && (
        <p className="muted small">
          Text used ({source === "ocr" ? "read by OCR" : "typed"}): &ldquo;{textUsed}&rdquo;
        </p>
      )}
    </section>
  );
}
