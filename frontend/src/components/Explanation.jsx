// Occlusion explanations from the backend.
// score > 0: removing this word / blanking this region LOWERED the predicted class
// probability, so it supported the prediction. score < 0: it worked against it.

function strength(score, max) {
  if (!max) return 0;
  return Math.min(1, Math.abs(score) / max);
}

export default function Explanation({ explanation, preview }) {
  const { words, image_grid: grid } = explanation;

  const maxWord = words ? Math.max(...words.map((w) => Math.abs(w.score)), 1e-9) : 0;
  const flat = grid ? grid.flat() : [];
  const maxCell = flat.length ? Math.max(...flat.map((v) => Math.max(v, 0)), 1e-9) : 0;

  return (
    <section className="card">
      <h2>Why this prediction?</h2>

      {words && (
        <div className="expl-block">
          <h3>Words</h3>
          <div className="chips">
            {words.map((w, i) => {
              const a = strength(w.score, maxWord);
              const color = w.score >= 0 ? "var(--support)" : "var(--oppose)";
              return (
                <span
                  key={`${w.word}-${i}`}
                  className="chip"
                  style={{ background: `color-mix(in srgb, ${color} ${Math.round(a * 70)}%, transparent)` }}
                  title={`${w.score >= 0 ? "+" : ""}${(w.score * 100).toFixed(1)} pts`}
                >
                  {w.word}
                </span>
              );
            })}
          </div>
          <p className="muted small">
            Blue words supported the prediction; orange words argued against it. Only the
            first 20 words are explained.
          </p>
        </div>
      )}

      {grid && preview && (
        <div className="expl-block">
          <h3>Image regions</h3>
          <div className="heat">
            <img src={preview} alt="Meme with importance overlay" />
            <div className="heat-grid" style={{ gridTemplateColumns: `repeat(${grid.length}, 1fr)` }}>
              {flat.map((v, i) => (
                <div
                  key={i}
                  style={{
                    background: `color-mix(in srgb, var(--support) ${Math.round(strength(Math.max(v, 0), maxCell) * 65)}%, transparent)`,
                  }}
                  title={`${v >= 0 ? "+" : ""}${(v * 100).toFixed(1)} pts`}
                />
              ))}
            </div>
          </div>
          <p className="muted small">
            Darker cells are regions whose removal lowered the model&apos;s confidence the most.
          </p>
        </div>
      )}

      {!words && !grid && <p className="muted">No explanation is available for this input.</p>}
    </section>
  );
}
