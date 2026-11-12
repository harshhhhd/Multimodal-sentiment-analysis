export default function StatusBar({ health, down, onRetry }) {
  if (down) {
    return (
      <div className="status bad">
        <span>
          Backend is not reachable. Start it with{" "}
          <code>uvicorn backend.main:app --port 8000</code> from the project root.
        </span>
        <button className="link" onClick={onRetry}>
          Retry
        </button>
      </div>
    );
  }
  if (!health) {
    return <div className="status">Connecting to backend...</div>;
  }

  const n = health.available_models.length;
  return (
    <div className={`status ${n ? "ok" : "warn"}`}>
      <span>
        Backend online &middot; device: {health.device} &middot; trained models: {n}/4 &middot;
        OCR: {health.ocr_available ? "on" : "off"}
      </span>
      {n === 0 && (
        <span>
          {" "}
          No checkpoints yet. Run <code>python -m src.train --fusion_type text_only</code>.
        </span>
      )}
    </div>
  );
}
