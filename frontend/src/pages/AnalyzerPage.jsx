import { useCallback, useEffect, useState } from "react";
import { compare, getHealth, getModels, ocr, predict } from "../api.js";
import { useAuth } from "../auth/AuthContext.jsx";
import StatusBar from "../components/StatusBar.jsx";
import UploadPanel from "../components/UploadPanel.jsx";
import ModelPicker from "../components/ModelPicker.jsx";
import ResultCard from "../components/ResultCard.jsx";
import Explanation from "../components/Explanation.jsx";
import ComparisonTable from "../components/ComparisonTable.jsx";

// The meme analyser (previously the whole App). Only reachable when signed in.
export default function AnalyzerPage() {
  const { user, logout } = useAuth();

  const [health, setHealth] = useState(null);
  const [backendDown, setBackendDown] = useState(false);
  const [models, setModels] = useState([]);

  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [text, setText] = useState("");
  const [fusionType, setFusionType] = useState("text_only");
  const [explain, setExplain] = useState(false);

  const [busy, setBusy] = useState(null); // "predict" | "compare" | "ocr" | null
  const [result, setResult] = useState(null);
  const [comparison, setComparison] = useState(null);
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    try {
      const [h, m] = await Promise.all([getHealth(), getModels()]);
      setHealth(h);
      setModels(m);
      setBackendDown(false);
      setFusionType((current) => {
        if (m.find((x) => x.id === current && x.available)) return current;
        const first = m.find((x) => x.available);
        return first ? first.id : current;
      });
    } catch {
      setBackendDown(true);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useEffect(() => {
    if (!file) {
      setPreview(null);
      return undefined;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const handleFile = (f) => {
    setFile(f);
    setResult(null);
    setComparison(null);
    setError("");
  };

  const run = async (kind) => {
    setBusy(kind);
    setError("");
    try {
      if (kind === "ocr") {
        const out = await ocr(file);
        if (!out.ocr_available) {
          setError("OCR is not available on the server (install Tesseract).");
        } else if (!out.text) {
          setError("No text could be read from this image. Type the text manually.");
        } else {
          setText(out.text);
        }
      } else if (kind === "predict") {
        setComparison(null);
        setResult(await predict({ image: file, text, fusionType, explain }));
      } else {
        setResult(null);
        setComparison(await compare({ image: file, text }));
      }
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setBusy(null);
    }
  };

  const canRun = !busy && !backendDown && (file || text.trim());

  return (
    <div className="page">
      <header className="hero hero-row">
        <div>
          <h1>IndicSent-M</h1>
          <p>Multimodal sentiment analysis of memes from their text and image.</p>
        </div>
        <div className="userbox">
          <span className="who">
            Signed in as <strong>{user.name}</strong>
          </span>
          <button className="btn" onClick={logout}>
            Log out
          </button>
        </div>
      </header>

      <StatusBar health={health} down={backendDown} onRetry={refresh} />

      <main className="grid">
        <div className="col">
          <UploadPanel
            preview={preview}
            text={text}
            onText={setText}
            onFile={handleFile}
            onOcr={() => run("ocr")}
            busy={busy}
            ocrAvailable={health ? health.ocr_available : true}
          />

          <section className="card">
            <h2>2. Model</h2>
            <ModelPicker models={models} value={fusionType} onChange={setFusionType} />
            <label className="check">
              <input
                type="checkbox"
                checked={explain}
                onChange={(e) => setExplain(e.target.checked)}
              />
              Explain the prediction (slower on CPU)
            </label>
            <div className="actions">
              <button className="btn primary" disabled={!canRun} onClick={() => run("predict")}>
                {busy === "predict" ? "Analysing..." : "Analyse sentiment"}
              </button>
              <button className="btn" disabled={!canRun} onClick={() => run("compare")}>
                {busy === "compare" ? "Comparing..." : "Compare all models"}
              </button>
            </div>
          </section>
        </div>

        <div className="col">
          {error && (
            <div className="alert" role="alert">
              {error}
            </div>
          )}

          {!result && !comparison && !error && (
            <section className="card placeholder">
              <h2>Result</h2>
              <p>
                Upload a meme (and optionally its text), pick a model, and press
                <strong> Analyse sentiment</strong>.
              </p>
            </section>
          )}

          {result && (
            <>
              <ResultCard result={result} />
              {result.explanation && (
                <Explanation explanation={result.explanation} preview={preview} />
              )}
            </>
          )}

          {comparison && <ComparisonTable comparison={comparison} />}
        </div>
      </main>

      <footer className="foot">
        Labels in the training data are model-generated (weak) labels, so treat
        predictions as a research prototype, not ground truth.
      </footer>
    </div>
  );
}
