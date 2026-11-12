import { useRef, useState } from "react";

export default function UploadPanel({
  preview,
  text,
  onText,
  onFile,
  onOcr,
  busy,
  ocrAvailable,
}) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f && f.type.startsWith("image/")) onFile(f);
  };

  const openPicker = () => inputRef.current && inputRef.current.click();

  return (
    <section className="card">
      <h2>1. Meme</h2>

      <div
        className={`dropzone ${dragging ? "drag" : ""}`}
        role="button"
        tabIndex={0}
        onClick={openPicker}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            openPicker();
          }
        }}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
      >
        {preview ? (
          <img src={preview} alt="Uploaded meme preview" />
        ) : (
          <p>Drop a meme image here, or click to browse</p>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          hidden
          onChange={(e) => {
            const f = e.target.files && e.target.files[0];
            if (f) onFile(f);
            e.target.value = "";
          }}
        />
      </div>

      <label htmlFor="meme-text">Meme text</label>
      <textarea
        id="meme-text"
        rows={4}
        value={text}
        onChange={(e) => onText(e.target.value)}
        placeholder="Leave empty to read the text from the image with OCR"
      />

      <button
        className="btn ghost"
        disabled={!preview || Boolean(busy) || !ocrAvailable}
        onClick={onOcr}
        title={ocrAvailable ? "" : "Tesseract is not available on the server"}
      >
        {busy === "ocr" ? "Extracting..." : "Extract text from image"}
      </button>
    </section>
  );
}
