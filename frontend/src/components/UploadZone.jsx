import { useRef, useState } from 'react';
import { processDocument } from '../api.js';

export default function UploadZone({ processing, setProcessing, setError, onResult }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  async function handleFile(file) {
    if (!file || processing) return;
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setError('Only PDF files are supported.');
      return;
    }
    setProcessing(true);
    setError(null);
    try {
      const data = await processDocument(file);
      onResult(file, data);
    } catch (err) {
      setError(err.message);
    } finally {
      setProcessing(false);
    }
  }

  return (
    <div>
      <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
        Upload
      </h2>
      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          handleFile(e.dataTransfer.files[0]);
        }}
        className={`cursor-pointer rounded-lg border-2 border-dashed p-6 text-center text-sm transition-colors ${
          dragging
            ? 'border-blue-400 bg-blue-50 text-blue-600'
            : 'border-slate-300 text-slate-500 hover:border-slate-400'
        }`}
      >
        {processing ? 'Processing… (first run loads Docling models)' : 'Drop judgment PDF here or click to browse'}
      </div>
      <input
        ref={inputRef}
        type="file"
        accept="application/pdf"
        className="hidden"
        onChange={(e) => handleFile(e.target.files[0])}
      />
    </div>
  );
}
