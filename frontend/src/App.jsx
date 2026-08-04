import { useState } from 'react';
import UploadZone from './components/UploadZone.jsx';
import DrivePanel from './components/DrivePanel.jsx';
import PdfPanel from './components/PdfPanel.jsx';
import MarkdownPanel from './components/MarkdownPanel.jsx';
import MetadataCard from './components/MetadataCard.jsx';
import ChunkViewer from './components/ChunkViewer.jsx';
import SearchPanel from './components/SearchPanel.jsx';
import DashboardPanel from './components/DashboardPanel.jsx';
import IndexStatusBadge from './components/IndexStatusBadge.jsx';
import { documentPdfUrl } from './api.js';

const TABS = ['Dashboard', 'Search', 'Markdown', 'Metadata', 'Chunks'];

export default function App() {
  const [result, setResult] = useState(null);
  const [pdfUrl, setPdfUrl] = useState(null);
  const [tab, setTab] = useState('Dashboard');
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState(null);

  // Upload flow already has the file locally; Drive flow fetches it back
  // from the backend's persisted copy.
  function onUploadResult(file, data) {
    setResult(data);
    setPdfUrl(URL.createObjectURL(file));
    setError(null);
  }

  function onDriveResult(data) {
    setResult(data);
    setPdfUrl(documentPdfUrl(data.doc_id));
    setError(null);
  }

  return (
    <div className="flex h-full flex-col bg-slate-100 text-slate-900">
      <header className="flex items-center justify-between border-b border-slate-200 bg-white px-4 py-2">
        <div>
          <h1 className="text-lg font-semibold">Vakil AI — Document Processing Studio</h1>
          <p className="text-xs text-slate-500">
            parse → metadata → chunks → embeddings → semantic search
          </p>
        </div>
        {result && (
          <div className="text-right text-xs text-slate-500">
            <div className="flex items-center justify-end gap-2">
              <span className="font-medium text-slate-700">{result.doc_id}</span>
              <IndexStatusBadge docId={result.doc_id} />
            </div>
            <div>
              {result.stats.total_chunks} chunks · {result.stats.total_tokens} tokens
            </div>
          </div>
        )}
      </header>

      <div className="flex min-h-0 flex-1">
        <aside className="flex w-72 shrink-0 flex-col gap-4 overflow-y-auto border-r border-slate-200 bg-white p-4">
          <UploadZone
            processing={processing}
            setProcessing={setProcessing}
            setError={setError}
            onResult={onUploadResult}
          />
          <DrivePanel
            processing={processing}
            setProcessing={setProcessing}
            setError={setError}
            onResult={onDriveResult}
          />
          {error && (
            <div className="rounded border border-red-200 bg-red-50 p-2 text-xs text-red-700">
              {error}
            </div>
          )}
        </aside>

        <main className="flex min-w-0 flex-1">
          <section className="min-w-0 flex-1 border-r border-slate-200 bg-slate-50">
            <PdfPanel url={pdfUrl} />
          </section>

          <section className="flex min-w-0 flex-1 flex-col bg-white">
            <nav className="flex gap-1 border-b border-slate-200 px-2 pt-2">
              {TABS.map((name) => (
                <button
                  key={name}
                  onClick={() => setTab(name)}
                  className={`rounded-t px-3 py-1.5 text-sm ${
                    tab === name
                      ? 'border border-b-0 border-slate-200 bg-white font-medium'
                      : 'text-slate-500 hover:text-slate-800'
                  }`}
                >
                  {name}
                </button>
              ))}
            </nav>
            <div className="min-h-0 flex-1 overflow-hidden">
              {tab === 'Dashboard' ? (
                // corpus-wide — works without a loaded document
                <DashboardPanel />
              ) : tab === 'Search' ? (
                <SearchPanel />
              ) : !result ? (
                <div className="flex h-full items-center justify-center text-sm text-slate-400">
                  Upload a judgment PDF or pick one from Drive.
                </div>
              ) : tab === 'Markdown' ? (
                <MarkdownPanel markdown={result.markdown} />
              ) : tab === 'Metadata' ? (
                <MetadataCard metadata={result.metadata} stats={result.stats} />
              ) : (
                <ChunkViewer chunks={result.chunks} stats={result.stats} />
              )}
            </div>
          </section>
        </main>
      </div>
    </div>
  );
}
