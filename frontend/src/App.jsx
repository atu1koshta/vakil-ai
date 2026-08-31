import { useEffect, useState } from 'react';
import UploadZone from './components/UploadZone.jsx';
import DrivePanel from './components/DrivePanel.jsx';
import PdfPanel from './components/PdfPanel.jsx';
import MarkdownPanel from './components/MarkdownPanel.jsx';
import MetadataCard from './components/MetadataCard.jsx';
import ChunkViewer from './components/ChunkViewer.jsx';
import SearchPanel from './components/SearchPanel.jsx';
import AgentPanel from './components/AgentPanel.jsx';
import CitationGraphPanel from './components/CitationGraphPanel.jsx';
import DashboardPanel from './components/DashboardPanel.jsx';
import ProgressPage from './components/ProgressPage.jsx';
import GrowthPage from './components/GrowthPage.jsx';
import IndexStatusBadge from './components/IndexStatusBadge.jsx';
import ProfileBadge from './components/ProfileBadge.jsx';
import { documentPdfUrl, listProfiles } from './api.js';

const TABS = ['Dashboard', 'Search', 'Agent', 'Citations', 'Markdown', 'Metadata', 'Chunks'];

export default function App() {
  // Poor-man's routing: static pages render by pathname, everything else the
  // studio. All pages are flat and param-free, so a router still buys nothing.
  if (window.location.pathname === '/progress') return <ProgressPage />;
  if (window.location.pathname === '/growth') return <GrowthPage />;

  return <Studio />;
}

function Studio() {
  const [result, setResult] = useState(null);
  const [pdfUrl, setPdfUrl] = useState(null);
  const [tab, setTab] = useState('Dashboard');
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState(null);
  // {active, profiles: {name: snapshot}} from /profiles; null until loaded,
  // stays null if the backend is down — profile UI just doesn't render.
  const [profileInfo, setProfileInfo] = useState(null);

  useEffect(() => {
    listProfiles().then(setProfileInfo).catch(() => setProfileInfo(null));
  }, []);

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
        <div className="flex items-center gap-3">
          <div>
            <h1 className="text-lg font-semibold">Vakil AI — Document Processing Studio</h1>
            <p className="text-xs text-slate-500">
              parse → metadata → chunks → embeddings → semantic search
            </p>
          </div>
          <ProfileBadge profileInfo={profileInfo} />
          <a
            href="/progress"
            className="rounded border border-slate-200 px-2 py-1 text-xs font-medium text-slate-500 hover:border-sky-300 hover:text-sky-700"
          >
            Progress
          </a>
          <a
            href="/growth"
            className="rounded border border-slate-200 px-2 py-1 text-xs font-medium text-slate-500 hover:border-sky-300 hover:text-sky-700"
          >
            Growth
          </a>
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
                <SearchPanel profileInfo={profileInfo} />
              ) : tab === 'Agent' ? (
                // agentic QA over the whole corpus — works without a loaded document
                <AgentPanel />
              ) : tab === 'Citations' ? (
                // citation graph — has its own doc selector, preseeded with
                // the loaded document when there is one
                <CitationGraphPanel key={result?.doc_id} initialDocId={result?.doc_id} />
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
