import { useCallback, useEffect, useState } from 'react';
import { driveSyncStart, driveSyncStatus, listDocuments } from '../api.js';

// Poll fast while a sweep runs, slow when idle — the status endpoint is cheap.
const POLL_RUNNING_MS = 3000;
const POLL_IDLE_MS = 15000;

export default function DashboardPanel() {
  const [sync, setSync] = useState(null);
  const [docs, setDocs] = useState([]);
  const [error, setError] = useState(null);
  const [starting, setStarting] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const [s, d] = await Promise.all([driveSyncStatus(), listDocuments()]);
      setSync(s);
      setDocs(d);
      setError(null);
    } catch (e) {
      setError(e.message);
    }
  }, []);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, sync?.running ? POLL_RUNNING_MS : POLL_IDLE_MS);
    return () => clearInterval(interval);
  }, [refresh, sync?.running]);

  async function startSync() {
    setStarting(true);
    try {
      await driveSyncStart();
      await refresh();
    } catch (e) {
      setError(e.message);
    } finally {
      setStarting(false);
    }
  }

  const total = sync ? sync.done + sync.pending + sync.failed : 0;
  const pct = total ? Math.round((sync.done / total) * 100) : 0;

  return (
    <div className="flex h-full flex-col gap-4 overflow-y-auto p-4">
      {error && (
        <div className="rounded border border-red-200 bg-red-50 p-2 text-xs text-red-700">
          {error}
        </div>
      )}

      <section className="rounded border border-slate-200 p-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold">Drive bulk sync</h2>
          <div className="flex items-center gap-2">
            <button
              onClick={refresh}
              className="rounded border border-slate-300 px-3 py-1 text-xs text-slate-600 hover:border-slate-500"
            >
              Refresh ⟳
            </button>
            <button
              onClick={startSync}
              disabled={starting || sync?.running}
              className="rounded bg-slate-800 px-3 py-1 text-xs font-medium text-white disabled:opacity-50"
            >
              {sync?.running ? 'Sync running…' : 'Start sync'}
            </button>
          </div>
        </div>

        {sync && (
          <>
            <div className="mt-3 flex items-center gap-4 text-xs">
              <Stat label="done" value={sync.done} className="text-green-700" />
              <Stat label="pending" value={sync.pending} className="text-amber-700" />
              <Stat label="failed" value={sync.failed} className="text-red-700" />
              <span className="ml-auto text-slate-500">{pct}%</span>
            </div>
            <div className="mt-2 h-2 overflow-hidden rounded bg-slate-100">
              <div className="h-2 bg-green-500 transition-all" style={{ width: `${pct}%` }} />
            </div>
            {sync.running && sync.current && (
              <p className="mt-2 text-xs text-slate-600">
                <span className="mr-1 inline-block h-2 w-2 animate-pulse rounded-full bg-amber-500" />
                processing: <span className="font-mono">{sync.current}</span>
              </p>
            )}
            {sync.recent_failures?.length > 0 && (
              <details className="mt-3 text-xs">
                <summary className="cursor-pointer text-red-700">
                  {sync.recent_failures.length} recent failure(s)
                </summary>
                <ul className="mt-1 space-y-1">
                  {sync.recent_failures.map((f) => (
                    <li key={f.file_id} className="rounded bg-red-50 p-1.5">
                      <span className="font-mono">{f.name}</span>
                      <span className="text-slate-500"> · {f.attempts} attempt(s) · </span>
                      {f.error}
                    </li>
                  ))}
                </ul>
              </details>
            )}
          </>
        )}
      </section>

      <section className="rounded border border-slate-200 p-4">
        <h2 className="text-sm font-semibold">
          Processed documents <span className="font-normal text-slate-400">({docs.length})</span>
        </h2>
        <table className="mt-2 w-full text-xs">
          <thead>
            <tr className="text-left text-slate-500">
              <th className="py-1 pr-2">Document</th>
              <th className="py-1 pr-2">Status</th>
              <th className="py-1 pr-2">Chunks</th>
              <th className="py-1 pr-2">Tokens</th>
              <th className="py-1 pr-2">v</th>
              <th className="py-1">Processed</th>
            </tr>
          </thead>
          <tbody>
            {docs.map((d) => (
              <tr key={d.doc_id} className="border-t border-slate-100">
                <td className="max-w-[16rem] truncate py-1 pr-2 font-medium text-slate-700">
                  {d.doc_id}
                </td>
                <td className="py-1 pr-2">
                  <span
                    className={`rounded px-1.5 py-0.5 ${
                      d.status === 'processed'
                        ? 'bg-green-100 text-green-800'
                        : 'bg-red-100 text-red-700'
                    }`}
                  >
                    {d.status}
                  </span>
                </td>
                <td className="py-1 pr-2">{d.chunk_count ?? '—'}</td>
                <td className="py-1 pr-2">{d.total_tokens ?? '—'}</td>
                <td className="py-1 pr-2 text-slate-400">v{d.pipeline_version}</td>
                <td className="py-1 text-slate-500">{(d.processed_at || '').slice(0, 16)}</td>
              </tr>
            ))}
            {docs.length === 0 && (
              <tr>
                <td colSpan="6" className="py-2 text-slate-400">
                  nothing processed yet
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </section>
    </div>
  );
}

function Stat({ label, value, className }) {
  return (
    <span>
      <span className={`font-semibold ${className}`}>{value}</span>
      <span className="ml-1 text-slate-500">{label}</span>
    </span>
  );
}
