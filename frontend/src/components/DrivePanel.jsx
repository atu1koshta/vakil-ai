import { useEffect, useState } from 'react';
import {
  driveAuthStatus,
  driveAuthUrl,
  driveDisconnect,
  listDriveFiles,
  processDriveFile,
} from '../api.js';

export default function DrivePanel({ processing, setProcessing, setError, onResult }) {
  const [authorized, setAuthorized] = useState(null); // null = unknown
  const [files, setFiles] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeId, setActiveId] = useState(null);

  useEffect(() => {
    // Returning from the OAuth redirect: ?drive=connected|error
    const params = new URLSearchParams(window.location.search);
    const driveParam = params.get('drive');
    if (driveParam) {
      window.history.replaceState({}, '', window.location.pathname);
      if (driveParam === 'error') setError('Drive: authorization failed, try again.');
    }
    driveAuthStatus()
      .then(({ authorized: ok }) => {
        setAuthorized(ok);
        if (ok) refresh();
      })
      .catch(() => setAuthorized(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function connect() {
    setError(null);
    try {
      const { url } = await driveAuthUrl();
      window.location.href = url; // Google consent screen, redirects back via backend
    } catch (err) {
      setError(`Drive: ${err.message}`);
    }
  }

  async function disconnect() {
    await driveDisconnect().catch(() => {});
    setAuthorized(false);
    setFiles(null);
  }

  async function refresh() {
    setLoading(true);
    setError(null);
    try {
      setFiles(await listDriveFiles());
      setAuthorized(true);
    } catch (err) {
      setError(`Drive: ${err.message}`);
      setFiles(null);
    } finally {
      setLoading(false);
    }
  }

  async function process(file) {
    if (processing) return;
    setProcessing(true);
    setActiveId(file.id);
    setError(null);
    try {
      onResult(await processDriveFile(file.id));
    } catch (err) {
      setError(`Drive: ${err.message}`);
    } finally {
      setProcessing(false);
      setActiveId(null);
    }
  }

  return (
    <div>
      <div className="mb-2 flex items-center justify-between">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
          Google Drive
        </h2>
        {authorized ? (
          <div className="flex gap-1">
            <button
              onClick={refresh}
              disabled={loading}
              className="rounded bg-slate-800 px-2 py-1 text-xs text-white hover:bg-slate-700 disabled:opacity-50"
            >
              {loading ? 'Loading…' : 'Refresh'}
            </button>
            <button
              onClick={disconnect}
              className="rounded border border-slate-300 px-2 py-1 text-xs text-slate-500 hover:bg-slate-50"
              title="Remove stored Drive token"
            >
              ✕
            </button>
          </div>
        ) : (
          <button
            onClick={connect}
            disabled={authorized === null}
            className="rounded bg-blue-600 px-2 py-1 text-xs text-white hover:bg-blue-500 disabled:opacity-50"
          >
            Connect Drive
          </button>
        )}
      </div>

      {!authorized ? (
        <p className="text-xs text-slate-400">
          Connect your Google account to list PDFs from the configured folder
          (read-only access).
        </p>
      ) : files === null ? (
        <p className="text-xs text-slate-400">Connected. Hit Refresh to list PDFs.</p>
      ) : files.length === 0 ? (
        <p className="text-xs text-slate-400">No PDFs in the folder.</p>
      ) : (
        <ul className="space-y-1">
          {files.map((file) => (
            <li key={file.id}>
              <button
                onClick={() => process(file)}
                disabled={processing}
                className="group flex w-full items-center gap-2 rounded border border-slate-200 px-2 py-1.5 text-left text-xs hover:border-blue-300 hover:bg-blue-50 disabled:opacity-50"
                title={`Process ${file.name}`}
              >
                <span className="min-w-0 flex-1 truncate">{file.name}</span>
                <span
                  className={`shrink-0 font-medium ${
                    activeId === file.id
                      ? 'animate-pulse text-blue-600'
                      : 'text-blue-600 opacity-0 group-hover:opacity-100'
                  }`}
                >
                  {activeId === file.id ? 'Processing…' : 'Process ▶'}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
