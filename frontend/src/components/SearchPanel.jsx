import { useState } from 'react';
import { searchChunks } from '../api.js';

// Cosine similarity bands (empirical for nomic-embed-text on this corpus):
// >0.75 strong match, 0.6-0.75 related, below = weak.
function scoreBadgeClass(score) {
  if (score > 0.75) return 'bg-green-100 text-green-800';
  if (score > 0.6) return 'bg-amber-100 text-amber-800';
  return 'bg-slate-100 text-slate-500';
}

const EXAMPLES = [
  'was the Preventive Detention Act held valid?',
  'meaning of procedure established by law in Article 21',
  'dissenting opinion on personal liberty',
];

export default function SearchPanel() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [openKey, setOpenKey] = useState(null);

  async function run(q) {
    const text = (q ?? query).trim();
    if (!text) return;
    setQuery(text);
    setLoading(true);
    setError(null);
    try {
      setResults(await searchChunks(text, 8));
    } catch (e) {
      setError(e.message);
      setResults(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex h-full flex-col">
      <form
        className="flex gap-2 border-b border-slate-200 p-3"
        onSubmit={(e) => {
          e.preventDefault();
          run();
        }}
      >
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask in plain language — search is by meaning, not keywords"
          className="min-w-0 flex-1 rounded border border-slate-300 px-3 py-1.5 text-sm"
        />
        <button
          type="submit"
          disabled={loading}
          className="rounded bg-slate-800 px-4 py-1.5 text-sm font-medium text-white disabled:opacity-50"
        >
          {loading ? 'Searching…' : 'Search'}
        </button>
      </form>

      {error && (
        <div className="m-3 rounded border border-red-200 bg-red-50 p-2 text-xs text-red-700">
          {error} — is the backend running and the document indexed?
        </div>
      )}

      {results === null && !error && (
        <div className="flex flex-1 flex-col items-center justify-center gap-3 p-6 text-sm text-slate-400">
          <p>Semantic search over every indexed judgment chunk. Try:</p>
          <div className="flex flex-col gap-2">
            {EXAMPLES.map((ex) => (
              <button
                key={ex}
                onClick={() => run(ex)}
                className="rounded border border-slate-200 bg-white px-3 py-1.5 text-left text-xs text-slate-600 hover:border-slate-400"
              >
                {ex}
              </button>
            ))}
          </div>
        </div>
      )}

      {results !== null && (
        <ul className="flex-1 space-y-2 overflow-y-auto p-3">
          {results.length === 0 && (
            <li className="text-sm text-slate-400">No results — is anything indexed yet?</li>
          )}
          {results.map((r) => {
            const key = `${r.doc_id}:${r.chunk_id}`;
            const open = openKey === key;
            return (
              <li key={key} className="rounded border border-slate-200">
                <button
                  onClick={() => setOpenKey(open ? null : key)}
                  className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs hover:bg-slate-50"
                >
                  <span
                    className={`shrink-0 rounded px-1.5 py-0.5 font-mono ${scoreBadgeClass(r.score)}`}
                  >
                    {r.score.toFixed(3)}
                  </span>
                  <span className="truncate rounded bg-slate-100 px-1.5 py-0.5 font-medium text-slate-600">
                    {r.section}
                  </span>
                  <span className="truncate text-slate-500">{r.case_title}</span>
                  <span className="ml-auto shrink-0 font-mono text-slate-400">{r.chunk_id}</span>
                </button>
                <p className="border-t border-slate-100 px-3 py-2 text-xs leading-relaxed text-slate-700">
                  {open ? (
                    <span className="whitespace-pre-wrap">{r.text}</span>
                  ) : (
                    `${r.text.slice(0, 220)}${r.text.length > 220 ? '…' : ''}`
                  )}
                </p>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
