import { useMemo, useState } from 'react';

// Target band from the chunking strategy: 400-800 tokens.
function tokenBadgeClass(tokens) {
  if (tokens < 400) return 'bg-amber-100 text-amber-800';
  if (tokens <= 800) return 'bg-green-100 text-green-800';
  return 'bg-red-100 text-red-700';
}

export default function ChunkViewer({ chunks, stats }) {
  const [sectionFilter, setSectionFilter] = useState('All');
  const [openId, setOpenId] = useState(null);

  const visible = useMemo(
    () =>
      sectionFilter === 'All' ? chunks : chunks.filter((c) => c.section === sectionFilter),
    [chunks, sectionFilter]
  );
  const maxTokens = Math.max(...chunks.map((c) => c.token_count), 1);

  return (
    <div className="flex h-full flex-col">
      <div className="flex flex-wrap items-center gap-2 border-b border-slate-200 p-3 text-xs">
        <select
          value={sectionFilter}
          onChange={(e) => setSectionFilter(e.target.value)}
          className="rounded border border-slate-300 px-2 py-1"
        >
          <option>All</option>
          {stats.sections.map((section) => (
            <option key={section}>{section}</option>
          ))}
        </select>
        <span className="text-slate-500">
          {visible.length} / {stats.total_chunks} chunks · avg {stats.avg_tokens} tok · min{' '}
          {stats.min_tokens} · max {stats.max_tokens}
        </span>
        <span className="ml-auto flex items-center gap-2 text-slate-400">
          <span className="rounded bg-amber-100 px-1.5 text-amber-800">&lt;400</span>
          <span className="rounded bg-green-100 px-1.5 text-green-800">400–800</span>
          <span className="rounded bg-red-100 px-1.5 text-red-700">&gt;800</span>
        </span>
      </div>

      <ul className="flex-1 space-y-2 overflow-y-auto p-3">
        {visible.map((chunk) => {
          const open = openId === chunk.id;
          return (
            <li key={chunk.id} className="rounded border border-slate-200">
              <button
                onClick={() => setOpenId(open ? null : chunk.id)}
                className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs hover:bg-slate-50"
              >
                <span className="font-mono text-slate-400">{chunk.id}</span>
                <span className="truncate rounded bg-slate-100 px-1.5 py-0.5 font-medium text-slate-600">
                  {chunk.section}
                </span>
                <span
                  className={`ml-auto shrink-0 rounded px-1.5 py-0.5 font-mono ${tokenBadgeClass(
                    chunk.token_count
                  )}`}
                >
                  {chunk.token_count} tok
                </span>
              </button>
              <div className="h-1 bg-slate-100">
                <div
                  className="h-1 bg-slate-400"
                  style={{ width: `${(chunk.token_count / maxTokens) * 100}%` }}
                />
              </div>
              {open && (
                <pre className="max-h-96 overflow-auto whitespace-pre-wrap border-t border-slate-200 bg-slate-50 p-3 text-xs leading-relaxed text-slate-800">
                  {chunk.text}
                </pre>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
