import { useEffect, useMemo, useState } from 'react';
import { documentCitations, listDocuments } from '../api.js';

// Citation graph for one document, drawn as hand-rolled SVG (no chart lib:
// the layout is a fixed three-column fan, d3 would be ceremony).
//
// Reading order mirrors the mental model of the two tools:
//   PAST (left)            CENTER              FUTURE (right)
//   cases it cites  ◄──  selected doc  ◄──  cases citing it
//   (get_cited)                             (get_citing)
// In-corpus nodes are clickable and re-center the graph — that IS the
// precedent-chain walk the agent does with get_cited/get_citing.

const ROW_H = 40;
const NODE_W = 300;
const NODE_H = 30;
const VIEW_W = 1080;
const CENTER_W = 320;
const MIN_H = 360;
const LABEL_CHARS = 36;

function truncate(text, n = LABEL_CHARS) {
  return text.length > n ? text.slice(0, n - 1) + '…' : text;
}

// Filename slug -> readable-ish label when we have nothing better.
function slugLabel(docId) {
  return truncate(docId.replace(/-\d+$/, '').replace(/-/g, ' '));
}

function edgePath(x1, y1, x2, y2) {
  const mx = (x1 + x2) / 2;
  return `M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`;
}

export default function CitationGraphPanel({ initialDocId = null }) {
  const [docs, setDocs] = useState([]);
  const [docId, setDocId] = useState(initialDocId);
  const [graph, setGraph] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [hover, setHover] = useState(null); // node key being hovered

  useEffect(() => {
    listDocuments()
      .then((rows) => setDocs(rows.filter((r) => r.status === 'processed')))
      .catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    if (!docId) return;
    setLoading(true);
    setError(null);
    documentCitations(docId)
      .then(setGraph)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [docId]);

  const names = useMemo(
    () => Object.fromEntries(docs.map((d) => [d.doc_id, d.source_name])),
    [docs],
  );

  const label = (id) => (names[id] ? truncate(names[id].replace(/\.pdf$/i, '')) : slugLabel(id));

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-3 border-b border-slate-200 px-4 py-2">
        <label className="text-xs font-medium text-slate-500">Document</label>
        <select
          value={docId || ''}
          onChange={(e) => setDocId(e.target.value || null)}
          className="min-w-0 flex-1 rounded border border-slate-300 bg-white px-2 py-1 text-sm"
        >
          <option value="">— pick a judgment —</option>
          {docs.map((d) => (
            <option key={d.doc_id} value={d.doc_id}>
              {d.source_name}
            </option>
          ))}
        </select>
      </div>

      <div className="flex items-center gap-4 border-b border-slate-100 px-4 py-1.5 text-[11px] text-slate-500">
        <span className="flex items-center gap-1">
          <span className="inline-block h-2.5 w-2.5 rounded-sm bg-sky-500" /> selected
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-2.5 w-2.5 rounded-sm bg-emerald-500" /> cited, in corpus (click to walk)
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-2.5 w-2.5 rounded-sm bg-slate-300" /> cited, not in corpus
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-2.5 w-2.5 rounded-sm bg-violet-500" /> cites this doc (click to walk)
        </span>
        <span className="ml-auto">edge width = times cited</span>
      </div>

      <div className="min-h-0 flex-1 overflow-auto bg-slate-50">
        {!docId ? (
          <Empty text="Pick a judgment to see its citation graph." />
        ) : loading ? (
          <Empty text="Loading graph…" />
        ) : error ? (
          <Empty text={error} tone="error" />
        ) : graph ? (
          <Graph graph={graph} label={label} onWalk={setDocId} hover={hover} setHover={setHover} />
        ) : null}
      </div>
    </div>
  );
}

function Empty({ text, tone }) {
  return (
    <div
      className={`flex h-full items-center justify-center text-sm ${
        tone === 'error' ? 'text-red-600' : 'text-slate-400'
      }`}
    >
      {text}
    </div>
  );
}

function Graph({ graph, label, onWalk, hover, setHover }) {
  const cited = graph.cited;
  const citedBy = graph.cited_by;

  if (!cited.length && !citedBy.length) {
    return (
      <Empty text="No citation edges for this document — it cites no reported cases and nothing in the corpus cites it." />
    );
  }

  const rows = Math.max(cited.length, citedBy.length, 1);
  const height = Math.max(MIN_H, rows * ROW_H + 80);
  const centerY = height / 2;
  const centerX = VIEW_W / 2;
  const maxOcc = Math.max(1, ...cited.map((e) => e.occurrences), ...citedBy.map((e) => e.occurrences));
  const columnTop = (count) => centerY - ((count - 1) * ROW_H) / 2;

  const leftTop = columnTop(cited.length);
  const rightTop = columnTop(citedBy.length);
  const strokeFor = (occ) => 1 + (occ / maxOcc) * 4;
  const dimmed = (key) => hover !== null && hover !== key;

  return (
    <div className="p-4">
      <svg viewBox={`0 0 ${VIEW_W} ${height}`} className="w-full" style={{ minWidth: 900 }}>
        <defs>
          <marker id="arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">
            <path d="M 0 0 L 8 4 L 0 8 z" fill="#94a3b8" />
          </marker>
        </defs>

        {/* PAST -> arrows OUT: center cites left nodes */}
        {cited.map((e, i) => {
          const y = leftTop + i * ROW_H;
          const key = `out-${e.cited_ref}`;
          const inCorpus = Boolean(e.resolved_doc_id);
          return (
            <g
              key={key}
              opacity={dimmed(key) ? 0.25 : 1}
              onMouseEnter={() => setHover(key)}
              onMouseLeave={() => setHover(null)}
              onClick={() => inCorpus && onWalk(e.resolved_doc_id)}
              className={inCorpus ? 'cursor-pointer' : ''}
            >
              <path
                d={edgePath(centerX - CENTER_W / 2, centerY, 20 + NODE_W, y)}
                fill="none"
                stroke="#94a3b8"
                strokeWidth={strokeFor(e.occurrences)}
                markerEnd="url(#arrow)"
              />
              <rect
                x={20}
                y={y - NODE_H / 2}
                width={NODE_W}
                height={NODE_H}
                rx={6}
                fill={inCorpus ? '#10b981' : '#e2e8f0'}
                stroke={inCorpus ? '#059669' : '#cbd5e1'}
              />
              <text
                x={20 + NODE_W / 2}
                y={y + 4}
                textAnchor="middle"
                fontSize="11"
                fill={inCorpus ? 'white' : '#475569'}
              >
                {inCorpus ? label(e.resolved_doc_id) : truncate(e.raw_text, 40)}
              </text>
              <title>
                {e.raw_text} — cited {e.occurrences}x
                {inCorpus ? ` — in corpus: ${e.resolved_doc_id} (click to walk)` : ' — not in corpus'}
              </title>
            </g>
          );
        })}

        {/* FUTURE -> arrows IN: right nodes cite center */}
        {citedBy.map((e, i) => {
          const y = rightTop + i * ROW_H;
          const key = `in-${e.citing_doc_id}`;
          return (
            <g
              key={key}
              opacity={dimmed(key) ? 0.25 : 1}
              onMouseEnter={() => setHover(key)}
              onMouseLeave={() => setHover(null)}
              onClick={() => onWalk(e.citing_doc_id)}
              className="cursor-pointer"
            >
              <path
                d={edgePath(VIEW_W - 20 - NODE_W, y, centerX + CENTER_W / 2, centerY)}
                fill="none"
                stroke="#a78bfa"
                strokeWidth={strokeFor(e.occurrences)}
                markerEnd="url(#arrow)"
              />
              <rect
                x={VIEW_W - 20 - NODE_W}
                y={y - NODE_H / 2}
                width={NODE_W}
                height={NODE_H}
                rx={6}
                fill="#8b5cf6"
                stroke="#7c3aed"
              />
              <text
                x={VIEW_W - 20 - NODE_W / 2}
                y={y + 4}
                textAnchor="middle"
                fontSize="11"
                fill="white"
              >
                {label(e.citing_doc_id)}
              </text>
              <title>
                {e.citing_doc_id} cites this doc {e.occurrences}x as {e.raw_text} (click to walk)
              </title>
            </g>
          );
        })}

        {/* center node last: draws above edge ends */}
        <g>
          <rect
            x={centerX - CENTER_W / 2}
            y={centerY - 28}
            width={CENTER_W}
            height={56}
            rx={10}
            fill="#0ea5e9"
            stroke="#0284c7"
            strokeWidth={2}
          />
          <text x={centerX} y={centerY - 4} textAnchor="middle" fontSize="12" fontWeight="600" fill="white">
            {label(graph.doc_id)}
          </text>
          <text x={centerX} y={centerY + 14} textAnchor="middle" fontSize="10" fill="#e0f2fe">
            {graph.own.length ? `reported as ${graph.own.join(' · ')}` : 'no reporter citation recorded'}
          </text>
          <title>{graph.doc_id}</title>
        </g>

        {/* column captions */}
        <text x={20 + NODE_W / 2} y={24} textAnchor="middle" fontSize="11" fontWeight="600" fill="#64748b">
          PAST — cases it cites ({cited.length})
        </text>
        <text x={VIEW_W - 20 - NODE_W / 2} y={24} textAnchor="middle" fontSize="11" fontWeight="600" fill="#64748b">
          FUTURE — cases citing it ({citedBy.length})
        </text>
      </svg>
    </div>
  );
}
