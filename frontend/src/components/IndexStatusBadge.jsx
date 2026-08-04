import { useEffect, useState } from 'react';
import { indexStatus } from '../api.js';

// Embedding runs as a backend background task after processing (~1 min/doc);
// poll until the vector count catches up with the chunk count.
export default function IndexStatusBadge({ docId }) {
  const [status, setStatus] = useState(null);

  useEffect(() => {
    let timer;
    let cancelled = false;
    async function poll() {
      try {
        const s = await indexStatus(docId);
        if (cancelled) return;
        setStatus(s);
        if (!s.complete) timer = setTimeout(poll, 3000);
      } catch {
        if (!cancelled) timer = setTimeout(poll, 5000);
      }
    }
    poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [docId]);

  if (!status) return null;
  return status.complete ? (
    <span className="rounded bg-green-100 px-1.5 py-0.5 text-green-800">indexed ✓</span>
  ) : (
    <span className="rounded bg-amber-100 px-1.5 py-0.5 text-amber-800">
      indexing {status.indexed_chunks}/{status.total_chunks}…
    </span>
  );
}
