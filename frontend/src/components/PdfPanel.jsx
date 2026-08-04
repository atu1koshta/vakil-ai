import { useState } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import 'react-pdf/dist/Page/TextLayer.css';
import 'react-pdf/dist/Page/AnnotationLayer.css';

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.mjs',
  import.meta.url
).toString();

export default function PdfPanel({ url }) {
  const [numPages, setNumPages] = useState(null);
  const [page, setPage] = useState(1);

  if (!url) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-slate-400">
        Original PDF appears here.
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-center gap-3 border-b border-slate-200 bg-white py-1.5 text-xs">
        <button
          onClick={() => setPage((p) => Math.max(1, p - 1))}
          disabled={page <= 1}
          className="rounded border border-slate-300 px-2 py-0.5 disabled:opacity-40"
        >
          Prev
        </button>
        <span className="text-slate-600">
          Page {page} / {numPages ?? '…'}
        </span>
        <button
          onClick={() => setPage((p) => Math.min(numPages ?? p, p + 1))}
          disabled={numPages !== null && page >= numPages}
          className="rounded border border-slate-300 px-2 py-0.5 disabled:opacity-40"
        >
          Next
        </button>
      </div>
      <div className="flex flex-1 justify-center overflow-auto p-4">
        <Document
          file={url}
          onLoadSuccess={({ numPages: n }) => {
            setNumPages(n);
            setPage(1);
          }}
          loading={<div className="text-sm text-slate-400">Loading PDF…</div>}
          error={<div className="text-sm text-red-500">Failed to load PDF.</div>}
        >
          <Page pageNumber={page} width={560} className="shadow" />
        </Document>
      </div>
    </div>
  );
}
