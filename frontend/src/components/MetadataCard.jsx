function Field({ label, value }) {
  return (
    <div>
      <dt className="text-xs font-semibold uppercase tracking-wide text-slate-400">{label}</dt>
      <dd className="mt-0.5 text-sm text-slate-800">
        {value == null || (Array.isArray(value) && value.length === 0) ? (
          <span className="italic text-slate-400">not detected</span>
        ) : Array.isArray(value) ? (
          <ul className="list-inside list-disc space-y-0.5">
            {value.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        ) : (
          value
        )}
      </dd>
    </div>
  );
}

export default function MetadataCard({ metadata, stats }) {
  return (
    <div className="h-full overflow-y-auto p-4">
      <dl className="space-y-4">
        <Field label="Case title" value={metadata.case_title} />
        <Field label="Court" value={metadata.court} />
        <Field label="Date" value={metadata.date} />
        <Field label="Judges" value={metadata.judges} />
        <Field label="Citations" value={metadata.citations} />
        <Field label="Source file" value={metadata.source_file} />
        <Field label="Sections detected" value={stats.sections} />
      </dl>
      <p className="mt-6 text-xs text-slate-400">
        Extracted heuristically (regex over document head). Verify before trusting — extractor
        tuning is part of Milestone 2.
      </p>
    </div>
  );
}
