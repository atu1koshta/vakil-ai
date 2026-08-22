import { PHASES, STATUS_META } from '../progressData.js';

// Standalone page at /progress — a visual learning journal of the pipeline.
// Static by design (no backend): progress updates are edits to progressData.js,
// committed alongside the code they describe.

function StatusChip({ status }) {
  const meta = STATUS_META[status] || STATUS_META.pending;
  return (
    <span
      className={`inline-flex shrink-0 items-center gap-1.5 rounded-full border px-2 py-0.5 text-[11px] font-medium ${meta.chip}`}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${meta.dot}`} />
      {meta.label}
    </span>
  );
}

function ComponentRow({ comp }) {
  const meta = STATUS_META[comp.status] || STATUS_META.pending;
  return (
    <li className="flex items-baseline gap-2 text-xs">
      <span className={`mt-1 h-1.5 w-1.5 shrink-0 self-center rounded-full ${meta.dot}`} />
      <span className={comp.status === 'done' ? 'text-slate-700' : 'text-slate-500'}>
        {comp.name}
      </span>
      {comp.note && <span className="text-slate-400">— {comp.note}</span>}
    </li>
  );
}

function Trail({ trail }) {
  return (
    <div className="mt-2 space-y-1 rounded border border-amber-100 bg-amber-50/60 p-2 text-xs">
      {['symptom', 'diagnosis', 'cure'].map((k) => (
        <div key={k} className="flex gap-2">
          <span className="w-16 shrink-0 font-semibold uppercase tracking-wide text-amber-600">
            {k}
          </span>
          <span className="text-slate-600">{trail[k]}</span>
        </div>
      ))}
    </div>
  );
}

function StepCard({ step, isLast }) {
  const meta = STATUS_META[step.status] || STATUS_META.pending;
  return (
    <div className="relative pl-8">
      {/* timeline rail */}
      <span
        className={`absolute left-2 top-2 h-3.5 w-3.5 rounded-full border-2 border-white shadow ${meta.dot}`}
      />
      {!isLast && <span className="absolute bottom-0 left-[13px] top-6 w-px bg-slate-200" />}

      <div
        className={`mb-4 rounded-lg border bg-white p-3 shadow-sm ${
          step.status === 'next' ? 'border-sky-300 ring-1 ring-sky-100' : 'border-slate-200'
        }`}
      >
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h3 className="text-sm font-semibold text-slate-800">{step.title}</h3>
          <div className="flex items-center gap-2">
            {step.when && <span className="text-[11px] text-slate-400">{step.when}</span>}
            <StatusChip status={step.status} />
          </div>
        </div>

        {step.note && <p className="mt-1.5 text-xs text-slate-500">{step.note}</p>}

        {(step.docs || (step.doc ? [step.doc] : [])).map((doc) => (
          <a
            key={doc.href}
            href={doc.href}
            target="_blank"
            rel="noreferrer"
            className="mr-2 mt-2 inline-flex items-center gap-1.5 rounded border border-sky-200 bg-sky-50/60 px-2 py-1 text-xs font-medium text-sky-700 hover:bg-sky-100"
          >
            <span aria-hidden>📄</span> {doc.label}
          </a>
        ))}

        {step.trail && <Trail trail={step.trail} />}

        {step.lesson && (
          <div className="mt-2 rounded border border-emerald-100 bg-emerald-50/60 p-2 text-xs">
            <span className="font-semibold uppercase tracking-wide text-emerald-600">
              lesson banked{' '}
            </span>
            <span className="text-slate-600">{step.lesson}</span>
          </div>
        )}

        {step.components?.length > 0 && (
          <ul className="mt-2.5 space-y-1 border-t border-slate-100 pt-2">
            {step.components.map((c) => (
              <ComponentRow key={c.name} comp={c} />
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

export default function ProgressPage() {
  const allSteps = PHASES.flatMap((p) => p.steps);
  const doneCount = allSteps.filter((s) => s.status === 'done').length;

  return (
    <div className="min-h-full bg-slate-100 text-slate-900">
      <header className="border-b border-slate-200 bg-white px-6 py-3">
        <div className="mx-auto flex max-w-3xl items-center justify-between">
          <div>
            <h1 className="text-lg font-semibold">Pipeline Progress</h1>
            <p className="text-xs text-slate-500">
              learning journal · symptom → diagnosis → cure · {doneCount}/{allSteps.length} steps
              done
            </p>
          </div>
          <a href="/" className="text-xs font-medium text-sky-600 hover:text-sky-800">
            ← back to studio
          </a>
        </div>
      </header>

      <main className="mx-auto max-w-3xl px-6 py-6">
        <div className="mb-6 flex flex-wrap gap-3">
          {Object.entries(STATUS_META).map(([key, meta]) => (
            <span key={key} className="flex items-center gap-1.5 text-[11px] text-slate-500">
              <span className={`h-2 w-2 rounded-full ${meta.dot}`} /> {meta.label}
            </span>
          ))}
        </div>

        {PHASES.map((phase) => (
          <section key={phase.title} className="mb-8">
            <h2 className="mb-1 text-sm font-bold uppercase tracking-wide text-slate-600">
              {phase.title}
            </h2>
            <p className="mb-4 text-xs text-slate-400">{phase.blurb}</p>
            {phase.steps.map((step, i) => (
              <StepCard key={step.id} step={step} isLast={i === phase.steps.length - 1} />
            ))}
          </section>
        ))}
      </main>
    </div>
  );
}
