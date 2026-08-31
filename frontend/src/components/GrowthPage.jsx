import { INTRO, CHAPTERS, EPILOGUE } from '../growthData.js';

// Standalone page at /growth — the project's growth story for a newcomer:
// how it went from simple to sophisticated, one earned upgrade at a time.
// Static by design (no backend), same convention as ProgressPage: telling
// the story = editing growthData.js, committed alongside the code it describes.

const BLOCKS = [
  {
    key: 'start',
    label: 'start',
    box: 'border-slate-200 bg-slate-50/80',
    tag: 'text-slate-500',
  },
  {
    key: 'problem',
    label: 'problem',
    box: 'border-amber-100 bg-amber-50/60',
    tag: 'text-amber-600',
  },
  {
    key: 'solution',
    label: 'solution',
    box: 'border-sky-100 bg-sky-50/60',
    tag: 'text-sky-600',
  },
  {
    key: 'after',
    label: 'state after',
    box: 'border-emerald-100 bg-emerald-50/60',
    tag: 'text-emerald-600',
  },
];

function CommitChip({ commit }) {
  return (
    <span className="inline-flex items-baseline gap-1.5 rounded border border-slate-200 bg-white px-2 py-0.5 text-[11px] text-slate-500">
      <code className="font-mono text-slate-400">{commit.hash}</code>
      {commit.subject}
    </span>
  );
}

function ChapterCard({ chapter, index, isLast }) {
  return (
    <div className="relative pl-8">
      {/* timeline rail */}
      <span className="absolute left-0 top-1.5 flex h-7 w-7 items-center justify-center rounded-full border-2 border-white bg-slate-700 text-[11px] font-semibold text-white shadow">
        {index + 1}
      </span>
      {!isLast && <span className="absolute bottom-0 left-[13px] top-10 w-px bg-slate-200" />}

      <div className="mb-6 ml-2 rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h3 className="text-sm font-semibold text-slate-800">{chapter.title}</h3>
          {chapter.when && <span className="text-[11px] text-slate-400">{chapter.when}</span>}
        </div>

        <div className="mt-3 space-y-2">
          {BLOCKS.map(({ key, label, box, tag }) => (
            <div key={key} className={`flex gap-3 rounded border p-2.5 text-xs ${box}`}>
              <span
                className={`w-16 shrink-0 pt-px font-semibold uppercase tracking-wide ${tag}`}
              >
                {label}
              </span>
              <p className="leading-relaxed text-slate-600">{chapter[key]}</p>
            </div>
          ))}
        </div>

        {chapter.commits?.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-1.5 border-t border-slate-100 pt-2.5">
            {chapter.commits.map((c) => (
              <CommitChip key={c.hash} commit={c} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default function GrowthPage() {
  return (
    <div className="min-h-full bg-slate-100 text-slate-900">
      <header className="border-b border-slate-200 bg-white px-6 py-3">
        <div className="mx-auto flex max-w-3xl items-center justify-between">
          <div>
            <h1 className="text-lg font-semibold">How Vakil AI Took Shape</h1>
            <p className="text-xs text-slate-500">
              growth story · start → problem → solution → state after · {CHAPTERS.length}{' '}
              chapters
            </p>
          </div>
          <div className="flex items-center gap-3 text-xs font-medium">
            <a href="/progress" className="text-sky-600 hover:text-sky-800">
              learning journal
            </a>
            <a href="/" className="text-sky-600 hover:text-sky-800">
              ← back to studio
            </a>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-3xl px-6 py-6">
        <div className="mb-8 rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
          <h2 className="text-sm font-bold text-slate-800">{INTRO.title}</h2>
          <p className="mt-1.5 text-xs leading-relaxed text-slate-500">{INTRO.blurb}</p>
        </div>

        {CHAPTERS.map((chapter, i) => (
          <ChapterCard
            key={chapter.id}
            chapter={chapter}
            index={i}
            isLast={i === CHAPTERS.length - 1}
          />
        ))}

        <section className="mt-2 rounded-lg border border-emerald-200 bg-emerald-50/50 p-4">
          <h2 className="text-sm font-bold text-emerald-800">{EPILOGUE.title}</h2>
          <ul className="mt-2 space-y-1.5">
            {EPILOGUE.points.map((p) => (
              <li key={p} className="flex items-baseline gap-2 text-xs text-slate-600">
                <span className="h-1.5 w-1.5 shrink-0 self-center rounded-full bg-emerald-500" />
                {p}
              </li>
            ))}
          </ul>
        </section>
      </main>
    </div>
  );
}
