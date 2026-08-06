// Learning journal — the pipeline's state, hand-updated as each piece lands.
// This file is deliberately data-only so updating progress = editing one array.
//
// status: 'done' | 'next' | 'pending' | 'deferred'
// Deferred items carry the symptom → diagnosis → cure trail: we build the fix
// only after watching the failure happen (motivated learning).

export const PHASES = [
  {
    title: 'Phase 1 — Ingestion',
    blurb: 'PDF to searchable vectors. Built first, battle-tested, all idempotent.',
    steps: [
      {
        id: 'p1-pipeline',
        title: 'Processing pipeline',
        status: 'done',
        when: 'Phase 1',
        note: 'process_pdf: parse → metadata → chunk → persist to output/<doc>/. Embedding intentionally decoupled (background task).',
        components: [
          { name: 'parser (docling)', status: 'done', note: 'PDF → structure-aware Markdown; component package with ABC + registry' },
          { name: 'metadata (regex)', status: 'done', note: 'court / case title / date / citations from the document head' },
          { name: 'chunker (section-aware)', status: 'done', note: 'canonical judgment sections, numbered-paragraph boundaries, section-prefixed chunks' },
          { name: 'upload + Drive sync', status: 'done', note: 'resumable outbox pattern: pending → done | failed rows in vakil.db' },
        ],
      },
      {
        id: 'p1-index',
        title: 'Indexing + storage',
        status: 'done',
        when: 'Phase 1',
        note: 'Two DBs, two roles: vakil.db is the durable catalog, vectors*.db are disposable per-profile indexes.',
        components: [
          { name: 'embedder (nomic via Ollama)', status: 'done', note: 'asymmetric prefixes, L2-normalized at source, batched' },
          { name: 'enriched embedding', status: 'done', note: 'case title + section baked into vectors; raw text stored for display' },
          { name: 'vector store (SQLite + numpy)', status: 'done', note: 'exact brute-force cosine — ANN is a millions-of-vectors problem, not ours' },
          { name: 'registry (sha256 dedup, versioning)', status: 'done', note: 'PIPELINE_VERSION staleness + reprocess loop' },
          { name: 'experiment profiles', status: 'done', note: 'embedding/chunking knobs in config.yaml, one index per profile, fingerprint-verified on open' },
        ],
      },
      {
        id: 'p1-eval',
        title: 'Retrieval eval harness',
        status: 'done',
        when: 'Phase 1',
        note: 'Numbers before opinions: every retrieval change must move these metrics or it does not ship.',
        components: [
          { name: 'stratified eval set (decade × size)', status: 'done' },
          { name: 'doc_recall@k / passage_recall@k / MRR', status: 'done', note: 'passage_recall is the chunker signal — moves when boundaries move' },
          { name: 'compare (side-by-side runs)', status: 'done' },
        ],
      },
    ],
  },
  {
    title: 'Phase 2 — RAG loop',
    blurb: 'Connect retrieval to generation: question → grounded, cited answer. Walking skeleton first (steps 1-5), measured upgrades after (2b-2d).',
    steps: [
      {
        id: 's1-chat',
        title: 'Step 1 — Chat model layer',
        status: 'done',
        when: '2026-08-06 · commit 433aeb4',
        note: 'app/llm/ component package. Generation is top-level config, not a profile knob: the chat model only reads retrieved text, so swapping it never invalidates an index.',
        lesson:
          'Parametric leakage, witnessed live: asked both models about Chintaman Rao (which sits in our corpus). llama3.1 invented pardon powers; deepseek-r1 invented an RTE Act holding with a fabricated citation. Confident legal fiction from case-name recognition — the "before" picture RAG must fix.',
        components: [
          { name: 'ChatModel ABC (base.py)', status: 'done', note: 'callers depend on the interface, never a provider' },
          { name: 'ollama provider', status: 'done', note: 'native /api/chat for num_ctx control; silent-truncation gotcha documented' },
          { name: 'openai-compatible provider', status: 'done', note: 'one client for DeepSeek API / Groq / vLLM — de-facto standard wire format' },
          { name: 'named models + active switch', status: 'done', note: 'llama · deepseek (local r1:8b) · deepseek-api; arg > VAKIL_CHAT_MODEL > active' },
          { name: 'secrets discipline', status: 'done', note: 'config.yaml stores env var NAMES (api_key_env); keys live in .env, autoloaded for CLIs' },
          { name: '<think> stripping', status: 'done', note: 'reasoning-model chain-of-thought is a provider artifact, not part of the answer' },
        ],
      },
      {
        id: 's2-retrieve',
        title: 'Step 2 — retrieve(): the single retrieval path',
        status: 'done',
        when: '2026-08-06 · commit 61eae8d',
        note: 'Capability vs transport: /search endpoint, evals, and the future /ask all call the same function. Hybrid + rerank will land inside it — callers never change.',
        lesson:
          'Eval honesty: run_eval now measures the exact code path production uses, not a parallel reimplementation. Current strategy: enriched-dense-exact-topk.',
        components: [
          { name: 'app/retrieval.py', status: 'done', note: 'question → ranked rows; frozen contract (doc_id, chunk_id, section, case_title, text, score)' },
          { name: '/search as thin wrapper', status: 'done', note: 'HTTP error mapping + display rounding stay in main.py' },
          { name: 'run_eval on shared path', status: 'done' },
        ],
      },
      {
        id: 's3-context',
        title: 'Step 3 — Context assembly',
        status: 'next',
        note: 'Retrieved rows → labeled evidence block. Where k-vs-noise, lost-in-the-middle, and the token budget against num_ctx get decided.',
        components: [
          { name: 'provenance labels [doc:chunk | section]', status: 'next', note: 'without labels the model cannot cite' },
          { name: 'k selection / noise dilution', status: 'next' },
          { name: 'token budget accounting', status: 'next', note: 'system + chunks + question + answer headroom must fit num_ctx' },
        ],
      },
      {
        id: 's4-ask',
        title: 'Step 4 — Grounding prompt + /ask',
        status: 'pending',
        note: 'First end-to-end grounded answer. Prompt engineering debuts here.',
        components: [
          { name: 'restriction clause', status: 'pending', note: 'answer ONLY from provided excerpts' },
          { name: 'escape hatch', status: 'pending', note: '"if insufficient, say so" — the single biggest anti-hallucination lever' },
          { name: 'citation contract [doc_id:chunk_id]', status: 'pending' },
          { name: '/ask endpoint', status: 'pending' },
        ],
      },
      {
        id: 's5-break',
        title: 'Step 5 — Break-it lab',
        status: 'pending',
        note: 'Deliberately trigger each failure mode and watch it happen. Prompt engineering is learned here, not in step 4.',
        components: [
          { name: 'parametric leakage probe', status: 'pending', note: 'ask about a case NOT in the corpus' },
          { name: 'retrieval miss probe', status: 'pending', note: 'exact-identifier queries dense search should whiff' },
          { name: 'context overflow probe', status: 'pending', note: 'oversized k vs num_ctx truncation' },
        ],
      },
      {
        id: 's2b-hybrid',
        title: '2b — Hybrid retrieval (BM25 + RRF)',
        status: 'deferred',
        trail: {
          symptom: 'Dense search whiffs exact identifiers: "AIR 1951 SC 118", "Section 302 IPC" (to be witnessed in step 5).',
          diagnosis: 'Embeddings smear rare identifier strings into generic legalness; lexical BM25 rewards exactly those rare terms.',
          cure: 'SQLite FTS5 table beside chunk_vectors, fuse rankings with RRF (rank-based, no score normalization). Lands inside retrieve().',
        },
        note: 'Deferred until the skeleton runs and dense-only baseline numbers are frozen — a lift needs something to lift over.',
        components: [
          { name: 'FTS5 BM25 index', status: 'deferred' },
          { name: 'RRF fusion', status: 'deferred' },
          { name: 'before/after eval by question type', status: 'deferred', note: 'expect lift on citation queries, none on conceptual — the split is the lesson' },
        ],
      },
      {
        id: 's2c-rerank',
        title: '2c — Rerank + query rewrite',
        status: 'deferred',
        trail: {
          symptom: 'Top-k order puts marginally relevant chunks above the decisive one; user phrasing retrieves poorly.',
          diagnosis: 'Bi-encoders compress query and document separately — word-level interaction is lost at 768 dims. Recall stage needs a precision stage.',
          cure: 'Cross-encoder reranks top-50 → top-8 (retrieve wide, rank narrow); LLM rewrites the question into retrieval-friendly form.',
        },
        components: [
          { name: 'cross-encoder reranker', status: 'deferred' },
          { name: 'query rewriting', status: 'deferred' },
        ],
      },
      {
        id: 's2d-geneval',
        title: '2d — Generation eval (RAG triad)',
        status: 'deferred',
        trail: {
          symptom: 'Once /ask exists, retrieval metrics stop covering what users see — a perfect retrieval can still yield an unfaithful answer.',
          diagnosis: 'Generation adds its own failure modes: hallucination past the evidence, reasoning errors, dropped citations.',
          cure: 'Extend eval set with gold answers; LLM-judge scores faithfulness (claim-by-claim vs chunks) and correctness. Hand-rolled, not imported — the metric must be understood.',
        },
        components: [
          { name: 'gold answers in eval set', status: 'deferred' },
          { name: 'faithfulness judge', status: 'deferred' },
          { name: 'correctness judge', status: 'deferred' },
        ],
      },
    ],
  },
  {
    title: 'Phase 3 — Agent',
    blurb: 'From one-shot RAG to multi-step: the model decides what to retrieve, judges sufficiency, retries. Concepts sit on top of Phase 2.',
    steps: [
      {
        id: 's3a-loop',
        title: '3a — Hand-rolled tool loop',
        status: 'pending',
        note: 'Agent = LLM + tools + loop + state, built raw (~150 lines) before any framework, so the abstraction is understood from below.',
        components: [
          { name: 'search_chunks tool', status: 'pending', note: 'wraps retrieve()' },
          { name: 'filter_documents tool', status: 'pending', note: 'court / year over vakil.db metadata' },
          { name: 'read_document tool', status: 'pending', note: 'full section when a chunk is not enough' },
          { name: 'tool-call dispatch loop', status: 'pending' },
        ],
      },
      {
        id: 's3b-langgraph',
        title: '3b — LangGraph port + retrieval grading',
        status: 'pending',
        note: 'Same loop as a graph: nodes, edges, checkpointed state. Adds grade-retrieval → rewrite-and-retry (corrective RAG).',
        components: [
          { name: 'graph port of the loop', status: 'pending' },
          { name: 'retrieval sufficiency grading', status: 'pending' },
        ],
      },
      {
        id: 's3c-citations',
        title: '3c — Citation graph traversal',
        status: 'pending',
        note: 'Judgments cite judgments. Extract citation edges at pipeline time, expose get_citing/get_cited tools — the agent walks precedent chains, which flat RAG cannot.',
        components: [
          { name: 'citation edge extraction', status: 'pending' },
          { name: 'edge table in vakil.db', status: 'pending' },
          { name: 'traversal tools', status: 'pending' },
        ],
      },
    ],
  },
];

export const STATUS_META = {
  done: { label: 'done', dot: 'bg-emerald-500', chip: 'bg-emerald-50 text-emerald-700 border-emerald-200' },
  next: { label: 'up next', dot: 'bg-sky-500', chip: 'bg-sky-50 text-sky-700 border-sky-200' },
  pending: { label: 'pending', dot: 'bg-slate-300', chip: 'bg-slate-50 text-slate-500 border-slate-200' },
  deferred: { label: 'deferred — on purpose', dot: 'bg-amber-400', chip: 'bg-amber-50 text-amber-700 border-amber-200' },
};
