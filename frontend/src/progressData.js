// Learning journal — the pipeline's state, hand-updated as each piece lands.
// This file is deliberately data-only so updating progress = editing one array.
//
// status: 'done' | 'next' | 'pending' | 'deferred'
// Deferred items carry the symptom → diagnosis → cure trail: we build the fix
// only after watching the failure happen (motivated learning).

export const PHASES = [
  {
    title: "Phase 1 — Ingestion",
    blurb:
      "PDF to searchable vectors. Built first, battle-tested, all idempotent.",
    steps: [
      {
        id: "p1-pipeline",
        title: "Processing pipeline",
        status: "done",
        when: "Phase 1",
        note: "process_pdf: parse → metadata → chunk → persist to output/<doc>/. Embedding intentionally decoupled (background task).",
        components: [
          {
            name: "parser (docling)",
            status: "done",
            note: "PDF → structure-aware Markdown; component package with ABC + registry",
          },
          {
            name: "metadata (regex)",
            status: "done",
            note: "court / case title / date / citations from the document head",
          },
          {
            name: "chunker (section-aware)",
            status: "done",
            note: "canonical judgment sections, numbered-paragraph boundaries, section-prefixed chunks",
          },
          {
            name: "upload + Drive sync",
            status: "done",
            note: "resumable outbox pattern: pending → done | failed rows in vakil.db",
          },
        ],
      },
      {
        id: "p1-index",
        title: "Indexing + storage",
        status: "done",
        when: "Phase 1",
        note: "Two DBs, two roles: vakil.db is the durable catalog, vectors*.db are disposable per-profile indexes.",
        components: [
          {
            name: "embedder (nomic via Ollama)",
            status: "done",
            note: "asymmetric prefixes, L2-normalized at source, batched",
          },
          {
            name: "enriched embedding",
            status: "done",
            note: "case title + section baked into vectors; raw text stored for display",
          },
          {
            name: "vector store (SQLite + numpy)",
            status: "done",
            note: "exact brute-force cosine — ANN is a millions-of-vectors problem, not ours",
          },
          {
            name: "registry (sha256 dedup, versioning)",
            status: "done",
            note: "PIPELINE_VERSION staleness + reprocess loop",
          },
          {
            name: "experiment profiles",
            status: "done",
            note: "embedding/chunking knobs in config.yaml, one index per profile, fingerprint-verified on open",
          },
        ],
      },
      {
        id: "p1-eval",
        title: "Retrieval eval harness",
        status: "done",
        when: "Phase 1",
        note: "Numbers before opinions: every retrieval change must move these metrics or it does not ship.",
        doc: {
          label:
            "Measurement primer — how to read every scoreboard: metrics, slices, the 12-question caveat",
          // symlinked into frontend/public/docs/ so Vite serves the repo's docs/ copy
          href: "/docs/eval-measurement.md",
        },
        components: [
          { name: "stratified eval set (decade × size)", status: "done" },
          {
            name: "doc_recall@k / passage_recall@k / MRR",
            status: "done",
            note: "passage_recall is the chunker signal — moves when boundaries move",
          },
          { name: "compare (side-by-side runs)", status: "done" },
        ],
      },
    ],
  },
  {
    title: "Phase 2 — RAG loop",
    blurb:
      "Connect retrieval to generation: question → grounded, cited answer. Walking skeleton first (steps 1-5), measured upgrades after (2b-2d).",
    steps: [
      {
        id: "s1-chat",
        title: "Step 1 — Chat model layer",
        status: "done",
        when: "2026-08-06 · commit 433aeb4",
        note: "app/llm/ component package. Generation is top-level config, not a profile knob: the chat model only reads retrieved text, so swapping it never invalidates an index.",
        lesson:
          'Parametric leakage, witnessed live: asked both models about Chintaman Rao (which sits in our corpus). llama3.1 invented pardon powers; deepseek-r1 invented an RTE Act holding with a fabricated citation. Confident legal fiction from case-name recognition — the "before" picture RAG must fix.',
        components: [
          {
            name: "ChatModel ABC (base.py)",
            status: "done",
            note: "callers depend on the interface, never a provider",
          },
          {
            name: "ollama provider",
            status: "done",
            note: "native /api/chat for num_ctx control; silent-truncation gotcha documented",
          },
          {
            name: "openai-compatible provider",
            status: "done",
            note: "one client for DeepSeek API / Groq / vLLM — de-facto standard wire format",
          },
          {
            name: "named models + active switch",
            status: "done",
            note: "llama · deepseek (local r1:8b) · deepseek-api; arg > VAKIL_CHAT_MODEL > active",
          },
          {
            name: "secrets discipline",
            status: "done",
            note: "config.yaml stores env var NAMES (api_key_env); keys live in .env, autoloaded for CLIs",
          },
          {
            name: "<think> stripping",
            status: "done",
            note: "reasoning-model chain-of-thought is a provider artifact, not part of the answer",
          },
        ],
      },
      {
        id: "s2-retrieve",
        title: "Step 2 — retrieve(): the single retrieval path",
        status: "done",
        when: "2026-08-06 · commit 61eae8d",
        note: "Capability vs transport: /search endpoint, evals, and the future /ask all call the same function. Hybrid + rerank will land inside it — callers never change.",
        lesson:
          "Eval honesty: run_eval now measures the exact code path production uses, not a parallel reimplementation. Current strategy: enriched-dense-exact-topk.",
        components: [
          {
            name: "app/retrieval.py",
            status: "done",
            note: "question → ranked rows; frozen contract (doc_id, chunk_id, section, case_title, text, score)",
          },
          {
            name: "/search as thin wrapper",
            status: "done",
            note: "HTTP error mapping + display rounding stay in main.py",
          },
          { name: "run_eval on shared path", status: "done" },
        ],
      },
      {
        id: "s3-context",
        title: "Step 3 — Context assembly",
        status: "done",
        when: "2026-08-07",
        note: "app/context.py: assemble() is the single evidence-building path between retrieve() and the chat model — /ask and evals will share it, same seam pattern as retrieve().",
        lesson:
          "k cap and token budget are two knobs for two diseases: the budget guards num_ctx overflow (a hardware limit), the k cap guards noise dilution (a quality judgment). First real run proved they bind independently — 12 chunks retrieved, all would have FIT the 7k budget, but the k cap kept only 8. Eviction is whole-chunk from the worst-ranked tail: the exact opposite of Ollama’s silent front-truncation, which eats the system prompt first.",
        components: [
          {
            name: "provenance labels [doc:chunk | section]",
            status: "done",
            note: "format_chunk(): header + case title + raw text — the anchor the citation contract points at",
          },
          {
            name: "k selection / noise dilution",
            status: "done",
            note: "max_chunks cap, independent of budget; junk stays junk at any context size",
          },
          {
            name: "token budget accounting",
            status: "done",
            note: "budget = num_ctx − system − question − answer headroom; tiktoken cl100k as good-enough approximation, headroom absorbs the error",
          },
          {
            name: "AssembledContext accounting",
            status: "done",
            note: "kept/dropped/token counts returned, not just the string — /ask reports the cut, step 5 asserts on eviction",
          },
        ],
      },
      {
        id: "s4-ask",
        title: "Step 4 — Grounding prompt + /ask",
        status: "done",
        when: "2026-08-07",
        note: "app/rag.py: ask() = retrieve() → assemble() → chat(). First end-to-end grounded answer. /ask is a thin transport wrapper, same seam pattern — generation evals (2d) will score rag.ask() itself.",
        lesson:
          'First live run: escape hatch fired unprompted ("the excerpts do not establish whether...") and every claim carried a chunk citation — but the model substituted the human-readable case title for the doc_id slug in citations, keeping only chunk_id exact. "Use the exact ids from its header" was not literal enough. Fix deliberately deferred to step 5: prompt engineering is learned against witnessed failures.',
        components: [
          {
            name: "restriction clause",
            status: "done",
            note: "answer ONLY from provided excerpts; recognizing a case name ≠ permission to use memory",
          },
          {
            name: "escape hatch",
            status: "done",
            note: '"if insufficient, say so" — the single biggest anti-hallucination lever',
          },
          {
            name: "citation contract [doc_id:chunk_id]",
            status: "done",
            note: "points at the provenance labels assemble() writes; slug-vs-title drift witnessed (see lesson)",
          },
          {
            name: "/ask endpoint",
            status: "done",
            note: "reports kept/dropped ids + token accounting; canned refusal (no LLM call) when nothing retrievable",
          },
        ],
      },
      {
        id: "s5-break",
        title: "Step 5 — Break-it lab",
        status: "done",
        when: "2026-08-20",
        note: "Deliberately trigger each failure mode and watch it happen. Prompt engineering is learned here, not in step 4.",
        lesson:
          'Probes beat assumptions: 2 predictions confirmed, 1 overturned, 1 unplanned failure found. Overturned: current Ollama hard-rejects oversized prompts (exceed_context_size_error) instead of silently front-truncating — the budget now guards availability, not safety; dependency assumptions rot between versions. Unplanned: dissent-as-holding — model reported a dissenting opinion as the Court’s decision with perfectly accurate citations. Faithfulness ≠ correctness; opinion boundaries are destroyed at chunking time (all labeled "Judgment"), so the fix is pipeline-level, not prompt-level.',
        doc: {
          label:
            "Break-it lab knowledge base — probe theory, examples, witnessed-failure log",
          // symlinked into frontend/public/docs/ so Vite serves the repo's docs/ copy
          href: "/docs/break-it-lab.md",
        },
        components: [
          {
            name: "parametric leakage probe",
            status: "done",
            note: "3/3 passes: Kesavananda refused, Gopalan answered + cited, overruling gap held despite parametric knowledge",
          },
          {
            name: "retrieval miss probe",
            status: "done",
            note: 'witnessed: "AIR 1962 SC 406" absent from top-20 (score band 0.547–0.564); conceptual control 5/5 at 0.75–0.82 — 2b earned',
          },
          {
            name: "context overflow probe",
            status: "done",
            note: "k cap and budget proven independent (8 vs 14 kept); escape hatch held at 6.9k pressure; Ollama now errors loudly on overflow",
          },
          {
            name: "citation drift probe",
            status: "done",
            note: "measured 16/16 exact citations (0% drift, n=3 generations); validator queued with 2d instead of prompt surgery",
          },
        ],
      },
      {
        id: "s2b-hybrid",
        title: "2b — Hybrid retrieval (BM25 + RRF)",
        status: "done",
        when: "2026-08-22",
        trail: {
          symptom:
            'WITNESSED (step 5): "AIR 1962 SC 406" — its document absent from dense top-20, score band compressed to 0.017 wide; the same doc hit 5/5 on a conceptual query.',
          diagnosis:
            "Embeddings smear rare identifier strings into generic legalness; lexical BM25 rewards exactly those rare terms.",
          cure: "SQLite FTS5 table beside chunk_vectors, fuse rankings with RRF (rank-based, no score normalization). Lands inside retrieve().",
        },
        note: "Landed inside retrieve() — callers unchanged. Query-time knob (retrieval: in config.yaml), outside the index fingerprint: FTS5 derives from stored payloads, so dense<->hybrid flips never invalidate vectors.",
        lesson:
          "Citation recall@5: 0.0 dense → 0.333 hybrid at canonical RRF settings (c50/k60) → 0.833 at c20/k10; conceptual types rose to 1.0 (not flat as predicted). The 0.333 taught the real lesson: when one ranker is pure noise for a query class, deep candidate lists + flat RRF produce FALSE CONSENSUS — junk chunks matching mid-list in both rankers outrank the correct single-list #1 (witnessed live before the eval said so). Also: COUNT(*) on an FTS5 external-content table reads through to the content table — the index can look populated while never built; rebuild state needs its own stamp.",
        docs: [
          {
            label:
              "Hybrid retrieval knowledge base — dense vs lexical, BM25 built from repairs, RRF fusion",
            // symlinked into frontend/public/docs/ so Vite serves the repo's docs/ copy
            href: "/docs/hybrid-retrieval.md",
          },
          {
            label:
              "Implementation field guide — every file, every choice, both bugs, the knob sweep",
            href: "/docs/hybrid-implementation.md",
          },
        ],
        components: [
          {
            name: "citation-type eval questions",
            status: "done",
            note: "12 questions, each citation verified unique to one indexed doc; dense-baseline-v2 froze the 0.0",
          },
          {
            name: "FTS5 BM25 index",
            status: "done",
            note: "external-content table over chunk_vectors (title+section+text = enriched lexical); rebuild stamped in index_meta",
          },
          {
            name: "RRF fusion",
            status: "done",
            note: "inside retrieve(); rank-based, quoted-OR query sanitization; score is an RRF sum, not a similarity",
          },
          {
            name: "before/after eval by question type",
            status: "done",
            note: "knob sweep c50/k60→c20/k10 in evals/results/hybrid-*.json; residual misses (prose dilution, common-token citations) queued for 2c",
          },
        ],
      },
      {
        id: "s2c-rerank",
        title: "2c — Rerank + query rewrite",
        status: "done",
        when: "2026-08-23",
        trail: {
          symptom:
            'WITNESSED (2b evals): q004 "what does the judgment say about... 1952 SCR 135" — prose terms diluted the BM25 query, gold chunk below the c20 cutoff; q017 bare "(1995) 2 SCC 7" at rank 8 — tokens hyper-common, identity lives in adjacency the OR query cannot express. Baseline passage@1 .226: found, rarely first.',
          diagnosis:
            "Two diseases: query noise drowning signal (dilution) and signal the query language cannot express (adjacency). Plus a thesis: 2b's narrow c20/k10 compensated for a missing precision stage — a cross-encoder should let candidates re-widen.",
          cure: "Query ensemble (add, never replace): regex citation detector → FTS5 phrase list into RRF; LLM keyword rewrite as extra dense+lexical lists; cross-encoder rerank over the fused pool. Each behind its own flag, measured in isolation.",
        },
        note: "Only citation phrases landed enabled. LLM rewrite and reranker shipped as built, measured knobs — OFF, with the convicting numbers in config.yaml comments. Winning config is also the fastest thing measured (120ms/query).",
        lesson:
          'Citation phrases swept: all 12 citation questions to rank 1 (q004 AND q017 — q004\'s question quotes its citation, so the regex cure caught the LLM cure\'s assigned case), passage@1 .226→.419, mrr .835→.962, everything else byte-identical — inert-or-decisive held exactly. LLM rewrite: paraphrase .286→.429 but holding 1.0→.85 and +7.4s/query — "add never replace" bounds damage, it does not guarantee benefit. The reranker LOST with every model tried (MiniLM-L-6 passage@5 .871→.645; bge-reranker-base .774) and re-widening to c50/pool50 made it WORSE (.758): the compensation thesis was falsified on record — a cross-encoder rewards topical relevance, cannot tell "discusses the subject" from "contains the evidence", and overrides the phrase list\'s near-certain verdicts. Pairwise autopsy: gold-vs-promoted chunk, MiniLM picked gold 2/6, bge 5/6; truncation explained only 2 of 14 losses. Decisive lexical evidence is hard to out-judge.',
        docs: [
          {
            label:
              "Rerank + rewrite knowledge base — query ensembles, bi- vs cross-encoders, two-stage retrieval",
            // symlinked into frontend/public/docs/ so Vite serves the repo's docs/ copy
            href: "/docs/rerank-rewrite.md",
          },
          {
            label:
              "Implementation field guide — every file and decision, the ablation scoreboard, the reranker autopsy (pairwise probe, topical vs evidential relevance, decisive-signal override)",
            href: "/docs/rerank-implementation.md",
          },
        ],
        components: [
          {
            name: "citation phrase detector",
            status: "done",
            note: "regex over Indian reporter grammars → FTS5 phrase queries, own RRF list; normalized through the tokenizer's own split rule; citation@5 .833→1.0, zero non-citation drift",
          },
          {
            name: "LLM query rewrite",
            status: "done",
            note: "llama3.1 keyword rewrite via app/llm, None on any failure; net-negative here (holding regression + 7.4s) — built, measured, off",
          },
          {
            name: "cross-encoder reranker",
            status: "done",
            note: "app/rerank/ component package; MiniLM and bge both lost to no-rerank; off, one yaml edit from a retrial with a stronger judge",
          },
          {
            name: "knob re-sweep under reranker",
            status: "done",
            note: "c50/k60/pool50 WORSE than c20/pool30 — widening thesis falsified; c20/k10 stays",
          },
        ],
      },
      {
        id: "s2d-geneval",
        title: "2d — Generation eval (RAG triad)",
        status: "deferred",
        trail: {
          symptom:
            "Once /ask exists, retrieval metrics stop covering what users see — a perfect retrieval can still yield an unfaithful answer.",
          diagnosis:
            "Generation adds its own failure modes: hallucination past the evidence, reasoning errors, dropped citations.",
          cure: "Extend eval set with gold answers; LLM-judge scores faithfulness (claim-by-claim vs chunks) and correctness. Hand-rolled, not imported — the metric must be understood.",
        },
        components: [
          { name: "gold answers in eval set", status: "deferred" },
          { name: "faithfulness judge", status: "deferred" },
          { name: "correctness judge", status: "deferred" },
        ],
      },
    ],
  },
  {
    title: "Phase 3 — Agent",
    blurb:
      "From one-shot RAG to multi-step: the model decides what to retrieve, judges sufficiency, retries. Concepts sit on top of Phase 2.",
    steps: [
      {
        id: "s3a-loop",
        title: "3a — Hand-rolled tool loop",
        status: "done",
        when: "2026-08-24",
        note: "Agent = LLM + tools + loop + state, built raw (~150 lines) before any framework, so the abstraction is understood from below. ChatModel grew chat_tools() (messages list + tool schemas, neutral OpenAI-style format; Ollama wants dict arguments, OpenAI-compat JSON strings — adapters translate). Exposed as GET /agent/ask + an Agent studio tab showing the tool trace.",
        lesson:
          "Planning met the data twice: vakil.db has no court/year columns (they live only in metadata.json, noisy — so filter_documents matches substrings over 76 json files), and chunk section names are classify_section() outputs, not literal headings (so read_document re-splits markdown with the chunker's own functions). llama3.1 answers from tools fine but drops citations from final answers — exactly what 2d's faithfulness judge would catch.",
        components: [
          {
            name: "search_chunks tool",
            status: "done",
            note: "wraps retrieve(); capped output, [doc_id:chunk_id | SECTION] headers",
          },
          {
            name: "filter_documents tool",
            status: "done",
            note: "substring court/title + year regex over output/*/metadata.json",
          },
          {
            name: "read_document tool",
            status: "done",
            note: "list sections, then fetch one — reuses split_sections()",
          },
          {
            name: "tool-call dispatch loop",
            status: "done",
            note: "max_steps + repeat-call cache + forced final answer",
          },
        ],
      },
      {
        id: "s3b-langgraph",
        title: "3b — LangGraph port + retrieval grading",
        status: "done",
        note: "Same loop as a graph: nodes, edges, checkpointed state. Adds grade-retrieval → rewrite-and-retry (corrective RAG).",
        components: [
          {
            name: "graph port of the loop",
            status: "done",
            note: "StateGraph + ToolNode + tools_condition; @tool annotations derive the schemas tools.py writes by hand; MemorySaver thread_id = session_id replaces sessions.py; agent.kind in config.yaml / ?agent= switches loops",
          },
          {
            name: "retrieval sufficiency grading",
            status: "done",
            note: "grade node after each search_chunks round: structured verdict (sufficient / what's missing / ONE rewritten query) via function-calling; insufficient appends grader guidance and loops to agent, capped at 2 rewrites/turn — the gated version of 2c's net-negative blind rewrite. Grader fails open; verdicts land in the tool trace",
          },
        ],
      },
      {
        id: "s3c-citations",
        title: "3c — Citation graph traversal",
        status: "done",
        note: "Judgments cite judgments. Extract citation edges at pipeline time, expose get_citing/get_cited tools — the agent walks precedent chains, which flat RAG cannot.",
        components: [
          {
            name: "citation edge extraction",
            status: "done",
            note: "app/citations/ component package (base/regex/registry+factory); normalized reporter refs are the graph join key; refs in the first 2500 chars = the doc's OWN citation, excluded from edges; runs in process_pdf + python -m app.citations.backfill over existing markdown — no PIPELINE_VERSION bump, no Docling re-run",
          },
          {
            name: "edge table in vakil.db",
            status: "done",
            note: "citation_edges (citing_doc_id, cited_ref, occurrences) + doc_citation_keys (ref → doc reverse lookup); delete+insert per doc = idempotent re-extraction; GET /documents/{id}/citations exposes own/cited/cited_by, rendered by the studio Citations tab as a past ← doc ← future SVG graph with click-to-walk on in-corpus nodes",
          },
          {
            name: "traversal tools",
            status: "done",
            note: "get_cited(doc_id) = precedent basis with in-corpus resolution; get_citing(doc_id or reporter cite) = reverse lookup semantic search cannot enumerate; wired into both agent stacks + system prompt",
          },
          {
            name: "backfill run over corpus",
            status: "done",
            note: "76 docs, 0 failed — 312 edges, 81 own-citation keys. Only 1 in-corpus resolution (stratified random selection rarely self-references); full precedent-chain payoff needs a corpus of related cases",
          },
        ],
      },
    ],
  },
];

export const STATUS_META = {
  done: {
    label: "done",
    dot: "bg-emerald-500",
    chip: "bg-emerald-50 text-emerald-700 border-emerald-200",
  },
  next: {
    label: "up next",
    dot: "bg-sky-500",
    chip: "bg-sky-50 text-sky-700 border-sky-200",
  },
  "in-progress": {
    label: "in progress",
    dot: "bg-violet-500",
    chip: "bg-violet-50 text-violet-700 border-violet-200",
  },
  pending: {
    label: "pending",
    dot: "bg-slate-300",
    chip: "bg-slate-50 text-slate-500 border-slate-200",
  },
  deferred: {
    label: "deferred — on purpose",
    dot: "bg-amber-400",
    chip: "bg-amber-50 text-amber-700 border-amber-200",
  },
};
