// Growth story — how the project went from an empty repo to an agentic legal
// RAG system, told as chapters. Each chapter is the same shape:
//   start    — where the project stood before this work
//   problem  — the concrete pain (witnessed where possible, never hypothetical)
//   solution — what was built and the reasoning behind the shape it took
//   after    — the new state of the project once the change landed
//
// This file is deliberately data-only (same convention as progressData.js):
// telling the story = editing one array. Commits anchor every chapter to git.

export const INTRO = {
  title: "From an empty repo to a precedent-walking agent",
  blurb:
    "Vakil AI grew by one rule: build the simple thing first, watch it fail in a specific, witnessed way, then earn the upgrade. Every chapter below follows that loop — start, problem, solution, and the state it left behind.",
};

export const CHAPTERS = [
  {
    id: "g01-plan",
    title: "A plan and an empty repo",
    when: "2026-08-03",
    commits: [{ hash: "eb29e9e", subject: "Add initial project plan" }],
    start:
      "Nothing but a question: can retrieval-augmented generation answer legal questions over Indian court judgments without inventing law?",
    problem:
      "The target architecture (Qdrant, LangGraph, agents, citation graphs) is too big to build in one pass — build it all up front and every bug hides in a component you don't understand yet.",
    solution:
      "PLAN.md records the full target, then deliberately swaps in lighter stand-ins: SQLite + numpy where the plan says Qdrant, hand-rolled loops before any framework. Rule of the repo: upgrades must be motivated by a witnessed failure, not by architecture envy.",
    after:
      "A roadmap that permits starting simple without losing sight of the sophisticated end state. Everything that follows is that roadmap being cashed in, one earned step at a time.",
  },
  {
    id: "g02-ingest",
    title: "PDFs become structured text",
    when: "2026-08-03 → 08-04",
    commits: [
      { hash: "811632c", subject: "read-only Google Drive connector" },
      { hash: "785f308", subject: "PDF ingestion pipeline" },
    ],
    start: "Judgments live as layout-heavy PDFs in a Google Drive folder.",
    problem:
      "Naive text extraction destroys the structure legal reasoning depends on: judgment sections, numbered paragraphs, tables. A chunk that cuts a holding in half is worse than no chunk.",
    solution:
      "OAuth Drive connector for ingestion; Docling parses PDF into structure-aware Markdown; regex heuristics pull court, case title, date and citations from the document head; a section-aware chunker splits on canonical judgment sections and numbered legal paragraphs ('12. The next contention…'), prefixing every chunk with its section title.",
    after:
      "Every document persists as inspectable artifacts under output/<doc>/ — source.pdf, markdown.md, metadata.json, chunks. The pipeline's output is a durable asset the rest of the project keeps re-reading for free.",
  },
  {
    id: "g03-semantic",
    title: "Meaning-based search",
    when: "2026-08-04",
    commits: [
      { hash: "11c14b4", subject: "embedding layer + SQLite vector store" },
    ],
    start: "Well-structured chunks that nothing can query yet.",
    problem:
      "Legal questions are conceptual — 'freedom of trade under Article 19' should find a judgment that never uses those exact words. Keyword search cannot cross that paraphrase gap.",
    solution:
      "nomic-embed-text via Ollama with asymmetric prefixes (documents embed as search_document:, queries as search_query: — dropping them silently degrades retrieval), vectors L2-normalized at the source so cosine = dot product. The store is brute-force exact cosine in numpy over SQLite — deliberate: exact KNN is milliseconds at this corpus size, ANN is a millions-of-vectors problem this project does not have.",
    after:
      "Semantic /search over the corpus. The store hides behind a VectorIndex interface, so the day the corpus outgrows numpy, Qdrant is a new module and a registry entry — not a rewrite.",
  },
  {
    id: "g04-idempotency",
    title: "Run it twice, pay once",
    when: "2026-08-04",
    commits: [
      { hash: "dfcebc4", subject: "document registry + reprocess infrastructure" },
    ],
    start:
      "A working pipeline where every improvement means re-running Docling — minutes per PDF — over the whole corpus.",
    problem:
      "Iteration cost grows with the corpus. Re-uploads re-parse identical bytes; a parser fix gives no way to find which documents were built by the old version; a failed Drive sweep restarts from zero.",
    solution:
      "Four stacked idempotency layers: sha256 dedup in a registry catalog (same bytes never re-parse), PIPELINE_VERSION staleness so `reprocess` re-runs exactly the outdated docs, content-hash skip in the indexer so unchanged chunks never re-embed, and a resumable outbox for Drive sync (pending/done/failed rows that survive restarts).",
    after:
      "Two databases, two roles: vakil.db is the durable catalog, vectors*.db are disposable indexes rebuildable any time from output/. Re-running anything costs near zero — which is what made the heavy experimentation of later chapters affordable.",
  },
  {
    id: "g05-studio",
    title: "Seeing the pipeline",
    when: "2026-08-04",
    commits: [{ hash: "29a2343", subject: "React studio frontend" }],
    start: "A pipeline that only speaks through terminal logs and JSON files.",
    problem:
      "A bad chunk boundary or misparsed table is invisible until it surfaces much later as a mysterious retrieval failure. Debugging retrieval quality without seeing the artifacts is guesswork.",
    solution:
      "A React + Vite studio: upload or Drive-sync a PDF, then inspect the original PDF, parsed Markdown, extracted metadata, and every chunk side by side, with live search against the index.",
    after:
      "Every pipeline stage inspectable at a glance. The studio becomes the project's growth surface too — each later chapter adds a tab (Agent trace, Citations graph, this page).",
  },
  {
    id: "g06-profiles",
    title: "Experiments without lies",
    when: "2026-08-06",
    commits: [
      { hash: "0bd6f71", subject: "profile-based config + pluggable components" },
      { hash: "a2eeb2f", subject: "experiment profile UI" },
    ],
    start:
      "One hard-coded configuration; changing the embedding model or chunk size means editing code and hoping.",
    problem:
      "Vectors from different models or chunkers are incomparable — mixing them in one database produces garbage similarities that look like numbers. And comparing variants 'by feel' proves nothing.",
    solution:
      "Named experiment profiles in config.yaml (pydantic-validated, typos fail at startup); each profile indexes into its OWN vector DB; a fingerprint of the vector-identity fields is stamped into the index and verified on open — a mismatch raises IndexConfigMismatch instead of returning wrong similarities. Beside it, an eval harness: stratified doc selection, doc_recall@k / passage_recall@k / MRR, side-by-side compare.",
    after:
      "The comparison loop that governs everything after: edit config → index profile → run eval → compare. Numbers before opinions — a retrieval change ships only if the scoreboard moves.",
  },
  {
    id: "g07-rag",
    title: "Generation, and the case for RAG",
    when: "2026-08-06 → 08-07",
    commits: [
      { hash: "433aeb4", subject: "chat model layer" },
      { hash: "61eae8d", subject: "retrieve() as the single retrieval path" },
      { hash: "6722718", subject: "context assembly" },
      { hash: "0be1a8d", subject: "grounded QA — rag.ask() + /ask" },
    ],
    start: "Retrieval works; nobody gets an answer yet.",
    problem:
      "Witnessed, not assumed: asked two local LLMs about a case sitting in the corpus — llama3.1 invented pardon powers, deepseek-r1 fabricated a holding with a made-up citation. Confident legal fiction from mere case-name recognition. That is the disease RAG exists to cure.",
    solution:
      "Four seams, each single-path: a chat-model layer with a provider registry (generation is config, never a profile knob — swapping the LLM never invalidates an index); retrieve() as the ONE retrieval function shared by /search, evals and everything later, so evals measure the production path; assemble() packing evidence under two independent knobs (a token budget for the hardware limit, a k-cap for noise dilution) with provenance labels on every chunk; ask() wrapping it in a grounding prompt — answer only from excerpts, an explicit escape hatch ('if insufficient, say so'), and a [doc_id:chunk_id] citation contract.",
    after:
      "First end-to-end grounded answer. On its first live run the escape hatch fired unprompted and every claim carried a chunk citation. The same models that invented law now decline when the evidence is thin.",
  },
  {
    id: "g08-breakit",
    title: "Break it on purpose",
    when: "2026-08-19 → 08-20",
    commits: [
      { hash: "361f58d", subject: "break-it lab knowledge base" },
      { hash: "bda5330", subject: "probe 2 and 3 results" },
      { hash: "585b7a5", subject: "step 5 status + lessons" },
    ],
    start: "A working RAG loop with failure modes known only in theory.",
    problem:
      "You don't know where a system breaks until it breaks — and the worst place to learn that is from a user. Prompt-engineering against imagined failures is superstition.",
    solution:
      "A break-it lab: deliberately trigger each predicted failure and watch. Parametric-leakage probes, retrieval-miss probes, context-overflow pressure, citation-drift measurement — each with a written prediction before the run.",
    after:
      "Two predictions confirmed, one overturned (Ollama now hard-rejects oversized prompts instead of silently truncating — dependency assumptions rot), one unplanned failure found (a dissent reported as the Court's holding, with perfectly accurate citations — faithfulness ≠ correctness). Most consequential: the query 'AIR 1962 SC 406' could not surface its own document in the dense top-20. That witnessed miss is the ticket the next chapter cashes.",
  },
  {
    id: "g09-hybrid",
    title: "Retrieval learns to read citations",
    when: "2026-08-22 → 08-24",
    commits: [
      { hash: "0b75251", subject: "citation-type eval questions" },
      { hash: "a29f597", subject: "hybrid BM25 + RRF inside retrieve()" },
      { hash: "6bfae73", subject: "2c — citation phrases land; rewrite + rerank measured, off" },
    ],
    start:
      "Dense-only retrieval, and a new eval slice that froze the failure in a number: citation recall@5 = 0.0.",
    problem:
      "Embeddings smear rare identifier strings like 'AIR 1962 SC 406' into generic legalness — the exact term a lawyer searches by is the one signal a vector cannot hold. Lexical BM25 rewards precisely those rare terms.",
    solution:
      "Hybrid retrieval fused with RRF inside retrieve() — callers never changed. Then a precision pass, every idea behind its own measured flag: a regex citation-phrase detector (all 12 citation questions to rank 1, passage@1 .226 → .419, MRR .835 → .962); an LLM query rewrite (built, measured, net-negative: helped paraphrase, hurt holdings, +7.4s/query — OFF); a cross-encoder reranker (lost with every model tried — a cross-encoder rewards 'discusses the subject' and cannot tell it from 'contains the evidence' — OFF, one yaml edit from a retrial).",
    after:
      "Citation recall@5 0.0 → 0.833 → 1.0, and the winning configuration is also the fastest thing measured (~120ms/query). Just as important: two negative results kept on the books, with the convicting numbers in config comments. 'Built, measured, off' became a legitimate end state.",
  },
  {
    id: "g10-agent",
    title: "From one-shot RAG to an agent",
    when: "2026-08-24 → 08-31",
    commits: [
      { hash: "0a31de2", subject: "hand-rolled tool loop" },
      { hash: "6057cb4", subject: "agentic QA panel with tool trace" },
      { hash: "1217e16", subject: "multi-turn conversation with session history" },
      { hash: "8ab2aa4", subject: "LangGraph port — checkpointed chat" },
      { hash: "bb93108", subject: "corrective RAG — grade retrieval, gated rewrite" },
    ],
    start:
      "ask() answers in exactly one retrieval round — it cannot decide what to fetch, judge whether the evidence suffices, or try again.",
    problem:
      "Real legal questions need multi-step work: filter to a court and year, read a specific section, search again with a better query. One-shot RAG has no verbs for any of that. And adopting an agent framework before writing the loop by hand means never understanding what the framework abstracts.",
    solution:
      "First a hand-rolled ~150-line loop — LLM + tools (search_chunks, filter_documents, read_document) + dispatch + state — with the full tool trace rendered in a studio tab. Only then the LangGraph port: the same loop as a checkpointed graph, which pays for itself by adding what the raw loop couldn't cheaply hold — a retrieval-grading node that judges sufficiency after each search and loops back with ONE rewritten query, capped at two retries. That gated rewrite is the disciplined comeback of the blind rewrite chapter 9 measured and rejected.",
    after:
      "A corrective-RAG agent with multi-turn memory and a visible tool trace. The framework was adopted from below — understood as a formalization of code already written, not as magic.",
  },
  {
    id: "g11-citations",
    title: "Judgments cite judgments",
    when: "2026-08-31",
    commits: [
      { hash: "68647f5", subject: "citation graph — extraction, edges, traversal tools" },
      { hash: "cae1635", subject: "citation graph view — click-to-walk SVG" },
    ],
    start:
      "An agent that treats every document as an island — similar text is reachable, related law is not.",
    problem:
      "Precedent is a graph: which cases does this judgment rely on, and who has cited it since? 'Enumerate everything that cites X' is not a similarity question — flat RAG structurally cannot answer it.",
    solution:
      "Citation edges extracted at pipeline time (regex over Indian reporter grammars; a document's own citation, found in its head, is excluded from edges); edge + reverse-lookup tables in the catalog DB, idempotent per-doc re-extraction, and a backfill CLI that reuses existing markdown so Docling never re-runs. On top: get_cited / get_citing tools wired into both agent stacks, and a studio Citations tab drawing a past ← doc ← future SVG graph with click-to-walk on in-corpus nodes.",
    after:
      "76 documents, 312 edges, zero failures. The agent now walks precedent chains — a capability class semantic search cannot reach. Only 1 in-corpus resolution so far (a stratified random corpus rarely self-references): the machinery is ahead of the data, waiting for a corpus of related cases.",
  },
];

export const EPILOGUE = {
  title: "Where it stands",
  points: [
    "Question → hybrid retrieval → graded sufficiency → grounded, cited answer — or an honest refusal.",
    "An agent that filters, reads, re-queries and walks citation graphs, with every tool call visible in the studio.",
    "A scoreboard on every claim: retrieval changes ship only with eval numbers, and negative results stay recorded.",
    "Still deliberately simple where simple wins: SQLite + numpy over Qdrant, exact KNN over ANN, generation evals (the RAG triad) deferred until they're the binding constraint.",
  ],
};
