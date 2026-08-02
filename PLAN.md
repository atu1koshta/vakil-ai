# Vakil AI — Case-Aware Legal RAG System

A practice project: an agentic RAG system over Indian court judgments. Documents flow
through an event-driven processing pipeline (parse → extract metadata → chunk → embed →
store), then a hybrid-retrieval agent answers legal queries grounded in past judgments.

**Current scope: Phase 1 only — "Document Processing Studio".** Stop at chunking output.
No embeddings, no vector DB, no LLM yet.

---

## Full System Architecture (target, later phases)

Event-driven pipeline:

```
[Drive/Dropbox or manual upload]
        │
   Sync Service ──► Queue (RabbitMQ) ──► Parser (Docling, OCR fallback: PaddleOCR)
                                              │
                                        Metadata Extractor
                                              │
                                        Semantic Chunker
                                              │
                                        Embedder (BGE-M3, local)
                                              │
                              ┌───────────────┴───────────────┐
                          Qdrant (vectors)            PostgreSQL (metadata)
                              └───────────────┬───────────────┘
                                       Hybrid Retrieval
                                              │
                                  Agent Orchestrator (LangGraph)
                                              │
                                    LLM (Ollama: Qwen/Gemma)
```

### Stack decisions (locked in)

| Concern | Choice | Why |
|---|---|---|
| PDF parsing | Docling | Structure-aware, PDF → Markdown/JSON, preserves headings |
| OCR fallback | PaddleOCR | Scanned judgments |
| Chunking | Custom semantic chunker | Judgments have strong section structure |
| Embeddings | BGE-M3 (local) | Zero API cost, strong multilingual/legal retrieval |
| Vector DB | Qdrant | Hybrid search support, easy local Docker |
| Metadata DB | PostgreSQL | Filters: court, year, judge, statute |
| LLM | Ollama (Qwen / Gemma) | Zero API cost, local |
| Agent framework | LangGraph | Multi-step retrieval + reasoning graphs |
| Queue | RabbitMQ | Event-driven doc processing |

**Total API cost: $0** — everything runs locally.

### Data source

- Primary: [Indian Kanoon](https://indiankanoon.org) judgments, manual download for
  the initial corpus.
- Alternatives: Supreme Court of India website, eCourts portal, KanoonGPT open dataset.

### Chunking strategy (core design)

1. Split by judgment sections detected from headings: **Facts / Issues / Arguments /
   Analysis / Judgment (holding)**.
2. Sub-split any section over ~700–1000 tokens.
3. Prepend section title to every chunk (context preservation).
4. Target chunk size: **400–800 tokens**, overlap **50–100 tokens**.
5. Each chunk carries doc-level metadata (case name, court, date, citation).

---

## Phase 1 — Document Processing Studio (BUILD THIS NOW)

Goal: upload a judgment PDF, see exactly what the pipeline produces at every stage —
parsed Markdown, extracted metadata, semantic chunks with token counts. A debugging /
inspection tool for the pipeline before anything gets embedded.

### Backend — FastAPI

- Deps: `fastapi`, `uvicorn`, `docling`, `tiktoken`, `python-multipart`.
- One endpoint:

```
POST /process-document   (multipart PDF upload)
→ {
    "metadata":  { case_title, court, date, judges, citations, ... },
    "markdown":  "<full docling markdown>",
    "chunks":    [ { id, section, text, token_count, char_count }, ... ]
  }
```

- Pipeline inside the endpoint:
  1. Docling parse → Markdown + structure JSON.
  2. Metadata extractor — regex/heuristics over first pages (case title, court name,
     date, judge names, citation patterns like `AIR`, `SCC`, `(20XX) X SCC XXX`).
  3. Semantic chunker (strategy above), token counts via tiktoken.
- Persist per-document output dir:

```
output/<doc-slug>/
  markdown.md
  metadata.json
  chunks.json
  chunks/chunk_001.txt, chunk_002.txt, ...
```

### Frontend — React + Tailwind

- Panels:
  1. **Original PDF** — PDF.js viewer.
  2. **Parsed Markdown** — Monaco (read-only) side-by-side with PDF.
  3. **Metadata card** — extracted fields, editable for correction.
  4. **Chunk viewer** — list of chunks with section label, token count, boundaries
     visible; click chunk → highlight source region in Markdown.
- Upload dropzone → calls `POST /process-document` → populates all panels.

### Directory layout

```
vakil-ai/
  PLAN.md
  backend/
    app/
      main.py            # FastAPI app, /process-document
      parser.py          # Docling wrapper
      metadata.py        # heuristic extractor
      chunker.py         # semantic chunker
      models.py          # pydantic response models
    requirements.txt
    output/              # per-doc artifacts (gitignored)
  frontend/
    src/
      components/        # PdfPanel, MarkdownPanel, MetadataCard, ChunkViewer
      ...
    package.json
  data/                  # sample judgment PDFs (gitignored)
```

### Milestones

1. **M1 — Parse & view**: upload PDF, Docling parse, show PDF + Markdown side-by-side.
2. **M2 — Structure & metadata**: heading detection, metadata extraction, metadata card.
3. **M3 — Chunking**: semantic chunker, chunk viewer with token counts, export to
   `output/<doc>/` (markdown.md, metadata.json, chunks.json, chunks/*.txt).

### Drive connector (added to Phase 1 scope)

Read-only Google Drive connector — pull, not sync. User OAuth (user connects
their own Google account via consent screen; `GOOGLE_OAUTH_CLIENT_FILE`,
`DRIVE_FOLDER_ID`). Endpoints: `GET /drive/auth/status`, `GET /drive/auth/url`,
`GET /auth/google/callback` (token stored at `backend/credentials/token.json`,
auto-refreshed), `POST /drive/disconnect`, `GET /drive/files` (list PDFs in
folder), `POST /drive/process/{file_id}` (download + run pipeline).
Watcher/queue-based sync stays in Phase 5. Dropbox dropped.

### Explicitly out of scope for Phase 1

- Embeddings (BGE-M3), Qdrant, PostgreSQL, RabbitMQ, Ollama, LangGraph, Drive
  watcher/sync (pull-only connector IS in scope). All later phases.

---

## Later phases (roadmap)

- **Phase 2 — Index**: BGE-M3 embeddings, Qdrant + PostgreSQL storage, batch ingestion
  over the corpus.
- **Phase 3 — Retrieve**: hybrid retrieval (dense + sparse + metadata filters),
  retrieval evaluation set from known cases.
- **Phase 4 — Agent**: LangGraph orchestrator + Ollama LLM, cited answers, multi-hop
  ("find precedents cited by X").
- **Phase 5 — Sync**: Drive/Dropbox watcher, RabbitMQ event pipeline, incremental
  ingestion.
