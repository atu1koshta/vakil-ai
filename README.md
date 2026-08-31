# Vakil AI — Legal RAG over Indian Court Judgments

A learning project that builds a retrieval-augmented QA system over Indian court
judgment PDFs, end to end: parse → metadata → chunk → embed → hybrid semantic
search → agentic QA with an LLM tool loop, plus a React studio UI for inspecting
every artifact along the way. See [PLAN.md](PLAN.md) for the target architecture
and [CLAUDE.md](CLAUDE.md) for a deeper architecture tour.

## Run

```sh
make install   # one-time: venv + pip deps + npm install
make dev       # backend :8787 + frontend :5173 together, Ctrl-C stops both
make eval      # retrieval benchmark against the active profile
```

Individually: `make backend`, `make frontend`. Requires Ollama at
`localhost:11434` with `nomic-embed-text` pulled (plus a chat model for the
agent/QA endpoints). The first processed PDF downloads Docling layout models
(~1-2 min, one-time).

## Use

1. Drop a judgment PDF in the UI (grab samples from indiankanoon.org), or
2. Drive connector (user OAuth): copy `backend/.env.example` to `backend/.env`,
   follow the comments (OAuth client + redirect URI + folder id), then hit
   **Connect Drive** in the sidebar and consent. Token lands in
   `backend/credentials/token.json` (gitignored), auto-refreshes. **Sync** runs
   a resumable sweep of the whole folder.

Processing artifacts land in `backend/output/<doc-slug>/`: `source.pdf`,
`markdown.md`, `metadata.json`, `chunks.json`, `chunks/chunk_NNN.txt`.
Indexing (embedding into the profile's vector DB) runs as a background task
after processing.

## What's inside

- **Pipeline** — Docling PDF→Markdown, regex metadata extraction,
  section-aware chunking tuned for numbered legal paragraphs.
- **Retrieval** — hybrid dense + BM25 with RRF fusion; exact cosine over
  SQLite/numpy vectors; optional reranking and query rewrite (measured via
  evals, currently off).
- **Agentic QA** — `/agent/ask` runs an LLM tool loop (hand-rolled and
  LangGraph implementations) with semantic search, document filtering, and
  citation-graph traversal tools; multi-turn sessions with checkpointed chat
  history. Corrective RAG grades retrieval and gates a rewrite loop.
- **Citation graph** — citation extraction into edge tables, traversal tools
  for the agent, and a click-to-walk graph view in the studio.
- **Experiment profiles** — every retrieval knob (embedding model, chunking,
  enrichment) lives in a named profile in `backend/config.yaml`; each profile
  gets its own vector DB, so model/chunking comparisons are cheap and safe.
- **Evals** — stratified doc selection, validated question set, and
  `run_eval`/`compare` CLIs reporting doc recall, passage recall, and MRR per
  profile.
- **Studio UI** — inspect parsed Markdown, metadata, chunks, citations, ask
  questions with a visible tool trace, and follow the build story on the
  `/progress` and `/growth` pages.

## Backend CLIs

Run from `backend/` with `.venv/bin/python`:

```sh
python -m app.indexer [doc-slug] [--profile <p>]   # embed + index into a profile's DB
python -m app.reprocess [--all|<doc-id>]           # re-run pipeline for stale docs
python -m app.evals.run_eval [tag] [--profile <p>] # retrieval metrics
python -m app.evals.compare [tags...]              # side-by-side metrics table
```

See [CLAUDE.md](CLAUDE.md) for the full CLI list and architecture notes.
