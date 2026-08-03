# Vakil AI — Document Processing Studio (Phase 1)

Parse Indian court judgment PDFs, inspect extracted Markdown, metadata, and semantic
chunks before anything gets embedded. See [PLAN.md](PLAN.md) for the full roadmap.

## Run

```sh
make install   # one-time: venv + pip deps + npm install
make dev       # backend :8787 + frontend :5173 together, Ctrl-C stops both
```

Individually: `make backend`, `make frontend`. First processed PDF downloads
Docling layout models (~1-2 min, one-time).

## Use

1. Drop a judgment PDF in the UI (grab samples from indiankanoon.org), or
2. Drive connector (user OAuth): copy `backend/.env.example` to `backend/.env`,
   follow the comments (OAuth client + redirect URI + folder id), export the vars
   before starting uvicorn, then hit **Connect Drive** in the sidebar and consent.
   Token lands in `backend/credentials/token.json` (gitignored), auto-refreshes.

Artifacts land in `backend/output/<doc-slug>/`: `source.pdf`, `markdown.md`,
`metadata.json`, `chunks.json`, `chunks/chunk_NNN.txt`.

## Scope

Phase 1 stops at chunking output — no embeddings, vector DB, or LLM. Chunking knobs
live in `backend/app/chunker.py` (`TARGET_TOKENS`, `MAX_TOKENS`, `OVERLAP_TOKENS`).
