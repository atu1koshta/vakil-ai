# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Vakil AI — a legal RAG practice project over Indian court judgment PDFs. Current state: parse → metadata → chunk → embed → semantic search, with a React studio UI for inspecting artifacts. PLAN.md holds the full target architecture (Qdrant, LangGraph, Ollama LLM); the current implementation deliberately uses lighter stand-ins (SQLite + numpy, nomic-embed-text via Ollama).

## Commands

```sh
make install    # one-time: backend venv + pip deps, frontend npm install
make dev        # backend :8787 + frontend :5173 together (Ctrl-C stops both)
make backend    # uvicorn only (auto-loads backend/.env if present)
make frontend   # vite only
```

Backend CLI entrypoints (run from `backend/`, use `.venv/bin/python`):

```sh
python -m app.indexer                 # embed + index all docs under output/, active profile (needs Ollama)
python -m app.indexer <doc-slug>      # index one document
python -m app.indexer --profile <p>   # build a named profile's index (own vectors-<p>.db)
python -m app.reprocess               # re-run pipeline for stale docs (older pipeline_version or failed)
python -m app.reprocess --all         # everything
python -m app.reprocess <doc-id>      # one document
python -m app.evals.select_docs [N]   # pick stratified eval docs from Drive
python -m app.evals.fetch_texts       # download eval PDFs + plain text
python -m app.evals.process_selection # process + index the eval selection
python -m app.evals.build_eval_set    # validate question batches -> evals/eval_set.jsonl
python -m app.evals.run_eval [tag] [--profile <p>]  # retrieval metrics -> evals/results/<tag>.json
python -m app.evals.compare [tags...] # side-by-side metrics table across saved runs
```

There is no test suite or linter configured.

Embedding requires Ollama at `localhost:11434` with the `nomic-embed-text` model pulled. First PDF processed downloads Docling layout models (~1-2 min, one-time).

## Architecture

Backend is FastAPI (`backend/app/`), frontend is React + Vite + Tailwind (`frontend/src/`). The frontend talks to the backend via `frontend/src/api.js`.

### Component packages (interface programming)

Each pipeline component is a package with a fixed layout: `base.py` = interface (ABC) + shared errors, `<impl>.py` = one implementation per file, `__init__.py` = registry dict + factory — the only thing the rest of the app imports. Pipeline/indexer/API code depends on the ABCs via factories, never on concrete classes. Adding a backend = implement the ABC in a new module + one registry entry + name it in config.yaml; no pipeline edits.

| Package | Interface | Factory | Registry / config key | Level |
|---|---|---|---|---|
| `app/parser/` | `Parser` | `get_parser()` | `components.parser` | pipeline |
| `app/metadata/` | `MetadataExtractor` | `get_metadata_extractor()` | `components.metadata_extractor` | pipeline |
| `app/chunker/` | `Chunker` | `get_chunker(cfg)` | `chunking.strategy` (per profile) | profile |
| `app/embeddings/` | `Embedder` | `get_embedder(profile)` | `embedding.provider` (per profile) | profile |
| `app/vector_store/` | `VectorIndex` | `open_store(profile)` | `store` (per profile) | profile |
| `app/rerank/` | `Reranker` | `get_reranker(profile)` | `retrieval.rerank.provider` (per profile) | profile |

Pipeline-level components' output (`output/`) is shared by all profiles — changing them means bumping `PIPELINE_VERSION`. Profile-level components feed the per-profile index — their identity is in the profile fingerprint.

### Experiment profiles (`config.py`, `backend/config.yaml`)

Every retrieval-quality knob — embedding model (+ prefixes, dim), chunking sizes, index enrichment — lives in a named profile in `backend/config.yaml`, validated by pydantic (`extra="forbid"`: typos fail at startup). Each profile indexes into its OWN vector DB (`db_path`, default `vectors-<name>.db`) because vectors from different models/chunkers are incomparable. The API server and pipeline use `active_profile`; CLIs take `--profile`; env overrides: `VAKIL_PROFILE`, `VAKIL_CONFIG`. `Profile.fingerprint()` hashes only vector-identity fields (model, dim, prefixes, chunking, enrich) — operational knobs (base_url, batch_size, timeout) don't invalidate an index. Model comparison loop: edit config.yaml → `python -m app.indexer --profile X` → `python -m app.evals.run_eval --profile X` → `python -m app.evals.compare`.

### Processing pipeline (`pipeline.py`)

`process_pdf()` runs parse (`app/parser/`, Docling PDF→Markdown) → metadata (`app/metadata/`, regex heuristics over the document head) → table extraction + chunking (`app/chunker/`) and persists everything under `backend/output/<doc-slug>/` (`source.pdf`, `markdown.md`, `metadata.json`, `chunks.json`, `chunks/chunk_NNN.txt`). Embedding is NOT part of `process_pdf` — API routes queue `indexer.index_doc_by_id` as a FastAPI background task afterward (poll `/documents/{doc_id}/index-status`).

### Two SQLite databases, two roles

- `backend/vakil.db` (`registry.py`) — the durable **catalog**: which PDFs (by content sha256) were processed, by which `PIPELINE_VERSION`, status processed/failed. Also holds the `drive_sync` checkpoint table.
- `backend/vectors*.db` (`app/vector_store/`) — disposable **indexes**, one per profile (`vectors.db` belongs to `nomic-default`): float32 vectors + chunk payloads, rebuildable any time from `output/` via `python -m app.indexer --profile <p>`. Deleting one loses nothing. Each carries an `index_meta` stamp (model/dim/config fingerprint) verified on open — opening with a mismatched profile raises `IndexConfigMismatch` instead of returning garbage similarities.

The indexer chunks in memory from `output/<doc>/markdown.md` using the profile's chunking config — `chunks.json` is only the studio inspection artifact of the active profile. So comparing chunking variants never re-runs Docling.

### Idempotency layers (they stack)

1. **Registry sha256 dedup**: same PDF bytes + same/newer pipeline version → cached result from `output/`, Docling never runs. `force=True` bypasses.
2. **Pipeline versioning**: any improvement to parser/chunker/metadata must bump `PIPELINE_VERSION` in `pipeline.py`; old rows become stale and `python -m app.reprocess` re-runs exactly those (source PDFs are already in `output/`, nothing re-uploads).
3. **Indexer content_hash dedup**: unchanged chunks are skipped before any embedding call, so re-indexing is free.
4. **Drive sync outbox** (`drive_sync.py`): per-file pending/done/failed rows in vakil.db; a re-triggered sweep skips done, retries failed — resumable across server restarts.

### Embedding contract (`app/embeddings/`)

Embedders are built per profile via `get_embedder(profile)`; providers register in `_EMBEDDERS` (currently `ollama`). Prefixes are config, not code: nomic-embed-text documents embed as `search_document: ...`, queries as `search_query: ...` — dropping prefixes silently degrades retrieval; symmetric models use empty prefixes. Vectors are L2-normalized at the source so cosine = dot product downstream, and every returned vector's length is checked against the profile's declared `dim`. The indexer embeds ENRICHED text (case title + section + chunk text; `indexing.enrich`) but stores RAW text for display. A different model = a different profile = a separate index.

### Vector search (`app/vector_store/`)

Brute-force exact cosine in numpy (`SqliteVectorStore`) — intentional; exact KNN is milliseconds at this corpus size. Callers use the `VectorIndex` ABC via `open_store()`, so a Qdrant/ANN backend is a new module + registry entry. Built without sqlite-vec because python.org macOS sqlite3 lacks loadable-extension support. The `index_meta` stamp carries a `META_SCHEMA` version — fingerprint-recipe changes migrate old stamps instead of false-alarming a mismatch.

### Drive connector (`connectors/drive.py`)

User OAuth (config via `backend/.env`, see `.env.example`); token in `backend/credentials/token.json` (gitignored), auto-refreshed. Routes return 424 when unconfigured, 401 when unauthorized.

### Evals (`app/evals/`, data in `backend/evals/`)

Retrieval benchmark: stratified doc selection → plain-text extraction for question authoring (deliberately bypasses the real pipeline) → validated `eval_set.jsonl` → `run_eval` reporting doc_recall@k, passage_recall@k (single-chunk 5-gram coverage — the metric sensitive to chunk boundaries), and MRR. `run_eval --profile <p>` scores that profile's index and embeds the full profile snapshot in the result JSON for reproducibility; `compare` prints saved runs side by side with best-per-metric markers.

## Conventions

- Chunking knobs (`strategy`, `target_tokens`, `max_tokens`, `min_tokens`, `overlap_tokens`, `encoding`) live in `ChunkingConfig` (`config.py`), set per profile in `config.yaml`; `SectionAwareChunker` prefers numbered legal paragraphs ("12. The next contention...") over blank lines as sub-split boundaries, and every chunk is prefixed with its section title.
- Component packages follow the `base.py` / `<impl>.py` / `__init__.py` (registry + factory) layout — keep it when adding implementations.
- Batch/per-doc scripts use per-item failure isolation: one bad document prints FAILED and the run continues. Keep this pattern.
- Module docstrings carry "Learning notes" documenting non-obvious design decisions — maintain them when changing those modules.
