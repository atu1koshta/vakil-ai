import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger("vakil")

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse

from .connectors import drive
from .indexer import index_doc_by_id, index_status
from .models import DriveFile, ProcessResult
from .pipeline import get_source_pdf, process_pdf

app = FastAPI(title="Vakil AI — Document Processing Studio", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/profiles")
def list_profiles() -> dict:
    """Configured experiment profiles (config.yaml) and which one is active."""
    from .config import get_config, get_profile

    cfg = get_config()
    return {
        "active": get_profile().name,
        "profiles": {name: p.snapshot() for name, p in cfg.profiles.items()},
    }


@app.get("/search")
def search(q: str, k: int = 5, profile: str | None = None) -> list[dict]:
    """Semantic search over indexed chunks (run `python -m app.indexer` first).
    Thin transport wrapper over retrieval.retrieve() — the shared path /ask
    and evals also use. `profile` targets an experiment profile; default =
    active profile."""
    from .config import ConfigError
    from .embeddings import EmbeddingError
    from .retrieval import retrieve
    from .vector_store import IndexConfigMismatch

    try:
        rows = retrieve(q, k=k, profile=profile)
    except ConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (EmbeddingError, IndexConfigMismatch) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    for r in rows:
        r["score"] = round(r["score"], 4)  # cosine similarity, higher = closer
    return rows


@app.get("/ask")
def ask_question(
    q: str, k: int = 12, model: str | None = None, profile: str | None = None
) -> dict:
    """Grounded QA: retrieve -> assemble -> chat model, with citations.
    Thin transport wrapper over rag.ask() — the same path generation evals
    (2d) will score. `model` picks a generation.models entry; `profile`
    picks the index searched."""
    from .config import ConfigError
    from .embeddings import EmbeddingError
    from .llm import GenerationError
    from .rag import ask
    from .vector_store import IndexConfigMismatch

    try:
        result = ask(q, k=k, model=model, profile=profile)
    except ConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (EmbeddingError, IndexConfigMismatch) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except GenerationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    ctx = result.context
    return {
        "question": result.question,
        "answer": result.answer,
        "model": result.model,
        "grounded": result.grounded,
        "context": {
            "kept": ctx.kept_ids,
            "dropped": ctx.dropped_ids,
            "context_tokens": ctx.context_tokens,
            "budget_tokens": ctx.budget_tokens,
        },
    }


@app.post("/process-document", response_model=ProcessResult)
async def process_document(
    file: UploadFile, background_tasks: BackgroundTasks, force: bool = False
) -> ProcessResult:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(await file.read())
        result = process_pdf(Path(tmp_path), file.filename, force=force)
        if not result.cached:
            # embedding takes ~1 min/doc — run after the response is sent; poll
            # /documents/{doc_id}/index-status for completion
            background_tasks.add_task(index_doc_by_id, result.doc_id)
        return result
    except HTTPException:
        raise
    except Exception as exc:  # surface parse failures to the UI
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}") from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)


FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:5173")


@app.get("/drive/auth/status")
def drive_auth_status() -> dict:
    return {"authorized": drive.is_authorized()}


@app.get("/drive/auth/url")
def drive_auth_url() -> dict:
    try:
        return {"url": drive.build_auth_url()}
    except drive.DriveNotConfigured as exc:
        raise HTTPException(status_code=424, detail=str(exc)) from exc


@app.get("/auth/google/callback")
def drive_oauth_callback(
    code: str | None = None, state: str | None = None, error: str | None = None
) -> RedirectResponse:
    if error or not code:
        logger.warning("OAuth callback without code (error=%s)", error)
        return RedirectResponse(f"{FRONTEND_URL}/?drive=error")
    try:
        drive.handle_oauth_callback(code, state)
    except Exception:
        logger.exception("Drive OAuth token exchange failed")
        return RedirectResponse(f"{FRONTEND_URL}/?drive=error")
    return RedirectResponse(f"{FRONTEND_URL}/?drive=connected")


@app.post("/drive/disconnect")
def drive_disconnect() -> dict:
    drive.disconnect()
    return {"authorized": False}


@app.get("/drive/files", response_model=list[DriveFile])
def drive_files() -> list[DriveFile]:
    try:
        return drive.list_pdfs()
    except drive.DriveNotConfigured as exc:
        raise HTTPException(status_code=424, detail=str(exc)) from exc
    except drive.DriveNotAuthorized as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@app.post("/drive/process/{file_id}", response_model=ProcessResult)
def drive_process(
    file_id: str, background_tasks: BackgroundTasks, force: bool = False
) -> ProcessResult:
    try:
        tmp_path, name = drive.download_pdf(file_id)
    except drive.DriveNotConfigured as exc:
        raise HTTPException(status_code=424, detail=str(exc)) from exc
    except drive.DriveNotAuthorized as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    try:
        result = process_pdf(tmp_path, name, force=force)
        if not result.cached:
            background_tasks.add_task(index_doc_by_id, result.doc_id)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)


@app.get("/documents/{doc_id}/index-status")
def document_index_status(doc_id: str) -> dict:
    return index_status(doc_id)


@app.post("/drive/sync")
def drive_sync_start(background_tasks: BackgroundTasks) -> dict:
    """Process EVERY PDF in the Drive folder, sequentially, in the background.
    Resumable: done files are skipped, failed/pending retried — safe to call
    again any time (including after a server restart mid-sweep)."""
    from . import drive_sync

    if not drive.is_authorized():
        raise HTTPException(status_code=401, detail="Drive not connected.")
    if not drive_sync.start():
        raise HTTPException(status_code=409, detail="Sync already running.")
    background_tasks.add_task(drive_sync.run_sync)
    return {"started": True}


@app.get("/drive/sync/status")
def drive_sync_status() -> dict:
    from . import drive_sync

    return drive_sync.status()


@app.get("/documents")
def list_documents() -> list[dict]:
    """The processed-documents catalog (registry)."""
    from . import registry

    return [dict(r) for r in registry.list_documents(registry.connect())]


@app.get("/documents/{doc_id}/pdf")
def document_pdf(doc_id: str) -> FileResponse:
    path = get_source_pdf(doc_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return FileResponse(path, media_type="application/pdf")
