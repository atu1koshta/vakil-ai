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


@app.get("/search")
def search(q: str, k: int = 5) -> list[dict]:
    """Semantic search over indexed chunks (run `python -m app.indexer` first)."""
    from .embeddings import EmbeddingError, embed_query
    from .vector_store import connect as vec_connect
    from .vector_store import search as vec_search

    try:
        vector = embed_query(q)
    except EmbeddingError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    rows = vec_search(vec_connect(), vector, k=k)
    for r in rows:
        r["score"] = round(r["score"], 4)  # cosine similarity, higher = closer
    return rows


@app.post("/process-document", response_model=ProcessResult)
async def process_document(file: UploadFile, background_tasks: BackgroundTasks) -> ProcessResult:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(await file.read())
        result = process_pdf(Path(tmp_path), file.filename)
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
def drive_process(file_id: str, background_tasks: BackgroundTasks) -> ProcessResult:
    try:
        tmp_path, name = drive.download_pdf(file_id)
    except drive.DriveNotConfigured as exc:
        raise HTTPException(status_code=424, detail=str(exc)) from exc
    except drive.DriveNotAuthorized as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    try:
        result = process_pdf(tmp_path, name)
        background_tasks.add_task(index_doc_by_id, result.doc_id)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)


@app.get("/documents/{doc_id}/index-status")
def document_index_status(doc_id: str) -> dict:
    return index_status(doc_id)


@app.get("/documents/{doc_id}/pdf")
def document_pdf(doc_id: str) -> FileResponse:
    path = get_source_pdf(doc_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return FileResponse(path, media_type="application/pdf")
