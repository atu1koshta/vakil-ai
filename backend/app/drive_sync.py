"""Bulk Drive sync — process every PDF in the folder, sequentially, resumably.

The outbox pattern, keyed by Drive file id:
- `drive_sync` table (vakil.db) is the durable checkpoint: one row per Drive
  file, status pending -> done | failed. A restarted sweep skips done rows and
  retries pending/failed — "resume from where it stopped" is a DB query, not
  in-memory state.
- Network tolerance at two levels: each download gets 3 attempts with
  exponential backoff (transient blips), and any file that still fails is
  recorded and SKIPPED — one bad file never stops the sweep. Failures retry
  on the next sync run.
- Content dedup stacks underneath: a file whose bytes were already processed
  hits the registry sha256 cache inside process_pdf (instant), then its
  chunks hit the indexer hash cache. Re-syncing a mostly-done folder is cheap.
- The sweep runs as a FastAPI background task; if the server dies mid-sweep,
  nothing is lost — POST /drive/sync again and it continues.
"""
import threading
import time
import traceback

from . import registry
from .connectors import drive
from .indexer import index_document
from .pipeline import OUTPUT_DIR, process_pdf
from .vector_store import connect as vec_connect

DOWNLOAD_ATTEMPTS = 3
BACKOFF_BASE_SECONDS = 2

_lock = threading.Lock()
_state: dict = {"running": False, "current": None, "started_at": None}


def _table(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS drive_sync (
            file_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',   -- pending | done | failed
            doc_id TEXT,
            error TEXT,
            attempts INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.commit()


def refresh_catalog(conn) -> int:
    """Pull the Drive listing and add unknown files as pending. Returns new count."""
    _table(conn)
    added = 0
    for f in drive.list_pdfs():
        cur = conn.execute(
            "INSERT OR IGNORE INTO drive_sync (file_id, name, updated_at) VALUES (?, ?, ?)",
            (f.id, f.name, registry.utc_now()),
        )
        added += cur.rowcount
    conn.commit()
    return added


def _download_with_retry(file_id: str):
    last_error: Exception | None = None
    for attempt in range(DOWNLOAD_ATTEMPTS):
        try:
            return drive.download_pdf(file_id)
        except (drive.DriveNotConfigured, drive.DriveNotAuthorized):
            raise  # config problems don't heal with retries
        except Exception as e:  # network blips, API 5xx
            last_error = e
            time.sleep(BACKOFF_BASE_SECONDS * 2**attempt)
    raise last_error


def _process_one(conn, row) -> None:
    file_id, name = row["file_id"], row["name"]
    tmp_path = None
    try:
        tmp_path, original_name = _download_with_retry(file_id)
        result = process_pdf(tmp_path, original_name)  # registry sha dedup inside
        if not result.cached:
            index_document(vec_connect(), OUTPUT_DIR / result.doc_id)
        conn.execute(
            "UPDATE drive_sync SET status='done', doc_id=?, error=NULL, updated_at=? "
            "WHERE file_id=?",
            (result.doc_id, registry.utc_now(), file_id),
        )
    except Exception as e:
        traceback.print_exc()
        conn.execute(
            "UPDATE drive_sync SET status='failed', error=?, attempts=attempts+1, "
            "updated_at=? WHERE file_id=?",
            (str(e)[:500], registry.utc_now(), file_id),
        )
    finally:
        conn.commit()
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def run_sync() -> None:
    """The sweep. Runs in a background thread; sequential by design —
    docling + Ollama saturate one machine, parallelism buys nothing local.

    Work list is snapshotted once: every non-done file gets exactly one shot
    per sweep. A file that fails now is retried on the NEXT sweep, so the
    sweep always terminates."""
    conn = registry.connect()  # own connection: sqlite conns are per-thread
    try:
        refresh_catalog(conn)
        work = [
            r["file_id"]
            for r in conn.execute(
                "SELECT file_id FROM drive_sync WHERE status != 'done' ORDER BY name"
            ).fetchall()
        ]
        for file_id in work:
            row = conn.execute(
                "SELECT * FROM drive_sync WHERE file_id = ?", (file_id,)
            ).fetchone()
            if row is None or row["status"] == "done":
                continue
            _state["current"] = row["name"]
            _process_one(conn, row)
    finally:
        _state["running"] = False
        _state["current"] = None
        conn.close()


def start() -> bool:
    """Begin a sweep unless one is already running. Returns True if started."""
    with _lock:
        if _state["running"]:
            return False
        _state.update(running=True, current="listing folder…", started_at=registry.utc_now())
    return True


def status() -> dict:
    conn = registry.connect()
    _table(conn)
    counts = {
        r["status"]: r["n"]
        for r in conn.execute(
            "SELECT status, COUNT(*) AS n FROM drive_sync GROUP BY status"
        ).fetchall()
    }
    failures = [
        dict(r)
        for r in conn.execute(
            "SELECT file_id, name, error, attempts FROM drive_sync "
            "WHERE status='failed' ORDER BY updated_at DESC LIMIT 10"
        ).fetchall()
    ]
    conn.close()
    return {
        "running": _state["running"],
        "current": _state["current"],
        "pending": counts.get("pending", 0),
        "done": counts.get("done", 0),
        "failed": counts.get("failed", 0),
        "recent_failures": failures,
    }
