"""Download + process + index every judgment in evals/selection.json.

Resumable: process_pdf dedups on sha256 via the registry, index_document
dedups on chunk hash — rerunning after a crash skips finished docs.

Usage:
    python -m app.evals.process_selection
"""

import json
import time
import traceback
from pathlib import Path

from ..connectors import drive
from ..indexer import index_document
from ..pipeline import OUTPUT_DIR, process_pdf
from ..vector_store import connect as vec_connect

EVALS_DIR = Path(__file__).resolve().parent.parent.parent / "evals"


def main() -> None:
    selection = json.loads((EVALS_DIR / "selection.json").read_text())
    vec = vec_connect()
    done, failed = 0, []
    started = time.time()

    for i, entry in enumerate(selection, 1):
        name = entry["name"]
        tmp_path = None
        try:
            t0 = time.time()
            tmp_path, original_name = drive.download_pdf(entry["file_id"])
            result = process_pdf(tmp_path, original_name)
            index_document(vec, OUTPUT_DIR / result.doc_id)
            done += 1
            print(
                f"[{i}/{len(selection)}] {result.doc_id}: "
                f"{result.stats.total_chunks} chunks "
                f"({'cached' if result.cached else f'{time.time() - t0:.0f}s'})",
                flush=True,
            )
        except Exception as e:
            failed.append(name)
            print(f"[{i}/{len(selection)}] FAILED {name}: {e}", flush=True)
            traceback.print_exc()
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

    print(
        f"\nDone: {done}/{len(selection)} in {(time.time() - started) / 60:.1f} min"
    )
    if failed:
        print("Failed:")
        for name in failed:
            print(f"  {name}")


if __name__ == "__main__":
    main()
