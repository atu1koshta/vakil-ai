"""Pick N eval judgments from the Drive folder, stratified by era and size.

Strata: decade of the judgment (parsed from the Indian Kanoon-style filename,
"... on 19 May 1950.pdf") crossed with file size. Within each decade bucket
files are sorted by size and picked at even intervals, so the selection keeps
both long constitutional-bench judgments and short orders, old scans and
recent clean PDFs.

Usage:
    python -m app.evals.select_docs [N]

Writes evals/selection.json next to backend/.
"""

import json
import re
import sys
from pathlib import Path

from ..connectors import drive
from ..pipeline import slugify

EVALS_DIR = Path(__file__).resolve().parent.parent.parent / "evals"

YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")


def year_of(name: str) -> int | None:
    # Filenames use underscores ("on_18_February_1958_1.PDF") which are word
    # chars, so normalize to spaces before boundary matching.
    years = YEAR_RE.findall(name.replace("_", " "))
    return int(years[-1]) if years else None


def stratified_pick(files: list, n: int) -> list:
    buckets: dict[str, list] = {}
    for f in files:
        year = year_of(f.name)
        key = f"{(year // 10) * 10}s" if year else "unknown"
        buckets.setdefault(key, []).append(f)

    for bucket in buckets.values():
        bucket.sort(key=lambda f: f.size or 0)

    # Proportional allocation, at least 1 per non-empty bucket while room lasts.
    total = len(files)
    picked: list = []
    ordered = sorted(buckets.items())
    quotas = {
        key: max(1, round(len(bucket) / total * n)) for key, bucket in ordered
    }
    # Adjust quota sum to exactly n by trimming/padding the largest buckets.
    while sum(quotas.values()) > n:
        key = max(quotas, key=lambda k: quotas[k])
        quotas[key] -= 1
    while sum(quotas.values()) < n:
        key = max(ordered, key=lambda kv: len(kv[1]) - quotas[kv[0]])[0]
        quotas[key] += 1

    for key, bucket in ordered:
        quota = min(quotas[key], len(bucket))
        if quota == len(bucket):
            picked.extend(bucket)
            continue
        # Even spacing across the size-sorted bucket keeps size diversity.
        step = (len(bucket) - 1) / max(quota - 1, 1)
        indices = sorted({round(i * step) for i in range(quota)})
        picked.extend(bucket[i] for i in indices)
    return picked


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    files = drive.list_pdfs()
    # Drive folder holds duplicates differing only by extension case
    # (.PDF/.pdf) — same judgment, same slug. Keep the first of each slug.
    seen: set[str] = set()
    files = [f for f in files if not (slugify(f.name) in seen or seen.add(slugify(f.name)))]
    print(f"Drive folder: {len(files)} unique PDFs")
    picked = stratified_pick(files, n)

    EVALS_DIR.mkdir(exist_ok=True)
    out = EVALS_DIR / "selection.json"
    out.write_text(
        json.dumps(
            [
                {
                    "file_id": f.id,
                    "name": f.name,
                    "size": f.size,
                    "year": year_of(f.name),
                }
                for f in sorted(picked, key=lambda f: f.name)
            ],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    by_decade: dict[str, int] = {}
    for f in picked:
        year = year_of(f.name)
        key = f"{(year // 10) * 10}s" if year else "unknown"
        by_decade[key] = by_decade.get(key, 0) + 1
    print(f"Picked {len(picked)} -> {out}")
    for key in sorted(by_decade):
        print(f"  {key}: {by_decade[key]}")


if __name__ == "__main__":
    main()
