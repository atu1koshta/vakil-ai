"""Run the retrieval eval against the current vector store.

Metrics (k = 1, 3, 5, 10):
- doc_recall@k    — gold judgment appears among top-k chunks' doc_ids.
- passage_recall@k — some single top-k chunk covers >= COVER_THRESHOLD of the
  gold passage's word 5-gram shingles. Single-chunk coverage is the chunker
  signal: it asks "did one chunk deliver the evidence intact", so it moves
  when chunk boundaries move, unlike doc_recall.
- mrr — mean reciprocal rank of the gold doc over top-MAX_K.

Prints a summary table and writes evals/results/<tag>.json with per-question
detail for miss analysis. The result embeds the full profile config snapshot,
so every saved run is reproducible and comparable (see compare.py).

Usage:
    python -m app.evals.run_eval [tag] [--profile <name>]

Tag names the run (e.g. "baseline-700-80"); defaults to the profile name.
--profile scores that profile's index (build it first with
`python -m app.indexer --profile <name>`); default = active profile.
"""

import argparse
import json
import re
from pathlib import Path

from ..config import get_profile
from ..embeddings import get_embedder
from ..vector_store import open_store

EVALS_DIR = Path(__file__).resolve().parent.parent.parent / "evals"
RESULTS_DIR = EVALS_DIR / "results"

KS = (1, 3, 5, 10)
MAX_K = max(KS)
SHINGLE = 5
COVER_THRESHOLD = 0.5


def normalize_words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()


def shingles(words: list[str], n: int = SHINGLE) -> set[tuple[str, ...]]:
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def coverage(gold: set[tuple[str, ...]], chunk_text: str) -> float:
    """Fraction of the gold passage's shingles present in the chunk."""
    if not gold:
        return 0.0
    return len(gold & shingles(normalize_words(chunk_text))) / len(gold)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tag", nargs="?", help="run name (default: profile name)")
    ap.add_argument("--profile", help="profile to score (default: active)")
    args = ap.parse_args()

    profile = get_profile(args.profile)
    tag = args.tag or profile.name
    embedder = get_embedder(profile)
    questions = [
        json.loads(line)
        for line in (EVALS_DIR / "eval_set.jsonl").read_text().splitlines()
        if line.strip()
    ]
    store = open_store(profile)
    total_chunks = store.count_chunks()
    print(
        f"Eval: {len(questions)} questions against {total_chunks} chunks "
        f"(profile {profile.name}, model {profile.embedding.model})\n"
    )

    details = []
    for q in questions:
        hits = store.search(embedder.embed_query(q["question"]), k=MAX_K)
        gold = shingles(normalize_words(q["gold_passage"]))

        doc_rank = next(
            (i + 1 for i, h in enumerate(hits) if h["doc_id"] == q["doc_slug"]), None
        )
        covers = [coverage(gold, h["text"]) for h in hits]
        passage_rank = next(
            (i + 1 for i, c in enumerate(covers) if c >= COVER_THRESHOLD), None
        )
        details.append(
            {
                "id": q["id"],
                "doc_slug": q["doc_slug"],
                "qtype": q["qtype"],
                "question": q["question"],
                "doc_rank": doc_rank,
                "passage_rank": passage_rank,
                "best_coverage": round(max(covers, default=0.0), 3),
                "top_docs": [h["doc_id"] for h in hits[:5]],
            }
        )

    n = len(details)
    summary: dict = {
        "tag": tag,
        "profile": profile.snapshot(),
        "questions": n,
        "chunks": total_chunks,
    }
    for k in KS:
        summary[f"doc_recall@{k}"] = round(
            sum(1 for d in details if d["doc_rank"] and d["doc_rank"] <= k) / n, 3
        )
        summary[f"passage_recall@{k}"] = round(
            sum(1 for d in details if d["passage_rank"] and d["passage_rank"] <= k) / n, 3
        )
    summary["mrr"] = round(
        sum(1 / d["doc_rank"] for d in details if d["doc_rank"]) / n, 3
    )

    by_type: dict = {}
    for qtype in sorted({d["qtype"] for d in details}):
        subset = [d for d in details if d["qtype"] == qtype]
        by_type[qtype] = {
            "n": len(subset),
            "doc_recall@5": round(
                sum(1 for d in subset if d["doc_rank"] and d["doc_rank"] <= 5)
                / len(subset),
                3,
            ),
            "passage_recall@5": round(
                sum(1 for d in subset if d["passage_rank"] and d["passage_rank"] <= 5)
                / len(subset),
                3,
            ),
        }
    summary["by_type"] = by_type

    print(f"{'metric':<20} {'value':>7}")
    for key, value in summary.items():
        if isinstance(value, (int, float)) and key not in ("questions", "chunks"):
            print(f"{key:<20} {value:>7}")
    print("\nby qtype (doc_recall@5 / passage_recall@5):")
    for qtype, stats in by_type.items():
        print(
            f"  {qtype:<12} n={stats['n']:<3} "
            f"{stats['doc_recall@5']} / {stats['passage_recall@5']}"
        )

    misses = [d for d in details if not d["passage_rank"] or d["passage_rank"] > 5]
    if misses:
        print(f"\npassage@5 misses ({len(misses)}):")
        for d in misses:
            print(
                f"  {d['id']} [{d['qtype']}] doc_rank={d['doc_rank']} "
                f"best_cover={d['best_coverage']} {d['doc_slug'][:50]}"
            )

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{tag}.json"
    out.write_text(
        json.dumps({"summary": summary, "details": details}, indent=2), encoding="utf-8"
    )
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
