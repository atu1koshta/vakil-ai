"""Compare saved eval runs side by side.

    python -m app.evals.compare              # every run in evals/results/
    python -m app.evals.compare tagA tagB    # only these tags

Reads the JSON files run_eval writes; each carries its profile config
snapshot, so the table shows WHAT differed (model, chunk sizes, enrichment)
next to how the metrics moved.
"""

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "evals" / "results"

METRIC_KEYS = (
    "doc_recall@1", "doc_recall@3", "doc_recall@5", "doc_recall@10",
    "passage_recall@1", "passage_recall@3", "passage_recall@5", "passage_recall@10",
    "mrr",
)


def load_runs(tags: list[str]) -> list[dict]:
    paths = (
        [RESULTS_DIR / f"{t}.json" for t in tags]
        if tags
        else sorted(RESULTS_DIR.glob("*.json"))
    )
    runs = []
    for path in paths:
        if not path.exists():
            print(f"missing: {path}")
            continue
        summary = json.loads(path.read_text())["summary"]
        runs.append(summary)
    return runs


def describe(summary: dict) -> str:
    prof = summary.get("profile")
    if not prof:  # run predates config snapshots
        return "-"
    chunking = prof["chunking"]
    return (
        f"{prof['embedding']['model']} "
        f"{chunking['target_tokens']}/{chunking['overlap_tokens']}"
        f"{'' if prof['indexing']['enrich'] else ' no-enrich'}"
    )


def describe_retrieval(summary: dict) -> str:
    """Query-time strategy line, e.g. 'hybrid c20/k10 +cite +rw +rr50'.
    Everything via .get with defaults: old result files predate these keys
    and must still render."""
    prof = summary.get("profile")
    if not prof:
        return "-"
    r = prof.get("retrieval") or {}
    if r.get("strategy", "dense") == "dense":
        desc = "dense"
    else:
        desc = f"hybrid c{r.get('candidates', 50)}/k{r.get('rrf_k', 60)}"
    rewrite = r.get("rewrite") or {}
    if rewrite.get("citations"):
        desc += " +cite"
    if rewrite.get("llm"):
        desc += " +rw"
    rerank = r.get("rerank") or {}
    if rerank.get("enabled"):
        desc += f" +rr{rerank.get('pool', '')}"
    return desc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tags", nargs="*", help="run tags (default: all saved runs)")
    args = ap.parse_args()

    runs = load_runs(args.tags)
    if not runs:
        print(f"no results under {RESULTS_DIR}")
        return

    tag_width = (
        max(
            len("metric"),
            *(len(r["tag"]) for r in runs),
            *(len(describe(r)) for r in runs),
            *(len(describe_retrieval(r)) for r in runs),
        )
        + 2
    )
    header = "metric".ljust(24) + "".join(r["tag"].rjust(tag_width) for r in runs)
    print(header)
    print("config".ljust(24) + "".join(describe(r).rjust(tag_width) for r in runs))
    print(
        "retrieval".ljust(24)
        + "".join(describe_retrieval(r).rjust(tag_width) for r in runs)
    )
    print("chunks".ljust(24) + "".join(str(r["chunks"]).rjust(tag_width) for r in runs))
    for key in METRIC_KEYS:
        row = key.ljust(24)
        best = max((r.get(key, 0) for r in runs), default=0)
        for r in runs:
            value = r.get(key)
            cell = "-" if value is None else f"{value:.3f}" + ("*" if value == best and len(runs) > 1 else " ")
            row += cell.rjust(tag_width)
        print(row)
    # Latency: reported, never best-marked — lower is better, and old runs
    # predate the field.
    print(
        "mean_latency_ms".ljust(24)
        + "".join(
            (
                "-"
                if r.get("mean_latency_ms") is None
                else f"{r['mean_latency_ms']:.0f} "
            ).rjust(tag_width)
            for r in runs
        )
    )
    if len(runs) > 1:
        print("\n* best per metric")


if __name__ == "__main__":
    main()
