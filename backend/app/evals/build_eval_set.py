"""Merge question batches into evals/eval_set.jsonl with validation.

Checks per question:
- doc_slug is in the selection and its text file exists
- gold_passage actually occurs in the document text (whitespace-normalized —
  PDF extraction line breaks are noise)
- passage word count within bounds, qtype is known, one question per doc

Usage:
    python -m app.evals.build_eval_set
"""

import json
import re
import sys
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent.parent.parent / "evals"
TEXT_DIR = EVALS_DIR / "text"
QUESTIONS_DIR = EVALS_DIR / "questions"

QTYPES = {"holding", "facts", "principle", "paraphrase"}


def normalize(text: str) -> str:
    """Collapse all whitespace runs to single spaces for robust matching."""
    return re.sub(r"\s+", " ", text).strip().lower()


def main() -> None:
    doc_texts = {
        p.stem: normalize(p.read_text(encoding="utf-8"))
        for p in TEXT_DIR.glob("*.txt")
    }

    rows: list[dict] = []
    errors: list[str] = []
    seen_docs: set[str] = set()

    for batch in sorted(QUESTIONS_DIR.glob("batch_*.jsonl")):
        for line_no, line in enumerate(batch.read_text().splitlines(), 1):
            if not line.strip():
                continue
            where = f"{batch.name}:{line_no}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"{where}: bad JSON — {e}")
                continue

            slug = row.get("doc_slug", "")
            if slug not in doc_texts:
                errors.append(f"{where}: unknown doc_slug {slug!r}")
                continue
            if slug in seen_docs:
                errors.append(f"{where}: duplicate question for {slug}")
                continue
            if row.get("qtype") not in QTYPES:
                errors.append(f"{where}: bad qtype {row.get('qtype')!r}")
                continue

            passage = row.get("gold_passage", "")
            words = len(passage.split())
            if not 30 <= words <= 300:
                errors.append(f"{where}: gold_passage {words} words (want 30-300)")
            if normalize(passage) not in doc_texts[slug]:
                errors.append(f"{where}: gold_passage NOT verbatim in {slug}")
                continue
            if not row.get("question", "").strip() or not row.get("answer", "").strip():
                errors.append(f"{where}: empty question/answer")
                continue

            seen_docs.add(slug)
            rows.append(row)

    rows.sort(key=lambda r: r["doc_slug"])
    for i, row in enumerate(rows, 1):
        row["id"] = f"q{i:03d}"

    out = EVALS_DIR / "eval_set.jsonl"
    out.write_text(
        "\n".join(
            json.dumps(
                {k: row[k] for k in ("id", "doc_slug", "qtype", "question", "answer", "gold_passage")},
                ensure_ascii=False,
            )
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )

    from collections import Counter

    print(f"Valid questions: {len(rows)} -> {out}")
    print(f"Type mix: {dict(Counter(r['qtype'] for r in rows))}")
    if errors:
        print(f"\n{len(errors)} problems:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
