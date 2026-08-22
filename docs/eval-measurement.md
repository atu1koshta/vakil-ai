# Measuring Retrieval: How to Read Every Scoreboard in This Project

*The measurement primer behind every eval number in this repo — including the knob-sweep scoreboard in the [hybrid implementation field guide](/docs/hybrid-implementation.md) (Part 7.1). Nothing here is specific to hybrid retrieval; this is the ruler itself. Read this once and every `evals/results/*.json` file and every results table becomes legible.*

---

## Part 1 — What a run actually is

One eval run is a loop, nothing more:

1. Load `backend/evals/eval_set.jsonl` — currently **62 questions**. Each line carries a `question`, the `doc_slug` of the judgment that answers it (the **gold document**), a `gold_passage` (the exact text span holding the answer), and a `qtype` label.
2. For each question, call `retrieve(question, k=10)` — **the same function `/search` and `/ask` use**, not a parallel reimplementation. This is deliberate: the numbers must describe the code production runs. Change `config.yaml` and the eval measures the changed system automatically.
3. For each question, record two ranks: where the gold *document* first appeared in the top-10, and where a chunk *covering the gold passage* first appeared.
4. Aggregate those ranks into the metrics below; print a table; save everything (per-question detail included) to `evals/results/<tag>.json`.

Run one with `python -m app.evals.run_eval <tag> [--profile <p>]`; compare saved runs with `python -m app.evals.compare <tags...>`.

A **gold document** and a **gold passage** are different targets, and that difference is the whole reason there are two recall metrics.

---

## Part 2 — The three metrics

### 2.1 doc_recall@k — "did the right judgment show up at all?"

The fraction of questions where the gold document's `doc_id` appears anywhere in the top **k** retrieved chunks. Any chunk of the right judgment counts, even if it's the wrong paragraph.

This is the *coarse* metric. It moves when the embedding model, the enrichment, or the retrieval strategy changes — it barely notices the chunker, because any chunk of the document scores.

### 2.2 passage_recall@k — "did one chunk deliver the evidence intact?"

Stricter, and the interesting one. A question counts as a hit at k only if some **single** chunk in the top-k covers at least **50%** of the gold passage's word 5-grams.

Mechanics, because the definition does real work:

- The gold passage and each chunk are normalized (lowercase, punctuation stripped) and shredded into **5-gram shingles** — every run of 5 consecutive words. "the detention order was passed without application of mind" yields shingles like `(the, detention, order, was, passed)`, `(detention, order, was, passed, without)`, …
- Coverage = the fraction of the gold passage's shingles that also appear in the chunk. Shingles rather than bag-of-words because they demand the *phrasing in sequence*, not just shared vocabulary — a chunk that merely discusses the same topic scores near zero.
- The **single-chunk** requirement is a design choice: two half-chunks each holding half the evidence do **not** count. A downstream LLM reading fragmented evidence must stitch it back together; this metric refuses to give the retriever credit for that situation.

That makes passage_recall the **chunker's signal**: split a judgment mid-passage and passage_recall drops while doc_recall doesn't move. When comparing chunking variants, this is the column to watch. When comparing retrieval strategies (same chunks), a passage gap below the doc line means "we find the right judgment but surface the wrong part of it" — a ranking problem, not a chunking problem.

### 2.3 MRR — "how high, on average?"

Mean reciprocal rank of the gold *document*: a question answered at rank 1 contributes 1.0, rank 2 contributes 1/2, rank 3 contributes 1/3, a complete miss contributes 0. Averaged over all questions.

Recall@k is binary per question — rank 1 and rank 5 look identical to recall@5. MRR is the tiebreaker that sees inside the cutoff: two runs with equal recall@5 but different MRR differ in how often the gold doc sits at the *top*. That matters because context assembly keeps the best-ranked chunks; rank 1 vs rank 5 changes what the LLM actually reads.

### 2.4 Why k = 5 is the headline

Metrics are computed at k = 1, 3, 5, 10, but tables usually quote @5 — because 5-8 chunks is roughly what context assembly passes to the model. recall@10 is diagnostic ("was it *near*?"); recall@5 is "would the answer actually have been in the prompt?".

---

## Part 3 — Slices: the aggregate hides the story

Every question carries a `qtype`: **facts**, **holding**, **principle**, **paraphrase**, and — added when the step-5 probes exposed dense retrieval's identifier blindness — **citation** (questions that locate a judgment by its reporter citation, e.g. "AIR 1962 SC 406").

The run reports each metric overall *and* per type, and the per-type table is where diagnosis happens. The motivating example: before hybrid retrieval, aggregate doc_recall@5 was 0.774 — looks decent — while the citation slice sat at **0.000**. A whole query class was completely broken and the aggregate barely flinched, because citations are 12 of 62 questions. An aggregate is a weighted average of stories; slices are the stories.

Corollary: adding eval questions is itself a measured change. Adding the 12 citation questions moved the *aggregate* of every run before and after — which is why the dense baseline was re-frozen (`dense-baseline-v2`) on the new set. **Numbers are only comparable across runs scored on the same eval set.**

---

## Part 4 — Reading a scoreboard row

Concrete row from the hybrid knob sweep (field guide, Part 7.1):

| run | knobs | citation doc@5 | citation pass@5 | aggregate doc@5 | MRR |
|---|---|---|---|---|---|
| hybrid-c20-k10 | c20 / k10 | 0.833 | 0.833 | 0.968 | 0.835 |

Decoded, column by column:

- **run** — the tag passed to `run_eval`; also the filename under `evals/results/`. The JSON embeds the full profile snapshot, so a tag is a reproducible claim, not just a label.
- **knobs** — the config under test (here: RRF `candidates` and `rrf_k`). Everything else held fixed; a scoreboard is only meaningful when rows differ in exactly the thing being swept.
- **citation doc@5 = 0.833** — for 10 of the 12 citation questions, some chunk of the right judgment ranked in the top 5.
- **citation pass@5 = 0.833** — for those same questions, a single top-5 chunk covered ≥ half the gold passage's 5-gram shingles. When the two citation columns match, finding the doc and finding the passage succeed or fail together for this slice.
- **aggregate doc@5 = 0.968** — 60 of 62 questions across all types found their document in the top 5.
- **MRR = 0.835** — on average, the gold document sits between rank 1 and rank 2.

Reading *down* a scoreboard: find the column that motivated the experiment (here, the citation slice — the witnessed failure), confirm it moved; then scan the other columns for regressions the change might have caused elsewhere. A row that wins its target column but drops the aggregate is a trade, not a win.

## Part 5 — How much can 12 questions prove?

With n = 12, each question is worth **0.083** of the slice score — the difference between 0.750 and 0.833 is exactly *one question*. So small-slice deltas are hypothesis-sized evidence, not proof-sized. The project's standard for acting on them anyway: the *mechanism* behind the number must be independently witnessed (a smoke-test query, a traced ranking), and the change must target that mechanism. The number confirms the story; it is not, alone, the story. When the eval set grows, small-slice conclusions get re-checked.

The same discipline in the other direction: per-question `details` in the result JSON (`doc_rank`, `passage_rank`, `best_coverage`, `top_docs`) exist so every miss can be *explained*, not just counted. A metric that drops without an explainable miss is a bug in the eval before it's a bug in retrieval.

---

## Part 6 — The measurement loop, end to end

The workflow every retrieval change follows:

1. **Freeze a baseline** — run the eval on the current system under a durable tag (`dense-baseline-v2`). Numbers before opinions.
2. **Change one thing** — a config knob, a strategy, a chunker. One variable per run tag.
3. **Re-run under a new tag** — `python -m app.evals.run_eval hybrid-c20-k10 --profile <p>`.
4. **Compare** — `python -m app.evals.compare dense-baseline-v2 hybrid-v1 hybrid-c20-k10` prints the side-by-side with best-per-metric markers.
5. **Explain the residue** — the questions still missing define the *next* step's work (miss analysis from the details array), the same symptom → diagnosis → cure trail the progress page tracks.

The rule the harness enforces by existing: **every retrieval change must move these metrics or it does not ship.**

---

## Part 7 — Self-test

1. Why can passage_recall fall while doc_recall stays flat? Which component does that implicate?
2. A chunk contains every word of the gold passage but scrambled across unrelated sentences. Roughly what does 5-gram coverage score it, and why is that the right answer?
3. Two half-passages in two top-5 chunks: hit or miss for passage_recall@5? Defend the choice.
4. Two runs tie on doc_recall@5; MRR differs by 0.1. What is physically different about what the LLM will read?
5. Why was the dense baseline re-frozen as `dense-baseline-v2` when the citation questions landed?
6. Aggregate doc@5 of 0.774 coexisted with a 0.000 slice. What practice prevents this class of blindness?
7. The citation slice moved 0.750 → 0.833. How many questions is that, and what earns a change made on evidence that thin?
