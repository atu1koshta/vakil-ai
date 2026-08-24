# Rerank + Rewrite, Implemented: A Field Guide Including the Part That Failed

*The companion to the [rerank + rewrite knowledge base](/docs/rerank-rewrite.md). That document built the concepts — query ensembles, bi- vs cross-encoders, two-stage retrieval — and made predictions on record. This one walks what actually landed: every file, every decision, the ablation scoreboard, and the honest headline — one of the three mechanisms won decisively, and the step's central thesis was **falsified by measurement**. That is not a failed step; that is the eval harness doing exactly its job. Along the way this guide also teaches the concepts the KB doc did not cover, because they only appeared once reality pushed back: the pairwise autopsy, topical vs evidential relevance, the decisive-signal override, and domain shift.*

---

## Part 0 — The map and the scoreboard

Three mechanisms went in, each behind its own config flag, each measured in isolation. The ablation matrix — one mechanism per row — means every metric movement has exactly one owner:

| run | config | passage@1 | passage@5 | mrr | citation@5 | paraphrase@5 | latency |
|---|---|---|---|---|---|---|---|
| `2c-baseline` | hybrid c20/k10 | .226 | .839 | .835 | .833 | .286 | 172ms |
| **`2c-citations`** | **+ citation phrases** | **.419** | **.871** | **.962** | **1.0** | .286 | **120ms** |
| `2c-rewrite` | + LLM rewrite | .371 | .855 | .927 | 1.0 | .429 | 7.6s |
| `2c-rerank-c20` | + rerank (MiniLM-L-6, pool 30) | .242 | .645 | .930 | — | — | 8.4s |
| `2c-rerank-bge` | rerank model swap: bge-reranker-base | .387 | .774 | .954 | — | — | 7.6s |
| `2c-rerank-c50` | re-widened c50/k60, pool 50 | .387 | .758 | .954 | — | — | 8.1s |

All runs saved under `backend/evals/results/`, each embedding its full profile snapshot for reproducibility.

**What landed enabled: citation phrases only.** The LLM rewrite and the reranker landed as built, measured, documented knobs — turned **off**, with the numbers that convicted them written as comments next to the flags in `config.yaml`. Nothing was deleted: a stronger reranker is one yaml edit from a retrial, and the matrix that convicted this one is ready to hear the case again.

The three findings, one per mechanism:

1. **Citation phrases swept the board.** All 12 citation questions moved to rank 1 — including q004, whose cure was supposed to be the LLM rewrite (its question quotes the citation verbatim, so the phrase detector caught it first). Every other question was **byte-identical** to baseline: the inert-or-decisive prediction held exactly.
2. **The LLM rewrite is a real trade, and a bad one here.** Paraphrase recall rose as predicted (.286 → .429) — but holding recall fell (1.0 → .85), and every query pays 7.4 seconds. Net passage@5: down.
3. **The reranker made retrieval worse with every model tried, and widening made it worse still.** The KB doc's testable prediction — "the 2b knob values should stop being optimal once the reranker exists" — came out **false** at this corpus/model scale.

---

## Part 1 — Before any feature: make the ruler trustworthy

### 1.1 Reproduce the baseline first

The very first change of the step was not a feature. It was re-running the frozen 2b baseline (`hybrid-c20-k10`) under a new tag, `2c-baseline`, and checking it reproduced **exactly** — same .226/.839/.835, same q004 miss at doc_rank None, same q017 at rank 8.

Why bother? Because every claim this step makes is a *delta against that baseline*. If the baseline itself had drifted (a dependency upgrade, an index rebuild, a changed eval set), every later comparison would be attributing that drift to whichever feature happened to land next to it. A reproduction run costs three minutes and buys the right to trust every subsequent row of the scoreboard. It reproduced to the third decimal — deltas mean something.

### 1.2 Teach the eval to report the price, not just the quality

Two mechanisms in this step (LLM rewrite, cross-encoder) trade **latency for precision**. An eval that reports only quality would hide half of every transaction. So before the features existed, the harness learned to measure them:

- `run_eval.py` wraps each `retrieve()` call in `time.perf_counter()` and records `elapsed_ms` per question, plus `mean_latency_ms` in the summary. The timing wraps *exactly* the production path — the same honesty rule the eval has always followed (it measures `retrieve()`, never a parallel reimplementation).
- `compare.py` gained two rows. A **retrieval** row renders each run's query-time config as a compact signature (`hybrid c20/k10 +cite +rw +rr30`), because from this step on, runs differ by retrieval flags while the embedding/chunking line stays constant — the old config row could no longer tell runs apart. And a **latency** row, deliberately *excluded* from the best-per-metric starring: the star logic marks the maximum, and for latency lower is better; starring it would praise the slowest run.

One defensive detail worth copying elsewhere: the retrieval descriptor reads everything through `.get(...)` with defaults, because saved result files from 2b **predate these config keys** and must still render. Old data outlives new schemas; readers of old data must tolerate their absence.

### 1.3 A schema decision the plan got wrong, caught before landing

The plan drafted `RewriteConfig.citations` defaulting to `True` — while also claiming "profiles that never mention these keys retrieve identically to before this step." Both cannot be true: a `True` default silently changes every existing hybrid profile's behavior the moment the code lands, and worse, it breaks the ablation: the "citations" improvement would be smeared into whatever commit introduced the schema instead of the commit that flips the flag.

Resolution: **every new flag defaults `False` in the schema; the yaml flips it, per feature, with the evidence run named next to it.** The general rule: *a new mechanism's default must be "absent," so that enabling it is an explicit, attributable act.* This discipline is what makes the scoreboard in Part 0 readable at all.

---

## Part 2 — Citation phrase queries (the mechanism that won)

### 2.1 What was built

**`app/rewrite.py` — a plain module, deliberately not a component package.** The codebase's convention (CLAUDE.md) is ABC + registry + factory for anything with multiple plausible backends. Citation detection is one regex list; there is no second implementation on the horizon. A package here would be ceremony — structure signaling flexibility that nothing needs. (Contrast with `app/rerank/` in Part 4, which *is* a package, because reranker backends genuinely vary.) Knowing when *not* to apply your own pattern is part of owning it.

```python
def detect_citations(question: str) -> list[str]
```

Six regexes over Indian legal reporter grammars: `AIR <year> <court> <n>`, `(<year>) <n> SCC <n>` (with Cri/Civ/L&S variants), `<year> Supp (<n>) SCC <n>`, `<year> (<n>) SCC <n>`, `[<year>] <n> SCR <n>`, bare `<year> SCR <n>`. Why a regex and not a model: reporter citations are a *grammar* — a small, rigid, closed format. The right tool for a grammar is a grammar tool: deterministic, testable, zero latency, no model to be wrong. (The KB doc's §1.2 argument, and this step's results vindicated it emphatically — see Part 6.)

### 2.2 The one sharp edge: normalize through the tokenizer's own rule

Every matched span is normalized as:

```python
" ".join(re.findall(r"[A-Za-z0-9]+", span))
```

so `(1995) 2 SCC 7` becomes `1995 2 SCC 7`. This is not cosmetic. An FTS5 phrase query matches only if its tokens equal what the index's unicode61 tokenizer produced **at index time** — and the index stored `(1995)` as the bare token `1995`, parentheses stripped. A detector that emitted the phrase with parentheses intact would match *nothing, silently*: no error, no warning, just an empty list forever. The `[A-Za-z0-9]+` split is the exact rule `lexical_search` already used for OR queries, so both sides of the match agree by construction. 2b's lesson — "tokenization decides everything" — biting from the query side.

### 2.3 The store: an optional capability, and a safety property for free

**`vector_store/base.py`** gained `lexical_phrase_search(phrases, k)` beside `lexical_search`, with a default body raising `NotImplementedError`. Same optional-capability pattern as 2b: a backend without a positional text index simply doesn't claim the capability, and hybrid features fail loudly against it instead of silently degrading.

**`vector_store/sqlite.py`** got a small refactor before the feature: the staleness check + SELECT + row-mapping shared by both lexical methods moved into a private `_fts_query(match, k)`. Now `lexical_search` builds an OR-of-tokens match string and `lexical_phrase_search` builds an OR-of-quoted-phrases match string, and both hand off to the same plumbing. One code path for FTS5 mechanics means the self-healing rebuild logic exists exactly once.

The phrase method **re-sanitizes** every phrase through the same `[A-Za-z0-9]+` split before quoting — even though `detect_citations` already normalized it. This buys a security property, not just hygiene: *quoted FTS5 content can never be parsed as an operator*. Whatever arrives in `phrases` — a future caller passing raw user text, a detector bug — the tokens end up inside double quotes where FTS5 treats them as literal terms. No user-supplied string can inject `NEAR`, `NOT`, column filters, or any other FTS5 syntax. Sanitizing at the *boundary where the query is built*, rather than trusting upstream callers, is the same principle as parameterized SQL.

### 2.4 The wiring: a separate list, never a mutation

In `retrieve()`, detected citations become **their own RRF list**:

```python
if cfg.rewrite.citations:
    phrases = detect_citations(question)
    if phrases:
        rankings.append(store.lexical_phrase_search(phrases, k=cfg.candidates))
```

The alternative — appending phrase syntax into the existing OR query — was rejected because it would change that query's semantics for *every* question, and a phrase-heavy MATCH would let BM25's per-column weighting interact with the OR terms in ways nobody could attribute. As a separate list, the mechanism is **inert or decisive**: no chunk contains the exact phrase → empty list → RRF is arithmetically unchanged; some chunk contains it → that chunk arrives at rank 1 of a list RRF then amplifies. There is no middle mode where it injects plausible junk.

### 2.5 The evidence

`2c-citations` vs `2c-baseline`, per-question diff: **exactly the 12 citation questions moved — all to doc_rank 1 and passage_rank 1 — and zero non-citation questions changed by even one rank.** The inertness prediction was not just "other slices flat at @5"; it held byte-for-byte across the whole result file. Citation passage@5 .833 → 1.0, passage@1 .226 → .419, doc@1 .742 → .935, mrr .835 → .962. Latency *dropped* (172ms → 120ms — run-to-run noise plus the phrase list often resolving quickly; either way, the win costs nothing).

And a bonus the plan didn't predict: **q004 — assigned to the LLM-rewrite cure — was recovered by the phrase list.** Its question ("What does the judgment say about the decision reported in 1952 SCR 135?") *contains* the citation; the detector doesn't care that prose surrounds it. The regex cure ate the neural cure's lunch. Remember this when reading Part 3: by the time the LLM rewrite ran, its marquee case was already solved.

---

## Part 3 — The LLM rewrite (the mechanism that traded badly)

### 3.1 What was built

`llm_rewrite(question, model=None)` in `rewrite.py`, using the existing `app/llm` ChatModel layer (`get_chat_model` — so the rewrite inherits the whole provider registry, env overrides, everything). System prompt: rewrite into a short keyword query; keep case names, citations, statutes, sections, doctrines exactly as written; one line, nothing else. Model: `llama` (llama3.1) — deepseek-r1's thinking tokens are far too slow for a per-query call.

The guards are the design: strip, take the first line, then return `None` if the result is empty, longer than 200 chars, or identical to the input — and catch **every** exception (`GenerationError`, `ConfigError`, network) into `None` as well. The rule behind all the Nones: *retrieval must never break because an optional enhancement had a bad day.* A `None` rewrite means the ensemble is one query smaller — exactly yesterday's behavior.

### 3.2 The wiring: the ensemble loop

```python
queries = [question]
if cfg.rewrite.llm:
    rewritten = llm_rewrite(question, cfg.rewrite.model or None)
    if rewritten:
        queries.append(rewritten)
rankings = []
for q in queries:
    rankings.append(store.search(embedder.embed_query(q), k=cfg.candidates))
    rankings.append(store.lexical_search(q, k=cfg.candidates))
```

The raw question **always** contributes its dense+lexical pair; a successful rewrite adds one more pair. A bad rewrite can only inject extra candidates into fusion, never displace the raw lists — failure bounded by construction, not by hoping the LLM behaves. Sanity spot-check before the eval: "What does the judgment say about the decision reported in 1952 SCR 135?" → `Decision in 1952 SCR 135`. The mechanism works as specified.

### 3.3 The evidence, and the concept the KB doc undersold

`2c-rewrite` vs `2c-citations`: paraphrase@5 .286 → .429 (predicted — keyword rewrites land closer to the corpus's own vocabulary than conversational phrasing), principle@5 .8 → .9. But **holding@5 1.0 → .85**: three holding questions whose gold chunk had comfortably sat in the top-5 got pushed past it by candidates the rewrite lists dragged into fusion. Net passage@5 .871 → .855, mrr .962 → .927, and +7.4 seconds on every single query.

Here is the concept to take away, sharper than the KB doc stated it: **"add, never replace" bounds *damage*; it does not guarantee *benefit* — and in rank fusion, added lists are never free.** RRF is a voting system. Every list you add gets a vote on every chunk it contains, and a chunk that appears at mediocre ranks in *several* lists can out-vote a chunk one list ranked first (2b's false consensus). The rewrite lists are correlated with the raw lists by intent, but not perfectly — and their disagreement votes landed on the wrong chunks often enough to cost more at top-5 than the paraphrase gains paid back. The safety argument (worst case ≈ old behavior) and the value argument (average case > old behavior) need **separate evidence**, and this step is the witnessed example of a mechanism passing the first and failing the second.

Off by default in yaml, with these numbers in the comment. The paraphrase slice keeps the receipt for a future where paraphrase matters more than holding — or where a fusion scheme can weight the raw lists above the rewrite lists.

---

## Part 4 — The reranker (the mechanism that lost, and how we know why)

### 4.1 What was built — the package that earns its structure

**`app/rerank/`** follows the full CLAUDE.md component layout, and unlike `rewrite.py` it should: reranker backends genuinely vary (cross-encoders, LLM-as-judge, cloud APIs like Cohere), so the seam is worth paying for up front.

- `base.py` — `Reranker` ABC: `rerank(question, rows, k) -> list[dict]`, plus `RerankError`. The contract's one aggressive clause: the returned rows' `score` is **replaced** by the reranker's relevance score. The pool's RRF scores are meaningless once a better judge has read the actual pairs — carrying both forward would invite downstream code to compare incomparable scales.
- `cross_encoder.py` — `CrossEncoderReranker`. Two implementation decisions that matter:
  - **Lazy import.** `sentence_transformers` pulls in torch at import time. The import happens inside model loading, so a profile with rerank disabled never pays it; an `ImportError` becomes a `RerankError` naming the exact pip install.
  - **Module-level model cache keyed by model name.** `retrieve()` constructs a reranker per call (it's cheap — just config). Without the cache, every `/search` would reload ~80MB+ of weights from disk. With it, the first call per process loads; every later call reuses.
- `__init__.py` — `_RERANKERS = {"cross-encoder": CrossEncoderReranker}` and `get_reranker(profile)`, unknown provider → `ConfigError`, same contract as the embeddings factory.

`RerankConfig` (enabled / provider / model / pool / batch_size) sits under `RetrievalConfig` — **outside the fingerprint**, because a reranker re-scores rows the index already returned; it never changes what vectors exist, so flipping it must never invalidate an index. And rerank is a *flag*, not a new `strategy` value: a precision stage over a candidate pool doesn't care whether the pool came from dense or hybrid retrieval, so it composes with either.

### 4.2 The gotcha fix: a pool is not k

The dense branch used to fetch exactly `k` results. With rerank enabled that is quietly wrong: **reranking k rows can only reorder them — it can never recover anything below the cut.** The whole point of a precision stage is choosing k from a *larger* pool. So the dense branch now fetches `max(k, rerank.pool)` when rerank is on. (Hybrid already fetched `candidates`-deep lists, so its fused pool was naturally larger than k.)

The final orchestration, with the rerank tail:

```python
if not cfg.rerank.enabled:
    return pool[:k]
return get_reranker(prof).rerank(question, pool[:cfg.rerank.pool], k)
```

The reranker's query is the **raw question, never the rewrite** — the user's own words are the ground truth of intent; the rewrite is a retrieval aid, not a replacement statement of what the user wants.

A ripple this created in the public contract: `/search`'s `score` field now lives on a **third scale** — cosine similarity (dense), RRF sum (hybrid), or cross-encoder logit (reranked: raw, unbounded, possibly negative). The stale "cosine similarity" comment in `main.py` was corrected; the retrieval docstring now states the rule: scores are comparable only within one result list, never across strategies.

### 4.3 A dependency landed with a tripwire check

`sentence-transformers>=3.0` was the step's only new package — but the venv's torch 2.13 is a *docling transitive dependency*, and a silent torch upgrade could break PDF parsing, a completely different subsystem. So the install was preceded by `pip install --dry-run` and the output checked: torch 2.13.0 already satisfies `>=2.2`, only sentence-transformers + scikit-learn would be added. Verified again after the real install. The general habit: **when two features share a heavy dependency, check the intersection before installing, not after something unrelated breaks.**

One environment note the plan got pleasantly wrong: on this machine sentence-transformers picked `mps:0` (Apple Silicon GPU), not CPU — so all latency numbers here are *better* than the plan's CPU estimates, which only strengthens the verdict below (the reranker lost even with a hardware discount).

### 4.4 The evidence: it lost, twice, then lost the thesis too

`2c-rerank-c20` (MiniLM-L-6, pool 30, over the full ensemble): **passage@5 .855 → .645, passage@1 .371 → .242.** Not a wobble — a collapse, with a distinctive signature in the miss list: question after question showing `doc_rank=1` but `best_coverage≈0`. The reranker was putting the **right document** first while swapping the **wrong chunk of it** into the top ranks — and dropping the gold chunk out of the top-10 entirely.

---

## Part 5 — The autopsy (three concepts the KB doc doesn't teach)

The KB doc's risk list said "watch for rerank-induced passage drops" and named truncation as the suspect. Watching found the drops; the autopsy found the suspect mostly innocent. This part is the methodology, because the *way* the diagnosis was reached is more reusable than the diagnosis itself.

### 5.1 Concept: hypothesis-first debugging with a measurable discriminator

**Step one — test truncation, quantitatively.** The truncation hypothesis makes a specific prediction: gold chunks lost to reranking should hold their evidence *late* in the chunk, past the ~512-token joint-input ceiling. So: take the 14 questions whose passage was recalled before reranking and lost after; for each, locate the gold passage's token offset inside its gold chunk (tokenize the chunk, find where the passage's first 5-gram begins). Result: only **2 of 14** had the evidence past token 400. One clean witness — q017's gold chunk holds its citation at token 493 of 864 — but as the *main* explanation, truncation was acquitted. Twelve losses needed a different cause.

The reusable move: a hypothesis about *why* a model fails usually implies a measurable property of the failing cases. Compute the property before believing the hypothesis. Ten minutes of scripting prevented shipping a sliding-window mitigation that would have fixed 2 of 14 losses and left the rest standing.

### 5.2 Concept: the pairwise preference probe

**Step two — reduce the failure to its atom.** A full eval run costs ~8 minutes (the LLM rewrite dominates) and mixes every stage's behavior. But the reranker's failure has an atomic form: *given the gold chunk and the chunk the reranker actually promoted, which does the model score higher?* That's two `predict()` calls per question — a **pairwise preference probe**. Run it for six lost questions across three candidate models:

| model | gold wins | latency / 30 pairs |
|---|---|---|
| ms-marco-MiniLM-L-6-v2 (22M) | 2/6 | ~1.0s |
| ms-marco-MiniLM-L-12-v2 | 4/6 | ~1.3s |
| BAAI/bge-reranker-base | 5/6 | ~5.9s |

Two things fall out. First, a clean **capability gradient**: bigger/better-trained judges pick gold more often, which localizes the failure in *model judgment*, not in the pipeline plumbing around it (a plumbing bug wouldn't care which model runs). Second, a model-selection shortcut: the probe predicted bge would do best *before* paying for its full eval run — and the full run agreed (.774 vs .645). The general tool: **when a learned component misbehaves, extract the smallest input pair that exhibits the disagreement and sweep candidate models over it.** Seconds per model instead of minutes, and it separates "the model can't do this" from "we're calling the model wrong."

### 5.3 Concept: topical vs evidential relevance

Why does even the best judge err — and err with that `doc_rank=1, coverage=0` signature? Because "relevance" is two different questions wearing one word:

- **Topical relevance** — is this text *about* the subject the question raises? Every chunk of the right judgment tends to qualify: they share the parties, the statute, the vocabulary.
- **Evidential relevance** — does this text *contain the specific evidence* the question needs: the quoted passage, the exact holding, the cited paragraph?

Cross-encoders like ms-marco are trained on web search relevance labels — short passages judged "relevant to the query" in the topical sense. Handed twenty chunks of one judgment, such a model happily promotes the chunk that *discusses the topic most fluently* over the chunk that *contains the exact passage*, because nothing in its training distinguishes them. Our passage metric — and a RAG answer that must quote its evidence — lives entirely on the evidential side. The reranker was optimizing a neighboring objective, competently. This is also a **domain shift** story (web prose → Indian legal English, 60-word passages → 700-token chunks), but the topical/evidential split is the sharper lens: it explains the *signature* (right doc, wrong chunk), not just the general degradation.

### 5.4 Concept: the decisive-signal override

The most instructive individual failure: q017. The phrase list had placed its gold chunk at RRF rank 1 on the strength of an **exact phrase match** — a signal that is close to a certainty (the KB doc's own "decisive" mode). The reranker then re-scored the whole pool *from scratch*, gave that chunk a poor logit, and buried it below topically-pleasant neighbors. q022's gold document dropped out of the top-10 entirely, taking doc_recall@5 below what the un-reranked pool had.

Name the pattern: **a re-scoring stage that ignores the provenance of its candidates will override upstream signals stronger than its own judgment.** The pipeline *knew* — with near-certainty — that q017's chunk was right; the architecture handed final say to a component that couldn't know that and didn't ask. If a reranker returns to this codebase, the fix is architectural, not model-sized: exempt phrase-decisive candidates from re-scoring, or fuse the reranker's score with RRF evidence instead of replacing it. The KB doc taught "never replace, only add" for *queries*; this step learned the same principle applies to *scores*.

### 5.5 The thesis test: widening under the judge

One matrix row remained — the step's central prediction: with a precision stage downstream, wide candidates (c50/k60, pool 50) should finally beat 2b's narrow c20/k10, because the reranker absorbs the junk that widening lets in. Run with the best judge available (bge): **passage@5 .758 — worse than the same judge over the narrow pool (.774), and far below no reranker at all (.871).**

The compensation theory, falsified: widening only helps if the downstream judge ranks the pool *reliably*. An unreliable judge given a bigger pool simply has more junk to mis-promote. 2b's narrow knobs weren't compensating for a missing precision stage — they were the *correct* setting for a system whose best final judge is fusion plus a decisive phrase list. Written down as a prediction beforehand, the falsification took exactly one compare-table row to accept. That's what predictions-on-record buy: no motivated reasoning after the fact.

---

## Part 6 — What landed, and the shape of the final config

The evidence-backed landing in `config.yaml` (nomic-default):

```yaml
retrieval:
  strategy: hybrid
  candidates: 20        # c20/k10 stays — the re-sweep vindicated it
  rrf_k: 10
  rewrite:
    citations: true     # the winner: citation@5 .833→1.0, everything else byte-identical
    llm: false          # measured net-negative: holding 1.0→.85, +7.4s/query
    model: llama
  rerank:
    enabled: false      # measured net-negative with every model tried
    provider: cross-encoder
    model: BAAI/bge-reranker-base   # best of three, for whenever it's retried
    pool: 30
    batch_size: 16
```

Every `false` carries its convicting run's numbers in a yaml comment — the 2b convention of documenting knob decisions where the knobs live. The final spot checks: q004 and q017 both return their gold document at rank 1, under 100ms each; the API server boots clean (pydantic `extra="forbid"` validates every new key at startup, so a typo'd flag fails at boot, not at query time).

Also landed: the CLAUDE.md component table row for `app/rerank/`, the corrected score-scale comments, and `sentence-transformers` in requirements with the torch note.

**Why keep two disabled mechanisms in the codebase?** Because the conclusions are *scale-bound* and the falsification itself is hypothesis-sized: 62 questions, one corpus, small local models. The machinery is sound and behind flags that default off; the eval matrix that convicted it is a `run_eval` invocation away from a retrial with a citation-aware reranker, a larger judge, or a corpus where paraphrase queries dominate. Deleting working, measured, off-by-default code would trade a few hundred lines for re-doing this entire step the day circumstances change.

---

## Part 7 — Lessons that survive the step

1. **Reproduce the baseline before trusting any delta.** Three minutes of re-running the frozen run is what makes every subsequent comparison meaningful.
2. **New mechanisms default off; enabling is an explicit, attributable act.** The plan's own draft violated this and would have smeared the ablation. Schema defaults are part of experimental design.
3. **Predictions written down before running make falsification cheap.** The widening thesis died in one table row, with no room for motivated reasoning.
4. **A hypothesis about model failure implies a measurable property — measure it.** Truncation predicted late-in-chunk evidence; 2 of 14 losses had it. The obvious suspect was mostly innocent.
5. **The pairwise preference probe: reduce a reranking failure to its atomic pair and sweep models over it.** Seconds per model, localizes the fault, and predicts full-eval outcomes.
6. **Topical relevance is not evidential relevance.** Web-trained cross-encoders optimize the former; passage-level RAG needs the latter. "Right doc, wrong chunk" is the telltale signature.
7. **Don't let a weak judge override strong evidence.** Re-scoring from scratch discards signal provenance; an exact phrase match is a near-certainty no small cross-encoder should be allowed to veto. "Add, never replace" applies to scores, not just queries.
8. **Bounded damage is not net benefit.** The ensemble structure guaranteed the rewrite's worst case; only measurement could reveal its average case, and the average lost.
9. **Cheap deterministic mechanisms first.** The regex detector: zero latency, zero nondeterminism, swept its whole failure class *including a case assigned to the LLM cure*. The expensive learned mechanisms fought over the leftovers and lost on net.
10. **Latency is a metric, not a footnote.** The winning config is also the fastest thing measured all step. 120ms vs 7.6s is interactive vs batch — a product decision hiding in an eval column.

## Appendix — Vocabulary this step added (beyond the KB doc's)

| Term | The idea it names |
|---|---|
| Baseline reproduction | Re-running the frozen reference before a step, so every delta is attributable to the step |
| Defaults-off discipline | New flags default to absent/false in schema; yaml enables per feature, keeping the ablation clean |
| Pairwise preference probe | Scoring (gold, promoted-rival) pairs directly to localize and compare judge models in seconds |
| Capability gradient | Failure rate falling monotonically with judge strength — evidence the fault is model judgment, not plumbing |
| Topical vs evidential relevance | "Discusses the subject" vs "contains the exact evidence" — web-trained rerankers optimize the first, passage metrics demand the second |
| Decisive-signal override | A re-scoring stage with no notion of provenance vetoing upstream signals stronger than its own judgment |
| Bounded ≠ beneficial | Structural safety caps the worst case; only measurement reveals the average case |
| Dependency tripwire | `pip install --dry-run` before adding a package that shares heavy transitive deps with an unrelated subsystem |
