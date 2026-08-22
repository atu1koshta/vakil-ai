# Hybrid Retrieval, Implemented: A Field Guide to Every Choice

*The companion to the [hybrid retrieval knowledge base](/docs/hybrid-retrieval.md). That document builds the concepts — dense vs lexical, BM25, RRF. This one walks the actual change that landed in 2b: every file, every decision, both bugs, and the eval story. Read the concept doc first; by the end of this one you should be able to defend every line.*

---

## Part 0 — The map

The whole change is six files. Hold this table loosely now; each row becomes a chapter.

| File | What changed | The decision it embodies |
|---|---|---|
| `app/config.py` | New `RetrievalConfig` on `Profile` | Retrieval strategy is a *query-time* knob, outside the index fingerprint |
| `config.yaml` | `retrieval:` block on `nomic-default` | Knobs are declared, not coded — and their tuning story is documented where they live |
| `app/vector_store/base.py` | Two new methods with *default bodies* | Lexical search is an optional capability, not a new obligation on every backend |
| `app/vector_store/sqlite.py` | FTS5 table + `lexical_search` + rebuild stamping | Index the words without duplicating a byte; know whether the index is actually built |
| `app/retrieval.py` | Strategy dispatch + `_rrf()` | Fusion by rank; callers inherit hybrid without changing |
| `app/indexer.py` | One rebuild call per run | The write path owns index freshness; the read path only self-heals |

And one number to keep in mind as the destination: citation-query recall@5 went **0.0 → 0.833**, while every conceptual query type sits at 1.0.

---

## Part 1 — Why only one function changed behavior

### 1.1 Beginner: the seam

Imagine the codebase as plumbing. Questions flow in from three places — the `/search` endpoint, the `/ask` RAG composition, and the eval harness — and all three pipes were deliberately routed through a single junction back in Step 2: a plain function called `retrieve(question, k, profile)`. It takes a question, returns ranked chunk dicts. No HTTP, no display formatting — those live in the callers.

That junction existed *for this moment*. The Step 2 module docstring literally reserved the parking spot: "hybrid BM25+RRF, reranking, query rewriting all land INSIDE here." So the entire user-visible behavior change of 2b is inside one function body:

```python
with open_store(prof) as store:
    if cfg.strategy == "dense":
        return store.search(vector, k=k)
    dense = store.search(vector, k=cfg.candidates)
    lexical = store.lexical_search(question, k=cfg.candidates)
    return _rrf([dense, lexical], cfg.rrf_k)[:k]
```

Nothing in `/search`, `/ask`, or `run_eval` changed — and that's not a happy accident, it's the payoff of the seam. It also means the eval measures *exactly* the code path production uses, which is what makes the before/after numbers honest.

### 1.2 Intermediate: what the function promises (the contract)

`retrieve()` returns dicts shaped `(doc_id, chunk_id, section, case_title, text, score)`. That shape is a *contract*: downstream code builds on it, so it can be extended but never broken. Hybrid keeps the shape — but quietly changes the *meaning* of one field, and that deserves a hard look (Part 6.3).

### 1.3 Advanced: why not a new function, a flag argument, or a subclass?

Alternatives considered and rejected:

- **`retrieve_hybrid()` beside `retrieve()`** — now every caller must choose, which is exactly the coupling the seam exists to prevent. Strategy is a property of the *system configuration*, not of the call site.
- **A `strategy=` parameter on `retrieve()`** — same problem in different clothes: callers would need to know about strategies to pass one.
- **A `HybridRetriever` class hierarchy** — machinery without a need. There is one function, two branches, and profiles already carry configuration. Add classes when a third strategy makes the `if` painful, not before.

The chosen shape: strategy comes from the profile, callers stay ignorant. Flipping `dense` ↔ `hybrid` in `config.yaml` re-routes the whole application, evals included.

---

## Part 2 — Where the knobs live, and the fingerprint question

### 2.1 Beginner: three shelves for configuration

This codebase sorts every knob onto one of three shelves:

1. **Pipeline-level** (parser, metadata extractor) — changes what's in `output/`; shared by all profiles; changing it means bumping `PIPELINE_VERSION` and reprocessing.
2. **Profile-level** (embedding model, chunking, enrichment) — changes what's *in the index*; each combination gets its own vector DB.
3. **Operational** (batch sizes, timeouts, URLs) — changes how work is done, never what is produced.

Where does `retrieval.strategy` go? The test is one question: *if I flip this knob, is anything already stored now wrong?*

### 2.2 Intermediate: the fingerprint, from scratch

Each profile computes a `fingerprint()` — a hash of exactly the fields that determine the *identity of the stored vectors*: model, dimension, prefixes, chunking config, enrichment flag. The store stamps this hash into the DB on first open and verifies it on every open. Point a profile at a DB built with different settings and it fails loudly (`IndexConfigMismatch`) instead of returning garbage similarities.

So the question becomes concrete: does flipping `dense` ↔ `hybrid` invalidate stored vectors? **No.** The vectors are untouched, and the FTS5 index (Part 3) is *derived from payloads the store already holds* — it can be rebuilt from what's in the DB at any moment. Hybrid changes how the index is *queried*, not what it *contains*. Therefore `RetrievalConfig` stays **out** of `fingerprint()`, and flipping strategies never forces a re-embed.

The counterfactual sharpens it: if we had chosen to build the FTS index from text *not* stored in the DB (say, re-reading `output/`), the index contents would depend on external state and this reasoning would collapse. Deriving FTS strictly from stored payloads is what buys the "query-time knob" classification.

### 2.3 The config itself

```yaml
retrieval:
  strategy: hybrid   # dense | hybrid (BM25 + RRF)
  candidates: 20     # depth per ranker before fusion
  rrf_k: 10          # sharper than the Cormack 60 default — see note above
```

Three choices worth defending:

- **`strategy` is a `Literal["dense", "hybrid"]`** in pydantic, not a free string. Other component fields (parser, store) are plain strings validated against registries — because implementations plug in via registry entries. Strategies aren't registry-backed; they're branches in `retrieve()`. A `Literal` gives the same fail-at-startup property (`extra="forbid"` ethos) with zero registry machinery.
- **`candidates` (default 50)** — how deep each ranker's list goes before fusion. Retrieve wide, fuse, cut to k. Why it ended up at 20 is the Part 7 story.
- **`rrf_k` (default 60)** — the RRF flattening constant. Why it ended up at 10 is the same story.

The yaml comment above these knobs records the sweep results *in place* — the next reader shouldn't need to excavate git history to learn why the values deviate from canon.

---

## Part 3 — Building a word index without storing anything twice

### 3.1 Beginner: what FTS5 is

SQLite ships a full-text search engine called FTS5. You create a *virtual table* — something that speaks the table interface (SELECT, INSERT) but is powered by custom machinery, here an inverted index: for every token, a list of the rows containing it. Query it with `MATCH` and it returns matching rows *ranked by BM25*, built in. No new server, no new dependency, no new file — it lives inside the same `vectors.db` the vectors live in.

That locality is worth pausing on. The lexical index is part of the *disposable, per-profile index artifact*: delete `vectors.db`, re-run the indexer, and both the vectors and the FTS index come back from `output/`. One artifact, one lifecycle, one transaction scope.

### 3.2 Intermediate: the storage decision — external content

FTS5 has two storage modes, and the choice is a real fork:

- **Contentful (default):** the FTS table stores its own copy of every indexed string. Simple — inserts into the FTS table are self-contained — but every chunk's text would now live in the DB *twice*.
- **External content:** the FTS table stores *only the inverted index* and reads the actual text through to another table when needed:

```sql
CREATE VIRTUAL TABLE chunk_fts USING fts5(
    case_title, section, text,
    content='chunk_vectors', content_rowid='id',
    tokenize='unicode61'
);
```

`content='chunk_vectors'` says "the real rows live there"; `content_rowid='id'` maps FTS rowids onto `chunk_vectors.id` (which, being an `INTEGER PRIMARY KEY`, *is* SQLite's rowid — an alias, not a copy). Zero payload duplication; the index is pure derivation.

The price of external content is a responsibility: **the FTS table does not watch the content table.** Insert into `chunk_vectors` and the FTS index silently knows nothing about it. Someone must tell it to update. That responsibility — and the bug it produced — is Parts 3.5 and 5.

### 3.3 Which text gets indexed: the enrichment-parity decision

The dense side embeds *enriched* text — "case title — section" prepended to each chunk — so every chunk is findable by case name even when the chunk body never mentions it. The lexical side should be enriched *the same way*, or citation-in-title queries would work in one index and not the other.

Rather than concatenating strings, the FTS table indexes `case_title`, `section`, and `text` as three separate columns. BM25 in FTS5 scores across all columns of the table, so a case title match counts for every chunk of that document — lexical enrichment, achieved structurally instead of by string surgery. (Separate columns also leave the door open to per-column BM25 weights later, without reshaping the index.)

### 3.4 The tokenizer: `unicode61`, and deliberately no stemming

`tokenize='unicode61'` is FTS5's default: split on anything that isn't alphanumeric (per Unicode 6.1), fold case. For "AIR 1962 SC 406" that yields the tokens `air`, `1962`, `sc`, `406` — punctuation and spacing variants of citations (`(1992)4SCC736` vs `(1992) 4 SCC 736`) tokenize identically, which quietly solved a real inconsistency in the PDFs.

What we did *not* enable: the `porter` stemming tokenizer, which folds "detained/detention/detaining" into one token. Stemming raises recall on prose but *mangles identifiers* — and identifiers are the entire reason this index exists. For a legal corpus, exact-token conservatism is the defensible default; the eval can overturn it later if prose recall ever needs the help.

### 3.5 The rebuild lifecycle: who keeps the index fresh

External content means someone must sync the index. Two idiomatic options:

- **Triggers** on `chunk_vectors` (AFTER INSERT/UPDATE/DELETE) that feed the FTS table row by row. Always-fresh, but subtle: FTS5's update protocol via triggers is fiddly (special `delete` commands), and a bug corrupts the index silently.
- **Full rebuild:** `INSERT INTO chunk_fts(chunk_fts) VALUES('rebuild')` — FTS5 re-derives the entire index by scanning the content table.

At 1,412 chunks a rebuild is milliseconds. Simplicity wins: **the indexer rebuilds once per run** (after the doc loop; also after each background-task indexing), and the read path *self-heals* if it detects staleness (Part 5 explains how detection itself turned out to be the hard part). Triggers become worth their complexity at a corpus size this project doesn't have — the same argument that keeps brute-force KNN.

---

## Part 4 — Querying it without getting hurt

### 4.1 Beginner: MATCH is a language, and user questions are hostile input

`MATCH` doesn't take plain text — it takes a *query mini-language*: `"quoted phrases"`, `AND`, `OR`, `NOT`, `NEAR`, column filters, `*` prefixes. Feed it a raw user question — *"Which case discusses (2003) 11 SCC 590 and in what context?"* — and the parentheses alone are a syntax error; an uppercase "AND" would silently become an operator.

So the question is sanitized into the *minimal safe subset* of the language:

```python
tokens = re.findall(r"[A-Za-z0-9]+", query)
match = " OR ".join(f'"{t}"' for t in tokens)  # quoted: never operators
```

Every alphanumeric run becomes a token; each token is wrapped in quotes — a one-word "phrase", which FTS5 can never mistake for an operator — and the phrases are joined with `OR`.

### 4.2 Intermediate: why OR, when FTS5's default is AND

Multiple terms in an FTS5 query are implicitly ANDed: *every* term must appear in a matching row. For a ten-word prose question, demanding all ten words in one 700-token chunk is absurdly strict — most relevant chunks would be filtered out before ranking even begins.

`OR` flips the semantics to "any term may match", which is what BM25 expects: it's a *ranking* function, designed to score partial matches, with IDF ensuring that matching the rare terms ("406") counts enormously more than matching the common ones ("what", "case"). Filter loosely, rank sharply. (The trade: OR over prose queries lets common-word matches into the candidate list — the residue that resurfaces in Part 7's misses.)

### 4.3 The score that comes back, and the sign flip

FTS5 exposes `bm25(chunk_fts)` where **lower = better** (it's an internal rank cost). The dense convention everywhere else is higher = better, so `lexical_search` negates it before returning. The row shape matches `search()` exactly — same contract — with one loud caveat in the docstring: the dense and lexical `score` fields live on *unrelated scales*, which is precisely why fusion happens by rank and never by score.

The final query joins the index back to the payloads:

```sql
SELECT cv.doc_id, cv.chunk_id, cv.section, cv.case_title, cv.text,
       bm25(chunk_fts) AS rank
FROM chunk_fts JOIN chunk_vectors cv ON cv.id = chunk_fts.rowid
WHERE chunk_fts MATCH ? ORDER BY rank LIMIT ?
```

---

## Part 5 — The two bugs, witnessed and kept

Both bugs are more valuable than the code they broke. Neither made it to a commit; both changed the design.

### 5.1 The index that looked full and was empty

First implementation of staleness detection: compare `SELECT COUNT(*) FROM chunk_fts` against `count_chunks()`; rebuild on mismatch. Reasonable — and **tautological**. On an external-content table, `COUNT(*)` (like any SELECT of its columns) *reads through to the content table*. The counts are equal by construction, forever, whether or not the inverted index was ever built.

Witnessed as: FTS reports 1,412 rows, sample rows read back fine, and every `MATCH` — even for `detention` — returns zero. The table is a convincing façade: the *view* of the data works, the *index over it* was never constructed.

The mental model that prevents this class of bug: an external-content FTS table is **two things wearing one name** — a pass-through view (always "works") and an inverted index (works only after someone builds it). No query against the view can tell you about the index; emptiness and no-matches are indistinguishable from the outside.

Fix: make build-state explicit. `rebuild_lexical_index()` stamps the chunk count it indexed into the `index_meta` table (`fts_chunks`); `lexical_search` rebuilds when the stamp is missing (never built) or differs from the live count (chunks changed since). Deterministic, introspectable, no reliance on FTS internals.

### 5.2 The stamp that impersonated corruption

The stamp created a second bug immediately: `_verify_meta()` — the guard that detects a DB built by a different profile — compared the *entire* `index_meta` table against its four expected identity keys with `stored != expected`. A fifth key (`fts_chunks`) makes the dicts unequal, and the store would have raised `IndexConfigMismatch` on every open: operational state masquerading as identity corruption.

Fix: the comparison filters to identity keys first. The deeper lesson: `index_meta` now holds **two kinds of state** — *identity* (who built this index: model, dim, fingerprint; mismatch = fatal) and *operational* (what maintenance has run: rebuild stamps; mismatch = do maintenance). The verify path must only ever see the first kind. Any future stamp gets this for free.

---

## Part 6 — Fusion: twenty lines carrying the whole concept doc

### 6.1 The code

```python
def _rrf(rankings: list[list[dict]], rrf_k: int) -> list[dict]:
    fused: dict[tuple, float] = {}
    rows: dict[tuple, dict] = {}
    for ranking in rankings:
        for rank, row in enumerate(ranking, start=1):
            key = (row["doc_id"], row["chunk_id"])
            fused[key] = fused.get(key, 0.0) + 1.0 / (rrf_k + rank)
            rows.setdefault(key, row)
    order = sorted(fused, key=fused.__getitem__, reverse=True)
    return [rows[key] | {"score": fused[key]} for key in order]
```

### 6.2 Line-by-line choices

- **`key = (doc_id, chunk_id)`** — fusion identity is the *chunk*, not the document. Both rankers return chunks, so consensus means "both rankers surfaced this exact chunk". (Fusing at document granularity would be a different, blurrier design — and would break the contract, which returns chunks.)
- **`enumerate(..., start=1)`** — ranks are 1-based; rank 0 would inflate the top result's reward and silently shift the whole curve.
- **`fused.get(key, 0.0) + 1/(rrf_k + rank)`** — the formula itself. A chunk absent from a list simply contributes nothing from it; no missing-value handling needed.
- **`rows.setdefault(key, row)`** — when both rankers return the same chunk, keep the first-seen dict. The payloads are identical (same store row); only `score` differed, and that gets overwritten with the fused value anyway. Deliberately *not* preserving the per-ranker scores: the contract has one score field, and exposing two would invite downstream code to compare incomparables.
- **`[:k]` at the call site** — each ranker contributed `candidates` deep; fusion sees the wide lists; the caller's `k` cuts last. Retrieve wide, return narrow.

### 6.3 Advanced: the contract field that changed meaning

Under dense, `score` is a cosine similarity in [0,1] — "0.82" carries absolute meaning. Under hybrid, `score` is an RRF sum (~1/rrf_k scale, e.g. 0.09–0.16 at k=10): *comparable within one result list, meaningless across lists or strategies*.

Was any downstream consumer treating score as a similarity? Audited: `/search` rounds it for display (fine — relative order is what a human reads), `assemble()` selects by rank and budget, never by score value (fine), evals compute rank-based metrics (fine). Nothing broke — but that's an audit result, not a guarantee, so the module docstring now states it as a rule: **nothing downstream may treat `score` as a similarity.** The eval-honesty framing: the contract's *shape* was preserved; its *semantics* were versioned by documentation.

The alternative — adding a `dense_score`/`lexical_score`/`fused_score` triple — was rejected as invitation-to-misuse: the only legitimate downstream use of retrieval order is *as an order*, until 2d's generation evals give a reason to know more.

---

## Part 7 — The eval story: where canon failed and why

### 7.1 The scoreboard

*(Metric definitions, slices, and how to read a row live in the [measurement primer](/docs/eval-measurement.md) — this section assumes them.)*

| run | knobs | citation doc@5 | citation pass@5 | aggregate doc@5 | MRR |
|---|---|---|---|---|---|
| dense-baseline-v2 | — | 0.000 | 0.000 | 0.774 | 0.751 |
| hybrid-v1 | c50 / k60 (canonical) | 0.333 | 0.250 | 0.871 | 0.796 |
| hybrid-c20 | c20 / k60 | 0.750 | 0.750 | 0.952 | 0.831 |
| hybrid-k10 | c50 / k10 | 0.833 | 0.750 | 0.968 | 0.820 |
| **hybrid-c20-k10** | **c20 / k10** | **0.833** | **0.833** | **0.968** | **0.835** |

Two predictions from the concept doc, checked: citation queries jumped (0 → 0.833 — predicted). Conceptual queries were supposed to stay *flat*; they actually **improved** (facts 0.923 → 1.0, holding 0.95 → 1.0) — the lexical ranker contributes on ordinary questions too, whenever a question happens to reuse the judgment's own rare vocabulary. A prediction pleasantly overturned is still an overturned prediction; recorded as such.

### 7.2 False consensus: the failure canon didn't warn about

Canonical RRF (c50/k60) managed only 0.333 — and the mechanism was witnessed *before* the eval quantified it, in a single smoke test. Query "AIR 1962 SC 406": lexical ranked the gold chunk **#1**. The fused list's top-5 didn't contain it.

The arithmetic of the betrayal, with k=60:

- Gold chunk: lexical #1, absent from dense's 50 → fused = 1/61 ≈ **0.0164**
- Junk chunk: lexical #3 (it contains "AIR" and "SC" — every judgment does) *and* dense #40 (generic legalness) → fused = 1/63 + 1/100 ≈ **0.0259**

The junk wins. Three ingredients, each necessary:

1. **One ranker is pure noise for this query class** — dense's list for a citation query contains zero signal (the witnessed step-5 blindness). RRF's implicit assumption — every ranker carries *some* signal — is violated outright.
2. **Deep candidate lists (50)** hand that noise ranker fifty lottery tickets to coincidentally overlap with the lexical mid-list.
3. **A flat curve (k=60)** prices rank #1 at barely 1.8× rank #50 — so *any* two-list sum beats *any* single-list appearance.

The two knobs attack ingredients 2 and 3 directly: `candidates: 20` issues fewer noise tickets; `rrf_k: 10` re-prices rank #1 at ~5× rank #50, so a confident single-list #1 can survive coincidental consensus. Each knob alone recovered most of the loss (0.75 / 0.833); together, 0.833 with the best passage recall.

The honest caveat, recorded in the journal: this sweep tuned two knobs against **twelve questions**. That's hypothesis-sized evidence, not proof-sized. The settings are defensible because the *mechanism* was witnessed and the fix targets the mechanism — but the numbers deserve re-checking when the eval set grows.

### 7.3 The residue: two misses that define the next step

- **q004 — "What does the judgment say about the decision reported in 1952 SCR 135?"** The citation is in the doc, verbatim. But the *prose wrapper* dilutes the OR query: common words match everywhere, and under c20 the gold chunk falls below the candidate cutoff before fusion ever sees it. Cure: query rewriting (strip the wrapper, search the identifier) and/or wider recall followed by a precision stage.
- **q017 — "(1995) 2 SCC 7"** — reached rank 8, not top-5. Tokens `1995`, `2`, `7` are hyper-common; only bag-of-words BM25 is running, and it cannot demand the tokens appear *adjacent*. Cure: phrase queries for detected citation patterns (FTS5 supports adjacency; the sanitizer currently never emits multi-word phrases).

Both cures are precision-stage work — exactly 2c ("retrieve wide, rank narrow"). The residue isn't failure; it's the witnessed symptom the next step will be built against, same as the step-5 miss was for this one.

---

## Part 8 — The interface question: obligation or capability?

`VectorIndex` is an ABC; every store must implement `search`, `upsert_chunk`, etc. Should `lexical_search` join that list of `@abstractmethod`s?

Chosen: **no** — it's an optional capability with concrete defaults on the base class:

```python
def lexical_search(self, query: str, k: int = 5) -> list[dict]:
    raise NotImplementedError(
        f"{type(self).__name__} has no lexical index; hybrid retrieval "
        "needs a store that implements lexical_search"
    )

def rebuild_lexical_index(self) -> None:  # no-op where unsupported
    pass
```

The reasoning: an abstract method is a *tax on every future backend*. A Qdrant store written next month for dense-only comparison shouldn't be forced to fake a BM25 index just to instantiate. But the failure must stay *loud*: configuring `strategy: hybrid` against a store without lexical support raises immediately, with a message naming the store class and the missing capability — not a silent empty list that would masquerade as "no results". The rebuild hook, by contrast, defaults to a no-op: "keep your lexical index fresh" is meaningless where there is none, and the indexer shouldn't need to know which stores have one.

(Asymmetric defaults — one raises, one passes — because their *misuse costs* are asymmetric: a silent empty search corrupts results; a skipped no-op rebuild costs nothing.)

---

## Part 9 — Self-test: do you own this change?

Every answer is in this document or the concept doc. Ordered roughly easy → hard.

1. Which function changed behavior, and why did `/ask` inherit hybrid without a single edit?
2. Why does flipping `strategy: dense` → `hybrid` not require re-indexing? What one property of the FTS index makes that argument valid?
3. What does `content='chunk_vectors'` buy, and what responsibility does it impose?
4. Why are `case_title` and `section` separate FTS columns instead of concatenated into the text?
5. Why no stemming tokenizer, when stemming raises prose recall?
6. Why does `lexical_search` join tokens with `OR` when FTS5's default is `AND`?
7. Why is every token quoted in the MATCH string?
8. FTS5's `bm25()` returns lower-is-better. Where does the sign flip, and why bother?
9. Why did `COUNT(*) FROM chunk_fts` equal `count_chunks()` even when MATCH returned nothing for everything?
10. What are the two kinds of state in `index_meta`, and which one must `_verify_meta` never see?
11. In `_rrf`, why is the fusion key `(doc_id, chunk_id)` and not `doc_id`?
12. Why does `_rrf` throw away the per-ranker scores instead of returning all three?
13. What does `score` mean under hybrid, and what may downstream code never do with it?
14. State the three ingredients of false consensus. Which knob attacks which?
15. Compute it: with k=60, why does {lexical #3 + dense #40} beat {lexical #1 alone}? Redo it with k=10.
16. The concept doc predicted conceptual queries would stay flat under hybrid. What actually happened, and why?
17. Why is `rrf_k: 10` defensible despite deviating from the literature default — and what makes it *provisionally* defensible rather than settled?
18. Explain both residual misses (q004, q017) and why their cures belong to 2c, not to more knob-turning here.
19. Why is `lexical_search` a default-raising method instead of an `@abstractmethod` — and why does `rebuild_lexical_index` default to `pass` instead of raising?
20. A new teammate proposes min-max normalizing both score lists and averaging, "so we keep the confidence information RRF throws away." Give the two-part rebuttal, using this corpus's witnessed score band as evidence.

If question 15 takes you longer than a minute, reread Part 7.2 — it's the heart of the whole step.
