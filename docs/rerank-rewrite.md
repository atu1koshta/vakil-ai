# Rerank + Query Rewrite: Fixing the Question, Then Reading the Answers Twice

*A knowledge base for Step 2c of Vakil AI — from two witnessed retrieval misses to query ensembles and cross-encoder reranking, built up slowly enough that every design choice feels inevitable by the time it appears.*

---

## Part 0 — The two failures that motivate all of this

Step 2b (hybrid BM25 + RRF) lifted citation recall@5 from 0.0 to 0.833. That number contains its own homework: 0.833 means misses remain, and we can name two of them exactly (witnessed in the 2b eval, logged in the implementation field guide):

- **q004** — *"What does the judgment say about the decision reported in 1952 SCR 135?"* The citation is right there. But the question is a full English sentence, and every word of it becomes a BM25 query term. The gold chunk fell below the candidate cutoff before fusion ever saw it.
- **q017** — the bare citation *"(1995) 2 SCC 7"*. No noise at all, pure signal — and it still missed (gold chunk at rank 8, not top-5). The tokens `1995`, `2`, `7` are individually everywhere in a corpus of legal judgments; what makes the citation distinctive is the tokens *in that order*, and our lexical query has no way to say "in that order."

Two misses, two *different* diseases. One query carries too much noise; the other carries signal our query language can't express. Neither is fixed by tuning what we have — they need new mechanisms. And beyond the named misses, the frozen baseline shows structural headroom: the right chunk is usually *somewhere* in the top-5 (passage_recall@5 = .839) but rarely *first* (passage_recall@1 = .226), and paraphrased questions — worded nothing like the judgment — find their passage only 28.6% of the time. Finding things and ranking them well are turning out to be different skills.

The route:

| Part | Question it answers |
|------|--------------------|
| 1 | The query itself is an engineering surface. What can we do to a question before searching with it — mechanically, and with an LLM — without ever making things worse? |
| 2 | Why is our ranking mediocre at the very top, by construction? (This is where bi-encoders vs cross-encoders gets built.) |
| 3 | How the pieces assemble into one pipeline, and why the 2b knob settings were never "the right values" — just compensation. |
| 4 | What "done" looks like: attributing each metric movement to exactly one mechanism. |

---

## Part 1 — The query is not sacred

### 1.1 Beginner: the question as its own worst enemy

Until now we've treated the user's question as a fixed input: embed it, tokenize it, search. But look at what q004's question actually *is*, from BM25's point of view. The sanitizer turns it into an OR query:

```
what OR does OR the OR judgment OR say OR about OR the OR decision
OR reported OR in OR 1952 OR SCR OR 135
```

BM25 scores a chunk by summing a contribution per matching term. "Judgment", "decision", "reported", "say" — in a corpus made *entirely of court judgments*, these words match nearly every chunk. Each match is worth little (IDF discounts common words), but there are many of them, and chunks dense in generic legal prose collect enough small contributions to outscore the one chunk containing `1952 SCR 135`. The signal is present; it is *diluted*. The gold chunk sat below rank 20 — below the `candidates` cutoff — so fusion never even saw it.

The human phrasing — "what does the judgment say about..." — is politeness aimed at a human reader. To the search engine it is ballast. Which suggests something almost embarrassingly direct: *change the query*. Strip it to the words that discriminate: `1952 SCR 135 decision`. Query rewriting is exactly this — treating the question as raw material, not gospel.

### 1.2 Deterministic rewriting: when the pattern is a grammar, use a grammar

Now the opposite failure. q017's query is already minimal — `(1995) 2 SCC 7`, nothing to strip. The problem is that BM25 is a *bag of words*: a chunk containing "1995" in one sentence, "2" in another, and "7" in a page number scores the same as a chunk containing the contiguous citation. The information "these tokens are adjacent, in this order" exists in the query and is thrown away by the query language we're using.

But the query language has a richer construct we haven't used: the **phrase query**. Every serious lexical engine (including SQLite's FTS5) accepts a quoted sequence — `"1995 2 SCC 7"` — meaning *these tokens, consecutive, in order*. That's precisely the constraint a citation needs. The inverted index already stores positions; phrases just finally ask for them.

How do we know when to emit a phrase query? Here's the lucky break: legal citations are not free text. Indian reporters follow a small set of rigid formats — `AIR <year> <court> <n>`, `(<year>) <n> SCC <n>`, `[<year>] <n> SCR <n>`, a few variants. A rigid format is a *grammar*, and the right tool for a grammar is a regex, not a neural network. A handful of patterns detects citations in the question deterministically: same input, same output, testable, zero latency, no model to be wrong.

Two properties make citation phrase queries unusually safe to add:

- **Inert or decisive, nothing in between.** A phrase query is high-precision/low-recall by nature. If no chunk contains the exact phrase, the list comes back empty and contributes nothing. If a chunk does contain it, that chunk is almost certainly the right one, and it arrives at rank 1 of its list. There is no middle mode where it adds plausible junk.
- **One sharp edge: tokenizer alignment.** A phrase matches only if its tokens equal what the index's tokenizer produced at index time. The index saw `(1995)` and stored the token `1995` — parentheses stripped by its tokenization rules. If our detector emits the phrase with parentheses intact, the phrase matches *nothing*, silently. So the detector must normalize matched spans through the *same* token-splitting rule the index uses. This is the same lesson 2b's Part 2 taught ("tokenization decides everything"), now biting from the query side.

### 1.3 Learned rewriting: when the pattern isn't a grammar, ask a model

Citations have a grammar; conversational dilution doesn't. "Strip this question to its retrieval-worthy keywords" requires judging which words carry meaning — that's a language task, and we have a language model sitting locally. Hand it the question with narrow instructions: *rewrite as a short keyword query; keep case names, citations, statutes, doctrines; one line.* For q004 it should produce something like `1952 SCR 135 decision` — the query BM25 wanted all along. As a bonus, the same treatment helps paraphrase questions: a keyword rewrite often lands closer to the corpus's own vocabulary than the user's conversational phrasing did.

But a learned component earns its place only after an honest accounting of its failure modes, because it has all the ones the regex doesn't:

- It can produce a *bad* rewrite — hallucinated terms, a dropped constraint, an answer to the question instead of a query.
- It is nondeterministic in the tails (temperature 0 reduces but does not eliminate run-to-run variation), which means eval numbers wobble slightly.
- It costs real latency — a model call per question, every question.
- It can *fail outright* — model not loaded, timeout, garbage output.

None of these is a reason not to use it. They are the requirements list for *how* to use it, which is the next section.

### 1.4 The principle that makes rewriting safe: never replace, only add

Here is the design move the whole step rests on. The rewritten query does **not** replace the original. Both run. The raw question still feeds dense and lexical retrieval exactly as before; the rewrite adds *additional* ranked lists to the fusion; detected citations add one more. RRF was built to merge ranked lists — 2b used two, and nothing about the formula caps it there. The query becomes a query *ensemble*:

```
dense(raw) + lexical(raw)                    ← always, unconditionally
+ dense(rewritten) + lexical(rewritten)      ← if the rewrite produced something
+ lexical_phrase(citations)                  ← if the detector found any
```

Trace the failure modes through this structure. The rewrite is bad? Its lists contain junk — but junk entering RRF at mediocre ranks, competing against the raw question's untouched lists, and (Part 2) a precision stage waits downstream. The rewrite fails entirely? The ensemble is simply one query smaller — which is exactly yesterday's behavior. No citations in the question? The phrase list never exists. The *worst possible outcome* of every new mechanism is graceful degradation to 2b. That guarantee comes from the structure, not from hoping the LLM behaves — you cannot get it by prompt engineering, only by architecture.

(The same reasoning says a failed rewrite should degrade silently to `None`, never raise: no retrieval request should ever break because an *optional enhancement* had a bad day.)

One caution to keep on record: 2b taught that deep candidate lists can manufacture **false consensus** — junk that appears at mediocre ranks in *several* lists out-fuses a chunk one list ranked highly. Going from two lists to five re-opens that door. Two counterweights: these lists are *correlated by intent* (the same question, restated — not independent noise sources), and Part 2's reranker no longer takes RRF's word as final. But that's an argument, not a fact; the re-sweep in Part 4 measures it.

---

## Part 2 — Two readers is still one kind of reading

### 2.1 Beginner: what the fast reader never sees

Recall how dense retrieval works (2b, Part 1): a model reads each chunk *once, at index time*, and compresses it into a 768-number location. Later, the question gets its own location, and relevance is proximity. Crucially, **the question and the chunk never meet**. Each was summarized in total ignorance of the other. This architecture is called a **bi-encoder** — two independent encodings, compared only afterward as geometry.

That independence is what makes an index possible at all: chunk vectors are computed once and reused for every future question. But it puts a hard ceiling on quality. The chunk's summary had to be written to serve *any question anyone might ever ask* — one fixed 768-number answer to every possible question. When your specific question arrives, the details it cares about may be exactly what the summary shed. BM25 has the same shape, differently dressed: chunk statistics precomputed, query matched against them, no joint reading. Fast retrieval is fast *because nobody ever reads the question and the chunk together.*

Now the .226 stops being mysterious. Getting the gold chunk into the top-5 requires only that its summary land in roughly the right neighborhood — summaries can do that. Getting it to *rank first*, above four near-neighbors, requires fine distinctions between chunks that are all genuinely about the topic — and the information for those distinctions often lives precisely in the details the summaries discarded. Recall is a coarse skill; precision at the top is a fine one. We built a system with only the coarse skill and are now graded on both.

### 2.2 The cross-encoder: pay for a joint reading

The fix is direct: build the reader we said was impossible at index time. A **cross-encoder** takes the question and a chunk *together*, as one concatenated input, through one transformer. Every question token attends to every chunk token — "1952" in the query lines up against "1952" in the text, "the decision reported in" aligns with the sentence discussing it — and out comes a single number: how relevant is *this* text to *this* question. No summary, no geometry, no information discarded in advance.

The catch is the same fact read in the other direction. There's nothing to precompute — the input *is* the pair, and the pair doesn't exist until the question arrives. Scoring means a full model forward pass **per (question, chunk) pair, at query time**. Over a whole corpus, absurd. Over a shortlist of fifty, entirely affordable — a small cross-encoder handles a 50-pair pool in a second or two on CPU.

So neither architecture wins; they price differently. Bi-encoder: cheap per query, capped quality. Cross-encoder: expensive per pair, superior judgment. Which sets up the obvious composition —

### 2.3 Two-stage retrieval: the economics that decouple recall from precision

> **In plain terms — a hiring analogy.** Today there is one interviewer, a mediocre judge. We let them see only 20 resumes (2b's `candidates: 20`), because with 100 resumes they pick charming-but-wrong people. Small pile = fewer mistakes — but the best candidate is sometimes not in the 20 and is never even seen (that was q004). One knob controls two things: small pile = safer picks but misses people; big pile = catches everyone but picks worse. Now hire a second interviewer — much smarter, but slow, able to interview only ~50 people deeply. The jobs split: the first stage just *collects* — pull in 50 plausible candidates; it doesn't need to rank well, only to get the right one *somewhere* in the pile. The smart interviewer reads each candidate against the job and picks the final few. Widening no longer costs precision — junk that sneaks into the wide pool gets scored low downstream. Fast-dumb over everything, slow-smart over the shortlist.

Formally: retrieval so far has had one knob doing two jobs. Candidate depth controlled both *what could be found* (recall) and *how clean the final ranking was* (precision) — 2b's sweep landed on narrow c20/k10 precisely because narrowness suppressed false consensus, and paid for it with q004. That trade was never a preference; it was **compensation for a missing stage**. The moment a competent judge sits after fusion, the trade dissolves:

1. **Recall stage** — the query ensemble (Part 1) feeding dense + lexical + phrases into RRF, with `candidates` re-widened. Its only job: make sure the gold chunk is *somewhere* in the fused pool. Mediocre ordering inside the pool is now fine.
2. **Precision stage** — the cross-encoder reads each of the pool's top ~50 chunks jointly with the *raw* question (never the rewrite — the user's own words are the ground truth of intent) and keeps the best k.

Each stage does the one thing it's structurally good at. This "retrieve wide, rank narrow" pattern is the standard architecture of serious search systems — and now every part of it has been earned rather than asserted.

A testable prediction falls out, and it's the heart of 2c: *the 2b knob values should stop being optimal once the reranker exists.* Wide candidates, previously punished by false consensus, should now win — the reranker absorbs the junk that widening lets in. If a re-sweep under the reranker doesn't move the optimum wider, the compensation theory was wrong. Either way we learn something.

### 2.4 Advanced: what the smart reader still can't see

- **The input-length ceiling.** Small cross-encoders read at most 512 tokens of (question + chunk) combined; our chunks run ~700 tokens. The tail of each chunk is *silently truncated* — evidence living past the cutoff is invisible to the reranker, and a chunk the pool ranked highly can be demoted because its relevant sentence sits at the end. The symptom to watch for in evals: passages that were recalled *before* reranking and lost *after*. (Known mitigation — score overlapping windows of the chunk and take the max — deliberately out of scope for v1.)
- **A third score scale.** The `score` field's meaning now depends on the path taken: cosine similarity (dense), an RRF sum (hybrid), or a cross-encoder logit (reranked) — raw, unbounded, possibly negative, and meaningful only *within one result list*. Comparing scores across queries or configurations is a category error. 2b's lesson ("BM25 scores are not probabilities") now applies to our own API response.
- **Model choice is a CPU question first.** Reranker quality scales with model size, but so does per-pair latency, linearly with pool size. A ~22M-parameter MiniLM cross-encoder scores a 50-pair pool in ~0.5–2s on CPU; heavier rerankers (bge-reranker-base and up) are 5–10× slower for the same 512-token ceiling. On CPU, the small model is the honest choice — and since the model name is configuration, upgrading later is an edit, not a redesign.
- **Latency is now a first-class metric.** An LLM rewrite plus 50 cross-encoder passes per question trades seconds for precision. An eval that reports quality but not latency hides half the transaction — so per-question timing joins the eval output, and the price gets reported next to the gains.

---

## Part 3 — The assembled pipeline, and its edges

Each mechanism now maps onto the failure that demanded it:

```
question
  ├─ citation detector (regex)          ─ the q017 cure: adjacency, expressed at last
  ├─ LLM keyword rewrite                ─ the q004 cure: dilution, stripped
  ├─ RRF over the ensemble's lists      ─ recall stage, safely re-widened
  └─ cross-encoder over the fused pool  ─ precision stage: the joint reading
        └─ top-k out, same contract as always
```

Everything lands inside the single retrieval function — the seam established in Step 2 and already exploited once in 2b. Callers (`/search`, evals, `rag.ask`) never learn anything changed. The seam pattern pays rent a second time.

And the boundary of the technique, stated as plainly as its powers:

- **The reranker can only rerank what was recalled.** If the gold chunk isn't in the fused pool, no precision stage recovers it. Recall failures remain recall failures — that's why the ensemble work of Part 1 isn't made redundant by the reranker.
- **Chunk-boundary damage stays invisible.** The dissent-as-holding failure (Step 5) — a dissenting opinion chunked without its opinion label — will sail through rewriting, fusion, *and* joint reading, because the missing information was destroyed before any index existed. Pipeline disease needs pipeline medicine.
- **Rewriting can't rescue a corpus gap.** If the answer isn't in the corpus, a better query just misses more precisely — and nearest-neighbor search still returns k confident-looking neighbors (the "no relevance floor" problem from 2b, unchanged).

## Part 4 — What "done" looks like

Same discipline as 2b: numbers before opinions, and this time with *attribution*. Three mechanisms land in one step, so a single before/after comparison would tell us "something helped" without saying what. Instead, the eval matrix turns mechanisms on one at a time — baseline → +citation phrases → +LLM rewrite → +reranker → +re-widened knobs — so each metric movement between adjacent runs belongs to exactly one mechanism.

Predictions on record before running:

- Citation phrases: q017 recovered; citation-slice recall up; *everything else flat* — if a supposedly inert list moves unrelated metrics, it wasn't inert, and that's a finding.
- LLM rewrite: q004 recovered; paraphrase slice drifts up; any regression bounded (the ensemble guarantees it structurally).
- Reranker: passage_recall@1 up — the headline, because precision-at-the-top is exactly what a joint reading buys; MRR up; recall@5 holds.
- Re-sweep: wider candidates beat c20/k10 *only now* — vindicating (or falsifying) the compensation theory of §2.3.

With the standing caveat: the eval slices are small (12 citation questions, 7 paraphrase), so one question flipping moves a slice ~8 points. This is hypothesis-sized evidence, and gets reported as such.

### The one-paragraph version

Two witnessed misses, two diseases: conversational phrasing dilutes the lexical query (q004), and bag-of-words queries can't demand token adjacency (q017). Cures: strip the question to keywords with an LLM, and detect citations with a regex to emit exact-phrase queries — both *added* to the query ensemble, never replacing the raw question, so the worst case is always yesterday's behavior. Underneath, a structural upgrade: bi-encoders summarize question and chunk separately and never read them together, capping precision at the top; a cross-encoder reads each (question, chunk) pair jointly — too slow for a corpus, cheap for a shortlist. Retrieve wide with the ensemble, rank narrow with the cross-encoder, and recall and precision stop being one knob. Verify with an ablation matrix that attributes every moved metric to exactly one mechanism, with latency reported next to the gains — and remember the edges: a reranker can't recover what recall missed, and no retrieval stage heals damage done at chunking time.

---

## Appendix — Vocabulary earned along the way

| Term | The idea it names |
|------|-------------------|
| Query dilution | Conversational words becoming query terms that drown the discriminating ones |
| Phrase query | A quoted token sequence requiring adjacency and order — positions the inverted index already had |
| Query rewriting | Treating the question as raw material: deterministic (regex → phrases) or learned (LLM → keywords) |
| Query ensemble | Multiple restatements of one question, each contributing ranked lists to fusion — add, never replace |
| Graceful degradation | Every new mechanism's worst case is the old behavior, guaranteed by structure not by hope |
| Bi-encoder | Question and chunk encoded independently, compared only as geometry — indexable, quality-capped |
| Cross-encoder | Question and chunk read jointly through one model — superior judgment, nothing precomputable |
| Two-stage retrieval | Retrieve wide (cheap recall stage), rank narrow (expensive precision stage) |
| Recall/precision decoupling | With a downstream judge, candidate depth stops trading one for the other |
| False consensus | Junk at mediocre ranks in several lists out-fusing a single list's top pick (2b lesson, re-opened by more lists) |
| Truncation blindness | The cross-encoder's 512-token ceiling hiding evidence in chunk tails |
| Score scale | Cosine, RRF sum, or logit — comparable only within one result list |
| Ablation matrix | Turning mechanisms on one at a time so every metric movement has exactly one owner |
