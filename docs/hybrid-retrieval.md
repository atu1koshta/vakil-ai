# Hybrid Retrieval: Why One Search Engine Isn't Enough

*A knowledge base for Step 2b of Vakil AI — from "what is search, really?" to BM25 + RRF, built up slowly enough that every formula feels inevitable by the time it appears.*

---

## Part 0 — The failure that motivates all of this

In the break-it lab (Step 5) we ran a probe: ask the system for a judgment by its citation, **"AIR 1962 SC 406"**. The document is in the corpus. Its chunks contain that exact string. Retrieval returned twenty chunks — and not one of them came from that document.

Stranger still: the *scores* told a story. The top twenty chunks scored between 0.547 and 0.564 — a band just 0.017 wide. The system wasn't saying "here's a great match and some weaker ones." It was saying "everything looks equally, mediocrely relevant," which is what indifference looks like when your search engine is forced to rank something. Meanwhile, the *same document* was retrieved perfectly (5 of 5 relevant chunks, scores 0.75–0.82) when we asked a conceptual question about its subject matter.

So our retrieval isn't broken. It is *good at one kind of question and blind to another*. This document is about understanding exactly why that happens, what kind of search is good at the question we failed, and how to run both kinds at once without making a mess.

The route:

| Part | Question it answers |
|------|--------------------|
| 1 | How does our current search "read" text, and why does that reading smear "AIR 1962 SC 406" into fog? |
| 2 | What is the older way of reading — matching words themselves — and how do you make it rank well? (This is where BM25 gets built, piece by piece.) |
| 3 | Why keep both readers instead of picking a winner? |
| 4 | Two readers produce two ranked lists. How do you merge them honestly? (This is where RRF gets built.) |
| 5 | What it looks like inside Vakil AI, and where the edges of the technique are. |

---

## Part 1 — The reader we already have: search by meaning

### 1.1 Beginner: text as a location on a map

Forget vectors for a moment. Imagine a vast library where books are shelved not alphabetically but *by what they're about* — so two books on preventive detention sit next to each other even if one never uses the word "detention" and calls it "custody without trial" instead. To find something, you don't scan titles; you walk to the neighborhood where that topic lives and look around.

That is what our current retrieval does. A model (nomic-embed-text, in our case) reads a chunk of text and assigns it a *location* — a point described by 768 numbers. Texts that mean similar things get nearby locations. A question gets a location too, and "search" becomes: *which chunks live closest to where the question landed?* Closeness is measured with a similarity score between 0 and 1 — that's the number we saw compressed into the 0.547–0.564 band.

This is usually called **dense retrieval** or **semantic search**, but the mechanics matter more than the name: *search = geometry*. No words are ever compared. Only locations.

The superpower is obvious: it survives paraphrase. "Can the state take my land?" finds chunks about "compulsory acquisition of property" even with zero shared vocabulary. For conceptual legal questions — most of what users ask — this is exactly right.

### 1.2 Intermediate: what gets lost on the way to the map

The location a chunk gets is a *summary*. 768 numbers must capture an entire paragraph — its topic, tone, legal domain, era, everything. Summaries keep what's common and shed what's peculiar. And a citation string like "AIR 1962 SC 406" is *pure peculiarity*: it carries no theme, no sentiment, no topic. It is an identifier — its whole job is to be an arbitrary, unique label.

Three things conspire against it:

1. **The model never learned what it means.** The embedding model was trained by showing it pairs of texts and teaching it to place similar-*meaning* pairs close together. "AIR 1962 SC 406" versus "AIR 1962 SC 407" are, to that training, near-identical strings of legal boilerplate — even though to a lawyer they name completely different cases. The training signal rewards capturing *aboutness*, and identifiers have no aboutness.
2. **The string gets shredded before the model even sees it.** Models read text in sub-word pieces, so the citation arrives as something like `AIR`, `19`, `62`, `SC`, `4`, `06` — fragments that individually appear in thousands of unrelated contexts. The digits of a year carry almost no stable meaning on their own.
3. **The summary has no room for it.** Even if some trace survives, it's one faint feature among hundreds describing "formal mid-century Indian constitutional prose" — which describes most of our corpus. Hence the score band: every chunk is about equally "generically legal," so every chunk lands about equally far from the query. A 0.017-wide band across twenty results is the geometric signature of a query that mapped to *nowhere in particular*.

The lesson to internalize: **dense retrieval answers "what is this text about?", and some queries are not about anything — they name something.** Aboutness-search cannot find names. Not because the implementation is bad, but by construction.

### 1.3 Advanced: recognizing the failure from the outside

You will rarely get to inspect embeddings directly, so learn the observable symptoms:

- **Score-band compression.** Healthy retrieval separates: a few high scores, a falling tail. A flat, narrow band (ours: 0.017 across top-20) means the query vector isn't near *anything* — the ranking is noise wearing a ranking's clothes.
- **No relevance floor.** Nearest-neighbor search *always* returns k results. There is no built-in "nothing matched." (The break-it lab hit this too: a query about a case absent from the corpus still returned twenty semantic neighbors.) So a retrieval miss never announces itself — it hands you plausible-looking wrong chunks, and the generation layer downstream will faithfully summarize the wrong evidence.
- **Asymmetry across query types.** Same document: 5/5 on a conceptual query, 0/20 on its own citation. When you see per-query-type variance like that, suspect a representational blind spot, not a tuning problem. No amount of k-tweaking or prompt-prefix fiddling fixes "the information was never in the vector."

One more subtlety worth knowing exists: embedding spaces tend to be *anisotropic* — vectors bunch into a narrow cone rather than spreading over the whole space, which further compresses score differences and makes "everything scores ~0.5-ish" a common resting state for out-of-distribution queries. You don't need the math; you need the reflex: *flat scores = the query fell off the map.*

---

## Part 2 — The older reader: search by the words themselves

### 2.1 Beginner: the index at the back of the book

Long before anyone embedded anything, search worked the way a book's back-index works: for every word, keep a list of every place it appears. To search, look up your query's words and see which documents they point to. This family is called **lexical search** (also "keyword" or "sparse" retrieval — same idea), and its defining property is the mirror image of Part 1: *it compares the words themselves, never their meaning.*

For "AIR 1962 SC 406" this is trivially perfect. The string appears in exactly one document; the index points straight at it. What was invisible to geometry is a direct lookup for word-matching. The two readers fail in *opposite* places — hold that thought, it becomes Part 3.

But raw lookup only answers *which* documents match. Real queries match many documents, so the real problem is *ranking*: which match deserves to be first? Everything in the rest of Part 2 is the answer to that one question, built up one repair at a time. The finished product has a name — BM25 — but the name is the least important part.

### 2.2 First attempt: count the matches

Naive idea: score a document by how many times the query's words appear in it. More occurrences of "detention" = more about detention. This intuition (call it *term frequency*) is genuinely load-bearing — a judgment that says "detention" forty times probably is more about detention than one that says it once.

But it breaks immediately, in three ways. Each break motivates one repair, and the three repairs *are* BM25.

### 2.3 Repair #1: not all words deserve equal credit

Query: "grounds for preventive detention". A document mentioning "for" a hundred times should earn nothing; one mentioning "preventive" three times should earn a lot. Why? Because "for" appears in *every* document — matching it carries zero information — while "preventive" appears in few, so matching it is strong evidence of relevance.

So: weight each word by its *rarity across the whole collection*. A word found in nearly all documents gets weight near zero; a word found in a handful gets a large weight. This rarity weight is called **IDF** (inverse document frequency), and one standard form is:

```
IDF(word) = ln( (N − n + 0.5) / (n + 0.5) + 1 )
```

where `N` is how many documents exist and `n` is how many contain the word. Don't memorize it — read it: *as n shrinks, the weight grows.* The 0.5s just keep the math sane at the extremes (a word in zero documents, or in all of them).

Notice what this does for our failing query: "1962" is moderately rare, "406" is rare, and the exact phrase's co-occurrence is rarer still. Lexical ranking *automatically pours its credit onto identifiers* — precisely the tokens the embedding summary threw away. The two readers aren't just different; they are near-perfect complements, by construction.

### 2.4 Repair #2: the fortieth occurrence proves less than the second

Back to counting. Is a document with 80 occurrences of "detention" twice as relevant as one with 40? Plainly not — both are simply *about detention*, and after a point extra repetitions add no new evidence. Raw counts also let one obsessively repetitive document bully the ranking.

The repair: let each occurrence add credit, but with *diminishing returns* — a curve that rises steeply for the first few occurrences and then flattens toward a ceiling. The standard curve is:

```
credit(f) = f · (k1 + 1) / (f + k1)
```

where `f` is the occurrence count and `k1` is a dial (typically ~1.2–2.0) controlling how fast the curve saturates. At `f = 1` you get full marks for showing up; by `f = 10` you're scraping the ceiling. Set `k1 = 0` and counts stop mattering entirely (pure "does it appear?"); set it huge and you're back to raw counting. The dial exists because different corpora reward repetition differently.

### 2.5 Repair #3: long documents match everything

A 50-page judgment mentions almost every legal term at least once, purely by being long. Without correction, length itself becomes relevance, and your ranking degenerates into "return the longest documents." The repair: measure each document's length against the collection's *average* length, and shrink the credit of longer-than-average documents:

```
length_penalty = 1 − b + b · (doc_length / average_length)
```

with `b` (typically 0.75) dialing how aggressively to punish length — `b = 0` ignores length entirely, `b = 1` normalizes fully. This penalty is folded into the saturation curve's denominator, so long documents saturate *faster*.

(For us this matters less than usual — we rank *chunks*, and our chunker already targets a token range, so lengths are semi-uniform. Worth knowing the dial exists; worth not agonizing over it.)

### 2.6 The assembled machine

Put the three repairs together — rarity weighting × saturated counting × length correction, summed over the query's words:

```
score(D, Q) = Σ over words w in Q of:
    IDF(w) · ( f(w,D) · (k1 + 1) ) / ( f(w,D) + k1 · (1 − b + b · |D|/avgdl) )
```

This is **BM25** ("Best Match 25" — literally the 25th ranking function tried in a 1990s research system; the name is a lab notebook label, nothing more). It has been the backbone of serious search engines for thirty years, it is unreasonably hard to beat, and now you know *why* it looks the way it does: every piece is one of the three repairs you just watched being motivated.

What it still cannot do, and never will: connect "custody without trial" to "preventive detention". No shared words, no score. Meaning-blindness is the price of word-fidelity — the exact inverse of Part 1's trade.

### 2.7 Advanced: the details that bite in practice

- **Tokenization decides everything.** BM25 matches *tokens*, and what counts as a token is a policy choice made at index time. Does "AIR 1962 SC 406" become four tokens? Does "406." keep its period? Case-folding, punctuation stripping, and number handling silently determine whether your citation query can match at all. When lexical search underperforms, look at the tokenizer before the formula.
- **Phrase vs. bag.** Plain BM25 treats the query as a bag of independent words — a document containing "1962", "SC", and "406" scattered across unrelated sentences scores like one containing the contiguous citation. Phrase queries (requiring adjacency) restore that distinction; most engines, including SQLite's FTS5, support them.
- **Stemming is a trade, not a free win.** Collapsing "detained/detention/detaining" into one token raises recall on prose but can mangle identifiers and citations. For a legal corpus where exact strings matter, a conservative tokenizer with no stemming is a defensible default.
- **BM25 scores are not probabilities.** They're unbounded, corpus-dependent numbers — a "12.3" means nothing outside the index that produced it. File this away; it becomes the central problem of Part 4.

---

## Part 3 — Why keep both readers

### 3.1 Beginner: the query spectrum

Line up the questions users actually ask a legal corpus:

- *"What did the court say about equal pay for contractual workers?"* — pure aboutness. Dense excels; lexical limps (users rarely guess the judgment's exact vocabulary).
- *"AIR 1962 SC 406"* — pure name. Lexical is a lookup; dense is fog (witnessed, 0/20).
- *"the Maneka Gandhi standard for procedure under Article 21"* — a hybrid: a name ("Maneka Gandhi", "Article 21") wrapped in a concept ("standard for procedure"). Each reader sees its half.

Real query streams are mixtures like this, and the mixture is why "pick the better engine" is a category error. There is no better engine; there are two engines with disjoint blind spots. Choosing one means choosing which class of user question you're willing to systematically fail.

### 3.2 Intermediate: complementarity is measurable, not vibes

The honest way to hold this belief is as a *prediction about evals*, split by query type:

- On citation/identifier queries: hybrid should massively beat dense-only (dense contributes ~nothing there — witnessed).
- On conceptual queries: hybrid should roughly *match* dense-only (lexical contributes little, and the fusion shouldn't drag the good ranking down).
- Aggregate metrics will therefore *understate* the improvement — a big lift on 20% of queries averages into a modest overall bump. **Always split eval results by query type.** The split is the lesson; the aggregate hides it.

This is exactly why our 2b plan pins a frozen dense-only baseline (`step4-baseline.json`) before touching anything: the claim "hybrid helps" must cash out as *these specific queries moved, those didn't, as predicted*.

### 3.3 Advanced: when you'd legitimately skip hybrid

Discipline check — hybrid is not free (a second index to build, keep in sync, and reason about), so know the cases where it earns nothing:

- **Query stream has no identifiers.** A pure FAQ/chatbot corpus where nobody searches by code or citation: dense-only is simpler and sufficient.
- **Identifiers are structured metadata.** If citations were extracted at pipeline time into a queryable field, an *exact-match filter* beats fuzzy lexical ranking for them. (For us, metadata extraction does capture citations, but users type free-text questions — routing "detect citation in query → filter" is a heavier, more brittle design than fusing two rankers. Worth revisiting if citation queries dominate.)
- **A cross-encoder reranker already sits behind a wide recall stage** — it can partially compensate for either reader's weakness, though it can only rerank what was recalled: if the right document isn't in *anyone's* top-50, no reranker saves you. Hybrid widens *recall*; reranking sharpens *precision* (that's Step 2c, deliberately after this one).

The reason 2b is "next" and not "someday" in our journal: the symptom is witnessed, the query class (citations) is core to the legal domain, and the baseline is frozen. Symptom → diagnosis → cure, in that order.

---

## Part 4 — Merging two rankings without lying to yourself

### 4.1 Beginner: the tempting mistake

Two readers each return a ranked list with scores. The obvious move: average the scores per chunk, re-sort, done.

Look at what you'd actually be averaging. Dense scores are similarities in [0, 1] — ours cluster around 0.5–0.8. BM25 scores are unbounded and corpus-dependent — 4.2, 11.7, 23.9, whatever the index produces. Averaging them is averaging degrees Celsius with kilograms: the operation runs fine and the output means nothing. Whichever scale happens to have bigger numbers silently dominates the "combined" ranking. This is the single most common hybrid-retrieval bug, and it produces a system that *looks* hybrid while behaving as one engine with extra steps.

### 4.2 Intermediate: ranks instead of scores

You could try to fix the units — rescale each list's scores to [0, 1] (min-max normalization) and then blend. It works, sort of, but it's fragile in an instructive way: the rescaling depends on the min and max *of this particular result list*, so one freak outlier score reshapes everyone else's normalized value, and per-query score distributions shift enough that a blend weight tuned on Tuesday misbehaves on Thursday. Remember Part 1's score-band pathology — a compressed 0.017-wide band, min-max-stretched to fill [0, 1], manufactures confident-looking differences out of noise.

The robust move is to *discard the scores entirely* and keep only what each reader is actually qualified to say: the *order*. "This chunk is my #1, that one my #7." Ranks are unit-free — #3 means #3 in any list — so lists from incomparable scoring systems become instantly comparable.

Then reward chunks by their rank in each list, with high ranks worth a lot and the value falling away quickly: rank 1 worth much more than rank 5, worth somewhat more than rank 20, with deep-tail ranks worth nearly nothing but *never exactly nothing*. A natural shape for "large at 1, decaying, never zero" is the reciprocal — `1/rank`-ish — summed across the lists:

```
fused_score(chunk) = Σ over each list L of:  1 / (k + rank_L(chunk))
```

A chunk missing from a list simply contributes nothing from it. This is **Reciprocal Rank Fusion (RRF)** — and notice you've already understood it before being told its name. The formula is one line; the insight is *ranks, not scores*.

### 4.3 The constant k, and why it's 60

Without `k` (i.e., `k = 0`), rank 1 is worth 1.0 and rank 2 is worth 0.5 — a cliff. One reader's top pick would nearly always crush everything below, making fusion hypersensitive to exactly the noisy #1-vs-#2 distinctions we just said rankings aren't reliable about. Adding `k` flattens the top of the curve: with `k = 60`, rank 1 earns 1/61 ≈ 0.0164 and rank 5 earns 1/65 ≈ 0.0154 — close, as they should be, because the difference between #1 and #5 in a single ranking is weak evidence. The gap only becomes decisive when *both* readers agree.

Why 60 specifically? Empirical, from the 2009 paper that introduced RRF — it worked well across their benchmark sets and the method is insensitive to the exact value. It is a default, not a law: treat it as "flatten the top, keep the decay," and don't tune it until an eval tells you to.

What makes RRF the right default for us:

- **No tuning surface.** One constant with a robust default. Compare: score normalization needs per-engine calibration and a blend weight — three knobs before you've measured anything.
- **Consensus without fragility.** A chunk that's #2 dense and #3 lexical beats a chunk that's #1 in one list and absent from the other. Agreement between independent readers is the strongest relevance evidence available.
- **Blind-spot rescue built in.** Our citation document — absent from dense's list entirely — needs only its lexical rank to enter the fused list. The whole point of 2b falls out of the formula for free.

### 4.4 Advanced: what RRF costs, and the wider fusion landscape

Honesty about the trade: by discarding scores, RRF discards *confidence information*. Dense retrieval saying "0.82, 0.81, 0.55, ..." is telling you the third result is a cliff-edge worse — RRF flattens that cliff into "#2 vs #3." Weighted variants (multiply each list's contribution by a per-engine weight), calibrated score fusion, and learned fusion (train a model over both engines' features) all try to recover that information, at the cost of parameters that must be fitted and re-fitted. There's also the pragmatic knob of *how deep* each reader's list goes before fusing (fusing top-50 from each ≠ fusing top-500) — deeper lists widen recall and add tail noise.

The engineering judgment for a corpus of our size: start with unweighted RRF over modest-depth lists, measure, and let evals justify any added machinery. Every parameter you don't have is a parameter that can't be wrong.

One design decision RRF quietly forces: fusion needs both readers to talk about the *same units*. If dense ranks chunks and lexical ranked whole documents, "rank of this chunk" wouldn't exist in one of the lists. So the lexical index must be built at chunk granularity, mirroring the vector store.

---

## Part 5 — Hybrid in Vakil AI, and the edges of the technique

### 5.1 Where it lands

The architecture already left a parking spot. `retrieve()` (Step 2) is the single retrieval path — `/search`, evals, and `rag.ask()` all call it, and its contract (`doc_id, chunk_id, section, case_title, text, score`) is frozen. Hybrid lands *inside* that function:

- **The lexical index** is an FTS5 table (SQLite's built-in full-text engine, BM25 ranking included) living beside `chunk_vectors` in each profile's vector DB. It indexes the same chunks, keyed by the same ids. It's part of the disposable index, rebuilt by the indexer, never a second source of truth.
- **A design decision to make deliberately:** the vector store embeds *enriched* text (case title + section + chunk) but stores raw text. Should FTS5 index raw or enriched? Enriched, most likely — a citation or case name often lives in the title, not the body of every chunk — but this is exactly the kind of choice the before/after eval should confirm rather than assume.
- **Fusion is ~fifteen lines**: run both rankers, RRF, return the top-k under the same contract. Callers never learn anything changed — that's the seam pattern paying rent.
- **Profile identity:** retrieval strategy shapes what an index must contain, so hybrid-vs-dense belongs in the profile/fingerprint story like every other retrieval-quality knob — comparisons run as separate profiles against the same eval set.

### 5.2 What "done" looks like

Not "the code merged" — the journal's standard is *numbers before opinions*:

1. Dense-only baseline: frozen (`step4-baseline.json`, done in Step 5).
2. Build hybrid, run the same eval, **split by query type**.
3. Predictions on record before running: citation-type queries jump (the witnessed miss becomes a hit); conceptual queries hold steady (fusion doesn't degrade the good case); aggregate shows a modest bump that would have hidden the story.
4. A prediction overturned is a finding, not a failure — the break-it lab already taught us that (the Ollama truncation assumption died the same way).

### 5.3 Advanced: where hybrid stops helping

Knowing a technique means knowing its boundary. Failures hybrid does *not* fix:

- **Ordering among relevant chunks.** Hybrid widens what gets *found*; it doesn't sharpen fine-grained ranking between several plausible chunks. Both readers compress query and document into separate representations before comparing — neither ever reads them *together*, word against word. The tool that does is a cross-encoder reranker: Step 2c, "retrieve wide, rank narrow."
- **Badly phrased queries.** If the user's phrasing sits far from the corpus's phrasing in *both* vocabularies, both readers miss together. That's query rewriting (also 2c).
- **Chunk-boundary damage.** The dissent-as-holding failure from Step 5 (a dissenting opinion reported as the Court's ruling, with perfect citations) is invisible to *any* retriever — the opinion boundary was destroyed at chunking time, before either index existed. Pipeline-level fix, `PIPELINE_VERSION` bump. Retrieval cannot recover information the pipeline never preserved.
- **Scale.** Our exact brute-force cosine is milliseconds at 76 documents; FTS5 likewise. At millions of chunks both sides change character (approximate nearest-neighbor search, distributed lexical indexes) — but the *concepts* in this document survive the rewrite untouched, which is why they're worth learning at this size first.

### 5.4 The one-paragraph version

Dense retrieval reads for *meaning* and is blind to *names*; lexical retrieval reads *names* and is blind to *meaning*. Real query streams contain both, so run both readers. Their scores live on incomparable scales, so never average scores — convert to ranks and fuse with RRF, which rewards consensus, rescues each reader's blind spot, and has essentially nothing to tune. Verify with evals split by query type, against a frozen baseline, with predictions written down first. And remember what fusion cannot do: it widens recall — precision (reranking), phrasing (rewriting), and pipeline damage (chunking) are different diseases with different cures.

---

## Appendix — Vocabulary earned along the way

Terms introduced only after their concept was built, collected here for reference:

| Term | The idea it names |
|------|-------------------|
| Dense / semantic retrieval | Search as geometry: texts become locations, relevance becomes proximity |
| Embedding | The 768-number location summary a model assigns to a text |
| Lexical / sparse retrieval | Search by matching the words themselves via an inverted index |
| Term frequency | "More occurrences = more relevant" — the load-bearing naive idea |
| IDF | Credit weighted by a word's rarity across the collection |
| Saturation (k1) | Diminishing returns on repeated occurrences |
| Length normalization (b) | Correcting for long documents matching everything |
| BM25 | The three repairs assembled: IDF × saturated TF × length correction |
| Tokenization | The index-time policy deciding what counts as a matchable unit |
| Score-band compression | Flat, narrow score spread — the signature of a query that mapped nowhere |
| Rank fusion | Merging ranked lists by position, not score |
| RRF | Σ 1/(k + rank): reciprocal-rank reward, k≈60 flattening the top |
| Cross-encoder rerank | Reading query and document *together* for precision — the next step, not this one |
