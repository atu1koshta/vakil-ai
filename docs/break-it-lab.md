# The Break-It Lab: Four Ways a RAG System Lies to You

*A knowledge base for Step 5 of Vakil AI — why we deliberately break our own system before improving it.*

---

## Part 0 — Why break something that works?

By Step 4 our pipeline produced its first end-to-end grounded answer: a question went in, chunks came back from the vector index, a local LLM wrote an answer with citations. It *looked* correct. That is exactly the problem.

A RAG system has a property that ordinary software does not: **it fails fluently**. When a normal program breaks, you get a stack trace. When a RAG system breaks, you get a well-written, confident, grammatically perfect paragraph that happens to be wrong. There is no error message, because from the model's point of view nothing went wrong — it did exactly what it was trained to do: produce plausible text.

So "it works on the questions I tried" is close to meaningless. The only way to know how a RAG system fails is to *make* it fail, on purpose, one failure mode at a time, and watch what happens. That is the break-it lab. Each probe below targets one distinct way the system can lie, and each is tested in isolation before we build the cure — because a fix built before witnessing the disease is a fix you don't understand.

The four probes:

| # | Probe | The lie it exposes | Layer at fault |
|---|-------|--------------------|----------------|
| 1 | Parametric leakage | "I read this in your documents" (it didn't) | Generation |
| 2 | Retrieval miss | "Nothing relevant exists" (it does) | Retrieval |
| 3 | Context overflow | "I considered all the evidence" (some was silently cut) | Context assembly |
| 4 | Citation drift | "See source X" (X is misnamed, or not the source) | Generation ↔ contract |

---

## Part 1 — Probe 1: Parametric leakage

### 1.1 Beginner: two brains, one mouth

Start with the mental model. An LLM answering a RAG question has access to two completely different knowledge sources:

- **Parametric knowledge** — everything absorbed into the model's weights during training. Think of it as long-term memory: vast, frozen at training time, uncited, and — crucially — *reconstructive*. The model doesn't store facts; it stores statistical tendencies that usually reproduce facts.
- **Contextual knowledge** — the text physically present in the prompt window. In RAG, this is the retrieved chunks. It is current, inspectable, and citable.

RAG's entire value proposition is a promise: *the answer comes only from contextual knowledge*. That's what makes answers verifiable — every claim can be traced to a document a human can read.

**Parametric leakage** is the model breaking that promise: answering (fully or partly) from its weights while presenting the answer as if it came from your documents.

### 1.2 Intermediate: why the model leaks

The model is not "cheating" — it literally cannot tell its two knowledge sources apart at generation time. Every token it produces is sampled from one probability distribution shaped by *both* the prompt text and the weights. An instruction like "answer only from the excerpts" doesn't build a wall between the sources; it just adds weight to one side of the scale. On the other side of the scale sit:

- a strong trained prior to be helpful and complete rather than say "I don't know";
- pattern-completion pull: a famous name in the question activates every association the weights hold about it.

Legal text is a worst-case domain for this. Indian Supreme Court judgments are in every training corpus. Ask about a famous case and the model *recognizes* the name — but recognition is not recall. What the weights hold is an association soup ("Chintaman Rao… constitutional… Article 19… reasonable restrictions…"), and generation happily condenses that soup into a specific-sounding holding with an invented citation. We witnessed exactly this in Step 1, before RAG existed: llama3.1 invented pardon powers for Chintaman Rao; deepseek-r1 fabricated an RTE Act holding complete with a fake citation. Confident, fluent, wrong.

### 1.3 Advanced: the two flavors of leakage

This is the distinction most tutorials skip, and it matters because the two flavors need different tests and different cures.

**Total leakage.** Retrieval returns nothing relevant; the model answers anyway, entirely from memory. This is the crude flavor. The standard defense is the *escape hatch* — an explicit instruction that saying "the excerpts do not establish this" is an acceptable answer. The escape hatch is the single biggest anti-hallucination lever in prompt design, because it gives the helpfulness prior a legal exit.

**Blended leakage.** Retrieval returns *partially* relevant chunks. The model answers mostly from them — correctly, with citations — but pads the gaps from memory: an extra holding, an invented paragraph number, a "the court further observed…" that appears nowhere in the excerpts. This flavor is far more dangerous:

- The escape hatch never fires, because there *was* something to answer from.
- The leaked sentences sit next to genuinely cited ones and borrow their credibility. A fabrication inside a wall of real citations is more convincing than a fabrication alone — RAG can make hallucination *worse* than a plain chatbot if this goes unchecked.
- It is invisible to any whole-answer check. You only catch it by reading claim-by-claim: *for each sentence, which chunk supports it?* This is exactly why the generation-eval phase (2d) plans a claim-by-claim faithfulness judge rather than an answer-level score.

### 1.4 How we probe it

The design principle: **maximize the tension between the two knowledge sources**, then watch which one wins.

**Test A — total leakage (maximum temptation).** Ask about a case that is maximally famous (strong parametric pull) and confirmed absent from the corpus (zero contextual support). We used *Kesavananda Bharati* — the most famous Indian constitutional case in existence, and verifiably not among our 76 documents.

A subtlety discovered while designing this test: our retrieval has **no relevance floor**. Cosine similarity always ranks and returns top-k; there is no "nothing found" below some threshold. So the corpus returned *something* — semantic neighbors like *A.K. Gopalan* chunks (constitutional law, fundamental rights). The probe therefore tests the hard version of the question: does the escape hatch fire when the context is *irrelevant*, not merely *empty*? (The `/ask` endpoint's canned refusal only covers the empty case.)

*Witnessed result (pass):* the model refused cleanly — "The provided excerpts do not establish anything regarding Kesavananda Bharati" — and even named which cases the excerpts actually came from. Zero leakage: it didn't sneak in so much as "the basic structure case" as an aside.

**Test B — the control arm.** A system that refuses everything passes Test A trivially. The escape hatch is only *calibrated* if the model also answers when evidence exists. Same question shape, case present in corpus (*A.K. Gopalan*).

*Witnessed result (pass, with a discovery):* full answer, exact citations. But see §1.5 — the answer was faithful and still wrong.

**Test C — blended leakage.** Ask a question where the corpus holds partial evidence but the full answer lives only in the weights: *"Was A.K. Gopalan later overruled?"* The Gopalan chunks (1950) physically cannot contain the answer; the overruling (Maneka Gandhi, 1978) is not in the corpus; the model certainly knows it. The escape hatch cannot fire on empty retrieval — plenty of Gopalan text arrives. Pure test of restriction discipline under partial evidence.

*Witnessed result (pass):* "The provided excerpts do not contain information about whether A.K. Gopalan was later overruled… no subsequent judgments are referenced in the excerpts." The model held the line between what the chunks say and what it knows.

**Grading rule for all three:** read claim-by-claim, not by vibes. One sentence of parametric history inside an otherwise-cited answer is a leak. And one clean run is an anecdote, not a property — sampling is nondeterministic, so repeat the probes several times before trusting the result.

### 1.5 What the probe found that we weren't looking for

Two side-discoveries, both journal-worthy:

**Faithful-but-wrong (dissent-as-holding).** In Test B the model answered "what was decided in Gopalan" by reporting that sections 12 and 14 were void and the petitioner was to be released — citing chunk_135 accurately. We read chunk_135: it says exactly that. But it is a *dissenting opinion* ("in my opinion… I would accordingly order his release"). The actual majority upheld the Act, struck down only section 14, and dismissed the petition — Gopalan stayed in jail. The model reported the outcome backwards while citing perfectly.

This is a distinct failure mode: not leakage (no weights involved), not a retrieval miss (right document found). Every claim was traceable to a chunk, and the answer was still wrong. The root cause is upstream of the prompt: a judgment contains multiple opinions (majority, concurrences, dissents), and our chunker labels them all `## Judgment`. The opinion-author information is destroyed at chunking time — no prompt can recover metadata that isn't there. Candidate cure: opinion-boundary detection in the chunker (judge-name markers like "MUKHERJEA J.—" are regex-able), tagging chunks majority/dissent. That is pipeline-level work (a `PIPELINE_VERSION` bump), deferred with a symptom→diagnosis→cure trail like every other measured upgrade.

The general lesson, worth stating once precisely: **faithfulness (every claim is supported by a cited source) and correctness (the answer is actually right) are different properties, and a RAG system can have the first without the second.** This is why the 2d eval plans two separate judges.

**Template parroting.** In Test C the model appended a literal `[doc_id:chunk_id | SECTION]` — the placeholder from the prompt's citation-format example — with nothing to cite. Harmless decoration here, but it shows format examples in prompts are followed *as text*, not as intent. Goes on the probe-4 prompt-tightening pile.

### 1.6 Criticality and scale

Test at **every prompt change and every model swap** — restriction discipline is a property of the (model × prompt) pair, not of the system. Our probes cost three curl commands; there is no excuse to skip them. At production scale this becomes automated: a held-out set of absent-case and partial-evidence questions run on every change, with the claim-by-claim check done by an LLM judge (the 2d faithfulness judge is precisely this probe, industrialized). In a legal product this is the highest-stakes probe of the four: a fabricated holding delivered to a lawyer inside a credible-looking cited answer is the catastrophic outcome.

---

## Part 2 — Probe 2: Retrieval miss

### 2.1 Beginner: the librarian who reads for gist

Imagine a librarian who has read every book and remembers *what each one is about*, but not the exact words. Ask "which cases discuss whether fraud allegations can go to arbitration?" and this librarian is superb. Ask "which book contains the serial number AIR 1962 SC 406?" and the librarian shrugs — serial numbers don't have a *gist*.

Dense retrieval — embeddings plus cosine similarity — is that librarian. An embedding model compresses a text into a single fixed-length vector (768 numbers, in our case) that captures its overall meaning. Similar meanings land near each other; retrieval means embedding the question and finding the nearest chunk vectors.

The compression is the power and the blindness at once. 768 dimensions is plenty for "this text is about preventive detention and fundamental rights." It has no room for "this text contains the exact string AIR 1962 SC 406."

### 2.2 Intermediate: why identifiers smear

Legal questions come in two shapes, and they interact with embeddings oppositely:

- **Conceptual queries** — "when can a compromise decree be set aside?" These are what embeddings are made for: paraphrase-tolerant, meaning-driven.
- **Identifier queries** — "AIR 1962 SC 406", "Section 302 IPC", "Civil Appeal No. 3777 of 2014". These are rare, precise strings whose *entire information content is the exact token sequence*.

For an identifier query, the embedding has almost nothing semantic to grab. "AIR 1962 SC 406" and "AIR 1967 SC 1910" mean nearly the same *thing* to a meaning-based model — "a legal citation" — so both embed into the same generic legal-citation neighborhood. The distinguishing digits get smeared away. The result is a query vector roughly equidistant from half the corpus: cosine still dutifully ranks a top-k (there is no relevance floor), but the ranking is noise. A telltale signature of a miss is a **compressed score band** — every result at nearly the same similarity, no clear winner.

Meanwhile the old-school lexical approach — BM25, essentially a smarter grep — *rewards exactly what embeddings discard*: a term that is rare in the corpus but present in the query is BM25's favorite thing in the world, and "406" next to "1962" next to "AIR" is as rare as it gets.

Neither method dominates. They fail on opposite query shapes. That observation, once witnessed and measured, is the entire justification for hybrid retrieval (planned as 2b): run both, fuse the rankings with Reciprocal Rank Fusion (rank-based fusion, so the incomparable score scales of BM25 and cosine never need normalizing), inside `retrieve()` so no caller changes.

### 2.3 How we probe it

The design principle: **a miss only counts if the target exists**. Whiffing on something absent proves nothing. So first we grepped the corpus for ground truth:

- `AIR 1962 SC 406` appears 3× inside the *A. Ayyasamy* (2016) document — the primary target.
- `Section 302` appears in 5 known documents — the secondary target.

Then two paired queries against `/search` (retrieval only — no LLM in the loop, so nothing muddies the signal):

1. **Identifier query:** `AIR 1962 SC 406`. Grading: does the Ayyasamy document appear in the top-k at all, and at what rank? Prediction: whiff, with a compressed score band.
2. **Conceptual control:** "arbitrability of fraud allegations in arbitration agreements" — the *same* document's core subject. Prediction: clean hit.

The pair, not either query alone, is the lesson: same document, findable by meaning, invisible by identifier. When 2b lands, this exact pair becomes the before/after measurement — the eval set is split by question type precisely so we can see the lift appear on citation queries and *not* on conceptual ones. A lift needs a frozen baseline to lift over, which is why hybrid was deferred until the dense-only numbers were locked.

*Status: witnessed. The identifier query missed completely — the Ayyasamy document was absent even from the top-20, with a compressed score band (0.547–0.564 across five unrelated documents, a spread of 0.017). The conceptual control took all five top slots at 0.75–0.82; the gap between its rank 1 and rank 2 alone (0.037) was double the identifier query's entire band. Same document: findable by meaning, invisible by identifier. Hybrid retrieval (2b) is now earned.*

### 2.4 Criticality and scale

For a legal assistant this probe is not optional: **identifier lookup is a primary user behavior**. Lawyers search by citation, section number, and case number as often as by concept. A system that silently returns plausible-but-wrong documents for "Section 302 IPC" doesn't look broken — it looks like the corpus lacks the answer, which is the retrieval layer's version of a fluent lie ("nothing relevant exists").

Test whenever the retrieval recipe changes (embedding model, chunking, enrichment, hybrid weights). At small scale: a handful of paired identifier/conceptual queries with known ground truth. At production scale: an eval set stratified by query type, tracking doc_recall@k and MRR per type — a single blended average would hide exactly the split this probe exists to expose.

---

## Part 3 — Probe 3: Context overflow

### 3.1 Beginner: the window is smaller than you think

An LLM reads at most a fixed number of tokens — the context window (`num_ctx` in Ollama). Everything must fit inside it: system prompt, the question, all retrieved chunks, *and* room for the answer the model hasn't written yet. RAG systems are context-hungry by design — the whole idea is stuffing evidence into the prompt — so RAG runs into the wall constantly.

The dangerous part is not the wall. It's what happens at the wall.

### 3.2 Intermediate: who does the cutting decides what survives

When the assembled prompt exceeds the window, *something* gets cut. The question is who cuts, and by what rule:

- **Ollama's rule (the silent default):** front-truncation. Tokens beyond `num_ctx` are dropped *from the beginning* — no error, no warning. The beginning of a RAG prompt is the system prompt: the restriction clause, the escape hatch, the citation contract. So the failure is maximally perverse: **overflow deletes the safety instructions first**, leaving the model with a pile of legal text and no rules — which is a hallucination machine wearing a RAG costume. We documented this gotcha in Step 1 and it is the reason our LLM layer uses Ollama's native `/api/chat` (where `num_ctx` is controllable) rather than the OpenAI-compatible shim.
- **Our rule (`context.py`, Step 3):** deliberate whole-chunk eviction from the *worst-ranked tail*. The budget is computed honestly — `num_ctx` minus system prompt, minus question, minus answer headroom — and chunks that don't fit are dropped whole, lowest-ranked first, with the cut *reported* (`kept`/`dropped` lists and token counts in every `/ask` response). The exact opposite of silent front-truncation on every axis: what is cut (worst evidence, not instructions), how (whole chunks, not mid-sentence), and visibly (accounted, not silent).

One more distinction from Step 3, because it is easy to conflate: the token budget and the k cap are **two knobs for two diseases**. The budget guards against the hardware limit (window overflow). The k cap guards against *noise dilution* — a quality judgment that chunk #13 is more likely to distract than help, even when it fits. Our first real run proved they bind independently: 12 chunks retrieved, all 12 would have fit the 7k budget, but the k cap kept 8.

### 3.3 Advanced: overflow without truncation

Even when everything fits, position matters. Models attend unevenly across long contexts — strongest at the beginning and end, weakest in the middle (the "lost in the middle" effect). And token counting is itself approximate: we count with tiktoken's cl100k encoding as a stand-in for whatever the local model actually uses, and let the answer headroom absorb the error. Approximations like this are fine *precisely because* they're accounted for — an unaccounted approximation is a future silent failure.

### 3.4 How we probe it

Force the overflow deliberately: request oversized `k` so assembly must evict, and (separately) bypass the budget to watch Ollama's raw front-truncation eat the system prompt. Grading:

- Eviction path: `dropped` list populates from the worst-ranked tail; answer quality degrades gracefully or not at all; safety instructions provably intact (the escape hatch still fires on an absent-case question even at max context pressure).
- Truncation path: witness the "before" picture — the same probe-1 questions leaking once the restriction clause has been silently eaten. This is the probe that proves *why* the budget exists.

*Status: pending.*

### 3.5 Criticality and scale

Test whenever the context recipe changes: `num_ctx`, system prompt length, k, chunk sizes, model swap (different models, different windows). The insidious property of overflow bugs is that they are **load-dependent**: every demo works (short questions, few chunks) and production fails (long questions, big k), and the failure mode is not an error but *quietly worse answers* — often with the safety prompt gone. Any system that grows its prompt at runtime needs an overflow probe; the assertion "escape hatch still fires at max context pressure" belongs in automated tests, not folklore.

---

## Part 4 — Probe 4: Citation drift

### 4.1 Beginner: a citation is a promise

Our system prompt instructs the model to cite evidence as `[doc_id:chunk_id]`, pointing at the provenance labels that context assembly stamps onto every chunk. A citation is the system's verifiability promise: *follow this pointer and you can check me*.

Citation drift is the pointer decaying: the model cites, but the pointer is malformed, renamed, or aimed at a chunk that doesn't support the claim. The answer *looks* verifiable — which is worse than looking unverifiable, because nobody checks what looks checkable.

### 4.2 Intermediate: why models drift

The model has no symbol table. `a-k-gopalan-vs-the-state-of-madras-union-of-india-on-19-may-1950-1` is not an identifier to it — it's just tokens, and the model has a lifetime of training pressure toward *human-readable* text. So it helpfully "improves" the citation: substituting the readable case title for the ugly slug, reformatting, abbreviating. Each improvement breaks machine resolution of the pointer.

We witnessed this live in Step 4's first run: every claim carried a citation (the contract's *spirit* held), but the model swapped the doc-id slug for the case title, keeping only the chunk-id exact. The instruction "use the exact ids from its header" was not literal enough — models follow format instructions *approximately* unless the format is made trivially copyable.

The probe-1 runs added two more data points. In the Gopalan control run, citations were exact — so drift is *intermittent*, which makes it worse: a validator that runs sometimes is a validator you can't trust. And in the blended-leakage run the model appended a literal `[doc_id:chunk_id | SECTION]` placeholder with nothing to cite — format examples in prompts are imitated as text, not understood as intent.

### 4.3 Advanced: the three grades of drift

Worth separating, because they need different defenses:

1. **Syntactic drift** — malformed or renamed pointers (the slug-vs-title swap). Machine-detectable: parse every citation in the answer and check it against the `kept` list. Cheap to validate, cheap to fail loudly on.
2. **Referential drift** — well-formed pointer, wrong target: the citation resolves to a chunk that doesn't support the claim. Undetectable by parsing; this needs claim-vs-chunk comparison, which is the 2d faithfulness judge's job.
3. **Decorative citation** — citations emitted as texture rather than reference (the template-parroting artifact is the degenerate case). A model that has learned "answers end with bracketed ids" will decorate even when nothing supports the claim.

The general principle: **an instruction is not a mechanism**. The prompt asks for exact ids; only a validator *guarantees* them. Mature systems treat model output as untrusted input — parse the citations, resolve them against the actually-provided chunks, and reject or repair on failure. The prompt's job is merely to make compliance easy (ids trivially copyable, one obvious format); the validator's job is to make non-compliance visible.

### 4.4 How we probe it

Drift was witnessed before the probe existed, so probe 4 is less discovery than systematic characterization: run grounded questions repeatedly, parse every citation, measure the exact-match rate against the `kept` list. Then tighten the prompt against the witnessed failures specifically — literal copyability ("copy the id exactly as it appears between the brackets in the header"), and reframe the format example so it cannot be parroted as decoration — and measure the rate again. Prompt engineering against witnessed failures, with a number attached; never against imagined ones.

*Status: drift witnessed (step 4), intermittency witnessed (probe 1 runs), systematic measurement pending.*

### 4.5 Criticality and scale

In a legal product, citations are not decoration — they are the product. A lawyer will act on "see [case:chunk]"; a broken pointer that nobody can follow, or worse, a confident pointer at the wrong text, converts the system's core trust feature into a liability. Test on every prompt or model change (drift is a (model × prompt) property, like leakage). At scale, the syntactic validator runs on **every single production answer** — it's a string parse, effectively free — while referential checking runs as sampled offline evaluation via the faithfulness judge.

---

## Part 5 — The shape of the method

Stepping back, the four probes teach one meta-lesson each:

1. **Leakage:** an instruction is a weight on a scale, not a wall. Verify discipline empirically, per model, per prompt, repeatedly.
2. **Retrieval miss:** every retrieval method has a blind spot shaped like its representation. Find the blind spot with ground-truth-confirmed targets before buying the cure.
3. **Overflow:** whoever cuts the context decides what the model knows — make the cut deliberate, ranked, and reported, never silent.
4. **Drift:** model output is untrusted input. Prompts request; validators guarantee.

And one lesson across all four: **build the fix after witnessing the failure.** Hybrid retrieval, the faithfulness judge, opinion-aware chunking, the citation validator — each sits in the journal as a symptom→diagnosis→cure trail, waiting for its symptom to be witnessed and its baseline to be frozen. The alternative — installing every fashionable RAG technique on day one — produces a system where nobody knows which parts do anything, because nothing was ever measured falling over without them.

---

## Appendix — Witnessed-failure log

| Date | Probe | Question | Result |
|------|-------|----------|--------|
| Step 1 (pre-RAG) | leakage baseline | Chintaman Rao (in corpus, no RAG) | llama3.1 invented pardon powers; deepseek fabricated RTE holding + citation |
| Step 4 first run | drift | grounded question | title substituted for doc-id slug; chunk_id kept exact |
| Probe 1-A | total leakage | Kesavananda (absent, famous) | **Pass** — escape hatch fired; named what excerpts actually contain |
| Probe 1-B | control | Gopalan (present, famous) | **Pass** on calibration + exact citations; **new failure witnessed**: dissent reported as the holding (faithful-but-wrong) |
| Probe 1-C | blended leakage | "Was Gopalan overruled?" (partial evidence) | **Pass** — refused the gap despite knowing Maneka Gandhi; template placeholder parroted |
| Probe 2 | retrieval miss | `AIR 1962 SC 406` (present in Ayyasamy, 3×) vs conceptual control | **Witnessed** — target absent from top-20 on identifier query, score band compressed (0.547–0.564); conceptual control hit 5/5 at 0.75–0.82. 2b earned |
| Probe 3 | overflow | oversized k; raw truncation | pending |
| Probe 4 | drift measurement | repeated grounded runs, parse rate | pending |
