# Retrieval, Reranking, and RAG on SciFact — Technical Report

## 1. Problem statement

Build a small two-stage information retrieval system over a scientific corpus
and evaluate it rigorously. Specifically: given a scientific claim, retrieve
the most relevant abstracts from a fixed corpus, rerank the candidates with a
cross-encoder, and optionally generate a grounded answer that cites which
passages support its claims.

The goal is not to push state-of-the-art numbers — the dataset and models are
all small, public, and run on a CPU laptop. The goal is to compare four
configurations honestly under identical conditions, expose where each one
fails, and treat retrieval and generation as **separately measurable**
problems.

## 2. Why this design

A modern search stack is rarely a single model. The dominant pattern is:

1. **First stage** — fast, recall-oriented retrieval over the entire corpus.
2. **Second stage** — slow, precision-oriented reranking over a small
   candidate pool produced by stage one.

The reason is throughput. Cross-encoders score `(query, doc)` pairs by feeding
both through a transformer jointly. They are sharper than dual-encoder
embeddings (which encode query and doc independently and compare with a dot
product) but ~100× slower per pair. Running a cross-encoder over the entire
corpus per query is infeasible. Running it over the top-100 candidates from a
cheap first stage is fine.

This project compares **two first stages** (lexical BM25, dense MiniLM), each
optionally followed by the same cross-encoder. That gives four configurations
on a 2×2 grid, which is the minimum needed to disentangle two questions:

- Does the *first stage* matter? (BM25 vs dense, both unranked)
- Does *reranking* matter, and does its benefit depend on the first stage?
  (`+ce` rows vs unranked rows)

## 3. Experimental setup

### 3.1 Dataset

[BEIR/SciFact](https://github.com/beir-cellar/beir): 5,183 scientific abstracts
as the corpus, 300 test claims as queries, with binary relevance judgements
identifying which abstracts support or refute each claim. SciFact was chosen
because it is small enough to iterate on a laptop, has clean human-annotated
qrels, and has published baseline numbers I can sanity-check against.

### 3.2 Models

| Component | Model | Parameters |
|---|---|---|
| Lexical retriever | `bm25s` (Okapi BM25, k1=1.5, b=0.75) | n/a |
| Dense encoder | `sentence-transformers/all-MiniLM-L6-v2` | 22M |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22M |
| Generator (RAG) | `claude-haiku-4-5` via Anthropic API | n/a |

All transformer models were chosen for CPU friendliness. MiniLM-L-6 is the
smallest competitive option for both dense retrieval and cross-encoder
reranking on BEIR-style benchmarks.

### 3.3 Pipeline parameters

- Candidate pool size for reranking: `retrieve_k = 100`.
- Final ranked list size for evaluation: `top_k = 100` (metrics computed at
  cutoffs 10, 50, 100).
- FAISS index: `IndexFlatIP` over L2-normalised embeddings (so inner product =
  cosine). Exact search is fine at this corpus size; index-building takes a
  few seconds and adds zero variance.
- BM25 tokenisation: lowercase, English stopwords removed, no stemming.

### 3.4 Metrics

I implemented Recall@k, Precision@k, MRR@k, and nDCG@k from scratch in
`src/evaluation.py` rather than calling `pytrec_eval`. The implementation is
verified against hand-computed values in `tests/test_metrics.py` (perfect
ranking → nDCG = 1, no relevant doc in top k → MRR = 0, etc.).

The metrics are macro-averaged over queries: each query contributes equally
regardless of how many relevant documents it has. This matches the BEIR
convention.

## 4. Results

> Run `python -m scripts.evaluate --dataset scifact` to populate this section.
> Reference numbers from the BEIR leaderboard for sanity-checking are in
> brackets.

### 4.1 Quality

| Method | recall@10 | mrr@10 | ndcg@10 |
|---|---|---|---|
| bm25 | _TODO_ [≈0.66] | _TODO_ | _TODO_ |
| dense | _TODO_ [≈0.65] | _TODO_ | _TODO_ |
| bm25+ce | _TODO_ | _TODO_ | _TODO_ |
| dense+ce | _TODO_ | _TODO_ | _TODO_ |

Things to verify when filling this in:

- **BM25 nDCG@10 ≈ 0.66**, otherwise tokenisation or indexing is broken.
- Dense should be roughly comparable to BM25 on SciFact (it is *not* an easy
  win — SciFact has a lot of scientific jargon where lexical match is strong).
- Both `+ce` rows should improve over their respective base. The improvement
  is bounded above by recall@`retrieve_k` of the first stage: if BM25 only
  finds the gold passage in 80% of queries within the top 100, the reranker
  cannot recover the missing 20% no matter how good it is.

### 4.2 Latency

> Run `python -m scripts.benchmark --dataset scifact` to populate this.

Headline numbers (ms per query, end-to-end, `retrieve_k=100`, `top_k=10`):

| Method | latency |
|---|---|
| bm25 | _TODO_ |
| dense | _TODO_ |
| bm25+ce | _TODO_ |
| dense+ce | _TODO_ |

The dominant cost in `+ce` configurations is the cross-encoder forward pass
(100 query-doc pairs per query at ~10–30 ms each on CPU). First-stage
retrieval is sub-millisecond for BM25 and a few milliseconds for dense.
Indexing is one-off: BM25 in seconds, dense encoding the full corpus in 1–2
minutes on CPU.

The latency-vs-quality tradeoff has a clear shape. For each value of
`retrieve_k`:

- Larger pool → reranker has more chances to find the gold passage → better
  nDCG, up to a ceiling set by first-stage recall.
- Larger pool → linear increase in rerank time.
- The sweet spot on SciFact is around `retrieve_k=100`. Beyond that, recall
  improvements taper while latency keeps growing linearly.

## 5. Failure mode analysis

> Fill in with concrete query examples after running evaluation. The framework
> below is what to look for.

### 5.1 Where BM25 wins

Queries with **rare technical terms** that appear verbatim in the relevant
abstract: gene names, chemical formulas, specific drug names, citation
identifiers. Lexical match dominates because the vocabulary is narrow and
unambiguous. Dense models can be misled by surface-level semantic similarity
to *other* abstracts that discuss the same general topic but not the specific
entity in the query.

Look for queries where BM25 ranks the gold doc top-3 and dense ranks it
beyond top-50.

### 5.2 Where dense wins

Queries where the **claim is paraphrased** with little token overlap with the
abstract that supports it. Example: claim says "X reduces Y" while the
abstract uses synonyms ("X attenuates Y", "decline in Y observed in X group").
BM25 has nothing to match on; dense embeddings catch the semantic
relationship.

### 5.3 Where reranking helps

The cross-encoder reads query and passage **jointly**. Its win is greatest
when the gold passage is in the candidate pool but not at the top — for
example, BM25 ranked it #20 because of token sparsity, but the cross-encoder
recognises that this passage actually answers the query when it sees them
together.

Concrete pattern: BM25 nDCG@10 = 0 (gold not in top 10), but BM25+CE nDCG@10
> 0 (gold pulled into top 10 by reranker).

### 5.4 Where reranking is still weak

If the gold passage is **not in the candidate pool**, the reranker cannot
help. This is the recall ceiling I mentioned above. Fixes are: increase
`retrieve_k`, or fix the first stage (e.g. ensemble BM25 + dense scores
before reranking).

### 5.5 Process for filling this section in

For each method, dump the 10 queries with the lowest nDCG@10. For each, look
at:

1. Is the gold passage in the candidate pool? (Sets ceiling.)
2. What did the system rank above the gold? (Tells you what feature it's
   confusing.)
3. What's the linguistic relationship between query and gold? (Lexical?
   Paraphrase? Multi-hop?)

Five well-analysed examples beat fifty pages of prose.

## 6. Tradeoffs and design lessons

**Reranking is a precision booster, not a recall booster.** It can only
reorder what the first stage hands it. If first-stage recall@`retrieve_k` is
the ceiling, then improving the first stage is the higher-leverage move once
the reranker is "working."

**Dense doesn't always beat BM25.** On lexical-heavy domains (scientific
abstracts, legal text, code) BM25 is a strong baseline that's hard to beat
with off-the-shelf embeddings. The win for dense retrieval is in
paraphrase-heavy or open-domain settings. This is why the BEIR paper exists —
to expose that dense models trained on MS MARCO transfer unevenly across
domains.

**The cross-encoder is the latency bottleneck.** On CPU, ~10–30 ms per
`(query, doc)` pair × 100 candidates = 1–3 seconds of latency per query.
Production systems work around this with model distillation, ONNX
quantisation, or smaller candidate pools. None of those are explored here.

**RAG quality is bottlenecked by retrieval quality.** The generation step
cannot fix bad retrieval — it can only refuse to answer or hallucinate. This
is why I evaluate retrieval *separately* from RAG output, with hard metrics
on retrieval and a qualitative demo for generation.

## 7. Limitations

- **One dataset.** Numbers on SciFact don't transfer directly to other
  domains. A real evaluation would run multiple BEIR subsets and report
  averaged metrics. SciFact is the focused-but-honest single-domain case.
- **Binary relevance.** SciFact qrels are 0/1, so nDCG collapses to a function
  of rank position alone. Graded relevance (e.g. TREC-COVID's 0/1/2 scale)
  would give a richer picture.
- **No CV (cross-validation) on hyperparameters.** BM25's k1 and b are at
  defaults; the encoder model is fixed. A more rigorous study would search
  these. For this scope, defaults are honest because they match published
  baselines.
- **CPU-only evaluation latency.** GPU rerankers are 10–50× faster. The
  reported per-query times tell you the *qualitative* shape of the
  latency-quality tradeoff, not absolute production numbers.
- **No statistical significance tests.** With 300 queries, paired bootstrap or
  randomization tests would be appropriate before claiming one method beats
  another by a small margin. This is left as future work.

## 8. Future work

Roughly in order of impact-per-effort:

1. **Hybrid first stage**: combine BM25 and dense scores via reciprocal rank
   fusion before reranking. Often gives the biggest single gain on
   BEIR-style benchmarks.
2. **Add statistical significance**: paired bootstrap on per-query nDCG@10
   between methods. With 300 queries, differences of ~0.02 are within noise;
   anything smaller should not be claimed as an improvement.
3. **Larger reranker** (e.g. `bge-reranker-base`, ~280M params): probably
   adds another 0.02–0.04 nDCG@10 on SciFact at 5× the latency. Worth
   measuring the tradeoff explicitly.
4. **Multi-dataset evaluation**: extend the same pipeline to NFCorpus, FiQA,
   TREC-COVID. The code is already dataset-agnostic via the `--dataset` flag
   in the eval script.
5. **Quantise the encoder and reranker** with ONNX + int8 to recover most of
   the GPU latency on CPU. This is the production-engineering path.

## 9. Reproducing this report

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# Sanity-check the codebase
pytest

# Run the full evaluation (~30–60 min on CPU)
python -m scripts.evaluate --dataset scifact

# Run latency benchmarks (~10 min)
python -m scripts.benchmark --dataset scifact --n-queries 50 --reps 3

# Optional: interactive UI
streamlit run app.py

# Optional: RAG demo (requires ANTHROPIC_API_KEY)
python -m scripts.rag_demo --query "Does aspirin reduce heart attack risk?"
```

Outputs land in `results/`:
- `scifact_metrics.json` and `scifact_table.md` — evaluation results
- `scifact_latency.json` and `scifact_latency.md` — benchmark results

## 10. References

- Thakur, Reimers, Rücklé, Srivastava, Gurevych. *BEIR: A Heterogeneous
  Benchmark for Zero-shot Evaluation of Information Retrieval Models*.
  NeurIPS Datasets & Benchmarks, 2021.
- Wadden, Lin, Lo, Wang, van Zuylen, Cohan, Hajishirzi. *Fact or Fiction:
  Verifying Scientific Claims*. EMNLP, 2020.
- Reimers, Gurevych. *Sentence-BERT: Sentence Embeddings using Siamese
  BERT-Networks*. EMNLP, 2019.
- Nogueira, Cho. *Passage Re-ranking with BERT*. arXiv:1901.04085, 2019.
- Lewis et al. *Retrieval-Augmented Generation for Knowledge-Intensive NLP
  Tasks*. NeurIPS, 2020.
