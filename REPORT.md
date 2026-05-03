# Technical Report — Search, Rerank, RAG

## 1. Problem

Given a query, retrieve the most relevant passages from a corpus, rerank them for precision, and optionally generate a grounded answer with citations. Evaluate each stage independently with standard offline metrics.

## 2. Architecture

Two-stage retrieve-then-rerank, the dominant pattern in production search:

1. **First stage** — fast, recall-oriented. Retrieves top-N candidates from the full corpus.
2. **Second stage** — slow, precision-oriented. Cross-encoder rescores only those N candidates.

Four configurations on a 2×2 grid (two retrievers × with/without reranker) isolate whether improvement comes from the retriever or the reranker.

## 3. Setup

**Dataset.** BEIR/SciFact: 5,183 scientific abstracts, 300 test claims with binary relevance judgements. Chosen for clean qrels, small size (iterates on a laptop), and published baselines to sanity-check against.

**Models.**

| Component | Model | Parameters |
|---|---|---|
| Lexical retriever | BM25 (k1=1.5, b=0.75) via `bm25s` | — |
| Dense retriever | `all-MiniLM-L6-v2` + FAISS `IndexFlatIP` | 22M |
| Cross-encoder | `ms-marco-MiniLM-L-6-v2` | 22M |
| RAG generator | Llama 3.3 70B via Groq | — |

All models run on CPU. Embeddings are L2-normalised so inner product = cosine similarity. FAISS uses exact search (fine at 5K docs). The RAG layer is provider-agnostic — Anthropic and Gemini are also supported via a pluggable backend protocol.

**Metrics.** Recall@k, Precision@k, MRR@k, nDCG@k implemented from scratch and verified against hand-computed values in the test suite. Macro-averaged over queries (BEIR convention).

## 4. Results

### 4.1 Retrieval quality (300 queries)

| Method | Recall@10 | MRR@10 | nDCG@10 | Recall@50 | nDCG@50 | Recall@100 | nDCG@100 |
|---|---|---|---|---|---|---|---|
| BM25 | 0.774 | 0.631 | 0.662 | 0.869 | 0.685 | 0.876 | 0.686 |
| Dense | 0.788 | 0.607 | 0.648 | 0.889 | 0.672 | 0.925 | 0.678 |
| BM25+CE | 0.786 | 0.647 | 0.674 | 0.869 | 0.695 | 0.876 | 0.696 |
| **Dense+CE** | **0.809** | **0.656** | **0.687** | **0.900** | **0.709** | **0.925** | **0.713** |

BM25 nDCG@10 = 0.662 matches the BEIR leaderboard reference (~0.66), confirming correct implementation.

### 4.2 Latency (50 queries, 3 reps, CPU)

**Indexing (one-off):**

| Component | Time |
|---|---|
| BM25 | 0.70s |
| Dense (encode 5K docs + FAISS) | 141.5s |

**First-stage retrieval (ms/query):**

| retrieve_k | BM25 | Dense |
|---|---|---|
| 10 | 0.3 ± 0.1 | 9.0 ± 0.9 |
| 50 | 0.3 ± 0.0 | 8.8 ± 0.7 |
| 100 | 0.3 ± 0.0 | 8.8 ± 0.7 |
| 200 | 0.3 ± 0.0 | 8.6 ± 0.7 |

**Cross-encoder reranking (ms/query):**

| Pool size | Rerank time |
|---|---|
| 10 | 159.5 ± 25.1 |
| 50 | 706.6 ± 61.4 |
| 100 | 1435.7 ± 170.5 |
| 200 | 2838.0 ± 289.8 |

**End-to-end (retrieve_k=100, top_k=10):**

| Method | ms/query |
|---|---|
| BM25 | 0.2 ± 0.1 |
| Dense | 9.6 ± 13.9 |
| BM25+CE | 1506.5 ± 211.9 |
| Dense+CE | 1540.9 ± 291.2 |

## 5. Analysis

### BM25 beats dense at the top of the list

BM25 nDCG@10 (0.662) exceeds dense nDCG@10 (0.648) despite dense having better recall@100 (0.925 vs 0.876). Scientific abstracts are lexically precise — gene names, drug names, measurement units appear verbatim in both query and passage. BM25 matches these exactly. Dense retrieval captures semantic similarity but also retrieves thematically related passages that use different terminology, which are irrelevant under binary qrels.

Dense recovers at recall@100 because it casts a wider semantic net — it finds relevant passages BM25 misses when the claim is paraphrased. But BM25 ranks its finds higher.

### Dense gives the reranker more to work with

Reranking adds +0.012 nDCG@10 on BM25 but +0.038 on dense. The asymmetry is explained by recall@100: dense provides a 5% larger candidate pool (0.925 vs 0.876), giving the reranker more relevant passages to promote. First-stage recall is the ceiling for reranked quality — the reranker cannot recover passages the retriever missed.

### Reranking is ~7,500× slower for ~3.8% nDCG gain

Dense+CE achieves the best nDCG@10 (0.687) at 1541ms/query — versus 0.2ms for BM25 alone. Reranking latency scales linearly with pool size: 160ms for 10 candidates, 707ms for 50, 1436ms for 100, 2838ms for 200. The cross-encoder processes each (query, doc) pair sequentially on CPU with no batching benefit across queries.

BM25 retrieval is constant at 0.3ms regardless of retrieve_k. Dense retrieval is ~9ms, also roughly constant — FAISS exact search cost is dominated by the query encoding forward pass, not the search itself.

### Out-of-domain queries fail honestly

Queries outside SciFact's domain (e.g. "does aspirin reduce heart attack risk?") return the nearest-domain passages (aspirin + colorectal cancer studies). The cross-encoder assigns negative scores to these (-0.7, -2.6), which can serve as a low-confidence signal. A well-prompted RAG system should refuse to answer rather than confabulate — and the grounding prompt in this project does exactly that.

## 6. Design decisions

**Custom metrics over pytrec_eval.** Avoids the Java-via-pip dependency. Forces clarity about what each metric computes. Verified against hand-computed values including edge cases (empty qrels, perfect ranking).

**Provider-agnostic RAG.** Three-method `LLMBackend` protocol with Groq, Gemini, and Anthropic implementations. Lazy imports ensure each backend only requires its own SDK. Adding a fourth provider is a ~15-line class.

**Section-aware PDF chunking.** Lecture notes are split by numbered headings, not paragraphs. Splitting mid-proof loses context the retriever needs. The chunker filters false positives (page numbers, TOC entries) and skips the first 3 pages (table of contents).

**Index caching.** Dense index (FAISS + doc_ids) persisted to disk after the first build. Subsequent runs load in <1s instead of re-encoding the full corpus (~142s).

**TYPE_CHECKING import guards.** Heavy ML dependencies (torch, FAISS, BEIR) are behind `if TYPE_CHECKING:` in pipeline.py so the unit test suite runs in <0.1s with only pytest installed. CI needs no GPU, no torch, no FAISS.

## 7. Limitations

- **One dataset.** SciFact numbers don't transfer directly to other domains.
- **Binary relevance.** nDCG with 0/1 judgements collapses to a function of rank position. Graded relevance would give a richer picture.
- **Default hyperparameters.** BM25 k1/b and encoder/reranker models are all at defaults. Honest — matches published baselines — but leaves optimisation headroom unexplored.
- **CPU-only latency.** GPU cross-encoders are 10–50× faster. Reported numbers show the shape of the tradeoff, not production-grade absolutes.
- **No statistical significance.** With 300 queries, paired bootstrap tests would be needed before claiming differences below ~0.01 nDCG@10 are real.

## 8. Future work

1. **Hybrid retrieval** — reciprocal rank fusion of BM25 + dense before reranking. Typically the single biggest gain on BEIR.
2. **Larger reranker** — `bge-reranker-base` (~280M params) for +0.02–0.04 nDCG@10 at ~5× latency.
3. **Multi-dataset evaluation** — NFCorpus, FiQA, TREC-COVID via the existing `--dataset` flag.
4. **ONNX quantisation** — recover GPU-class latency on CPU.

## 9. Reproducing

```bash
pip install -r requirements.txt
pytest                                                    # 17 tests, <1s
python -m scripts.evaluate --dataset scifact              # ~30 min on CPU
python -m scripts.benchmark --dataset scifact             # ~10 min
streamlit run app.py                                      # interactive UI
```

## 10. References

- Thakur et al. *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.* NeurIPS 2021.
- Wadden et al. *Fact or Fiction: Verifying Scientific Claims.* EMNLP 2020.
- Reimers & Gurevych. *Sentence-BERT.* EMNLP 2019.
- Nogueira & Cho. *Passage Re-ranking with BERT.* arXiv:1901.04085.
