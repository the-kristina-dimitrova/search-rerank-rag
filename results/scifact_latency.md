# Latency benchmarks — scifact

Sampled 50 queries, 3 reps each. Times are wall-clock on the machine running this script.

## Indexing (one-off)

| Component | Time (s) |
|---|---|
| BM25 | 0.70 |
| Dense (encode + FAISS add) | 141.54 |

## First-stage retrieval (ms per query, mean ± std)

| retrieve_k | bm25 | dense |
|---|---|---|
| 10 | 0.3 ± 0.1 | 9.0 ± 0.9 |
| 50 | 0.3 ± 0.0 | 8.8 ± 0.7 |
| 100 | 0.3 ± 0.0 | 8.8 ± 0.7 |
| 200 | 0.3 ± 0.0 | 8.6 ± 0.7 |

## Reranking (ms per query, mean ± std)

| pool size | rerank time |
|---|---|
| 10 | 159.5 ± 25.1 |
| 50 | 706.6 ± 61.4 |
| 100 | 1435.7 ± 170.5 |
| 200 | 2838.0 ± 289.8 |

## End-to-end pipeline (retrieve_k=100, top_k=10)

| Method | ms / query |
|---|---|
| bm25 | 0.2 ± 0.1 |
| dense | 9.6 ± 13.9 |
| bm25+ce | 1506.5 ± 211.9 |
| dense+ce | 1540.9 ± 291.2 |
