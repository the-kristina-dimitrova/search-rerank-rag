# search-rerank-rag

Two-stage retrieval pipeline (BM25 + dense + cross-encoder reranking) evaluated on BEIR/SciFact, with RAG over custom document collections.

## How it works

1. **Retrieve** — BM25 (lexical) or sentence-transformer + FAISS (semantic) pulls top-N candidates from a unified corpus
2. **Rerank** — cross-encoder rescores candidates for precision at the top of the list
3. **Generate** — LLM produces a grounded answer with passage citations

## Results on SciFact (300 queries)

| Method | recall@10 | mrr@10 | ndcg@10 | recall@50 | mrr@50 | ndcg@50 | recall@100 | mrr@100 | ndcg@100 |
|---|---|---|---|---|---|---|---|---|---|
| BM25 | 0.7739 | 0.6312 | **0.6617** | 0.8686 | 0.6355 | 0.6846 | 0.8759 | 0.6356 | 0.6858 |
| dense | 0.7883 | 0.6068 | 0.6484 | 0.8893 | 0.6119 | 0.6723 | 0.9250 | 0.6123 | 0.6783 |
| BM25+reranker | 0.7861 | 0.6465 | 0.6741 | 0.8692 | 0.6509 | 0.6948 | 0.8759 | 0.6510 | 0.6959 |
| **dense+reranker** | **0.8089** | **0.6559** | **0.6868** | **0.9003** | **0.6604** | **0.7093** | **0.9250** | **0.6607** | **0.7134** |

BM25 nDCG@10 matches the [BEIR leaderboard](https://github.com/beir-cellar/beir) (~0.66). Reranking gains +0.038 on dense but only +0.012 on BM25 — first-stage recall@100 (0.925 vs 0.876) is the ceiling for reranked quality.

## Quick start

```bash
pip install -r requirements.txt

# Evaluate retrieval quality
python -m scripts.evaluate --dataset scifact

# Add your own documents and query them
mkdir -p data/notes && cp your_lectures.pdf data/notes/
python -m scripts.build_notes
python -m scripts.query_notes --query "What is VC dimension?"

# Interactive UI
streamlit run app.py
```

**Adding custom document collection:**Drop your PDF materials in data/notes/.

Then run:
```bash
python -m scripts.build_notes
```

## Key design choices

- **Metrics from scratch** — Recall@k, MRR@k, nDCG@k implemented and verified against hand-computed values, no pytrec_eval/Java dependency
- **Provider-agnostic RAG** — pluggable `LLMBackend` protocol with Groq, Gemini, and Anthropic backends; adding a new provider is ~15 lines
- **Section-aware PDF chunking** — splits by numbered headings to preserve mathematical context, not naive paragraph splitting
- **Index caching** — FAISS index persisted to disk after first build, sub-second loads on subsequent runs
- **CI with zero heavy deps** — `TYPE_CHECKING` guards keep torch/FAISS out of the test import graph; CI installs only pytest

## Repo layout

```
src/              Core library
  retrievers.py     BM25 + Dense (MiniLM/FAISS)
  reranker.py       Cross-encoder reranking
  pipeline.py       Retrieve → rerank orchestration
  evaluation.py     Offline metrics
  rag.py            Grounded generation with citations
  custom_data.py    PDF section chunker

scripts/          Entry points
  evaluate.py       Quality eval on BEIR datasets
  benchmark.py      Latency measurements with variance
  build_notes.py    Build unified corpus from SciFact + your PDFs
  query_notes.py    RAG demo (CLI)

tests/            Unit tests (17 tests, ~0.1s)
app.py            Streamlit UI
REPORT.md         Full technical writeup with analysis
```

📄 **[Full technical writeup → REPORT.md](REPORT.md)**
