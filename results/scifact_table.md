# Results on scifact (test)

Queries scored: 300

Encoder: `sentence-transformers/all-MiniLM-L6-v2`  
Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`  
Candidate pool: top-100

## Metrics

| Method | recall@10 | mrr@10 | ndcg@10 | recall@50 | mrr@50 | ndcg@50 | recall@100 | mrr@100 | ndcg@100 |
|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.7739 | 0.6312 | 0.6617 | 0.8686 | 0.6355 | 0.6846 | 0.8759 | 0.6356 | 0.6858 |
| dense | 0.7883 | 0.6068 | 0.6484 | 0.8893 | 0.6119 | 0.6723 | 0.9250 | 0.6123 | 0.6783 |
| bm25+ce | 0.7861 | 0.6465 | 0.6741 | 0.8692 | 0.6509 | 0.6948 | 0.8759 | 0.6510 | 0.6959 |
| dense+ce | 0.8089 | 0.6559 | 0.6868 | 0.9003 | 0.6604 | 0.7093 | 0.9250 | 0.6607 | 0.7134 |

## Latency

| Method | total (s) | per query (s) |
|---|---|---|
| bm25 | 0.0 | 0.000 |
| dense | 1.0 | 0.003 |
| bm25+ce | 441.6 | 1.472 |
| dense+ce | 433.0 | 1.443 |

