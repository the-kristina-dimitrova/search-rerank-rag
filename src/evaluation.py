"""Standard retrieval metrics: Recall@k, Precision@k, MRR@k, nDCG@k.

Implemented from scratch to avoid the pytrec_eval/Java dependency. Matches
the trec_eval conventions:

- Recall@k: |relevant docs in top k| / |relevant docs total|
- Precision@k: |relevant docs in top k| / k
- MRR@k: 1 / (rank of first relevant doc), 0 if no relevant in top k
- nDCG@k: DCG@k / IDCG@k, where DCG = sum(rel_i / log2(i+2))

Macro-averaging over queries (each query weighted equally), which is the
BEIR convention.
"""
from __future__ import annotations

import math
from typing import Iterable


# qrels: {query_id: {doc_id: relevance_score}}
# run:   {query_id: [(doc_id, score), ...]}  (sorted by descending score)
Qrels = dict[str, dict[str, int]]
Run = dict[str, list[tuple[str, float]]]


def _relevant_docs(qrels_q: dict[str, int], min_rel: int = 1) -> set[str]:
    return {d for d, r in qrels_q.items() if r >= min_rel}


def recall_at_k(ranked_doc_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = ranked_doc_ids[:k]
    hits = sum(1 for d in top_k if d in relevant)
    return hits / len(relevant)


def precision_at_k(ranked_doc_ids: list[str], relevant: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    top_k = ranked_doc_ids[:k]
    hits = sum(1 for d in top_k if d in relevant)
    return hits / k


def reciprocal_rank_at_k(ranked_doc_ids: list[str], relevant: set[str], k: int) -> float:
    for i, d in enumerate(ranked_doc_ids[:k]):
        if d in relevant:
            return 1.0 / (i + 1)
    return 0.0


def _dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked_doc_ids: list[str], qrels_q: dict[str, int], k: int) -> float:
    gains = [float(qrels_q.get(d, 0)) for d in ranked_doc_ids[:k]]
    ideal_gains = sorted(qrels_q.values(), reverse=True)[:k]
    ideal_gains_f = [float(g) for g in ideal_gains]
    idcg = _dcg(ideal_gains_f)
    if idcg == 0.0:
        return 0.0
    return _dcg(gains) / idcg


def evaluate_run(
    run: Run,
    qrels: Qrels,
    ks: Iterable[int] = (10, 50, 100),
) -> dict[str, float]:
    """Compute Recall@k, Precision@k, MRR@k, nDCG@k macro-averaged across queries.

    Only queries that appear in `qrels` are scored.
    """
    ks = list(ks)
    metrics: dict[str, list[float]] = {}
    for k in ks:
        metrics[f"recall@{k}"] = []
        metrics[f"precision@{k}"] = []
        metrics[f"mrr@{k}"] = []
        metrics[f"ndcg@{k}"] = []

    for qid, qrels_q in qrels.items():
        ranked = [d for d, _ in run.get(qid, [])]
        relevant = _relevant_docs(qrels_q)
        for k in ks:
            metrics[f"recall@{k}"].append(recall_at_k(ranked, relevant, k))
            metrics[f"precision@{k}"].append(precision_at_k(ranked, relevant, k))
            metrics[f"mrr@{k}"].append(reciprocal_rank_at_k(ranked, relevant, k))
            metrics[f"ndcg@{k}"].append(ndcg_at_k(ranked, qrels_q, k))

    return {name: sum(vals) / len(vals) if vals else 0.0 for name, vals in metrics.items()}


def format_results_table(
    results: dict[str, dict[str, float]],
    ks: Iterable[int] = (10, 50, 100),
) -> str:
    """Format a results table as Markdown.

    Args:
        results: {method_name: {metric: value}}
        ks: cutoffs to display
    """
    ks = list(ks)
    metric_names = []
    for k in ks:
        metric_names.extend([f"recall@{k}", f"mrr@{k}", f"ndcg@{k}"])

    header = "| Method | " + " | ".join(metric_names) + " |"
    sep = "|" + "---|" * (len(metric_names) + 1)
    lines = [header, sep]
    for method, m in results.items():
        row = f"| {method} | " + " | ".join(f"{m.get(n, 0.0):.4f}" for n in metric_names) + " |"
        lines.append(row)
    return "\n".join(lines)