"""Verify metric implementations against hand-computed values.

These checks exist because the metric code is the one piece where a subtle bug
(off-by-one in nDCG positions, wrong tie-breaking, division by zero on empty
qrels) would silently invalidate every number in the results table.
"""
import math

from src.evaluation import (
    evaluate_run,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)


# Shared fixture: 3 relevant docs, ranked list with relevant at positions 1, 3, 5.
RANKED = ["a", "b", "c", "d", "e", "f"]
RELEVANT = {"a", "c", "e"}


def test_recall_at_k():
    assert recall_at_k(RANKED, RELEVANT, 1) == 1 / 3
    assert recall_at_k(RANKED, RELEVANT, 3) == 2 / 3
    assert recall_at_k(RANKED, RELEVANT, 5) == 1.0


def test_recall_empty_relevant():
    # Avoid zero-division on queries with no positive judgements.
    assert recall_at_k(RANKED, set(), 10) == 0.0


def test_precision_at_k():
    assert precision_at_k(RANKED, RELEVANT, 1) == 1.0
    assert precision_at_k(RANKED, RELEVANT, 3) == 2 / 3
    assert precision_at_k(RANKED, RELEVANT, 5) == 3 / 5


def test_mrr_first_relevant_at_top():
    assert reciprocal_rank_at_k(RANKED, RELEVANT, 10) == 1.0


def test_mrr_first_relevant_lower():
    assert reciprocal_rank_at_k(["b", "a", "c"], RELEVANT, 10) == 0.5


def test_mrr_no_relevant_in_top_k():
    assert reciprocal_rank_at_k(["x", "y", "z"], RELEVANT, 10) == 0.0


def test_mrr_relevant_outside_cutoff():
    # Cutoff k=2 hides the relevant doc at rank 3.
    assert reciprocal_rank_at_k(["x", "y", "a"], RELEVANT, 2) == 0.0


def test_ndcg_against_hand_computed():
    qrels_q = {"a": 1, "c": 1, "e": 1}
    # Relevant at positions 1, 3, 5 (1-indexed).
    dcg = 1.0 / math.log2(2) + 1.0 / math.log2(4) + 1.0 / math.log2(6)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3) + 1.0 / math.log2(4)
    expected = dcg / idcg
    assert abs(ndcg_at_k(RANKED, qrels_q, 10) - expected) < 1e-9


def test_ndcg_perfect_ranking():
    qrels_q = {"a": 1, "c": 1, "e": 1}
    perfect = ["a", "c", "e", "b", "d"]
    assert abs(ndcg_at_k(perfect, qrels_q, 10) - 1.0) < 1e-9


def test_ndcg_no_relevant_returns_zero():
    # Empty qrels means IDCG=0 — guard against NaN.
    assert ndcg_at_k(RANKED, {}, 10) == 0.0


def test_evaluate_run_aggregates_macro_average():
    qrels_q = {"a": 1, "c": 1, "e": 1}
    run = {"q1": [(d, 1.0) for d in RANKED]}
    qrels = {"q1": qrels_q}
    agg = evaluate_run(run, qrels, ks=(10,))
    assert agg["recall@10"] == 1.0
    assert agg["mrr@10"] == 1.0
    # Two queries, one perfect, one empty -> mean is 0.5
    qrels2 = {"q1": qrels_q, "q2": {"x": 1}}
    run2 = {"q1": [(d, 1.0) for d in RANKED], "q2": []}
    agg2 = evaluate_run(run2, qrels2, ks=(10,))
    assert agg2["recall@10"] == 0.5
    assert agg2["mrr@10"] == 0.5