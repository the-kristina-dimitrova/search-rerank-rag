"""Wiring tests for the retrieve-then-rerank pipeline.

We use lightweight fakes (no torch/FAISS) to test that:
  - Stage 1 candidates are passed correctly to stage 2.
  - The reranker actually changes ordering.
  - top_k truncates the final list.
  - retrieve_k pulls a wider candidate pool than top_k when needed.
"""
from src.pipeline import Pipeline


class FakeRetriever:
    """Returns a fixed ranked list per query, ignoring the actual query text."""

    name = "fake"

    def __init__(self, ranked):
        # ranked: list[(doc_id, score)] sorted descending by score
        self._ranked = ranked
        self.last_k = None

    def retrieve_batch(self, queries, k=100):
        self.last_k = k
        return {qid: self._ranked[:k] for qid in queries}


class ReverseReranker:
    """Reverses whatever the first stage handed us — easy to assert against."""

    def rerank_batch(self, queries, candidates_per_query, corpus, top_k=None, show_progress=True):
        out = {}
        for qid, cands in candidates_per_query.items():
            reversed_cands = list(reversed(cands))
            if top_k is not None:
                reversed_cands = reversed_cands[:top_k]
            out[qid] = reversed_cands
        return out


def test_retriever_only():
    ranked = [("d1", 0.9), ("d2", 0.8), ("d3", 0.7)]
    pipe = Pipeline(retriever=FakeRetriever(ranked), reranker=None, retrieve_k=10)
    out = pipe.run({"q1": "anything"}, corpus={}, top_k=2)
    assert out["q1"] == [("d1", 0.9), ("d2", 0.8)]


def test_reranker_reorders_candidates():
    ranked = [("d1", 0.9), ("d2", 0.8), ("d3", 0.7)]
    pipe = Pipeline(
        retriever=FakeRetriever(ranked),
        reranker=ReverseReranker(),
        retrieve_k=10,
    )
    out = pipe.run({"q1": "anything"}, corpus={}, top_k=10)
    # Reverse reranker flips order — confirms candidates flowed through.
    assert [doc_id for doc_id, _ in out["q1"]] == ["d3", "d2", "d1"]


def test_top_k_truncates_after_reranking():
    ranked = [(f"d{i}", 1.0 - i * 0.1) for i in range(5)]
    pipe = Pipeline(
        retriever=FakeRetriever(ranked),
        reranker=ReverseReranker(),
        retrieve_k=10,
    )
    out = pipe.run({"q1": "anything"}, corpus={}, top_k=2)
    assert len(out["q1"]) == 2


def test_retrieve_k_wider_than_top_k():
    """Reranking benefits from a deeper candidate pool than the final top_k."""
    fake = FakeRetriever([(f"d{i}", 1.0 - i * 0.01) for i in range(100)])
    pipe = Pipeline(retriever=fake, reranker=ReverseReranker(), retrieve_k=50)
    pipe.run({"q1": "anything"}, corpus={}, top_k=10)
    assert fake.last_k == 50  # asked for 50 candidates, not 10


def test_top_k_above_retrieve_k_expands_pool():
    """If user wants top_k=200 but retrieve_k=100, we need to ask for >= 200."""
    fake = FakeRetriever([(f"d{i}", 1.0) for i in range(500)])
    pipe = Pipeline(retriever=fake, reranker=None, retrieve_k=100)
    pipe.run({"q1": "anything"}, corpus={}, top_k=200)
    assert fake.last_k == 200


def test_pipeline_name_reflects_components():
    fake = FakeRetriever([])
    assert Pipeline(retriever=fake, reranker=None).name == "fake"
    assert Pipeline(retriever=fake, reranker=ReverseReranker()).name == "fake+ce"
