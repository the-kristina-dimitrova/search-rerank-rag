"""End-to-end retrieve-then-rerank pipeline."""
from __future__ import annotations

from .data import Corpus
from .reranker import CrossEncoderReranker
from .retrievers import Retriever


class Pipeline:
    """Two-stage pipeline: first-stage retriever -> optional cross-encoder reranker.

    Standard pattern in production search:
      1. Retrieve N candidates with a fast lexical/dense method.
      2. Rerank those N pairs with a slower but more accurate cross-encoder.
      3. Return the top-k after reranking.

    The retrieve_k value (candidate pool size) matters: too small and you
    bottleneck on first-stage recall; too large and reranking gets slow.
    100 is a common sweet spot.
    """

    def __init__(
        self,
        retriever: Retriever,
        reranker: CrossEncoderReranker | None = None,
        retrieve_k: int = 100,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.retrieve_k = retrieve_k

    @property
    def name(self) -> str:
        if self.reranker is None:
            return self.retriever.name
        return f"{self.retriever.name}+ce"

    def run(
        self,
        queries: dict[str, str],
        corpus: Corpus,
        top_k: int = 100,
    ) -> dict[str, list[tuple[str, float]]]:
        """Retrieve, optionally rerank, return ranked (doc_id, score) lists."""

        first_k = max(self.retrieve_k, top_k)
        first_stage = self.retriever.retrieve_batch(queries, k=first_k)

        if self.reranker is None:
            return {qid: hits[:top_k] for qid, hits in first_stage.items()}

        reranked = self.reranker.rerank_batch(
            queries=queries,
            candidates_per_query=first_stage,
            corpus=corpus,
            top_k=top_k,
        )
        return reranked