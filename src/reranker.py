"""Second-stage cross-encoder reranker."""
from __future__ import annotations

from sentence_transformers import CrossEncoder
from tqdm.auto import tqdm

from .data import Corpus, doc_text


class CrossEncoderReranker:
    """Re-score (query, doc) pairs with a cross-encoder.

    Cross-encoders feed query and document through the same transformer,
    which is slow but produces sharper relevance scores than dual-encoder
    retrieval. The standard pattern is to retrieve N candidates cheaply,
    then rerank only those N pairs.

    Default model: ms-marco-MiniLM-L-6-v2 — small (~22M params), trained on
    MS MARCO passage ranking, runs comfortably on CPU.
    """

    name = "cross_encoder"

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name, max_length=max_length)

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, float]],
        corpus: Corpus,
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        """Rerank candidate (doc_id, score) pairs and return new sorted list."""
        if not candidates:
            return []
        doc_ids = [doc_id for doc_id, _ in candidates]
        pairs = [(query, doc_text(corpus[doc_id])) for doc_id in doc_ids]
        scores = self.model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )
        ranked = sorted(zip(doc_ids, scores.tolist()), key=lambda x: -x[1])
        if top_k is not None:
            ranked = ranked[:top_k]
        return [(doc_id, float(score)) for doc_id, score in ranked]

    def rerank_batch(
        self,
        queries: dict[str, str],
        candidates_per_query: dict[str, list[tuple[str, float]]],
        corpus: Corpus,
        top_k: int | None = None,
        show_progress: bool = True,
    ) -> dict[str, list[tuple[str, float]]]:
        """Rerank candidates for each query. Iterates query-by-query to keep
        memory bounded — each batch is one query's candidates."""
        out: dict[str, list[tuple[str, float]]] = {}
        iterator = queries.items()
        if show_progress:
            iterator = tqdm(iterator, total=len(queries), desc="Reranking")
        for qid, qtext in iterator:
            cands = candidates_per_query.get(qid, [])
            out[qid] = self.rerank(qtext, cands, corpus, top_k=top_k)
        return out