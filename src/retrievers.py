"""First-stage retrievers: BM25 (lexical) and Dense (sentence-transformer + FAISS)."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Protocol, runtime_checkable

import bm25s
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from .data import Corpus, doc_text


# ---------------------------------------------------------------------------
# Retriever protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Retriever(Protocol):
    """Common interface for first-stage retrievers."""

    name: str

    def index(self, corpus: Corpus) -> None:
        ...

    def retrieve(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        """Return top-k (doc_id, score) pairs sorted by descending score."""
        ...

    def retrieve_batch(
        self, queries: dict[str, str], k: int = 100
    ) -> dict[str, list[tuple[str, float]]]:
        ...


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------


class BM25Retriever:
    """BM25 retriever backed by `bm25s` (fast pure-Python implementation).

    Uses the default Okapi BM25 formula with k1=1.5, b=0.75. Stopwords are
    removed, tokens are lowercased.
    """

    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._retriever: bm25s.BM25 | None = None
        self._doc_ids: list[str] = []

    def index(self, corpus: Corpus) -> None:
        self._doc_ids = list(corpus.keys())
        texts = [doc_text(corpus[doc_id]) for doc_id in self._doc_ids]
        tokens = bm25s.tokenize(texts, stopwords="en", show_progress=False)
        self._retriever = bm25s.BM25(k1=self.k1, b=self.b)
        self._retriever.index(tokens, show_progress=False)

    def retrieve(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        return self.retrieve_batch({"_q": query}, k=k)["_q"]

    def retrieve_batch(
        self, queries: dict[str, str], k: int = 100
    ) -> dict[str, list[tuple[str, float]]]:
        if self._retriever is None:
            raise RuntimeError("Call .index(corpus) first.")
        qids = list(queries.keys())
        qtexts = [queries[qid] for qid in qids]
        qtokens = bm25s.tokenize(qtexts, stopwords="en", show_progress=False)
        # Cap k at corpus size — bm25s raises otherwise.
        k_eff = min(k, len(self._doc_ids))
        results, scores = self._retriever.retrieve(qtokens, k=k_eff, show_progress=False)
        out: dict[str, list[tuple[str, float]]] = {}
        for i, qid in enumerate(qids):
            out[qid] = [
                (self._doc_ids[int(idx)], float(scores[i, j]))
                for j, idx in enumerate(results[i])
            ]
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        assert self._retriever is not None
        self._retriever.save(str(path))
        with open(path / "doc_ids.pkl", "wb") as f:
            pickle.dump(self._doc_ids, f)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        self._retriever = bm25s.BM25.load(str(path))
        with open(path / "doc_ids.pkl", "rb") as f:
            self._doc_ids = pickle.load(f)


# ---------------------------------------------------------------------------
# Dense retrieval
# ---------------------------------------------------------------------------


class DenseRetriever:
    """Dense retriever: sentence-transformer encoder + FAISS exact inner product.

    For SciFact's ~5K docs, exact search (IndexFlatIP) is fine. If you swap in
    a larger corpus, replace with IVF/HNSW. We L2-normalise embeddings so
    inner product equals cosine similarity.
    """

    name = "dense"

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device="cpu")
        self._index: faiss.Index | None = None
        self._doc_ids: list[str] = []
        self._dim: int | None = None

    def _encode(self, texts: list[str], desc: str = "Encoding") -> np.ndarray:
        embs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        return np.ascontiguousarray(embs, dtype=np.float32)

    def index(self, corpus: Corpus) -> None:
        self._doc_ids = list(corpus.keys())
        texts = [doc_text(corpus[doc_id]) for doc_id in self._doc_ids]
        embs = self._encode(texts, desc="Encoding corpus")
        self._dim = embs.shape[1]
        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(embs)

    def retrieve(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        return self.retrieve_batch({"_q": query}, k=k)["_q"]

    def retrieve_batch(
        self, queries: dict[str, str], k: int = 100
    ) -> dict[str, list[tuple[str, float]]]:
        if self._index is None:
            raise RuntimeError("Call .index(corpus) first.")
        qids = list(queries.keys())
        qtexts = [queries[qid] for qid in qids]
        qembs = self._encode(qtexts, desc="Encoding queries")
        k_eff = min(k, len(self._doc_ids))
        scores, indices = self._index.search(qembs, k_eff)
        out: dict[str, list[tuple[str, float]]] = {}
        for i, qid in enumerate(qids):
            out[qid] = [
                (self._doc_ids[int(idx)], float(scores[i, j]))
                for j, idx in enumerate(indices[i])
                if idx != -1
            ]
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        assert self._index is not None
        faiss.write_index(self._index, str(path / "index.faiss"))
        with open(path / "doc_ids.pkl", "wb") as f:
            pickle.dump({"doc_ids": self._doc_ids, "model_name": self.model_name}, f)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        self._index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "doc_ids.pkl", "rb") as f:
            meta = pickle.load(f)
        self._doc_ids = meta["doc_ids"]
        if meta["model_name"] != self.model_name:
            print(
                f"Warning: index was built with {meta['model_name']!r} "
                f"but current model is {self.model_name!r}"
            )