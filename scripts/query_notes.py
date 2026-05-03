"""Query your study notes with RAG.

Usage:
    export GROQ_API_KEY=gsk_...
    python -m scripts.query_notes --query "What is VC dimension?"
    python -m scripts.query_notes                          # interactive REPL
    python -m scripts.query_notes --backend none --query "..."  # retrieval only
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
from pathlib import Path

from src.pipeline import Pipeline
from src.rag import RAG, make_backend
from src.reranker import CrossEncoderReranker
from src.retrievers import BM25Retriever, DenseRetriever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="indexes/unified")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieve-k", type=int, default=50)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--backend", default="groq",
                        choices=["groq", "anthropic", "gemini", "none"])
    parser.add_argument("--query", default=None)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    corpus_path = cache_dir / "corpus.json"

    if not corpus_path.exists():
        print("No notes indexed yet. Run:")
        print("  1. mkdir -p data/notes")
        print("  2. cp your_lectures.pdf data/notes/")
        print("  3. python -m scripts.build_notes")
        return

    print("Loading indexes...")
    with open(corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)
    print(f"  {len(corpus)} chunks")

    bm25 = BM25Retriever()
    bm25.load(cache_dir / "bm25")

    dense = DenseRetriever()
    dense.load(cache_dir / "dense")

    reranker = None if args.no_rerank else CrossEncoderReranker()

    pipeline = Pipeline(
        retriever=dense,
        reranker=reranker,
        retrieve_k=args.retrieve_k,
    )

    backend = make_backend(args.backend)
    rag = RAG(pipeline=pipeline, corpus=corpus, backend=backend)
    generate = backend is not None

    def run_query(q: str):
        resp = rag.answer(q, top_k=args.top_k, generate=generate)
        print(f"\n{resp.pretty()}\n")

    if args.query:
        run_query(args.query)
        return

    print("\nStudy notes RAG ready. Type a question (Ctrl-D to exit).")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q:
            run_query(q)


if __name__ == "__main__":
    main()
