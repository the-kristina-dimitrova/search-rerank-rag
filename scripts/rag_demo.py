"""Interactive RAG demo on the indexed BEIR corpus.

Usage:
    # With Groq (free, works globally — recommended)
    export GROQ_API_KEY=gsk_...
    python -m scripts.rag_demo --backend groq --query "..."

    # With Gemini (free key from https://aistudio.google.com/app/apikey,
    # may be region-restricted)
    export GEMINI_API_KEY=AIza...
    python -m scripts.rag_demo --backend gemini --query "..."

    # With Anthropic (requires billing)
    export ANTHROPIC_API_KEY=sk-ant-...
    python -m scripts.rag_demo --backend anthropic --query "..."

    # Retrieval only, no LLM call, no key needed
    python -m scripts.rag_demo --backend none --query "..."

The dense index is cached to disk after the first run (default: indexes/).
Subsequent runs skip the ~70s encoding step and load in under a second.
"""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
from pathlib import Path

from src.data import load_beir
from src.pipeline import Pipeline
from src.rag import RAG, make_backend
from src.reranker import CrossEncoderReranker
from src.retrievers import DenseRetriever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--split", default="test")
    parser.add_argument("--top-k", type=int, default=5,
                        help="How many passages to feed to the LLM")
    parser.add_argument("--retrieve-k", type=int, default=50,
                        help="First-stage candidate pool")
    parser.add_argument("--no-rerank", action="store_true",
                        help="Skip cross-encoder reranking")
    parser.add_argument(
        "--backend",
        default="groq",
        choices=["anthropic", "gemini", "groq", "none"],
        help="Which LLM backend to use. 'none' = retrieval only.",
    )
    parser.add_argument("--query", default=None,
                        help="Single query to run; if omitted, REPL")
    parser.add_argument("--cache-dir", default="indexes",
                        help="Directory for cached FAISS indexes (default: indexes/)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Build pipeline
    # ------------------------------------------------------------------
    corpus, _, _ = load_beir(args.dataset, split=args.split)

    cache = Path(args.cache_dir) / args.dataset / "dense"
    dense = DenseRetriever()
    if cache.exists():
        print(f"Loading dense index from cache ({cache})...")
        dense.load(cache)
    else:
        print("Building dense index (first run only — will be cached)...")
        dense.index(corpus)
        dense.save(cache)
        print(f"Index saved to {cache}")

    reranker = None if args.no_rerank else CrossEncoderReranker()

    pipeline = Pipeline(
        retriever=dense, reranker=reranker, retrieve_k=args.retrieve_k
    )

    backend = make_backend(args.backend)
    rag = RAG(pipeline=pipeline, corpus=corpus, backend=backend)

    generate = backend is not None
    if not generate:
        print("(retrieval-only mode — no answer will be generated)")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run_query(q: str) -> None:
        resp = rag.answer(q, top_k=args.top_k, generate=generate)
        print("\n" + resp.pretty() + "\n")

    if args.query:
        run_query(args.query)
        return

    print("\nRAG demo ready. Type a question (Ctrl-D to exit).")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        run_query(q)


if __name__ == "__main__":
    main()