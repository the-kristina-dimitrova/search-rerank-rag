"""Evaluate all retrieval configurations on a BEIR dataset.

Usage:
    python -m scripts.evaluate --dataset scifact --retrieve-k 100

Produces a Markdown table of metrics for:
  - BM25
  - Dense
  - BM25 + cross-encoder rerank
  - Dense + cross-encoder rerank

Outputs:
  - results/{dataset}_metrics.json — raw numbers
  - results/{dataset}_table.md     — formatted Markdown table

The dense index is cached to disk after the first build (default: indexes/).
Subsequent runs load from cache and skip the encoding step.
"""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
import time
from pathlib import Path

from src.data import load_beir
from src.evaluation import evaluate_run, format_results_table
from src.pipeline import Pipeline
from src.reranker import CrossEncoderReranker
from src.retrievers import BM25Retriever, DenseRetriever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--split", default="test")
    parser.add_argument("--retrieve-k", type=int, default=100,
                        help="Candidate pool size for reranking")
    parser.add_argument("--top-k", type=int, default=100,
                        help="Final ranked list size")
    parser.add_argument(
        "--encoder",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformer model for dense retrieval",
    )
    parser.add_argument(
        "--reranker",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Cross-encoder model for reranking",
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--cache-dir", default="indexes",
                        help="Directory for cached FAISS indexes (default: indexes/)")
    parser.add_argument("--limit-queries", type=int, default=None,
                        help="Evaluate on only the first N queries (for quick smoke tests)")
    args = parser.parse_args()
 
    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    corpus, queries, qrels = load_beir(args.dataset, split=args.split)
 
    if args.limit_queries is not None:
        qids = list(qrels.keys())[: args.limit_queries]
        qrels = {qid: qrels[qid] for qid in qids}
        queries = {qid: queries[qid] for qid in qids if qid in queries}
        print(f"Limiting to {len(queries)} queries for quick eval")
 
    # ------------------------------------------------------------------
    # Build retrievers
    # ------------------------------------------------------------------
    print("\n=== Indexing BM25 ===")
    t0 = time.time()
    bm25 = BM25Retriever()
    bm25.index(corpus)
    print(f"  BM25 indexed in {time.time() - t0:.1f}s")
 
    print("\n=== Indexing Dense ===")
    t0 = time.time()
    cache = Path(args.cache_dir) / args.dataset / "dense"
    dense = DenseRetriever(model_name=args.encoder)
    if cache.exists():
        print(f"  Loading from cache ({cache})...")
        dense.load(cache)
    else:
        dense.index(corpus)
        dense.save(cache)
        print(f"  Index saved to {cache}")
    print(f"  Dense ready in {time.time() - t0:.1f}s")
 
    print("\n=== Loading cross-encoder ===")
    reranker = CrossEncoderReranker(model_name=args.reranker)
 
    # ------------------------------------------------------------------
    # Run all four configurations
    # ------------------------------------------------------------------
    configs = {
        "bm25": Pipeline(retriever=bm25, retrieve_k=args.retrieve_k),
        "dense": Pipeline(retriever=dense, retrieve_k=args.retrieve_k),
        "bm25+ce": Pipeline(retriever=bm25, reranker=reranker, retrieve_k=args.retrieve_k),
        "dense+ce": Pipeline(retriever=dense, reranker=reranker, retrieve_k=args.retrieve_k),
    }
 
    runs: dict[str, dict] = {}
    timings: dict[str, float] = {}
    for name, pipeline in configs.items():
        print(f"\n=== Running {name} ===")
        t0 = time.time()
        run = pipeline.run(queries, corpus, top_k=args.top_k)
        elapsed = time.time() - t0
        runs[name] = run
        timings[name] = elapsed
        print(f"  {name} finished in {elapsed:.1f}s ({elapsed / max(len(queries), 1):.2f}s/query)")
 
    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    print("\n=== Computing metrics ===")
    ks = (10, 50, 100)
    results: dict[str, dict[str, float]] = {}
    for name, run in runs.items():
        results[name] = evaluate_run(run, qrels, ks=ks)
 
    # ------------------------------------------------------------------
    # Print + save
    # ------------------------------------------------------------------
    table = format_results_table(results, ks=ks)
    print("\n" + table + "\n")
 
    timing_table = "| Method | total (s) | per query (s) |\n|---|---|---|\n"
    for name, t in timings.items():
        timing_table += f"| {name} | {t:.1f} | {t / max(len(queries), 1):.3f} |\n"
    print("Latency:\n" + timing_table)
 
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.dataset}_metrics.json"
    md_path = out_dir / f"{args.dataset}_table.md"
    with open(json_path, "w") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "split": args.split,
                "n_queries": len(queries),
                "encoder": args.encoder,
                "reranker": args.reranker,
                "retrieve_k": args.retrieve_k,
                "metrics": results,
                "timings_seconds": timings,
            },
            f,
            indent=2,
        )
    with open(md_path, "w") as f:
        f.write(f"# Results on {args.dataset} ({args.split})\n\n")
        f.write(f"Queries scored: {len(queries)}\n\n")
        f.write(f"Encoder: `{args.encoder}`  \n")
        f.write(f"Reranker: `{args.reranker}`  \n")
        f.write(f"Candidate pool: top-{args.retrieve_k}\n\n")
        f.write("## Metrics\n\n")
        f.write(table + "\n\n")
        f.write("## Latency\n\n")
        f.write(timing_table + "\n")
 
    print(f"Wrote {json_path} and {md_path}")
 
 
if __name__ == "__main__":
    main()
 