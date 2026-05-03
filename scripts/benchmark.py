"""Latency benchmarks for the retrieval pipeline.

Measures wall-clock time on:
  - One-off indexing (BM25 + Dense)
  - Per-query first-stage retrieval at varying retrieve_k
  - Per-query cross-encoder reranking at varying pool sizes
  - End-to-end pipeline latency for each of the four configs

Sample size: N queries (default 50) drawn from the test set, M repetitions
(default 3) per measurement. Reports mean and standard deviation in
milliseconds.

Usage:
    python -m scripts.benchmark --dataset scifact --n-queries 50 --reps 3
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

from src.data import load_beir
from src.pipeline import Pipeline
from src.reranker import CrossEncoderReranker
from src.retrievers import BM25Retriever, DenseRetriever

import os
os.environ["OMP_NUM_THREADS"] = "1"


def time_call(fn, *, reps: int) -> tuple[float, float]:
    """Run fn() `reps` times, return (mean_ms, std_ms)."""
    timings = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        timings.append((time.perf_counter() - t0) * 1000.0)
    mean = statistics.mean(timings)
    std = statistics.stdev(timings) if len(timings) > 1 else 0.0
    return mean, std


def fmt(mean_std: tuple[float, float]) -> str:
    mean, std = mean_std
    return f"{mean:.1f} ± {std:.1f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-queries", type=int, default=50,
                        help="Number of queries to sample for per-query measurements")
    parser.add_argument("--reps", type=int, default=3,
                        help="Repetitions per query for variance estimation")
    parser.add_argument("--retrieve-ks", type=int, nargs="+",
                        default=[10, 50, 100, 200],
                        help="Candidate pool sizes to sweep")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    random.seed(args.seed)

    # ------------------------------------------------------------------
    # Load data + sample queries
    # ------------------------------------------------------------------
    corpus, queries, qrels = load_beir(args.dataset, split=args.split)
    qids = sorted(queries.keys())
    sample_qids = random.sample(qids, k=min(args.n_queries, len(qids)))
    sample_queries = {qid: queries[qid] for qid in sample_qids}
    print(f"Sampled {len(sample_queries)} queries; {args.reps} reps each.")

    out: dict = {
        "dataset": args.dataset,
        "n_queries": len(sample_queries),
        "reps": args.reps,
        "indexing_ms": {},
        "first_stage_ms_per_query": {},     # method -> retrieve_k -> {mean, std}
        "rerank_ms_per_query": {},          # pool_size -> {mean, std}
        "end_to_end_ms_per_query": {},      # method -> {mean, std}
    }

    # ------------------------------------------------------------------
    # Indexing (one-off; measured once each)
    # ------------------------------------------------------------------
    print("\n[1/4] Indexing")

    t0 = time.perf_counter()
    bm25 = BM25Retriever()
    bm25.index(corpus)
    out["indexing_ms"]["bm25"] = (time.perf_counter() - t0) * 1000.0
    print(f"  bm25  : {out['indexing_ms']['bm25']:.0f} ms")

    t0 = time.perf_counter()
    dense = DenseRetriever()
    dense.index(corpus)
    out["indexing_ms"]["dense"] = (time.perf_counter() - t0) * 1000.0
    print(f"  dense : {out['indexing_ms']['dense']:.0f} ms")

    print("  (loading reranker)")
    reranker = CrossEncoderReranker()

    # ------------------------------------------------------------------
    # First-stage retrieval at different retrieve_k
    # ------------------------------------------------------------------
    print("\n[2/4] First-stage retrieval (per query, varying retrieve_k)")
    out["first_stage_ms_per_query"]["bm25"] = {}
    out["first_stage_ms_per_query"]["dense"] = {}

    for k in args.retrieve_ks:
        bm25_per_q = []
        dense_per_q = []
        for _ in range(args.reps):
            for qid, q in sample_queries.items():
                t0 = time.perf_counter()
                bm25.retrieve(q, k=k)
                bm25_per_q.append((time.perf_counter() - t0) * 1000.0)

                t0 = time.perf_counter()
                dense.retrieve(q, k=k)
                dense_per_q.append((time.perf_counter() - t0) * 1000.0)
        bm25_stats = (statistics.mean(bm25_per_q), statistics.stdev(bm25_per_q))
        dense_stats = (statistics.mean(dense_per_q), statistics.stdev(dense_per_q))
        out["first_stage_ms_per_query"]["bm25"][str(k)] = {
            "mean": bm25_stats[0], "std": bm25_stats[1]
        }
        out["first_stage_ms_per_query"]["dense"][str(k)] = {
            "mean": dense_stats[0], "std": dense_stats[1]
        }
        print(f"  retrieve_k={k:>3}  bm25 {fmt(bm25_stats):>14}  dense {fmt(dense_stats):>14}  ms")

    # ------------------------------------------------------------------
    # Reranking at different pool sizes
    # ------------------------------------------------------------------
    print("\n[3/4] Cross-encoder reranking (per query, varying pool size)")
    for pool in args.retrieve_ks:
        rerank_per_q = []
        for _ in range(args.reps):
            for qid, q in sample_queries.items():
                cands = bm25.retrieve(q, k=pool)
                t0 = time.perf_counter()
                reranker.rerank(q, cands, corpus, top_k=pool)
                rerank_per_q.append((time.perf_counter() - t0) * 1000.0)
        rr_stats = (statistics.mean(rerank_per_q), statistics.stdev(rerank_per_q))
        out["rerank_ms_per_query"][str(pool)] = {"mean": rr_stats[0], "std": rr_stats[1]}
        print(f"  pool={pool:>3}  rerank {fmt(rr_stats):>14}  ms")

    # ------------------------------------------------------------------
    # End-to-end pipeline (default retrieve_k=100, top_k=10)
    # ------------------------------------------------------------------
    print("\n[4/4] End-to-end pipeline latency (retrieve_k=100, top_k=10)")
    pipelines = {
        "bm25": Pipeline(retriever=bm25, retrieve_k=100),
        "dense": Pipeline(retriever=dense, retrieve_k=100),
        "bm25+ce": Pipeline(retriever=bm25, reranker=reranker, retrieve_k=100),
        "dense+ce": Pipeline(retriever=dense, reranker=reranker, retrieve_k=100),
    }
    for name, pipe in pipelines.items():
        per_q = []
        for _ in range(args.reps):
            for qid, q in sample_queries.items():
                t0 = time.perf_counter()
                pipe.run({"_q": q}, corpus, top_k=10)
                per_q.append((time.perf_counter() - t0) * 1000.0)
        stats = (statistics.mean(per_q), statistics.stdev(per_q))
        out["end_to_end_ms_per_query"][name] = {"mean": stats[0], "std": stats[1]}
        print(f"  {name:<10} {fmt(stats):>14}  ms / query")

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.dataset}_latency.json"
    md_path = out_dir / f"{args.dataset}_latency.md"

    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    lines: list[str] = []
    lines.append(f"# Latency benchmarks — {args.dataset}\n")
    lines.append(
        f"Sampled {out['n_queries']} queries, {out['reps']} reps each. "
        "Times are wall-clock on the machine running this script.\n"
    )

    lines.append("## Indexing (one-off)\n")
    lines.append("| Component | Time (s) |")
    lines.append("|---|---|")
    lines.append(f"| BM25 | {out['indexing_ms']['bm25'] / 1000:.2f} |")
    lines.append(f"| Dense (encode + FAISS add) | {out['indexing_ms']['dense'] / 1000:.2f} |\n")

    lines.append("## First-stage retrieval (ms per query, mean ± std)\n")
    header = "| retrieve_k | " + " | ".join(["bm25", "dense"]) + " |"
    sep = "|" + "---|" * 3
    lines.append(header)
    lines.append(sep)
    for k in args.retrieve_ks:
        bs = out["first_stage_ms_per_query"]["bm25"][str(k)]
        ds = out["first_stage_ms_per_query"]["dense"][str(k)]
        lines.append(
            f"| {k} | {bs['mean']:.1f} ± {bs['std']:.1f} | {ds['mean']:.1f} ± {ds['std']:.1f} |"
        )
    lines.append("")

    lines.append("## Reranking (ms per query, mean ± std)\n")
    lines.append("| pool size | rerank time |")
    lines.append("|---|---|")
    for pool in args.retrieve_ks:
        rs = out["rerank_ms_per_query"][str(pool)]
        lines.append(f"| {pool} | {rs['mean']:.1f} ± {rs['std']:.1f} |")
    lines.append("")

    lines.append("## End-to-end pipeline (retrieve_k=100, top_k=10)\n")
    lines.append("| Method | ms / query |")
    lines.append("|---|---|")
    for name in ["bm25", "dense", "bm25+ce", "dense+ce"]:
        s = out["end_to_end_ms_per_query"][name]
        lines.append(f"| {name} | {s['mean']:.1f} ± {s['std']:.1f} |")
    lines.append("")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nWrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
