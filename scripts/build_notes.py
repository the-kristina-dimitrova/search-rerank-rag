"""Build a unified corpus from SciFact + your PDFs, then index everything.

Usage:
    1. Drop PDFs into data/notes/
    2. Run: python -m scripts.build_notes
    3. Query: python -m scripts.query_notes  OR  streamlit run app.py

Adding new material: drop another PDF, re-run step 2. SciFact is always
included as a base so the system has broad scientific coverage. Your
lecture notes and papers are layered on top.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
from pathlib import Path

from src.custom_data import load_notes_folder
from src.data import load_beir, doc_text
from src.retrievers import BM25Retriever, DenseRetriever


def main():
    parser = argparse.ArgumentParser(
        description="Build unified corpus: SciFact + PDFs in data/notes/."
    )
    parser.add_argument("--notes-dir", default="data/notes",
                        help="Folder containing your PDF files (default: data/notes/)")
    parser.add_argument("--cache-dir", default="indexes/unified",
                        help="Where to save built indexes (default: indexes/unified/)")
    parser.add_argument("--skip-scifact", action="store_true",
                        help="Only index your PDFs, skip SciFact")
    parser.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()

    corpus: dict[str, dict[str, str]] = {}

    # ------------------------------------------------------------------
    # 1. SciFact base corpus
    # ------------------------------------------------------------------
    if not args.skip_scifact:
        print("Loading SciFact corpus...")
        sf_corpus, _, _ = load_beir("scifact", split="test")
        for doc_id, doc in sf_corpus.items():
            corpus[f"scifact:{doc_id}"] = {
                "title": doc.get("title") or "(no title)",
                "text": doc.get("text") or "",
            }
        print(f"  SciFact: {len(sf_corpus)} documents")

    # ------------------------------------------------------------------
    # 2. Your PDFs
    # ------------------------------------------------------------------
    notes = load_notes_folder(args.notes_dir)
    corpus.update(notes)

    if not corpus:
        print("\nNothing to index.")
        return

    # ------------------------------------------------------------------
    # 3. Save corpus + build indexes
    # ------------------------------------------------------------------
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = cache_dir / "corpus.json"
    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    print(f"\nCorpus saved: {len(corpus)} total chunks")

    print("\nBuilding BM25 index...")
    bm25 = BM25Retriever()
    bm25.index(corpus)
    bm25.save(cache_dir / "bm25")
    print("  BM25 done")

    print("\nBuilding Dense index (encoding all chunks)...")
    dense = DenseRetriever(model_name=args.encoder)
    dense.index(corpus)
    dense.save(cache_dir / "dense")
    print("  Dense done")

    # Count sources
    n_scifact = sum(1 for k in corpus if k.startswith("scifact:"))
    n_notes = len(corpus) - n_scifact
    pdf_count = len(list(Path(args.notes_dir).glob("*.pdf"))) if Path(args.notes_dir).exists() else 0

    print(f"\n{'='*50}")
    print(f"  Ready! {len(corpus)} chunks indexed:")
    print(f"    SciFact: {n_scifact} abstracts")
    print(f"    Notes:   {n_notes} sections from {pdf_count} PDF(s)")
    print(f"  Query: python -m scripts.query_notes")
    print(f"  Or:    streamlit run app.py")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
