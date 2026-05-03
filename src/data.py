"""Load BEIR datasets (default: SciFact)."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

from beir import util
from beir.datasets.data_loader import GenericDataLoader

# Standard BEIR mirror.
_BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


# Type aliases for clarity.
Corpus = dict[str, dict[str, str]]      # doc_id -> {"title": str, "text": str}
Queries = dict[str, str]                # query_id -> query text
Qrels = dict[str, dict[str, int]]       # query_id -> {doc_id: relevance}


def load_beir(
    name: str = "scifact",
    data_root: str | Path = "datasets",
    split: str = "test",
) -> Tuple[Corpus, Queries, Qrels]:
    """Download (if needed) and load a BEIR dataset.

    Args:
        name: BEIR dataset name (e.g. "scifact", "nfcorpus", "fiqa").
        data_root: Directory where datasets are cached.
        split: One of "train", "dev", "test".

    Returns:
        (corpus, queries, qrels) tuple.
    """
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    data_path = data_root / name

    if not data_path.exists():
        url = _BEIR_URL.format(name=name)
        print(f"Downloading {name} from {url}")
        util.download_and_unzip(url, str(data_root))

    corpus, queries, qrels = GenericDataLoader(data_folder=str(data_path)).load(split=split)
    print(
        f"Loaded {name}/{split}: "
        f"{len(corpus):,} docs, {len(queries):,} queries, {len(qrels):,} queries with qrels"
    )
    return corpus, queries, qrels


def doc_text(doc: dict[str, str]) -> str:
    """Concatenate title + text for a corpus entry."""
    title = doc.get("title", "") or ""
    text = doc.get("text", "") or ""
    return f"{title}. {text}".strip()