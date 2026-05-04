"""Streamlit UI for the unified retrieval + RAG system.

Run with:
    streamlit run app.py

Searches a single combined corpus (SciFact + your study PDFs).
Build the index first with: python -m scripts.build_notes
"""
from __future__ import annotations

import json
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import time
from contextlib import contextmanager
from pathlib import Path

import streamlit as st

from src.pipeline import Pipeline
from src.reranker import CrossEncoderReranker
from src.retrievers import BM25Retriever, DenseRetriever


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_CACHE_DIR = Path("indexes/unified")


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading cross-encoder reranker...")
def get_reranker():
    return CrossEncoderReranker()


@st.cache_resource(show_spinner="Loading corpus...")
def get_corpus():
    corpus_path = _CACHE_DIR / "corpus.json"
    if not corpus_path.exists():
        return None
    with open(corpus_path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner="Loading BM25 index...")
def get_bm25():
    bm25 = BM25Retriever()
    bm25.load(_CACHE_DIR / "bm25")
    return bm25


@st.cache_resource(show_spinner="Loading dense index...")
def get_dense():
    dense = DenseRetriever()
    dense.load(_CACHE_DIR / "dense")
    return dense


@contextmanager
def measure(store: dict, key: str):
    t0 = time.perf_counter()
    yield
    store[key] = time.perf_counter() - t0


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Retrieval + RAG", layout="wide")

corpus = get_corpus()

if corpus is None:
    st.title("Retrieval + RAG")
    st.error(
        "No index found. Build it first:\n\n"
        "```\n"
        "mkdir -p data/notes\n"
        "cp your_lectures.pdf data/notes/\n"
        "python -m scripts.build_notes\n"
        "```"
    )
    st.stop()

n_scifact = sum(1 for k in corpus if k.startswith("scifact:"))
n_notes = len(corpus) - n_scifact

st.title("Two-Stage Retrieval + RAG")
parts = []
if n_scifact:
    parts.append(f"{n_scifact:,} SciFact abstracts")
if n_notes:
    parts.append(f"{n_notes} study note sections")
st.caption(f"Searching {' + '.join(parts)}. Best results from any source.")

# --- Sidebar ---------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")

    method = st.selectbox(
        "Retrieval method",
        options=["bm25", "dense", "bm25+ce", "dense+ce"],
        index=3,
        format_func=lambda m: {
            "bm25": "BM25",
            "dense": "Dense (MiniLM + FAISS)",
            "bm25+ce": "BM25 → cross-encoder",
            "dense+ce": "Dense → cross-encoder",
        }[m],
    )

    top_k = st.slider("Passages to display", min_value=1, max_value=20, value=5)

    if "+ce" in method:
        retrieve_k = st.slider(
            "Candidate pool size (retrieve_k)",
            min_value=20, max_value=200, value=100, step=10,
        )
    else:
        retrieve_k = top_k

    st.divider()
    st.subheader("RAG")

    backend_choice = st.selectbox(
        "Backend",
        options=["none", "groq", "gemini", "anthropic"],
        index=1,
        format_func=lambda b: {
            "none": "None",
            "groq": "Groq ",
            "gemini": "Gemini ",
            "anthropic": "Anthropic Claude",
        }[b],
    )

    generate = backend_choice != "none"
    env_var = {
        "groq": "GROQ_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }.get(backend_choice)
    api_key_set = bool(env_var and os.environ.get(env_var))
    if generate and not api_key_set:
        st.warning(f"{env_var} not set; generation will fail.")

    st.divider()
    st.caption(f"Corpus: {len(corpus):,} chunks total")


# --- Build pipeline --------------------------------------------------------

def build_pipeline(m: str) -> Pipeline:
    bm25 = get_bm25()
    dense = get_dense()
    reranker = get_reranker() if "+ce" in m else None
    retriever = bm25 if m.startswith("bm25") else dense
    return Pipeline(retriever=retriever, reranker=reranker, retrieve_k=retrieve_k)


# --- Example queries -------------------------------------------------------

EXAMPLE_QUERIES = [
    "What is VC dimension?",
    "What is Rademacher complexity?",
    "Can a neural network approximate any function?",
    "Is there a link between gut microbiome and obesity?",
]

# --- Query input -----------------------------------------------------------

if "query" not in st.session_state:
    st.session_state.query = ""
if "submitted_query" not in st.session_state:
    st.session_state.submitted_query = ""

def _submit_from_input():
    st.session_state.submitted_query = st.session_state.query

def _submit_example(text: str):
    st.session_state.query = text
    st.session_state.submitted_query = text

col_q, col_btn = st.columns([5, 1])
with col_q:
    st.text_input(
        "Query",
        key="query",
        placeholder="Type a question, then press Enter…",
        label_visibility="collapsed",
        on_change=_submit_from_input,
    )
with col_btn:
    if st.button("Search", type="primary", use_container_width=True):
        st.session_state.submitted_query = st.session_state.query

with st.expander("Try an example query"):
    cols = st.columns(len(EXAMPLE_QUERIES))
    for col, ex in zip(cols, EXAMPLE_QUERIES):
        col.button(ex, key=f"ex_{ex}", on_click=_submit_example, args=(ex,))

query = st.session_state.submitted_query
if query:
    st.session_state.submitted_query = ""

# --- Search + display ------------------------------------------------------

if query and query.strip():
    pipeline = build_pipeline(method)
    timings: dict[str, float] = {}

    with measure(timings, "retrieve"):
        first_stage = pipeline.retriever.retrieve_batch(
            {"_q": query}, k=pipeline.retrieve_k
        )

    if pipeline.reranker is not None:
        with measure(timings, "rerank"):
            reranked = pipeline.reranker.rerank_batch(
                queries={"_q": query},
                candidates_per_query=first_stage,
                corpus=corpus,
                top_k=top_k,
                show_progress=False,
            )
        hits = reranked["_q"]
    else:
        hits = first_stage["_q"][:top_k]

    doc_ids = [d for d, _ in hits]
    scores = [s for _, s in hits]
    passages = [corpus[d] for d in doc_ids]

    answer_text = None
    backend_used = None
    if generate and api_key_set:
        from src.rag import _SYSTEM_PROMPT, build_user_prompt, make_backend

        backend = make_backend(backend_choice)
        backend_used = backend.name
        user_prompt = build_user_prompt(query, passages)
        with measure(timings, "generate"):
            answer_text = backend.generate(
                system=_SYSTEM_PROMPT, user=user_prompt, max_tokens=1024
            )

    # --- Display ---------------------------------------------------------
    st.divider()

    parts = [f"**{k}**: {v * 1000:.0f} ms" for k, v in timings.items()]
    parts.append(f"**total**: {sum(timings.values()) * 1000:.0f} ms")
    st.markdown(" · ".join(parts))

    if answer_text is not None:
        header = f"Generated answer ({backend_used})" if backend_used else "Generated answer"
        st.subheader(header)
        st.markdown(answer_text)
        st.caption("Citation markers `[P1]`, `[P2]`… refer to the passages below.")

    st.subheader(f"Top {len(hits)} passages — `{method}`")
    for i, (doc_id, score, p) in enumerate(zip(doc_ids, scores, passages), start=1):
        title = p.get("title") or "(no title)"
        # Show source tag (scifact vs lecture file name)
        source = doc_id.split(":")[0]
        with st.container(border=True):
            cols = st.columns([0.6, 0.4])
            cols[0].markdown(f"**[P{i}] {title}**")
            cols[1].markdown(
                f"<div style='text-align:right;font-family:monospace;color:#888'>"
                f"<span style='background:#333;padding:2px 6px;border-radius:3px'>{source}</span>"
                f" · score {score:.3f}</div>",
                unsafe_allow_html=True,
            )
            st.write(p.get("text", ""))
