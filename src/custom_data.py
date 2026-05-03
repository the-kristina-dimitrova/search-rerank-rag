"""Unified corpus loader for personal study materials.

Design principle: one folder of PDFs → one combined corpus. Adding new
material means dropping a PDF in the folder and re-running the build
script. No code changes, no flags.

Each PDF is split into sections using numbered heading detection.
Section-level chunking preserves mathematical context that would be
lost with naive paragraph splitting.
"""
from __future__ import annotations

import re
from pathlib import Path

import pdfplumber

# Type alias — same format the rest of the pipeline expects.
Corpus = dict[str, dict[str, str]]  # doc_id -> {"title": str, "text": str}


# ---------------------------------------------------------------------------
# Section-aware PDF chunker
# ---------------------------------------------------------------------------

# Matches headings like "1 Introduction", "1.1 Algorithm", "10.3 The Function"
_SECTION_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+([A-Z][A-Za-z].*)", re.MULTILINE)


def _is_real_heading(sec_num: str, title: str) -> bool:
    """Filter out false positives: TOC entries, page numbers, math artifacts."""
    if ". . ." in title or "(cid:" in title:
        return False
    words = title.split()
    if len(words) < 2:
        return False
    if title.startswith(("See [", "Proof")):
        return False
    # When section number has no dot (e.g. "5", "13", "38"), it could be a
    # page number. Reject lines starting with common non-heading patterns.
    if "." not in sec_num:
        non_heading = (
            "Definition", "Lemma", "Theorem", "Proposition", "Corollary",
            "Remark", "Property", "Step", "From", "More", "Since", "Let",
            "We", "The real", "Therefore", "Empirical",
        )
        if title.startswith(non_heading):
            return False
    return True


def pdf_to_sections(pdf_path: str | Path, source_tag: str = "") -> Corpus:
    """Extract text from a PDF and split into sections by numbered headings.

    Args:
        pdf_path: Path to the PDF file.
        source_tag: Short string prepended to doc_ids for disambiguation
                    when combining multiple PDFs (e.g. "lectures_en").

    Returns:
        Corpus dict: {doc_id: {"title": str, "text": str}}
    """
    pdf_path = Path(pdf_path)
    tag = source_tag or pdf_path.stem

    # Skip first 3 pages (typically TOC / table of definitions) to avoid
    # duplicate heading detection from TOC entries.
    pages_text: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        toc_pages = min(3, len(pdf.pages))
        for page in pdf.pages[toc_pages:]:
            t = page.extract_text() or ""
            pages_text.append(t)
    full_text = "\n".join(pages_text)

    # Find all section headings and their character positions.
    headings: list[tuple[int, str, str]] = []
    for m in _SECTION_RE.finditer(full_text):
        sec_num = m.group(1)
        title_text = m.group(2).strip()
        if _is_real_heading(sec_num, title_text):
            headings.append((m.start(), sec_num, title_text))

    if not headings:
        # Fallback: treat entire document as one chunk.
        return {f"{tag}:full": {"title": pdf_path.name, "text": full_text.strip()}}

    corpus: Corpus = {}
    for i, (start, sec_num, title) in enumerate(headings):
        end = headings[i + 1][0] if i + 1 < len(headings) else len(full_text)
        chunk_text = full_text[start:end].strip()

        # Remove the heading line from the body to avoid duplication.
        lines = chunk_text.split("\n")
        body = "\n".join(lines[1:]).strip() if len(lines) > 1 else chunk_text

        if len(body) < 30:
            continue

        doc_id = f"{tag}:{sec_num}"
        full_title = f"[{tag}] {sec_num} {title}"
        corpus[doc_id] = {"title": full_title, "text": body}

    print(f"  {pdf_path.name}: {len(corpus)} section chunks")
    return corpus


# ---------------------------------------------------------------------------
# Folder scanner — the main entry point
# ---------------------------------------------------------------------------

def load_notes_folder(
    folder: str | Path = "data/notes",
) -> Corpus:
    """Scan a folder for PDFs and combine into a single corpus.

    Each PDF is chunked by section headings. Doc IDs are prefixed with
    the PDF filename (stem) so chunks from different files don't collide.

    Args:
        folder: Path to the folder containing PDF files.

    Returns:
        Combined Corpus dict.
    """
    folder = Path(folder)
    if not folder.exists():
        print(f"Notes folder does not exist: {folder}")
        print(f"Create it and drop your PDFs there:")
        print(f"  mkdir -p {folder}")
        return {}

    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        print(f"No PDF files found in {folder}/")
        return {}

    print(f"Found {len(pdfs)} PDF(s) in {folder}/:")
    corpus: Corpus = {}
    for pdf_path in pdfs:
        tag = pdf_path.stem.replace(" ", "_").lower()
        chunks = pdf_to_sections(pdf_path, source_tag=tag)
        corpus.update(chunks)

    print(f"\nTotal: {len(corpus)} chunks from {len(pdfs)} PDF(s)")
    return corpus
