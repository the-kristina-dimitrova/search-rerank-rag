"""RAG layer: retrieve top passages, generate a grounded answer with citations.

Pluggable backends — pick at runtime:
  - "anthropic" : Claude via the Anthropic API (needs ANTHROPIC_API_KEY)
  - "gemini"    : Gemini via the Google GenAI SDK  (needs GEMINI_API_KEY)
  - "none"      : retrieval only, no generation (no key needed)

Adding a new provider means implementing the LLMBackend protocol — the rest
of the pipeline stays unchanged.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol

from .data import Corpus
from .pipeline import Pipeline


_SYSTEM_PROMPT = """You are a careful research assistant answering questions \
strictly from a set of provided passages.

Rules:
- Use ONLY information stated in the passages. Do not invent facts.
- Cite passages inline using [P1], [P2], etc. matching the passage labels in the prompt.
- Every factual claim should have at least one citation.
- If the passages do not contain enough information to answer, say so clearly \
and explain what is missing. Do not guess.
- Keep the answer concise and focused on the question."""


def build_user_prompt(query: str, passages: list[dict]) -> str:
    """Format query + numbered passages into a single user message."""
    blocks = []
    for i, p in enumerate(passages, start=1):
        title = p.get("title") or "(no title)"
        text = p.get("text") or ""
        blocks.append(f"[P{i}] {title}\n{text}")
    context = "\n\n".join(blocks)
    return (
        f"Question: {query}\n\n"
        f"Passages:\n{context}\n\n"
        "Answer the question using only the passages above. "
        "Cite supporting passages with [P1], [P2], etc."
    )


# ---------------------------------------------------------------------------
# Backend protocol — anything implementing .generate(...) is a valid backend
# ---------------------------------------------------------------------------


class LLMBackend(Protocol):
    name: str

    def generate(self, system: str, user: str, max_tokens: int) -> str:
        ...


class AnthropicBackend:
    """Claude via the Anthropic Python SDK."""

    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
    ):
        import anthropic  # lazy import — users without the SDK can still import this module

        self.model = model
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

    def generate(self, system: str, user: str, max_tokens: int) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(
            b.text for b in resp.content if getattr(b, "type", None) == "text"
        )


class GeminiBackend:
    """Gemini via the google-genai Python SDK."""

    name = "gemini"

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: str | None = None,
    ):
        from google import genai  # lazy import

        self.model = model
        self._client = genai.Client(
            api_key=api_key or os.environ.get("GEMINI_API_KEY")
        )

    def generate(self, system: str, user: str, max_tokens: int) -> str:
        from google.genai import types

        resp = self._client.models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
            ),
        )
        return resp.text or ""


class GroqBackend:
    """Llama 3.3 70B (and others) via the Groq API.

    Groq's free tier works globally, with no card required. Their hardware
    serves open-weight models with very low latency — usually <1s end-to-end
    on top of retrieval.
    """

    name = "groq"

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: str | None = None,
    ):
        from groq import Groq  # lazy import

        self.model = model
        self._client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))

    def generate(self, system: str, user: str, max_tokens: int) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


def make_backend(name: str, **kwargs) -> LLMBackend | None:
    """Factory: name -> backend instance. 'none' returns None (retrieval only)."""
    name = name.lower()
    if name in ("none", "off", "retrieval-only"):
        return None
    if name == "anthropic":
        return AnthropicBackend(**kwargs)
    if name == "gemini":
        return GeminiBackend(**kwargs)
    if name == "groq":
        return GroqBackend(**kwargs)
    raise ValueError(
        f"Unknown backend: {name!r}. Pick one of: anthropic, gemini, groq, none."
    )


# ---------------------------------------------------------------------------
# RAG response + orchestrator
# ---------------------------------------------------------------------------


@dataclass
class RAGResponse:
    query: str
    answer: str
    passages: list[dict] = field(default_factory=list)
    doc_ids: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    backend: str = ""

    def pretty(self) -> str:
        lines = [f"Q: {self.query}", "", f"A: {self.answer}"]
        if self.backend:
            lines.append(f"   (generated by: {self.backend})")
        lines += ["", "Sources:"]
        for i, (doc_id, p, score) in enumerate(
            zip(self.doc_ids, self.passages, self.scores), start=1
        ):
            title = p.get("title") or "(no title)"
            lines.append(f"  [P{i}] {doc_id}  score={score:.3f}  {title}")
        return "\n".join(lines)


class RAG:
    def __init__(
        self,
        pipeline: Pipeline,
        corpus: Corpus,
        backend: LLMBackend | None = None,
        max_tokens: int = 1024,
    ):
        self.pipeline = pipeline
        self.corpus = corpus
        self.backend = backend
        self.max_tokens = max_tokens

    def answer(self, query: str, top_k: int = 5, generate: bool = True) -> RAGResponse:
        """Retrieve and (optionally) generate."""
        run = self.pipeline.run({"_q": query}, self.corpus, top_k=top_k)
        hits = run["_q"]
        doc_ids = [d for d, _ in hits]
        scores = [s for _, s in hits]
        passages = [self.corpus[d] for d in doc_ids]

        if not generate or self.backend is None:
            return RAGResponse(
                query=query,
                answer="(generation skipped)",
                passages=passages,
                doc_ids=doc_ids,
                scores=scores,
                backend="" if self.backend is None else self.backend.name,
            )

        user_prompt = build_user_prompt(query, passages)
        answer_text = self.backend.generate(
            system=_SYSTEM_PROMPT, user=user_prompt, max_tokens=self.max_tokens
        )
        return RAGResponse(
            query=query,
            answer=answer_text,
            passages=passages,
            doc_ids=doc_ids,
            scores=scores,
            backend=self.backend.name,
        )
