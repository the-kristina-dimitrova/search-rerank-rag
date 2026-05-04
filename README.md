# search-rerank-rag

Two-stage retrieval pipeline (BM25 + dense + cross-encoder reranking) evaluated on BEIR/SciFact, with RAG over custom document collections.

## How it works

1. **Retrieve** — BM25 (lexical) or sentence-transformer + FAISS (semantic) pulls top-N candidates from a unified corpus
2. **Rerank** — cross-encoder rescores candidates for precision at the top of the list
3. **Generate** — LLM produces a grounded answer with passage citations

### Example 1 — BM25 wins on an exact ML theory term

**Query**  
> What is VC dimension?

#### BM25 (SciFact + my lecture notes)

**Generated answer (Groq)**  
> The VC dimension, also known as the Vapnik–Chervonenkis dimension, is defined as “the Vapnik–Chervonenkis dimension for Λ and h: VCdim(Λ) = max{m | ΠΛ(m) = 2^m}” [P1]. It measures the “expressibility” of the class Λ, or how many points can be “shattered” in an arbitrary way [P1]. A large VC dimension means a rich class that can overfit, while a small one means a restricted class with better generalization [P1].

Citation markers [P1], [P2]… refer to the passages below.

**Top passage — BM25**

[P1] **[lectures_en] 3 Growth Function and Vapnik–Chervonenkis Dimension**  
> Definition 3.2 (VCdim): The Vapnik–Chervonenkis dimension for Λ and h: VCdim(Λ) = max{m | ΠΛ(m) = 2^m}.  
> Remark 3.1: Intuition: VCdim measures the “expressibility” of the class Λ — how many points can be “shattered” in an arbitrary way. A large VCdim means a rich class (can overfit), a small one — a restricted class (better generalization). Practically: For a linear classifier in ℝ^d we have VCdim = d + 1 — i.e. the number of parameters (weights) gives a direct measure of complexity. For neural networks VCdim grows as O(|W| log |W|). [...]

BM25 jumps directly to my own ML theory lecture notes, which contain the formal definition, intuition, and examples. The answer is precise, uses standard notation, and cites the exact section I would use when explaining VC dimension in an exam setting.

#### dense + reranker (SciFact only)

**Generated answer (Groq)**  
> The VC dimension, also known as Vapnik–Chervonenkis dimension, is defined as the maximum number of points that can be “shattered” in an arbitrary way by a class of functions Λ [P2]. It is formally defined as VCdim(Λ) = max{m | Π(m) = 2^m}, where Π(m) is the growth function [P2]. [...]  

**Top passages — dense + CE**

[P1] **Bone loss and vascular calcification: A bi-directional interplay?** (irrelevant biomedical paper about vascular calcification “VC”)  
> Vascular calcification (VC) represents a recognized adverse predictor for cardiovascular morbidity and mortality. [...]  

[P2] **[lectures_en] 3 Growth Function and Vapnik–Chervonenkis Dimension** (same lecture notes as above, but assigned a lower score).

Dense retrieval over SciFact initially promotes an off‑topic biomedical article about **vascular calcification (VC)**, because “VC” appears frequently and is strongly associated with that abbreviation in the corpus. The cross‑encoder reranker corrects this partially by surfacing my lecture notes, but noise from the biomedical sense of “VC” makes the top of the ranking less clean than BM25 over the unified corpus.

#### Why BM25 is better here
In this case, a sparse lexical index that includes my own notes (BM25) is strictly better: it treats “VC dimension” as a rare, highly discriminative phrase and jumps straight to the correct mathematical definition, while dense retrieval struggles with an overloaded acronym in the SciFact abstracts.

### Example 2 — dense + reranker wins on a high-level biomedical question

**Query**  
> Is there a link between gut microbiome and obesity?

#### BM25 (for comparison)

**Generated answer (Groq)**  
> There is a link between the gut microbiome and obesity, as suggested by [P5], which states that changes in gut microbiota are accompanied by an increase in insulin resistance, glucose intolerance, and later, obesity in TLR2-deficient mice. Additionally, [P5] mentions that the sequence of events, including obesity, was reproduced in WT mice by microbiota transplantation and was also reversed by antibiotics, highlighting the role of gut microbiota in obesity. While [P1] does not directly discuss obesity, it mentions that the effect of gut microbiome composition on arterial stiffness is only minimally mediated by metabolic syndrome, which includes obesity as a component [P1].

**Top 5 passages — BM25**

[P1] *Gut microbial diversity is associated with lower arterial stiffness in women*  
> [...] Our aim was to examine whether gut microbial composition is associated with arterial stiffness. [...] Gut microbiome diversity is inversely associated with arterial stiffness in women. The effect of gut microbiome composition on PWV is only minimally mediated by MetS. [...]

[P5] … (TLR2-deficient mice, microbiota transplantation, obesity)

#### dense + cross-encoder

**Generated answer (Groq)**  
> Yes, there is a link between the gut microbiome and obesity. According to [P1], gut bacteria-derived lipopolysaccharide (LPS) can initiate obesity and insulin resistance in mice. Additionally, [P2] states that the presence of gut microbiota can induce insulin resistance and obesity in TLR2-deficient mice, characterized by changes in the gut microbiota and increased LPS absorption. [P3] also suggests that the intestinal immune system and gut microbiota play a role in metabolic disease, including obesity and insulin resistance. Furthermore, [P5] found that intestinal methane production, which is associated with the gut microbiome, is correlated with a higher body mass index (BMI) in obese individuals. Overall, these passages support the idea that the gut microbiome is linked to obesity, although the exact mechanisms and relationships are complex and multifaceted [P1, P2, P3, P5].

**Top 5 passages — dense + CE**

[P1] *Regulation of Serum Amyloid A3 (SAA3) in Mouse Colonic Epithelium and Adipose Tissue by the Intestinal Microbiota*  
> The gut microbiota has been proposed as an environmental factor that affects the development of metabolic and inflammatory diseases in mammals. Recent reports indicate that gut bacteria-derived lipopolysaccharide (LPS) can initiate obesity and insulin resistance in mice; however, the molecular interactions responsible for microbial regulation of host metabolism and mediators of inflammation have not been studied in detail. [...]

[P2] …  
[P3] …  
[P5] … (intestinal methane production correlated with BMI in obese individuals)

#### Why dense + reranker is better here

The query is broad and semantic rather than an exact claim from SciFact. Dense + cross-encoder surfaces passages that explicitly discuss **obesity and insulin resistance in relation to gut microbiota**, including mouse models where microbiota changes induce obesity and human data linking intestinal methane production to BMI. BM25, driven by exact term overlap, focuses on a paper about **arterial stiffness** whose connection to obesity is only indirect via metabolic syndrome, and mixes it with one relevant obesity-related passage. In this case, dense + CE provides a more on-topic evidence set and a clearer grounded answer.

## Results on SciFact (300 queries)

| Method | recall@10 | mrr@10 | ndcg@10 | recall@50 | mrr@50 | ndcg@50 | recall@100 | mrr@100 | ndcg@100 |
|---|---|---|---|---|---|---|---|---|---|
| BM25 | 0.7739 | 0.6312 | **0.6617** | 0.8686 | 0.6355 | 0.6846 | 0.8759 | 0.6356 | 0.6858 |
| dense | 0.7883 | 0.6068 | 0.6484 | 0.8893 | 0.6119 | 0.6723 | 0.9250 | 0.6123 | 0.6783 |
| BM25+reranker | 0.7861 | 0.6465 | 0.6741 | 0.8692 | 0.6509 | 0.6948 | 0.8759 | 0.6510 | 0.6959 |
| **dense+reranker** | **0.8089** | **0.6559** | **0.6868** | **0.9003** | **0.6604** | **0.7093** | **0.9250** | **0.6607** | **0.7134** |

BM25 nDCG@10 matches the [BEIR leaderboard](https://github.com/beir-cellar/beir) (~0.66). Reranking gains +0.038 on dense but only +0.012 on BM25 — first-stage recall@100 (0.925 vs 0.876) is the ceiling for reranked quality.

## Quick start

```bash
pip install -r requirements.txt

# Evaluate retrieval quality
python -m scripts.evaluate --dataset scifact

# Add your own documents and query them
mkdir -p data/notes && cp your_lectures.pdf data/notes/
python -m scripts.build_notes
python -m scripts.query_notes --query "What is VC dimension?"

# Interactive UI
streamlit run app.py
```

**Adding custom document collection:**Drop your PDF materials in data/notes/.

Then run:
```bash
python -m scripts.build_notes
```

## Key design choices

- **Metrics from scratch** — Recall@k, MRR@k, nDCG@k implemented and verified against hand-computed values, no pytrec_eval/Java dependency
- **Provider-agnostic RAG** — pluggable `LLMBackend` protocol with Groq, Gemini, and Anthropic backends; adding a new provider is ~15 lines
- **Section-aware PDF chunking** — splits by numbered headings to preserve mathematical context, not naive paragraph splitting
- **Index caching** — FAISS index persisted to disk after first build, sub-second loads on subsequent runs
- **CI with zero heavy deps** — `TYPE_CHECKING` guards keep torch/FAISS out of the test import graph; CI installs only pytest

## Repo layout

```
src/              Core library
  retrievers.py     BM25 + Dense (MiniLM/FAISS)
  reranker.py       Cross-encoder reranking
  pipeline.py       Retrieve → rerank orchestration
  evaluation.py     Offline metrics
  rag.py            Grounded generation with citations
  custom_data.py    PDF section chunker

scripts/          Entry points
  evaluate.py       Quality eval on BEIR datasets
  benchmark.py      Latency measurements with variance
  build_notes.py    Build unified corpus from SciFact + your PDFs
  query_notes.py    RAG demo (CLI)

tests/            Unit tests (17 tests, ~0.1s)
app.py            Streamlit UI
REPORT.md         Full technical writeup with analysis
```

📄 **[Full technical writeup → REPORT.md](REPORT.md)**
