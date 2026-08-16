# Safe-RAG: CV & Portfolio Master Metric Registry

This document serves as the **Single Source of Truth** for all verified, reproducible metrics across the Safe-RAG legal contract analysis system.

---

## 1. Core Evaluation Matrix

### A. Document-Scoped Hybrid Retrieval (Phase 4.2 Canonical Baseline)
- **Dataset**: CUAD Held-Out Split ($N = 294$ answerable queries across 25 unseen contracts)
- **Retriever Configuration**: BGE-M3 (Dense) + BM25Okapi (Sparse) + Reciprocal Rank Fusion ($k=60$) + TinyBERT CrossEncoder Reranker
- **Chunk Hierarchy**: Child ~250 tokens / Parent ~1,200 tokens

| Metric | Target | Final Achieved | Verification Status |
|---|---|---|---|
| **Child HitRate@5** | $\ge 60.0\%$ | **68.71%** | **CV_SAFE** |
| **Child HitRate@10** | $\ge 75.0\%$ | **81.97%** | **CV_SAFE** |
| **Mean Reciprocal Rank (MRR)** | $\ge 0.450$ | **0.5214** | **CV_SAFE** |
| **Parent HitRate@10** | $\ge 90.0\%$ | **94.90%** | **CV_SAFE** |

---

### B. End-to-End Real API Generation & Citation Integrity (Phase 6.1 Frozen Benchmark)
- **Dataset**: Custom CUAD Holdout v2 ($N = 200$ stratified queries: 100 Answerable, 100 Unanswerable across 25 unseen contracts)
- **System Architecture**: Full Bounded Multi-Agent Stack (`FULL_BOUNDED_MULTI_AGENT`)
- **API Engine**: Real Google GenAI API (`gemma-4-26b-a4b-it`)
- **Protocol**: Strict Layer A execution (Zero Gold Access) / Strict Layer B offline rescoring (Zero Top-1 Fallback)

| Metric | Measured Value | Denominator Scope | Verification Status |
|---|---|---|---|
| **Strict Balanced Answerability Accuracy** | **72.50%** | Sentinel-only prefix requirement ($N=200$) | **CV_SAFE** |
| **Inclusive Balanced Answerability Accuracy** | **74.50%** | Prose-aware refusal accounting ($N=200$) | **CV_SAFE** |
| **Strict Unanswerable Refusal Rate** | **78.00%** | 78 / 100 strict sentinel refusals | **CV_SAFE** |
| **Inclusive Unanswerable Refusal Rate** | **82.00%** | 82 / 100 total unanswerable refusals | **CV_SAFE** |
| **Answerable Acceptance Rate** | **67.00%** | 67 / 100 answerable queries answered | **CV_SAFE** |
| **Valid Explicit Citation Compliance** | **98.51%** | 84 / 85 accepted answers contained valid citations | **CV_SAFE** |
| **Child Citation Hit Rate** | **85.07%** | 58 / 67 accepted answerable responses cite gold child | **CV_SAFE** |
| **End-to-End Child Citation Coverage** | **62.00%** | 58 / 100 total answerable queries cite gold child | **CV_SAFE** |
| **Parent Citation Hit Rate** | **92.54%** | 63 / 67 accepted answerable responses cite gold parent | **CV_SAFE** |
| **Parent Citation Coverage** | **68.00%** | 63 / 100 total answerable queries cite gold parent | **CV_SAFE** |
| **Citation Precision (Macro)** | **80.97%** | Mean precision across cited answers | **CV_SAFE** |
| **Citation Precision (Micro)** | **73.53%** | Micro-averaged citation precision | **CV_SAFE** |
| **Citation Recall (Macro)** | **63.00%** | Mean recall across gold spans | **CV_SAFE** |
| **Wrong-Document Citations** | **0 / 140 observed** | 0 cross-document citations observed | **CV_SAFE** |
| **Invalid Citation Mentions** | **0 / 140 observed** | 0 invalid reference indices or chunk IDs | **CV_SAFE** |
| **Production API Calls / Query** | **3.42 calls** | Mean calls across $N=200$ test queries | **CV_SAFE** |
| **Total Tokens / Query** | **3,971.9 tokens** | Mean tokens across $N=200$ test queries | **CV_SAFE** |
| **End-to-End Latency (P50)** | **32.62 s** | Median latency across $N=200$ test queries | **CV_SAFE** |

---

### C. Independent Blinded Judge Evaluation (LLM Judge)
- **Evaluator Model**: `gemma-4-26b-a4b-it` (100.0% coverage across 85 accepted answers)
- **Groundedness Scope**: Judged strictly against retrieved context supplied to the generator.
- **Semantic Correctness Scope**: Judged strictly against verified gold references.

| Metric | Measured Value | Scope | Classification |
|---|---|---|---|
| **Grounded Material Claim Rate** | **97.93%** | 142 / 145 material claims supported by retrieved context | **JUDGE_BASED** |
| **Unsupported Claim Rate** | **2.07%** | 3 / 145 claims | **JUDGE_BASED** |
| **Contradicted Claims** | **0.00%** | 0 / 145 claims | **JUDGE_BASED** |
| **Semantic Correctness** | **92.54%** | Mean score 1.85 / 2.0 against gold evidence | **JUDGE_BASED** |
| **Contradiction Rate (vs Gold)** | **1.49%** | 1 / 67 accepted answerable responses | **JUDGE_BASED** |

---

## 2. Recommended Resume / CV Bullets

### Recommended Bullet 1 (Retrieval Engineering):
> **"Engineered a document-scoped legal retrieval pipeline using BGE-M3, BM25, Reciprocal Rank Fusion, and CrossEncoder reranking, achieving 81.97% strict child HitRate@10 and 0.5214 MRR across 294 held-out CUAD queries from 25 contracts."**

### Recommended Bullet 2 (Systems & Evidence-Bounded Generation):
> **"Built an evidence-bounded multi-agent legal RAG system evaluated with real Google GenAI on 200 held-out queries, achieving 72.5% strict balanced answerability accuracy and 80.97% macro citation precision, with 0/140 wrong-document citations observed."**
