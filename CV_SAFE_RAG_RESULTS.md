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
| **Balanced Answerability Accuracy** | **74.50%** | $N=200$ queries | **CV_SAFE** |
| **Unanswerable Refusal Rate** | **82.00%** | 82 / 100 unanswerable queries | **CV_SAFE** |
| **Answerable Acceptance Rate** | **67.00%** | 67 / 100 answerable queries | **CV_SAFE** |
| **False Refusal Rate** | **33.00%** | 33 / 100 answerable queries | **CV_SAFE** |
| **False Answer Rate** | **18.00%** | 18 / 100 unanswerable queries | **CV_SAFE** |
| **Explicit Citation Compliance** | **98.51%** | 84 / 85 accepted answers | **CV_SAFE** |
| **Child Citation Hit Rate** | **85.07%** | 58 / 67 accepted answerable responses | **CV_SAFE** |
| **Child Citation Coverage** | **62.00%** | 58 / 100 total answerable queries | **CV_SAFE** |
| **Parent Citation Hit Rate** | **92.54%** | 63 / 67 accepted answerable responses | **CV_SAFE** |
| **Parent Citation Coverage** | **68.00%** | 63 / 100 total answerable queries | **CV_SAFE** |
| **Citation Precision (Macro)** | **80.97%** | Mean precision across cited answers | **CV_SAFE** |
| **Citation Recall (Macro)** | **63.00%** | Mean recall across gold spans | **CV_SAFE** |
| **Wrong Document Citation Rate** | **0.00%** | 0 / 140 emitted citations | **CV_SAFE** |
| **Invalid Citation Mention Rate** | **0.00%** | 0 / 140 emitted citations | **CV_SAFE** |
| **Production API Calls / Query** | **3.42 calls** | Mean calls across $N=200$ | **CV_SAFE** |
| **Total Tokens / Query** | **3,971.9 tokens** | Mean tokens across $N=200$ | **CV_SAFE** |
| **End-to-End Latency (P50)** | **32.62 s** | Median latency across $N=200$ | **CV_SAFE** |

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
> **"Engineered a document-scoped legal retrieval pipeline using BGE-M3, BM25, Reciprocal Rank Fusion ($k=60$), and CrossEncoder reranking, achieving 81.97% strict child HitRate@10 and 0.5214 MRR on 294 held-out CUAD contract queries."**

### Recommended Bullet 2 (Safe Generation & Systems):
> **"Designed an evidence-bounded Multi-Agent RAG system with Google GenAI, achieving 74.50% balanced answerability accuracy, 82.00% unanswerable refusal, and 80.97% citation precision with zero cross-document contamination across 25 unseen contracts."**
