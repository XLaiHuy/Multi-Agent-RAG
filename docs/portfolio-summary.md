# Safe-RAG: Portfolio Technical Summary

This document presents a structured engineering overview of the Safe-RAG legal contract analysis system for technical recruiters, engineering managers, and ML/RAG practitioners.

---

## 1. Project Summary

Safe-RAG is a specialized Retrieval-Augmented Generation system designed for contract question answering and risk analysis, where cross-document hallucination, ambiguous refusal, and unsourced assertions carry legal risk.

The system is evaluated across two distinct, independently frozen benchmarks:
1. **Document-Scoped Hybrid Retrieval Benchmark**: $N = 294$ held-out answerable questions across 25 legal contracts.
2. **Real API End-to-End Generation Benchmark**: $N = 200$ stratified queries (100 Answerable, 100 Unanswerable) across 25 unseen contracts using real Google GenAI API calls (`gemma-4-26b-a4b-it`).

---

## 2. Architecture Overview

```
User Query ──► Document Boundary ──► Structure-Aware Chunker (Child ~250 tok / Parent ~1200 tok)
                                           │
       ┌───────────────────────────────────┴───────────────────────────────────┐
       ▼                                                                       ▼
BGE-M3 Dense Retrieval (Top-20)                             BM25Okapi Sparse Retrieval (Top-20)
       └───────────────────────────────────┬───────────────────────────────────┘
                                           ▼
                       Reciprocal Rank Fusion (k=60)
                                           ▼
                      TinyBERT CrossEncoder Reranker (Top-5)
                                           ▼
              Multi-Agent Stack: Planner ──► Critic ──► Generator ──► Verifier
                                           ▼
                     Verified Answer + In-Text Clause Citations
                                          or
                             INSUFFICIENT_EVIDENCE Refusal
```

---

## 3. Final Frozen Retrieval Metrics ($N = 294$ Answerable Queries)

- **Dataset**: CUAD Held-Out Split ($N = 294$ answerable questions from 25 contracts)
- **Strict Child HitRate@10**: **81.97%**
- **Strict Child HitRate@5**: **68.71%**
- **Mean Reciprocal Rank (MRR)**: **0.5214**
- **Parent Section HitRate@10**: **94.90%**
- **Online CPU Latency**: **586 ms P50** / **820 ms P95** (BGE-M3 dense search, BM25, RRF, TinyBERT reranking on 4 CPU threads)
- **Corpus-Wide Collision Penalty**: When searching across all 25 contracts without document scoping, HitRate@10 drops from 81.97% to **28.67%**, showing that document scoping substantially reduces cross-contract clause collisions under this evaluation protocol.

---

## 4. Final Frozen Real API Metrics ($N = 200$ Queries)

- **Dataset**: Custom CUAD Holdout v2 ($N = 200$: 100 Answerable, 100 Unanswerable from 25 unseen contracts)
- **Strict Balanced Answerability Accuracy**: **72.50%** (Sentinel-only prefix requirement)
- **Inclusive Balanced Answerability Accuracy**: **74.50%** (Prose-aware refusal accounting)
- **Strict Unanswerable Refusal Rate**: **78.00%** (78 / 100 strict sentinel refusals)
- **Inclusive Unanswerable Refusal Rate**: **82.00%** (82 / 100 total refusals: 78 strict + 4 prose)
- **Answerable Acceptance Rate**: **67.00%** (67 / 100 answered queries)
- **Valid Explicit Citation Compliance**: **98.51%** (84 / 85 accepted answers contain valid in-text citations)
- **Child Citation Hit Rate (accepted)**: **85.07%** (58 / 67 accepted answerable responses cite verified gold child clause)
- **End-to-End Child Citation Coverage**: **62.00%** (58 / 100 total answerable queries cite verified gold child clause)
- **Parent Citation Hit Rate (accepted)**: **92.54%** (63 / 67 accepted answerable responses cite verified parent section)
- **Parent Citation Coverage**: **68.00%** (63 / 100 total answerable queries cite verified parent section)
- **Citation Precision (Macro)**: **80.97%**
- **Citation Precision (Micro)**: **73.53%**
- **Citation Recall (Macro)**: **63.00%**
- **Wrong-Document Citations**: **0 / 140 observed** (0 cross-document contamination)
- **Invalid Citation Mentions**: **0 / 140 observed** (0 non-existent chunk IDs)
- **Real API Telemetry**: **3.42 calls/query**, **3,971.9 tokens/query**, **32.62s P50 latency**, **0 rate-limit errors**.

---

## 5. Judge-Based Metrics (`JUDGE-BASED`)

Evaluated by `gemma-4-26b-a4b-it` across all 85 accepted answers (100.0% coverage):
- **Grounded Material Claim Rate**: **97.93%** (142 / 145 material claims supported by retrieved context supplied to generator)
- **Unsupported Claim Rate**: **2.07%** (3 / 145 claims)
- **Contradicted Claims**: **0.00%** (0 / 145 claims)
- **Semantic Correctness**: **92.54%** (Mean score 1.85 / 2.0 against gold reference text)
- **Contradiction Rate (vs Gold)**: **1.49%** (1 / 67 accepted answerable responses)

---

## 6. Key Engineering Decisions

1. **Document-Scoped Indexing**: Enforcing query execution boundaries per contract prevents multi-contract keyword collisions and preserves clause precision.
2. **Structure-Aware Parent-Child Chunking**: Splitting into ~250-token child chunks for dense/sparse indexing, then expanding to ~1,200-token parent chunks for generator context, achieves optimal retrieval resolution without truncating context.
3. **Non-Parametric Rank Fusion**: RRF ($k=60$) balances dense semantic representations and exact sparse keyword matching without requiring task-specific weight tuning.
4. **Deterministic In-Text Citation Extraction**: Removing rank-based fallback ensures that citation compliance reflects model behavior rather than evaluation heuristics.

---

## 7. Evaluation Boundaries & Limitations

- **Official LegalBench-RAG Benchmark**: `NOT_RUN` — Evaluated on `CUSTOM_CUAD_HOLDOUT_V2`.
- **Groundedness / Semantic Correctness**: Evaluated via same-model LLM judge (`gemma-4-26b-a4b-it`).
- **Planner Causal Attribution**: Planner performed structured query analysis; retrieval query was passed verbatim (`PLANNER_PRESENT_NO_ISOLATED_CAUSAL_EFFECT`).
- **Real API Telemetry**: Telemetry measured directly via benchmark API client.

---

## 8. Recommended Resume / CV Bullets

### Primary Recommendation 1 (Retrieval Engineering):
> **"Engineered a document-scoped legal retrieval pipeline using BGE-M3, BM25, Reciprocal Rank Fusion, and CrossEncoder reranking, achieving 81.97% strict child HitRate@10 and 0.5214 MRR across 294 held-out CUAD queries from 25 contracts."**

### Primary Recommendation 2 (Systems & Evidence-Bounded Generation):
> **"Built an evidence-bounded multi-agent legal RAG system evaluated with real Google GenAI on 200 held-out queries, achieving 72.5% strict balanced answerability accuracy and 80.97% macro citation precision, with 0/140 wrong-document citations observed."**

---

## 9. Interview Talking Points

- **Why Document Scoping?**: In legal contract QA, users ask questions about an active agreement. Global corpus search drops HitRate@10 from 81.97% to 28.67% due to boilerplate overlap across agreements.
- **Why RRF over Simple Linear Weighted Sum?**: RRF is non-parametric and invariant to score distributions across dense cosine similarity and sparse BM25 scores.
- **Why Strict Child Mapping?**: Credit is granted only when the retrieved ~250-token chunk directly overlaps the gold annotated clause, avoiding inflated scores from sibling propagation.

- [Master Interview Study Guide & Deep-Dive Curriculum](interview-study-guide.md)
