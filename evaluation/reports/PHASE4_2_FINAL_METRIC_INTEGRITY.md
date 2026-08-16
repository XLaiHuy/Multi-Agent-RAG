# Phase 4.2 — Final Metric Integrity Gate & Benchmark Freeze

**Date:** 2026-08-16  
**Status:** COMPLETE & FROZEN  
**Benchmark Split:** `CUSTOM_CUAD_HOLDOUT_V2` ($N=294$ Answerable Queries, 25 Unseen Contracts)  
**Configuration Protocol:** `retrieval_metric_protocol_v4_2.json` (v4.2.0)  

---

## Executive Summary

Phase 4.2 closes the final metric-integrity risks identified during scientific auditing:
1. **Rebuilt Gold Mapping from Scratch**: Eliminated parent-to-sibling relevance leakage. A child chunk (~250 tokens) is relevant if and only if the gold evidence overlaps the child itself. Sibling chunks never inherit relevance.
2. **Measured True Online Retrieval Latency**: Timed end-to-end online query execution starting with runtime BGE-M3 query embedding on CPU through candidate generation and CrossEncoder reranking.
3. **Committed Deterministic Cache Benchmark**: Verified cold vs warm execution on an identical workload with exact SHA-256 result fingerprint matching.

---

## 1. Metric Comparison: Phase 4.1 vs Phase 4.2

| Evaluation Target | Metric | Phase 4.1 (Parent-Propagated Gold) | Phase 4.2 (Strict Child Gold) | Delta | Classification |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Child Candidate Pool** | **CandidateHitRate@20** | 98.29% | **92.86%** | $-5.43\%$ | Verified Candidate Recovery |
| **Child Chunk Retrieval** | **HitRate@1** | — | **39.12%** | — | Strict Child Precision |
| **Child Chunk Retrieval** | **HitRate@5** | 82.94% | **68.71%** | $-14.23\%$ | Strict Child Retrieval |
| **Child Chunk Retrieval** | **HitRate@10** | 94.54% | **81.97%** | $-12.57\%$ | Strict Child Retrieval |
| **Child Chunk Retrieval** | **MRR** | 0.6418 | **0.5214** | $-0.1204$ | Strict Mean Reciprocal Rank |
| **Child Chunk Retrieval** | **nDCG@5** | 0.3585 | **0.4906** | $+0.1321$ | Strict Child Ranking Quality |
| **Child Chunk Retrieval** | **TrueChunkRecall@10** | — | **69.04%** | — | Exact Evidence Coverage |
| **Parent Context Expansion** | **ParentHitRate@5** | — | **83.67%** | — | Hierarchical Parent Context |
| **Parent Context Expansion** | **ParentHitRate@10** | — | **94.90%** | — | Hierarchical Parent Context |

> **Scientific Finding**: When relevance is strictly confined to the exact ~250-token child chunk containing the evidence, the system achieves **81.97% Child HitRate@10** and **0.5214 MRR**. When evaluating at the hierarchical Parent Context level (~1,200 tokens expanded for LLM generation), the retriever recovers **94.90% ParentHitRate@10**.

---

## 2. Gold Evidence Mapping Audit

- **Total Answerable Queries on Holdout ($N=294$)**:
  - `MAPPED_CHILD_EXACT`: **290 queries** (98.6%)
  - `MAPPED_CHILD_SPAN_OVERLAP`: **4 queries** (1.4%)
  - `UNMAPPED`: **0 queries** (0.0%)
- **Average Relevant Child Chunks Per Query**: **3.31** (reduced from 7.88 in Phase 4.1).
- **Sibling Relevance Propagation**: **0% (Completely Eliminated)**.

---

## 3. End-to-End Online Latency Profile (CPU Only, 4 Threads)

Hardware: Intel/AMD x86_64 CPU (8 logical cores, PyTorch thread limit = 4, zero GPU).

| Processing Stage | Implementation | P50 (ms) | P95 (ms) | P99 (ms) | Mean (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **1. Online Query Embedding** | `BAAI/bge-m3` (570M, CPU) | **437.03** | **620.21** | 734.30 | 426.57 |
| **2. Document Scope Prefilter** | In-Memory Tenant/Doc Index | 0.02 | 0.06 | 0.13 | 0.03 |
| **3. Scoped Dense Search** | NumPy Slice Dot Product ($N=45-60$) | 0.55 | 1.53 | 3.40 | 0.81 |
| **4. Scoped BM25 Search** | BM25Okapi Document Index | 0.74 | 2.35 | 4.98 | 1.05 |
| **5. RRF Fusion & Dedup** | Reciprocal Rank Fusion ($k=60$) | 0.10 | 0.15 | 0.20 | 0.11 |
| **6. CrossEncoder Reranking** | `ms-marco-TinyBERT-L-2-v2` (top 20) | **164.84** | **255.84** | 320.60 | 152.99 |
| **Post-Embedding Retrieval** | Stages 2 through 6 | **166.35** | **267.04** | 322.83 | 155.04 |
| **TOTAL ONLINE RETRIEVAL** | **Stages 1 through 6 (End-to-End)** | **586.11** | **819.96** | **902.97** | **581.60** |

---

## 4. Evaluation Cache Speedup Benchmark

Reproduced via committed runner `evaluation/scripts/benchmark_eval_cache.py`:
- **Workload**: 25 CUAD DEV queries across 3 canonical contracts (90 chunks) evaluated under true document-scoped hybrid retrieval + TinyBERT reranker.
- **Cold Runtime (Seconds)**: **179.74 s** (Full canonical parsing + BGE-M3 document & query embedding + retrieval & reranking).
- **Warm Runtime (Seconds)**: **1.5384 s** (Loading cached embeddings + retrieval & reranking).
- **Measured Acceleration**: **116.84x Speedup**.
- **Cold Result SHA-256**: `666c72934e39b40b8ce9689eed398c51ac1c114e8dffe47e6c7478a470d42c1c`
- **Warm Result SHA-256**: `666c72934e39b40b8ce9689eed398c51ac1c114e8dffe47e6c7478a470d42c1c`
- **Result Identity Verified**: **YES (Exact Match)**.

---

## 5. Master Claim Classification

### CV_SAFE & README_SAFE (Defensible Headline Claims)
1. **Document-Scoped QA Accuracy**: **81.97% Strict Child HitRate@10**, **68.71% HitRate@5**, **0.5214 MRR** on held-out CUAD ($N=294$, 25 unseen contracts).
2. **Hierarchical Context Expansion**: **94.90% ParentHitRate@10** and **83.67% ParentHitRate@5** for LLM generation context.
3. **Measured Online Latency**: P50 = **586.11 ms**, P95 = **819.96 ms** on CPU (including online BGE-M3 query embedding).
4. **Evaluation Cache Acceleration**: **116.8x speedup** (Cold 179.7s $	o$ Warm 1.54s) with verified SHA-256 result fingerprint identity.
5. **Zero Tenant Leakage**: Observed zero cross-tenant retrieval leakage across 7 ACL regression suites.

### HISTORICAL_SUPERSEDED
- Phase 4.1 Parent-Propagated Child HitRate@10 (94.54%) $	o$ Superseded by Phase 4.2 strict child gold (**81.97%**).
- Phase 4.1 Post-Embedding Only Latency P50 (68.89 ms) $	o$ Clarified as Post-Embedding Latency, while End-to-End Online Latency is **586.11 ms**.

### INVALIDATED_PROHIBITED
- 100% unanswerable refusal accuracy (`REAL_API NOT_RUN`).
- 40.5% LLM cost reduction (`REAL_API NOT_RUN`).
- 94.74% generation faithfulness (`REAL_API NOT_RUN`).
- Official LegalBench-RAG results (`NOT_RUN`).
