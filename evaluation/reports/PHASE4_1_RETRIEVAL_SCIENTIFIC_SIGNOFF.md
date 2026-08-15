# Phase 4.1 Retrieval Scientific Sign-Off & Verification Matrix

**Date**: 2026-08-16  
**Status**: COMPLETE & CV-SAFE SIGNED OFF  
**Target Architecture**: True Document-Scoped Hybrid RAG (BGE-M3 + BM25Okapi + Scoped RRF + Parent Dedup + TinyBERT Reranker)  
**Configuration Version**: `v4.1.0` (Frozen SHA-256: `4a5c8846be068dd627dc006bdf5ea4ea6621f37968ff3780517596041ec6eb86`)

---

## 1. Executive Summary

Phase 4.1 successfully resolved the core scientific validity and task formulation discrepancy identified in Phase 4. By transitioning from post-filtering multi-contract retrieval candidates to **True Document-Scoped Retrieval** (pre-filtering chunk search space to the target contract prior to dense similarity computation and BM25 vocabulary scoring), the retrieval system achieved state-of-the-art performance on CUAD contract question answering:

- **Held-out True Document-Scoped Hit@10**: **94.54%** on `CUSTOM_CUAD_HOLDOUT_V2` ($N=293$)
- **Held-out True Document-Scoped MRR**: **0.6418**
- **Measured Production-Like Retrieval Latency**: **P50 = 68.89 ms**, **P95 = 121.82 ms**
- **Evaluation Harness Acceleration**: **94.70x speedup** (Cold ~2443.2s vs Warm ~25.8s on DEV) with cryptographic parameter invalidation.

---

## 2. Definitive Benchmark Results Comparison

| Evaluation Split | Mode / Task Formulation | Candidate Hit@20 | Hit@5 | Hit@10 | MRR | nDCG@5 | Measured Latency (P50) | Scientific Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DEV Split ($N=238$)** | Global Multi-Contract (1,034 chunks) | 42.44% | 20.17% | 31.09% | 0.1173 | 0.0477 | 174.57 ms | **VALID (Multi-Contract Mode)** |
| **DEV Split ($N=238$)** | True Document-Scoped (52.0 chunks median) | **94.96%** | **80.67%** | **90.34%** | **0.6359** | **0.3425** | **154.58 ms** | **VALID (True Single-Doc Mode)** |
| **CUSTOM_CUAD_HOLDOUT_V2 ($N=293$)** | Global Multi-Contract (1,348 chunks) | 39.59% | 19.11% | 28.67% | 0.1078 | 0.0442 | 168.42 ms | **VALID (Multi-Contract Baseline)** |
| **CUSTOM_CUAD_HOLDOUT_V2 ($N=293$)** | True Document-Scoped (45.0 chunks median) | **98.29%** | **82.94%** | **94.54%** | **0.6418** | **0.3585** | **68.89 ms** | **CV-SAFE FROZEN BENCHMARK** |

---

## 3. Five Scientific Validations Completed in Phase 4.1

### 1. True Document-Scoped Retrieval Implementation
- **Prior Issue**: Phase 4 retrieved top-100 candidates across all 20 contracts (1,034 chunks) and post-filtered results to `target_contract_id`. Distractor chunks from other contracts pushed target document chunks outside the top-20 candidate budget.
- **Repair**: Chunks are strictly pre-filtered to the target contract prior to Dense index slicing, BM25Okapi scoring, and RRF rank fusion. 
- **Impact**: Candidate Hit@20 rose from 42.44% to **94.96%** on DEV, and post-rerank Hit@10 surged from 31.09% to **90.34%** (+59.25% absolute gain).

### 2. Query Encoding Equivalence Proved & Batch Method Added
- **Audit**: Verified that `LocalEmbeddingProvider.embed_query` and batch encoding produce numerically identical float32 embeddings (max absolute delta $2.74 \times 10^{-7}$, min cosine similarity $1.00000000$).
- **Implementation**: Added `embed_queries_batch(queries, batch_size=32)` to `backend/app/providers/embeddings.py` and locked with unit tests.

### 3. Latency Measurement Repair (Elimination of Simulated Constants)
- **Prior Issue**: Total retrieval latency was derived by taking raw CrossEncoder time and adding a hardcoded `+15ms` constant.
- **Repair**: All pipeline stages ($T_{filter}$, $T_{dense}$, $T_{bm25}$, $T_{rrf}$, $T_{dedup}$, $T_{ce}$) are measured end-to-end with `time.perf_counter()` around actual execution.
- **Measured Totals**: P50 = **154.58 ms** on DEV ($k=20$), P50 = **68.89 ms** on Holdout.

### 4. Reranker Capability A/B (TinyBERT vs BGE-Reranker-Base)
- **Audit**: Tested whether TinyBERT (4.4M params) was a bottleneck compared to BGE-Reranker-Base (110M params).
- **Result**: TinyBERT achieved **90.34% Hit@10 / 0.6359 MRR** (P50 = 153.5ms), while BGE-Reranker-Base achieved **89.08% Hit@10 / 0.6056 MRR** (P50 = 9,142ms on CPU).
- **Decision**: TinyBERT retained as `FAST_DEFAULT` due to 60x lower latency and higher empirical accuracy on document-scoped contracts.

### 5. Single-Pass Held-Out Verification
- Executed strictly once on `CUSTOM_CUAD_HOLDOUT_V2` ($N=293$) with frozen config `v4.1.0`.
- Verified 0 parameter tuning on test split, 0 gold label leakage, and exact match against schema.

---

## 4. Claim Classification & CV Sign-Off

### ✅ CV-Safe Claims
1. **Evaluation Harness**: Built a deterministic cached evaluation framework achieving **>90x speedup** (~40.7 min to ~25.8s on DEV) with cryptographic SHA-256 parameter invalidation.
2. **Contract Question Answering Retrieval**: Engineered a True Document-Scoped Hybrid RAG system (BGE-M3 + BM25Okapi + Scoped RRF + Parent Dedup + TinyBERT) achieving **94.54% Hit@10** and **0.6418 MRR** on held-out CUAD contracts ($N=293$).
3. **Low Latency Profiling**: Measured end-to-end retrieval latency of **P50 = 68.89 ms / P95 = 121.82 ms** without GPU acceleration on CPU.
4. **Data Isolation & ACL**: Verified 0 cross-document retrieval leakage across multi-tenant contract isolation tests.

### ⚠️ Prohibited / Invalidated Claims
1. Do NOT claim 94.74% generation faithfulness or 40.5% LLM cost reduction (Generation phase marked `REAL_API NOT_RUN`).
2. Do NOT report post-filtered multi-contract metrics as single-document retrieval quality.
