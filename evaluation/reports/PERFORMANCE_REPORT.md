# System Performance & Latency Report

**Evaluation Date**: August 14, 2026  
**Auditor Mode**: STRICT BENCHMARK-INTEGRITY REPAIR MODE  
**Environment**: Local x86_64 CPU (Single Node, Windows 11, PyTorch CPU Backend)  

---

## 1. Latency Profile by Architectural Stage

| Pipeline Stage | Implementation Component | P50 Latency (ms) | P95 Latency (ms) | Processing Throughput |
| :--- | :--- | :---: | :---: | :---: |
| **BM25 Lexical Retrieval** | In-memory Rank-BM25 | **3.11 ms** | **4.36 ms** | ~290–2,900 QPS |
| **Dense Vector Retrieval** | BAAI/bge-small-en-v1.5 (Cosine Sim) | **2.59 ms** | **3.79 ms** | ~350 QPS |
| **Hybrid RRF Fusion** | Reciprocal Rank Fusion ($k=60$) | **0.46 ms** | **0.70 ms** | >1,000 QPS |
| **Neural CrossEncoder Reranking** | `cross-encoder/ms-marco-TinyBERT-L-2-v2` | **3,770 ms** | **6,450 ms** | ~0.25 QPS (CPU sequential) |
| **Fixed Full Pipeline (Total)** | Planner + Hybrid + Rerank + Critic + Verifier | **3,775.92 ms** | **6,449.40 ms** | 4.0 LLM calls/query |
| **Adaptive Multi-Agent (Total)** | Dynamic Confidence Routing (L1 / L2 / L3) | **2,386.89 ms** | **4,391.88 ms** | **2.36 LLM calls/query** |

---

## 2. LLM Invocation & Resource Efficiency

| Pipeline Strategy | Avg LLM Invocations / Query | Invocations Reduction vs Fixed | Avg Estimated Tokens / Query |
| :--- | :---: | :---: | :---: |
| **Fixed Full Pipeline** | **4.00** | 0.0% (Baseline) | 2,080 tokens |
| **Adaptive Multi-Agent** | **2.36** | **-41.0%** | **1,227 tokens** |

> [!NOTE]
> **Audit Note on Efficiency Claims**:  
> The legacy claim of *"62.5% cost reduction"* is replaced with the empirically measured **41.0% LLM invocation reduction** (from 4.00 to 2.36 invocations/query).

---

## 3. Scaling Curve Across Contract Corpus Scales

**Run ID**: `microbench_run_20260814_180526_e5f66a`  

```
Corpus Scaling Curve (P50 Latency vs Contract Count):
------------------------------------------------------
1 Contract (16 chunks)   : 0.31 ms  [████████████████████████] 2,907 QPS
5 Contracts (71 chunks)  : 0.60 ms  [████████████] 1,545 QPS
10 Contracts (585 chunks): 3.45 ms  [███] 291 QPS
```

| Corpus Scale | Child Chunks | Parent Blocks | Parsing Time (s) | BM25 Index Time (s) | P50 (ms) | P95 (ms) | Memory (RAM) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 Contract** | 16 | 3 | 0.003s | 0.0025s | 0.31 ms | 0.64 ms | 180.0 MB |
| **5 Contracts** | 71 | 15 | 0.016s | 0.0131s | 0.60 ms | 0.99 ms | 180.0 MB |
| **10 Contracts** | 585 | 110 | 0.243s | 0.1093s | 3.45 ms | 5.46 ms | 180.0 MB |
