# Locked Test V1 Retrieval Benchmark Report

**Dataset:** CUAD Locked Test V1 (10 Contracts, 19 Answerable Queries)  
**Dense Model:** `BAAI/bge-m3` (1024-d)  
**Reranker Model:** `cross-encoder/ms-marco-TinyBERT-L-2-v2` (strict=True, full text input)  
**Timestamp:** 2026-08-15 19:28:43Z  
**Runtime:** 1093.39s  

---

## 1. Measured Retrieval Performance

| Metric | Measured Value | Scope |
|:---|:---:|:---|
| **CandidateHitRate@100** | **78.95%** | First-Stage $RRF_{60}$ Broad Pool |
| **HitRate@1** | **15.79%** | Top-1 Exact Clause Accuracy |
| **HitRate@5** | **26.32%** | Top-5 Context Window |
| **HitRate@10** | **47.37%** | Top-10 Output |
| **MRR** | **0.2241** | Mean Reciprocal Rank |
