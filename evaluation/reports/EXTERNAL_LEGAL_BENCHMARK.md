# Custom CUAD Holdout v2 Evaluation Report

> **Dataset Classification Notice:**  
> This benchmark evaluates a custom 25-contract holdout split from **CUAD v1** (`evaluation/manifests/cuad_locked_test_v2_manifest.json`).  
> It is designated as CUSTOM_CUAD_HOLDOUT_V2 (not an external third-party suite). Unverified external published baselines have been removed.

**Corpus Scope:** 25 Holdout Commercial Contracts (1,221 child chunks, 293 evaluated answerable queries)  
**Dense Model:** `BAAI/bge-m3` (1024-d)  
**Reranker Model:** `cross-encoder/ms-marco-TinyBERT-L-2-v2` (strict mode, full text input)  
**Timestamp:** 2026-08-15 20:08:24Z  
**Runtime:** 2359.18s  

---

## 1. Measured Performance on CUSTOM_CUAD_HOLDOUT_V2

| Metric | Measured Value | Scope / Definition |
|:---|:---:|:---|
| **CandidateHitRate@100** | **74.06%** | First-Stage $RRF_{60}$ Broad Pool ($k=100$) |
| **HitRate@1** | **4.78%** | Post-Rerank Top-1 Exact Accuracy |
| **HitRate@5** | **19.11%** | Post-Rerank Top-5 Context Window |
| **HitRate@10** | **28.67%** | Post-Rerank Top-10 Output |
| **MRR** | **0.1078** | Mean Reciprocal Rank over final reranked list |
