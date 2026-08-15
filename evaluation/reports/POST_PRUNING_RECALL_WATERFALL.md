# Post-Pruning Recall & HitRate Waterfall Audit

**Evaluation Dataset:** CUAD DEV Split (20 Contracts, 238 Evaluated Answerable Queries)  
**Dense Model:** `BAAI/bge-m3` (1024-d)  
**Reranker Model:** `cross-encoder/ms-marco-TinyBERT-L-2-v2` (`strict=True`, `max_seq_length=512`, full text passed)  
**Reranker Failures:** `0`  
**Timestamp:** 2026-08-15 21:46:40Z  
**Runtime:** 1965.18s  

---

## 1. Step-by-Step Recall & Coverage Waterfall Funnel

| Funnel Stage | Description | CandidateHitRate (Any-Gold) | TrueChunkRecall (All-Gold) | HitRate Loss vs Prev | HitRate Retention vs Prev |
|:---|:---|:---:|:---:|:---:|:---:|
| **Stage 1** | Raw Top-100 First-Stage Retrieval ($RRF_{60}$) | **77.31%** | **29.93%** | Baseline | 100.0% |
| **Stage 2** | After Parent Deduplication (Max 2 chunks/parent) | **77.31%** | **24.66%** | +0.00% | 100.0% |
| **Stage 3** | After Top-20 Truncation (RRF-Order Budget Input) | **43.28%** | **10.64%** | +34.03% | 56.0% |
| **Stage 4** | After CrossEncoder Reranking (Top-10 Output) | **31.09%** | **7.08%** | +12.18% | 71.8% |
| **Stage 5** | Final Top-5 Context Window (Top-5 Output) | **20.17%** | **4.23%** | +10.92% | 46.6% |
