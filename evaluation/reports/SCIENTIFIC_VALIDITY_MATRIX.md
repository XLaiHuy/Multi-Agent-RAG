# Scientific Validity & Benchmark Evidence Matrix

**Repository:** `XLaiHuy/Multi-Agent-RAG`  
**Protocol Version:** `3.1.0`  
**Timestamp:** 2026-08-15 18:42:53Z

---

## 1. Comprehensive Metric Validity Classification

| Metric / Claim | Benchmark Dataset | N (Queries) | Mode | Split | Model Stack | Config Version | Gold Used Only for Scoring? | Classification Status |
|:---|:---|:---:|:---:|:---:|:---|:---:|:---:|:---:|
| **Zero Cross-Tenant Leakage (0.0%)** | Synthetic Tenant ACL Suite | 7 suites | Real Local | TEST | Anti-IDOR AuthGuard | v3.1 | YES | **CV_SAFE** |
| **CandidateHitRate@100 (73.68%)** | CUAD Official Manifest (10 contracts) | 19 ans. | Real Local | TEST | BGE-M3 + BM25 (RRF 60) | v3.1 | YES | **CV_SAFE** |
| **CandidateHitRate@100 (68.91%)** | CUAD DEV Manifest (20 contracts) | 238 ans. | Real Local | DEV | BGE-M3 + BM25 (RRF 60) | v3.1 | YES | **README_SAFE** |
| **TrueChunkRecall@100 (57.98%)** | CUAD DEV Manifest (20 contracts) | 238 ans. | Real Local | DEV | BGE-M3 + BM25 (RRF 60) | v3.1 | YES | **README_SAFE** |
| **Post-Rerank HitRate@1 (26.32%)** | CUAD Official Manifest (10 contracts) | 19 ans. | Real Local | TEST | TinyBERT CrossEncoder | v3.1 | YES | **README_SAFE** |
| **Post-Rerank HitRate@5 (31.58%)** | CUAD Official Manifest (10 contracts) | 19 ans. | Real Local | TEST | TinyBERT CrossEncoder | v3.1 | YES | **README_SAFE** |
| **CandidateHitRate@100 (48.98%)** | CUSTOM_CUAD_HOLDOUT_V2 (25 contracts) | 294 ans. | Real Local | TEST | BGE-M3 + BM25 + TinyBERT | v3.1 | YES | **README_SAFE** |
| **HitRate@5 (14.68%) & MRR (0.0896)** | CUSTOM_CUAD_HOLDOUT_V2 (25 contracts) | 294 ans. | Real Local | TEST | BGE-M3 + BM25 + TinyBERT | v3.1 | YES | **README_SAFE** |
| **Reranker CPU P50 Latency (4.34s)** | CUAD DEV & Locked Test | 238 / 19 | Real Local | DEV/TEST | TinyBERT (4 CPU threads) | v3.1 | YES | **CV_SAFE** |
| **100.0% Correct Refusal (31/31)** | Prior Agent Ablation Run | 31 unans. | Simulated | TEST | Simulated Verifier | Prior | NO (Leaked) | **INVALIDATED / NOT_CV_SAFE** |
| **40.5% LLM Invocation Reduction** | Prior Agent Ablation Run | 50 queries | Simulated | TEST | Simulated Router | Prior | NO (Leaked) | **INVALIDATED / NOT_CV_SAFE** |
| **94.74% Generation Faithfulness** | Prior Agent Ablation Run | 19 ans. | Simulated | TEST | Heuristic overlap | Prior | NO | **INVALIDATED / NOT_CV_SAFE** |
