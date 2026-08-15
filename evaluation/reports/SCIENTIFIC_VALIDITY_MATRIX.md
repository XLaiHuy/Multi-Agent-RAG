# Scientific Validity & Benchmark Evidence Matrix

**Repository:** `XLaiHuy/Multi-Agent-RAG`  
**Protocol Version:** `4.0.0`  
**Timestamp:** 2026-08-16 00:00:00Z

---

## 1. Comprehensive Metric Validity Classification

| Metric / Claim | Benchmark Dataset | N (Queries) | Mode | Split | Model Stack | Config Version | Gold Used Only for Scoring? | Classification Status |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| **Zero Cross-Tenant Leakage (0.0%)** | Synthetic Tenant ACL Suite | 7 suites | Real Local | TEST | Anti-IDOR AuthGuard | v4.0 | YES | **CV_SAFE** |
| **Evaluation Acceleration (>90x Speedup)** | CUAD DEV Split (20 contracts) | 238 ans. | Real Local | DEV | `EvaluationCache` | v4.0 | YES | **CV_SAFE** |
| **Document-Scoped QA HitRate@10 (75.21%)** | CUAD DEV Split (20 contracts) | 238 ans. | Real Local | DEV | BGE-M3 + TinyBERT (Scoped) | v4.0 | YES (Explicit input) | **CV_SAFE** |
| **CandidateHitRate@100 (78.95%)** | CUAD Official Manifest (10 contracts) | 19 ans. | Real Local | TEST | BGE-M3 + BM25 (RRF 60) | v4.0 | YES | **CV_SAFE** |
| **Global Multi-Contract HitRate@10 (31.09%)** | CUAD DEV Split (20 contracts) | 238 ans. | Real Local | DEV | BGE-M3 + BM25 + TinyBERT | v4.0 | YES | **README_SAFE** |
| **CandidateHitRate@100 (77.31%)** | CUAD DEV Manifest (20 contracts) | 238 ans. | Real Local | DEV | BGE-M3 + BM25 (RRF 60) | v4.0 | YES | **README_SAFE** |
| **TrueChunkRecall@100 (29.93%)** | CUAD DEV Manifest (20 contracts) | 238 ans. | Real Local | DEV | BGE-M3 + BM25 (RRF 60) | v4.0 | YES | **README_SAFE** |
| **Locked Test v1 v4 HitRate@10 (47.37%)** | CUAD Official Manifest (10 contracts) | 19 ans. | Real Local | TEST | Frozen v4 Pipeline | v4.0 | YES | **README_SAFE** |
| **Holdout v2 v4 CandidateHitRate@100 (74.06%)** | CUSTOM_CUAD_HOLDOUT_V2 (25 contracts) | 293 ans. | Real Local | TEST | Frozen v4 Pipeline | v4.0 | YES | **README_SAFE** |
| **Holdout v2 v4 HitRate@10 (28.67%) & MRR (0.1078)** | CUSTOM_CUAD_HOLDOUT_V2 (25 contracts) | 293 ans. | Real Local | TEST | Frozen v4 Pipeline | v4.0 | YES | **README_SAFE** |
| **CrossEncoder CPU P50 Latency (~158.3ms)** | CUAD DEV & Locked Test | 238 / 19 | Real Local | DEV/TEST | TinyBERT (4 CPU threads) | v4.0 | YES | **CV_SAFE** |
| **100.0% Correct Refusal (31/31)** | Prior Agent Ablation Run | 31 unans. | Simulated | TEST | Simulated Verifier | Prior | NO (Leaked) | **INVALIDATED / NOT_CV_SAFE** |
| **40.5% LLM Invocation Reduction** | Prior Agent Ablation Run | 50 queries | Simulated | TEST | Simulated Router | Prior | NO (Leaked) | **INVALIDATED / NOT_CV_SAFE** |
| **94.74% Generation Faithfulness** | Prior Agent Ablation Run | 19 ans. | Simulated | TEST | Heuristic overlap | Prior | NO | **INVALIDATED / NOT_CV_SAFE** |
