# CV-Safe & Scientifically Defensible RAG Evaluation Results

**Frozen as of Phase 4.2 Gate:** 2026-08-16  
**Status:** CV_SAFE, README_SAFE, AUDITED  

---

## 1. Verified Headline Metrics (CUSTOM_CUAD_HOLDOUT_V2, N=294)

| Metric Category | Metric Name | Measured Value | Scope & Evidence Definition |
| :--- | :--- | :---: | :--- |
| **Strict Child Retrieval** | **HitRate@10** | **81.97%** | Exact ~250-token child contains gold evidence |
| **Strict Child Retrieval** | **HitRate@5** | **68.71%** | Exact ~250-token child contains gold evidence |
| **Strict Child Retrieval** | **HitRate@1** | **39.12%** | Exact ~250-token child contains gold evidence |
| **Strict Child Retrieval** | **MRR** | **0.5214** | Strict Mean Reciprocal Rank |
| **Strict Child Retrieval** | **nDCG@5** | **0.4906** | Normalized Discounted Cumulative Gain |
| **Candidate Recovery** | **CandidateHitRate@20** | **92.86%** | BGE-M3 + BM25 + RRF Candidate Pool ($k=20$) |
| **Parent Context Recovery** | **ParentHitRate@10** | **94.90%** | Expanded parent context (~1200 tokens) contains clause |
| **Parent Context Recovery** | **ParentHitRate@5** | **83.67%** | Expanded parent context (~1200 tokens) contains clause |
| **Online Latency** | **Total Online P50 / P95** | **586 ms / 820 ms** | End-to-end CPU (including online BGE-M3 query embed) |
| **Post-Embed Latency** | **Post-Embedding P50** | **166 ms** | Scoped search + BM25 + RRF + TinyBERT |
| **Evaluation Acceleration** | **Cache Speedup** | **116.8x** | Cold 179.7s $	o$ Warm 1.54s (SHA-256 fingerprint verified) |
| **Tenant Isolation** | **Security Leakage** | **0.0%** | 7 / 7 ACL & security regression suites passing |

---

## 2. Superseded & Non-Claim Boundaries

- **Phase 4.1 Parent-Propagated Gold**: Hit@10 = 94.54% is **SUPERSEDED** by Phase 4.2 strict child Hit@10 = **81.97%**.
- **Phase 4.1 Post-Embedding Latency**: 68.89 ms is clarified as post-embedding, while true online latency is **586 ms P50**.
- **Real API End-to-End Generation**: Refusal rate, faithfulness (94.74%), and cost reduction (40.5%) remain **NOT_RUN**.
