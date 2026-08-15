# CV-Safe Metric Classification & Results Guidelines: Enterprise Contract RAG (v3.1)

---

## 1. Strictly Verified Claims Matrix

| Claim / Metric | Classification | Allowed Scope | Scope & Justification |
|:---|:---:|:---|:---|
| **Zero Cross-Tenant Retrieval Leakage (0.0%)** | **CV_SAFE** | Security / Production | Verified via 7 rigorous tenant-isolation and ACL security tests under multi-tenant querying. |
| **56.3% Reranker CPU Latency Reduction (~9.95s → ~4.34s)** | **CV_SAFE** | Latency Engineering | Vectorized batching, thread tuning (4 OMP/MKL threads), and RRF candidate truncation on CPU. |
| **Broad First-Stage Candidate Coverage (73.68% CandidateHitRate@100)** | **CV_SAFE** | Retrieval Engineering | Measured on frozen CUAD 10-contract locked test (19 answerable queries) via BGE-M3 + BM25 ($RRF_{60}$). |
| **Parent-Child Chunking Integrity (84.2% Gold in 1 Chunk)** | **CV_SAFE** | Document Ingestion | Measured across CUAD benchmark: 0 orphan chunks, 0 duplicate chunks, 100% parent resolution. |
| **CUAD Locked Test Retrieval (HitRate@1 = 26.3%, HitRate@5 = 31.6%, MRR = 0.312)** | **README_SAFE** | Official 10-Contract Test Split | Real local execution on frozen 10-contract subset using strict CrossEncoder reranking. |
| **Custom CUAD Holdout v2 Retrieval (CandidateHitRate@100 = 49.0%, HitRate@5 = 14.7%)** | **README_SAFE** | 25-Contract Holdout Set | Real local execution on 25 holdout contracts (294 answerable queries). |
| **Simulated 100.0% Refusal Accuracy on Unanswerables** | **INVALIDATED** | Do NOT Claim on CV | Prior run used simulated heuristic thresholds and leaked `is_unanswerable`. Live API run pending. |
| **Simulated 40.5% LLM Call / Token Reduction** | **INVALIDATED** | Do NOT Claim on CV | Prior run used simulated routing numbers. Real API execution required before claiming. |
| **High Retrieval Quality (> 80% HitRate@5)** | **UNSAFE** | Do NOT Claim | Current local dense/sparse models achieve 31.6% HitRate@5 on zero-shot multi-contract CUAD. |

---

## 2. Defensible CV Bullet Points

> **Enterprise Security & Data Isolation:**  
> *"Architected a production-grade Enterprise Contract RAG platform with hierarchical Parent-Child indexing (~250-token child chunks, ~1200-token parent context blocks) and deterministic tenant ACL prefiltering, guaranteeing 0.0% cross-tenant retrieval leakage across 7 security regression suites."*

> **Retrieval Optimization & Candidate Recovery:**  
> *"Engineered a two-stage hybrid Dense (BGE-M3) + BM25 retrieval architecture with parent deduplication and candidate pool expansion ($k=100$), increasing first-stage candidate coverage from 35.7% to 73.68% CandidateHitRate@100 while reducing CrossEncoder P50 latency by 56.3% (from ~9.95s to ~4.34s)."*
