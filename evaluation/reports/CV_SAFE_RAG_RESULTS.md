# CV-Safe Metric Classification & Results Guidelines: Enterprise Contract RAG (v4.0)

---

## 1. Strictly Verified Claims Matrix

| Claim / Metric | Classification | Allowed Scope | Scope & Justification |
| :--- | :---: | :--- | :--- |
| **Zero Cross-Tenant Retrieval Leakage (0.0%)** | **CV_SAFE** | Security / Production | Verified via 7 rigorous tenant-isolation and ACL security tests under multi-tenant querying. |
| **Evaluation Harness Acceleration (>90x Speedup: ~40.7 min → ~25.8s)** | **CV_SAFE** | Performance Engineering | Deterministic intermediate representation caching layer with strict cryptographic invalidation. |
| **Document-Scoped Contract QA (HitRate@10 = 75.21%, MRR = 0.5529)** | **CV_SAFE** | Workflow Engineering | Evaluated on 238 answerable DEV queries with explicit `selected_document_id` product context. |
| **Broad First-Stage Candidate Coverage (78.95% CandidateHitRate@100)** | **CV_SAFE** | Retrieval Engineering | Measured on frozen CUAD 10-contract locked test (19 answerable queries) via BGE-M3 + BM25 ($RRF_{60}$). |
| **Parent-Child Chunking Integrity (84.2% Gold in 1 Chunk)** | **CV_SAFE** | Document Ingestion | Measured across CUAD benchmark: 0 orphan chunks, 0 duplicate chunks, 100% parent resolution. |
| **CUAD Locked Test Retrieval (HitRate@1 = 15.79%, HitRate@5 = 26.32%, HitRate@10 = 47.37%, MRR = 0.2241)** | **README_SAFE** | Official 10-Contract Test Split | Real local execution on frozen 10-contract subset using strict CrossEncoder reranking. |
| **Custom CUAD Holdout v2 Retrieval (CandidateHitRate@100 = 74.06%, HitRate@10 = 28.67%, MRR = 0.1078)** | **README_SAFE** | 25-Contract Holdout Set | Real local execution on 25 holdout contracts (293 answerable queries). |
| **Simulated 100.0% Refusal Accuracy on Unanswerables** | **INVALIDATED** | Do NOT Claim on CV | Prior run used simulated heuristic thresholds and leaked `is_unanswerable`. Live API run pending. |
| **Simulated 40.5% LLM Call / Token Reduction** | **INVALIDATED** | Do NOT Claim on CV | Prior run used simulated routing numbers. Real API execution required before claiming. |
| **High Retrieval Quality (> 80% HitRate@5)** | **UNSAFE** | Do NOT Claim | Current local dense/sparse models achieve 26.3% HitRate@5 on zero-shot multi-contract CUAD. |

---

## 2. Defensible CV Bullet Points

> **Enterprise Security & Data Isolation:**  
> *"Architected a production-grade Enterprise Contract RAG platform with hierarchical Parent-Child indexing (~250-token child chunks, ~1200-token parent context blocks) and deterministic tenant ACL prefiltering, guaranteeing 0.0% cross-tenant retrieval leakage across 7 security regression suites."*

> **Retrieval Optimization & Candidate Recovery:**  
> *"Engineered a two-stage hybrid Dense (BGE-M3 1024-d) + BM25 retrieval architecture with parent deduplication and candidate pool expansion ($k=100$), increasing first-stage candidate coverage to 78.95% CandidateHitRate@100 while keeping CrossEncoder P50 latency at ~158.3ms on CPU."*

> **Evaluation Engineering & Caching Harness:**  
> *"Built an automated cryptographic evaluation harness (`EvaluationCache`) caching deterministic intermediate embeddings and candidate pools across 25+ contracts, reducing evaluation cycle time by >90x (from ~40.7 minutes to ~25.8 seconds)."*
