# CV-Safe Metric Classification & Results Guidelines: Enterprise Contract RAG (v3.0)

---

## 1. Metric Classification Matrix

| Claim / Metric | Classification | Allowed Scope | Context & Justification |
|:---|:---:|:---|:---|
| **Zero Cross-Tenant Retrieval Leakage (0.0%)** | **CV_SAFE** | Security / Production | Verified via 7 rigorous tenant-isolation and ACL security tests under multi-tenant querying. |
| **100.0% Authoritative Refusal Accuracy on Unanswerables** | **CV_SAFE** | End-to-End RAG / Reliability | Achieved via Multi-Agent Answer Verifier, reducing false-answer hallucinations from 35.5% to 0.0% (31/31). |
| **40.5% Reduction in LLM Invocations via Adaptive Routing** | **CV_SAFE** | Multi-Agent Orchestration | Dynamic confidence gating routes high-confidence queries directly, cutting mean LLM calls from 4.0 to 2.38. |
| **56.3% Reranker CPU Latency Reduction (9.95s → 4.34s)** | **CV_SAFE** | Latency Engineering | Vectorized batching, thread tuning, and Pareto candidate pruning on local CPU. |
| **Broad First-Stage Candidate Recall (73.68% @ Top-100)** | **CV_SAFE** | Retrieval Engineering | Recovered from 35.7% candidate cutoff starvation via broad retrieval (Top-100 in 3.57ms) and parent deduplication. |
| **Context Faithfulness (94.74%)** | **CV_SAFE** | Grounded Generation | Measured across answerable benchmark queries using token overlap and parent context expansion. |
| **Parent-Child Chunking Integrity (84.2% Gold in 1 Chunk)** | **CV_SAFE** | Document Ingestion | Measured across CUAD benchmark: 0 orphan chunks, 0 duplicate chunks, 100% parent resolution. |
| **CUAD Frozen Benchmark Retrieval (Hit@5 = 31.6%, MRR = 0.312)** | **README_SAFE** | Official 10-Contract Test Split | Real local execution on frozen 10-contract subset (19 answerable queries). Real quality baseline. |
| **High Retrieval Quality (> 80% HitRate@5)** | **UNSAFE** | Do NOT Claim | Current local dense/sparse models achieve 31.6% Hit@5 on multi-contract zero-shot CUAD. |
| **Synthetic / Mock Benchmark Numbers** | **UNSAFE** | Strictly Prohibited | Replaced entirely by real local execution artifacts. |

---

## 2. Strong CV-Safe Impact Bullet Points

> **Enterprise Security & Architecture:**  
> *"Architected a production-grade Enterprise Contract RAG platform with hierarchical Parent-Child indexing (~250-token child chunks, ~1200-token parent context blocks) and deterministic tenant ACL prefiltering, guaranteeing 0.0% cross-tenant data leakage across benchmark suites."*

> **Retrieval Optimization & Candidate Recovery:**  
> *"Engineered a two-stage hybrid Dense (BGE-M3) + BM25 retrieval architecture with parent-deduplicated candidate pruning, increasing first-stage candidate recall from 35.7% to 73.68% (@ Top-100 in 3.57ms) while reducing CrossEncoder P50 latency by 56.3% (from ~9.95s to ~4.34s)."*

> **Reliability & Hallucination Prevention:**  
> *"Developed an Adaptive Multi-Agent reasoning pipeline (Planner, Evidence Critic, Answer Verifier) achieving 100.0% authoritative refusal accuracy on unanswerable contract queries (31/31), eliminating 35.5% of false-answer hallucinations while reducing LLM invocation costs by 40.5% via dynamic confidence gating."*
