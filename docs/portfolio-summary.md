# Portfolio Summary: Enterprise Multi-Agent Contract Intelligence Platform

**Author:** AI Engineer Portfolio  
**Repository:** [XLaiHuy/Multi-Agent-RAG](https://github.com/XLaiHuy/Multi-Agent-RAG)  
**Status:** Phase 4.2 Frozen Scientific Evaluation  

---

## Executive Summary for Recruiters & Interviewers

This project implements a **production-grade, tenant-isolated contract intelligence platform** featuring:
1. **Document-Scoped Hybrid Retrieval**: Combines dense semantic search (`BAAI/bge-m3`), sparse keyword retrieval (`BM25Okapi`), Reciprocal Rank Fusion ($k=60$), and lightweight CrossEncoder reranking (`ms-marco-TinyBERT-L-2-v2`).
2. **Hierarchical Structure-Aware Chunking**: Creates ~250-token child chunks for precision indexing and ~1,200-token parent chunks for context synthesis.
3. **Empirically Defensible Accuracy**: Evaluated on $N=294$ answerable queries across 25 unseen contracts (`CUSTOM_CUAD_HOLDOUT_V2`) under strict child gold evidence mapping (zero parent-to-sibling leakage):
   - **Strict Child HitRate@10**: **81.97%**
   - **Strict Child HitRate@5**: **68.71%**
   - **Strict Child MRR**: **0.5214**
   - **Parent Context HitRate@10**: **94.90%**
4. **Sub-Second Online CPU Latency**: End-to-end online query execution (including BGE-M3 query embedding + search + TinyBERT rerank) completes in **586 ms P50** / **820 ms P95** on standard CPU.
5. **Deterministic Cache Acceleration**: Accelerates repeated evaluation runs by **116.8x** (Cold 179.7s $	o$ Warm 1.54s) while proving exact float32 result identity via SHA-256 fingerprint matching.

---

## Top 5 Architectural & Engineering Decisions

1. **Document-Scoped QA Formulation vs Global Search**
   - *Problem*: Global multi-contract vector search on legal contracts suffers severe cross-agreement clause collisions, yielding only 28.67% Hit@10.
   - *Decision*: Bound retrieval slices to the user's active agreement, achieving **81.97% Strict Child Hit@10** and **94.90% Parent Hit@10**.

2. **Hierarchical Parent-Child Indexing & Strict Evidence Evaluation**
   - *Problem*: Large chunks dilute dense vectors; small chunks truncate vital surrounding legal clauses.
   - *Decision*: Index child chunks (~250 tokens) for search, expand to parent chunks (~1,200 tokens) for synthesis. Evaluated under strict child evidence matching to eliminate sibling false positives.

3. **TinyBERT FAST_DEFAULT vs Heavy Cross-Encoders**
   - *Problem*: 110M parameter rerankers (`bge-reranker-base`) introduce 9+ seconds CPU latency.
   - *Decision*: Selected `ms-marco-TinyBERT-L-2-v2` (4.4M params), reducing reranker latency to 164 ms (60x faster) with identical top-10 accuracy.

4. **Deterministic Evaluation Caching with Fingerprint Identity**
   - *Problem*: Repeated parameter sweeps spent 40+ minutes recomputing identical embeddings.
   - *Decision*: Built a cryptographically keyed cache hashing 11 pipeline parameters, reducing runtime by **116.8x** with verified SHA-256 ranking identity.

5. **Tenant Isolation & Security-First Ingestion**
   - *Problem*: Multi-tenant RAG systems risk IDOR attacks and cross-tenant leakage.
   - *Decision*: Mandatory tenant prefiltering enforced at query entry; verified with zero leakage across 7 security regression suites.

---

## Verified Resume / CV Bullets

### 1. Retrieval & Machine Learning Focus
> *"Architected a document-scoped hybrid legal retriever combining BGE-M3, BM25, RRF, and CrossEncoder reranking, achieving 81.97% strict child HitRate@10, 94.90% parent clause recovery, and 0.5214 MRR on 294 held-out CUAD queries across 25 unseen contracts."*

### 2. Systems Performance & Latency Focus
> *"Optimized end-to-end CPU retrieval and reranking latency to 586 ms P50 / 820 ms P95 (including online BGE-M3 query embedding), and engineered a deterministic caching framework delivering 116.8x evaluation speedup with verified result fingerprint matching."*

### 3. Full-Stack & Enterprise Security Focus
> *"Developed a tenant-aware enterprise contract intelligence platform with FastAPI, React, hierarchical parent-child indexing, bounded 3-agent verification, and zero cross-tenant retrieval leakage across 7 security regression suites."*

---

## 5 Technical Interview Talking Points

1. **How do you prevent cross-contract clause collisions in legal RAG?**  
   *Answer*: In legal contracts, boilerplates like "Governing Law" or "Termination" appear identically across contracts. By enforcing pre-retrieval document scoping to the active contract ID, search precision jumps from 28.67% to 81.97% Hit@10.

2. **Why separate Child HitRate from Parent HitRate?**  
   *Answer*: A child chunk (~250 tokens) contains the exact sentence match. Sibling child chunks sharing the same parent contain unrelated text. Strict child metrics ensure the retriever hits the exact clause, while parent metrics measure synthesis context adequacy.

3. **How was sub-second online CPU latency achieved?**  
   *Answer*: BGE-M3 encodes the single incoming query in 437 ms on CPU. Document-scoped vector dot products and BM25 take $<2$ ms on 50 chunks. TinyBERT (4.4M parameters) reranks top-20 candidates in 164 ms, yielding 586 ms P50 total.

4. **How does evaluation caching guarantee scientific validity?**  
   *Answer*: Cache keys hash manifest checksums, chunk sizes, and embedding models. Cold vs warm outputs are SHA-256 fingerprinted, guaranteeing identical rankings while eliminating 99% of compute overhead.

5. **How is multi-tenant isolation enforced?**  
   *Answer*: Tenant IDs are enforced as mandatory prefilters before any index lookups occur, preventing IDOR vulnerabilities.
