# Senior AI Engineer Portfolio Summary & Interview Guide

This document provides a concise source-of-truth for technical recruiters, hiring managers, and AI engineering interviewers.

---

## 1. Project Overview

**Project**: Enterprise Contract Intelligence Platform  
**Repository**: [https://github.com/XLaiHuy/Multi-Agent-RAG](https://github.com/XLaiHuy/Multi-Agent-RAG)  
**Domain**: Enterprise Legal AI, Contract Question Answering, Clause Comparison & Risk Review  
**Architecture**: Role-Aware Multi-Agent RAG with Document-Scoped Hybrid Retrieval & Hierarchical Context Expansion

---

## 2. Verified Headline Metrics (CUAD Held-Out Split, $N=293$)

*Evaluated on 25 unseen contracts, 293 answerable queries under frozen config `v4.1.0` on commodity 4-thread CPU:*

| Metric | Result | Benchmark Significance |
| :--- | :---: | :--- |
| **HitRate@10** | **94.54%** | Gold clause retrieved in top-10 in 94.5% of questions |
| **HitRate@5** | **82.94%** | High top-rank precision for single-pass context feeding |
| **MRR (Mean Reciprocal Rank)** | **0.6418** | High reciprocal ranking position across diverse categories |
| **CandidateHitRate@20** | **98.29%** | First-stage hybrid recall before reranking truncation |
| **Measured Latency (P50)** | **68.89 ms** | Sub-100ms retrieval on 4 CPU threads (zero GPU required) |
| **Evaluation Acceleration** | **94.70x** | Harness sweep time reduced from 40.7 min to 25.8s |
| **Security Isolation** | **0 Leaks** | Zero cross-tenant retrieval leakage observed across 7 ACL suites |

---

## 3. Five Key Engineering & Architectural Decisions

1. **True Document-Scoped Retrieval vs Multi-Contract Post-Filtering**:
   - *Problem*: Multi-contract corpora create cross-agreement distractor collisions on common legal questions (*"Governing law?"*).
   - *Solution*: Explicitly bounded search space to the active document prior to Dense dot-product and BM25 scoring.
   - *Impact*: Lifted Hit@10 from 28.67% to **94.54%** (+65.87 pp) on held-out contracts.
2. **BGE-M3 + BM25Okapi Hybrid Fusion with Equal RRF ($k=60$)**:
   - Combines 1024-dim dense semantic embeddings with lexical exact match to capture both conceptual obligations and statutory keyword phrases.
3. **Structure-Aware Parent-Child Chunking**:
   - Indexes child chunks (~250 tokens) with section breadcrumbs for high retrieval precision, expanding to parent context (~1200 tokens) for synthesis.
4. **Empirical Reranker Optimization (TinyBERT vs BGE-Reranker-Base)**:
   - Verified that `ms-marco-TinyBERT-L-2-v2` (4.4M params) achieves equal or superior precision (90.34% vs 89.08% Hit@10) while running **60x faster on CPU** (153ms vs 9,142ms).
5. **Deterministic Parameter-Sensitive Evaluation Caching**:
   - Built a cryptographic cache invalidating on all tuning parameters, accelerating iteration cycles by **94.7x** with identical float32 metric outputs.

---

## 4. Candidate Resume / CV Bullets

### Candidate A (Retrieval & ML Focus):
> *"Built a document-scoped hybrid legal RAG retriever using BGE-M3, BM25, RRF and CrossEncoder reranking, achieving 94.54% HitRate@10 and 0.6418 MRR across 293 held-out CUAD questions from 25 unseen contracts."*

### Candidate B (Systems & Performance Focus):
> *"Engineered deterministic embedding and candidate caching for RAG evaluation that reduced sweep latency from 40.7 minutes to 25.8 seconds (94.7x speedup) while preserving exact float32 metric identity."*

### Candidate C (Full-Stack & Security Focus):
> *"Designed a tenant-aware contract intelligence platform with hierarchical Parent-Child retrieval, bounded Planner/Critic/Verifier reasoning and ACL prefiltering, with zero cross-tenant retrieval leakage observed across 7 security regression tests."*

### Recommended 2-Bullet Combination for 1-Page Resume:
- **Bullet 1**: *Candidate A (Retrieval & ML Focus)*
- **Bullet 2**: *Candidate C (Full-Stack & Security Focus)*

---

## 5. Technical Interview Talking Points & Deep Dives

1. **Why Equal RRF over Learned Reciprocal Rank Weights?**
   - Equal RRF ($k=60$) avoids overfitting to training split distribution shifts and provides scale-invariant fusion across disparate score distributions.
2. **Why Parent-Child Chunking over Large Sliding Windows?**
   - Small child chunks prevent semantic dilution in dense embeddings; hierarchical parent expansion ensures the LLM sees full clauses without window boundary chopping.
3. **Why Bounded 3-Agent Orchestration over Open-Ended ReAct Loops?**
   - Open-ended multi-agent loops introduce latency variance and runaway LLM costs. Bounding to Planner $	o$ Critic (max 2 loops) $	o$ Verifier guarantees bounded execution budgets.
4. **How Is Anti-IDOR Enforced at the Data Layer?**
   - Predicates enforce `(tenant_id, accessible_role)` at the SQL query level and in-memory vector slicing, verified by automated security regression tests.
5. **Why Was TinyBERT Kept over BGE-Reranker-Base?**
   - For scoped single-document candidates ($k=20$), TinyBERT's distillation on MS-MARCO provides sufficient discriminative power without the 9-second CPU latency penalty of a 110M model.
