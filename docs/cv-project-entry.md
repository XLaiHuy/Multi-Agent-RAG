# Safe-RAG: CV & Portfolio Entry Source

This document is the **Single Verified Source** for copying Safe-RAG project details and bullet points into resumes, CVs, and portfolio profiles.

---

## Project Title
**Multi-Agent Safe-RAG: Document-Scoped Legal Intelligence Pipeline**

## One-Line Project Description
*An enterprise-grade, evidence-bounded legal contract analysis system combining document-scoped hybrid retrieval (BGE-M3 + BM25 + RRF + CrossEncoder) with a bounded multi-agent verification stack on Google GenAI.*

---

## Recommended Resume Bullets (Select Top 2)

### Bullet 1 — Retrieval Engineering (Recommended):
> **"Engineered a document-scoped legal retrieval pipeline using BGE-M3, BM25, Reciprocal Rank Fusion, and CrossEncoder reranking, achieving 81.97% strict child HitRate@10 and 0.5214 MRR across 294 held-out CUAD queries from 25 contracts."**

### Bullet 2 — Systems & Evidence-Bounded Generation (Recommended):
> **"Built an evidence-bounded multi-agent legal RAG system evaluated with real Google GenAI on 200 held-out queries, achieving 72.5% strict balanced answerability accuracy and 80.97% macro citation precision, with 0/140 wrong-document citations observed."**

### Bullet 3 — API Efficiency & Telemetry (Optional Alternative):
> **"Implemented a reproducible real-API RAG evaluation pipeline measuring 3.42 LLM calls and 3,971.9 tokens per query across 200 held-out queries, with deterministic citation, refusal, retrieval, and latency telemetry."**

---

## Verified Tech Stack Keywords
`Python` • `FastAPI` • `React` • `Google GenAI` • `BGE-M3` • `BM25` • `CrossEncoder` • `ChromaDB` • `Pytest` • `Docker`

---

## Key Metrics Quick Reference for Technical Interviews

| Dimension | Metric Name | Verified Value | Benchmark Scope |
|---|---|---|---|
| **Retrieval** | Strict Child HitRate@10 | **81.97%** | $N = 294$ held-out CUAD queries across 25 contracts |
| **Retrieval** | Mean Reciprocal Rank (MRR) | **0.5214** | $N = 294$ held-out CUAD queries across 25 contracts |
| **Retrieval** | Parent Section HitRate@10 | **94.90%** | $N = 294$ held-out CUAD queries across 25 contracts |
| **Retrieval** | Online CPU Latency (P50) | **586 ms** | Dense encoding + BM25 + RRF + TinyBERT on 4 CPU threads |
| **Generation** | Strict Balanced Accuracy | **72.50%** | $N = 200$ (100 answerable, 100 unanswerable, sentinel-only) |
| **Generation** | Inclusive Balanced Accuracy | **74.50%** | $N = 200$ (100 answerable, 100 unanswerable, prose-aware) |
| **Refusal** | Strict Unanswerable Refusal | **78.00%** | 78 / 100 strict sentinel refusals (`INSUFFICIENT_EVIDENCE:`) |
| **Refusal** | Inclusive Unanswerable Refusal | **82.00%** | 82 / 100 total unanswerable refusals (78 strict + 4 prose) |
| **Citation** | Valid Citation Compliance | **98.51%** | 84 / 85 accepted answers contained valid in-text citations |
| **Citation** | Child Citation Hit Rate | **85.07%** | 58 / 67 accepted answerable responses cite verified gold child |
| **Citation** | End-to-End Child Coverage | **62.00%** | 58 / 100 total answerable queries cite verified gold child |
| **Citation** | Citation Precision (Macro) | **80.97%** | Exact match against verified contract clause chunks |
| **Integrity** | Wrong-Document Citations | **0 / 140 observed** | 0 cross-document clause contamination |
| **Integrity** | Invalid Citation Mentions | **0 / 140 observed** | 0 non-existent chunk IDs |
| **Judge** | Grounded Material Claim Rate | **97.93%** | 142 / 145 claims supported by retrieved context (`JUDGE-BASED`) |
| **Judge** | Semantic Correctness | **92.54%** | Mean score 1.85 / 2.0 against gold evidence (`JUDGE-BASED`) |
| **Telemetry** | Calls / Query | **3.42 calls** | Mean calls across $N = 200$ test queries |
| **Telemetry** | Total Tokens / Query | **3,971.9 tokens** | Mean tokens across $N = 200$ test queries |
| **Telemetry** | Latency (P50) | **32.62 s** | Median latency across $N = 200$ test queries |
