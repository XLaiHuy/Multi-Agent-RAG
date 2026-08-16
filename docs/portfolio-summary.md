# Safe-RAG: Portfolio Technical Summary

This document presents a structured engineering overview of the Safe-RAG legal contract analysis system for technical recruiters, engineering managers, and ML/RAG practitioners.

---

## 1. Executive Summary

Safe-RAG is a specialized Retrieval-Augmented Generation system designed to solve high-stakes legal contract analysis where cross-document hallucination, ambiguous refusal, and unsourced assertions are unacceptable.

The system is evaluated across two distinct, independently frozen benchmarks:
1. **Document-Scoped Hybrid Retrieval Benchmark**: $N = 294$ held-out answerable questions across 25 legal contracts.
2. **Real API End-to-End Generation Benchmark**: $N = 200$ stratified queries (100 Answerable, 100 Unanswerable) across 25 unseen contracts using real Google GenAI API calls (`gemma-4-26b-a4b-it`).

---

## 2. Key Empirical Results

### A. Document-Scoped Hybrid Retrieval ($N = 294$ Answerable Queries)
- **Strict Child HitRate@10**: **81.97%**
- **Strict Child HitRate@5**: **68.71%**
- **Mean Reciprocal Rank (MRR)**: **0.5214**
- **Parent Section HitRate@10**: **94.90%**
- **Corpus-Wide Collision Penalty**: When searching across all 25 contracts without document scoping, HitRate@10 drops from 81.97% to **28.67%**, proving that document-level isolation is a mathematical requirement for contract QA.

### B. Real API End-to-End Evaluation ($N = 200$ Queries)
- **Inclusive Balanced Answerability Accuracy**: **74.50%** (Prose-aware)
- **Strict Balanced Answerability Accuracy**: **72.50%** (Sentinel-only)
- **Unanswerable Refusal Rate**: **82.00%** (82 / 100 queries refused; 78 strict sentinel, 4 prose)
- **Answerable Acceptance Rate**: **67.00%** (67 / 100 queries answered with citations)
- **Valid Explicit Citation Compliance**: **98.51%** (84 / 85 accepted answers contained explicit citations)
- **Child Citation Hit Rate (accepted)**: **85.07%** (58 / 67 accepted answers cite exact gold child clause)
- **End-to-End Child Citation Coverage**: **62.00%** (58 / 100 total answerable queries cite exact gold child clause)
- **Citation Precision (Macro)**: **80.97%**
- **Citation Recall (Macro)**: **63.00%**
- **Wrong-Document Citation Rate**: **0.00%** (0 / 140 citations crossed contract boundaries)
- **Invalid Citation Rate**: **0.00%** (0 / 140 non-existent chunk IDs)
- **Grounded Material Claim Rate**: **97.93%** (142 / 145 claims supported by retrieved context, judged by `gemma-4-26b-a4b-it`)
- **Semantic Correctness**: **92.54%** (Mean score 1.85 / 2.0 against gold evidence, judged by `gemma-4-26b-a4b-it`)
- **API Efficiency**: **3.42 calls/query**, **3,971.9 tokens/query**, **32.62s P50 latency**, **0 rate-limit errors**.

---

## 3. Recommended Resume / Portfolio Bullets

### Primary Recommendation 1 (Retrieval Engineering):
> **"Engineered a document-scoped legal retrieval pipeline using BGE-M3, BM25, Reciprocal Rank Fusion ($k=60$), and CrossEncoder reranking, achieving 81.97% strict child HitRate@10 and 0.5214 MRR on 294 held-out CUAD contract queries."**

### Primary Recommendation 2 (Systems & Safe Generation):
> **"Designed an evidence-bounded Multi-Agent RAG system with Google GenAI, achieving 74.50% balanced answerability accuracy, 82.00% unanswerable refusal, and 80.97% citation precision with zero cross-document contamination across 25 unseen contracts."**
