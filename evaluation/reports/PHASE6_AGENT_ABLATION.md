# Phase 6 Agent Architecture Ablation Study

## Executive Summary
This document presents the rigorous comparative empirical study across three distinct RAG generation architectures evaluated on the **Phase 6 Development Split ($N=80$ queries: 40 Answerable, 40 Unanswerable)** using real Google GenAI API calls. All systems operated under strict **Layer A Zero Gold Access** isolation.

---

## 1. System Architecture Variants
1. **Variant A: `BASE_RAG` (Single-Call Baseline)**
   - Single LLM generation call over top-5 reranked contract chunks.
   - Relies strictly on prompt-guided in-context citation and structured refusal syntax.
2. **Variant B: `RAG_PLUS_VERIFIER` (Two-Stage Generation & Grounding Audit)**
   - Generator generates answer with citations.
   - Verifier Agent audits answer against evidence chunks; triggers single bounded regeneration if ungrounded claims are detected.
3. **Variant C: `FULL_BOUNDED_MULTI_AGENT` (Flagship Multi-Agent System)**
   - **Planner Agent**: Classifies query complexity and selects domain retrieval parameterization.
   - **Retrieval Engine**: Phase 4.2 Locked Protocol (BGE-M3 + BM25 + RRF + TinyBERT Cross-Encoder).
   - **Evidence Critic Agent**: Audits retrieved chunks; executes bounded query expansion if evidence is insufficient.
   - **Generator Agent**: Generates legally grounded answers with structured citations.
   - **Answer Verifier Agent**: Audits claims against source clauses with qualification / refusal / bounded regeneration logic.

---

## 2. Empirical Benchmark Results ($N=80$, Seed=42)

| Metric | `BASE_RAG` | `RAG_PLUS_VERIFIER` | `FULL_BOUNDED_MULTI_AGENT` (Winner) | Multi-Agent Delta |
|---|---|---|---|---|
| **Balanced Accuracy** | 76.25% | 75.00% | **78.75%** | **+2.50%** |
| **Answerable Acceptance Rate** | 60.00% | 57.50% | **65.00%** | **+5.00%** |
| **Unanswerable Refusal Rate** | 92.50% | 92.50% | **92.50%** | $\pm 0.00\%$ |
| **False Refusal Rate** | 40.00% | 42.50% | **35.00%** | **-5.00%** |
| **False Answer Rate** | 7.50% | 7.50% | **7.50%** | $\pm 0.00\%$ |
| **Child Citation Hit Rate** | 87.50% | 86.96% | **92.31%** | **+4.81%** |
| **Citation Precision** | 85.42% | 84.78% | **90.38%** | **+4.96%** |
| **Citation Recall** | 43.74% | 42.97% | **48.27%** | **+4.53%** |
| **Parent Citation Hit Rate** | 91.67% | 91.30% | **96.15%** | **+4.48%** |
| **Grounded Claim Rate** | 100.0% | 100.0% | **100.0%** | $\pm 0.00\%$ |
| **Production Calls / Query** | 1.00 | 1.32 | 3.38 | +2.38 calls |
| **Total Tokens / Query** | 1,617.9 | 2,038.0 | 4,091.8 | +2,473.9 tokens |
| **P50 Latency** | 3,303.7 ms | 5,702.0 ms | 36,194.8 ms | +32.89 s |

---

## 3. Pareto Frontier & Trade-off Analysis
- **Quality Dominance**: `FULL_BOUNDED_MULTI_AGENT` achieved Pareto-optimal performance for high-stakes legal contracts, increasing Child Citation Hit Rate from 87.5% to **92.31%** and Citation Precision from 85.42% to **90.38%**, while reducing false refusals on complex questions by 5.0%.
- **Cost & Latency Considerations**: `BASE_RAG` offers lightweight 3.3s P50 latency for interactive low-budget deployments, while `FULL_BOUNDED_MULTI_AGENT` is selected for the production flagship standard where legal accuracy and auditable agent traces are non-negotiable.

---

## 4. Frozen Selection
`FULL_BOUNDED_MULTI_AGENT` is formally selected and frozen in `evaluation/configs/generation_final_config_v6.json` as the production generation architecture for the Final Held-Out Benchmark.
