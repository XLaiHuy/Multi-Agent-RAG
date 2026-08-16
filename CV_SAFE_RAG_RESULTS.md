# CV-Safe Empirical Results & Scientific Metrics

## Executive Overview
All performance numbers reported below are generated through **strict reproducible offline execution scripts** and **real Google GenAI API end-to-end evaluations** under **Layer A Zero Gold Access** isolation.

---

## 1. End-to-End Generation & Agent Performance (Phase 6 Final Benchmark)
- **Evaluation Set**: $N=200$ Stratified Held-Out Queries (100 Answerable, 100 Unanswerable) across 25 unseen contracts.
- **Architecture**: `FULL_BOUNDED_MULTI_AGENT` (Planner + Hybrid Retrieval + Critic + Generator + Verifier).
- **Model Engine**: Google GenAI `gemma-4-26b-a4b-it` (Real API Calls).

| Metric | Target | Final Held-Out Measured Result | Verification |
|---|---|---|---|
| **Balanced Answerability Accuracy** | $\ge 70.0\%$ | **74.50%** | Exceeded |
| **Unanswerable Refusal Rate** | $\ge 80.0\%$ | **82.00%** | Met |
| **Answerable Acceptance Rate** | $\ge 60.0\%$ | **67.00%** | Met |
| **Child Citation Hit Rate** | $\ge 80.0\%$ | **86.57%** | Exceeded |
| **Parent Citation Hit Rate** | $\ge 85.0\%$ | **94.03%** | Exceeded |
| **Citation Precision** | $\ge 75.0\%$ | **82.84%** | Exceeded |
| **Citation Recall** | $\ge 50.0\%$ | **63.50%** | Exceeded |
| **Grounded Claim Rate** | $100.0\%$ | **100.0%** | Zero Hallucination |
| **Wrong Document Citation Rate** | $0.0\%$ | **0.00%** | Clean Scoping |
| **Production Calls / Query** | $\le 4.0$ | **3.42** | Bounded Budget |
| **Total Tokens / Query** | $\le 5,000$ | **3,971.9** | Efficient Context |
| **End-to-End Latency P50** | $\le 45.0	ext{ s}$ | **32.62 s** | Real API Latency |

---

## 2. Multi-Agent Architecture Ablation ($N=80$ DEV Split)

| Metric | `BASE_RAG` (1 call) | `RAG_PLUS_VERIFIER` (~1.3 calls) | `FULL_BOUNDED_MULTI_AGENT` (Flagship) |
|---|---|---|---|
| **Balanced Accuracy** | 76.25% | 75.00% | **78.75%** (+2.50%) |
| **Child Citation Hit Rate** | 87.50% | 86.96% | **92.31%** (+4.81%) |
| **Citation Precision** | 85.42% | 84.78% | **90.38%** (+4.96%) |
| **False Refusal Rate** | 40.00% | 42.50% | **35.00%** (-5.00%) |
| **Calls / Query** | 1.00 | 1.32 | 3.38 |
| **Tokens / Query** | 1,617.9 | 2,038.0 | 4,091.8 |
| **Latency P50** | 3.30 s | 5.70 s | 36.19 s |

---

## 3. Retrieval Performance Summary (Phase 4.2 Locked Benchmark)
- **Benchmark Corpus**: CUAD 25 Unseen Held-Out Contracts ($N=294$ queries).
- **Child HitRate@5**: **87.76%** (MRR@5: 0.7788)
- **Parent HitRate@5**: **94.22%** (MRR@5: 0.8878)
- **Child HitRate@10**: **92.18%**
- **Candidate Pool Complementarity**: 97.62%
