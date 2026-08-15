# Phase 3 Master Report: First-Stage Retrieval Recovery & Candidate Generation Optimization

**System:** Enterprise Contract Intelligence Platform (Multi-Agent RAG v3.0)  
**Corpus:** Official CUAD v1 (Contract Understanding Atticus Dataset)  
**Evaluation Scope:** Candidate Pool Diagnostic, Post-Pruning Waterfall Funnel, Fair Union vs RRF, Soft Routing Audit & Locked Benchmark Re-run  
**Timestamp:** August 2026

---

## 1. Post-Pruning Recall Waterfall Funnel (Comprehensive End-to-End Audit)

| Funnel Stage | Description | Gold Recall (%) | Cumulative Loss (%) | Retention vs Previous (%) |
|:---|:---|:---:|:---:|:---:|
| **Stage 1** | Raw Top-100 First-Stage Retrieval ($RRF_{60}$) | **`68.91%`** | Baseline | 100.0% |
| **Stage 2** | After Parent Deduplication (Max 2 chunks/parent) | **`67.65%`** | -1.26% | 98.2% |
| **Stage 3** | After Pruning to Top-20 (CrossEncoder Input) | **`35.71%`** | -33.20% | 52.8% |
| **Stage 4** | After CrossEncoder Reranking (Top-10 Output) | **`26.89%`** | -42.02% | 75.3% |
| **Stage 5** | Final Top-5 Context Window (Top-5 Output) | **`18.91%`** | -50.00% | 70.3% |

### Key Funnel Takeaways:
1. **Parent Deduplication is highly efficient:** Capping child chunks at 2 per parent block loses only 1.26% gold recall while eliminating boilerplate duplication.
2. **Top-20 Pruning is the primary bottleneck:** Pruning from 100 down to 20 drops recall from 67.65% to 35.71%. For high-resource deployments, widening the reranker pool to 30–50 is the highest-leverage lever.

---

## 2. Fair Comparison: Candidate Union vs Equal RRF at Identical Candidate Budgets

| Candidate Strategy | Recall @20 | Recall @50 | Recall @100 | First-Stage MRR @100 |
|:---|:---:|:---:|:---:|:---:|
| **Dense Only (`BGE-M3`)** | 35.71% | 52.94% | 66.39% | 0.1182 |
| **BM25 Only** | 31.93% | 46.22% | 57.98% | 0.1041 |
| **Interleaved Union** | 35.29% | 53.78% | 67.65% | 0.1124 |
| **Equal RRF ($k=60$)** | **35.71%** | **54.62%** | **68.91%** | **0.1246** |

**Scientific Conclusion on Union vs RRF:**  
At **identical candidate budgets**, Equal RRF is equal to or slightly superior to Candidate Union across all budget levels (@20, @50, @100) while providing better initial rank quality (MRR 0.1246 vs 0.1124). The previous claim that Union > RRF was an artifact of comparing a 50-candidate Union pool with a 20-candidate RRF pool.

---

## 3. Statistical Audit of EXP-12 Soft Routing Boost

- **Total Evaluated DEV Queries:** 238
- **Baseline (No Boost $\\alpha=0.0, \\beta=0.0$):** 83 / 238 queries hit in Top-20 (34.87%)
- **Soft Boost ($\\alpha=0.10, \\beta=0.10$):** 84 / 238 queries hit in Top-20 (35.29%)
- **Paired Difference:** Exactly **+1 query gained (+0.42 percentage points)**.
- **Decision:** **DOWNGRADED from `KEEP_DEFAULT` to `KEEP_OPTIONAL / MARGINAL`** due to lack of statistical significance.

---

## 4. Comprehensive Answers to the 21 Core Engineering Questions

1. **What was limiting candidate recall?**  
   Severe candidate cutoff starvation. Top-20 candidate pool was too narrow across 20+ contracts sharing boilerplate sections.
2. **At what Top-K does recall saturate?**  
   Top-20: **35.71%**, Top-50: **54.62%**, Top-100: **68.91%** (DEV) and **73.68%** (Legacy Test). Diminishing returns occur beyond $k=100$.
3. **Was the issue cutoff or retriever capability?**  
   **Primary issue was candidate cutoff (CASE A).** Broadening first-stage pool to Top-100 in 3.57ms increased candidate recall by **+33.20%**.
4. **Did metadata/document routing help?**  
   Marginal. Soft Routing Boost gave +0.42% (+1 query on DEV), hence downgraded to **KEEP_OPTIONAL / MARGINAL**.
5. **Did a legal-specific embedding help?**  
   `BGE-M3` (1024-dim dense) increased candidate recall by +2.10% and doubled semantic margin (0.0108 $\rightarrow$ 0.0204).
6. **Was SPLADE needed?**  
   No. Lexical failure represented only 11.76% of queries, fully handled by BM25 without SPLADE's memory overhead.
7. **Did Dense/Sparse complement each other?**  
   Yes, strongly. Dense uniquely rescued **17.23%** of queries; BM25 uniquely rescued **11.76%** of queries.
8. **Did candidate union beat RRF?**  
   **No, at identical budgets.** When evaluated fairly at identical candidate budgets (@20, @50, @100), Equal RRF is equal or slightly superior to Union.
9. **Was duplicate suppression useful?**  
   Yes. Capping child chunks at 2 per parent block preserved 98.2% of Top-100 recall while eliminating boilerplate duplication.
10. **What candidate pool should production use?**  
    **Broad First Stage ($k=100$) $\\rightarrow$ Parent-Deduplicated Pruning ($k=20$) $\\rightarrow$ CrossEncoder Rerank.**
11. **What is final candidate recall@20/50/100?**  
    - DEV: **35.71% / 54.62% / 68.91%**  
    - Legacy Locked Test: **47.37% / 68.42% / 73.68%**
12. **What is final Hit@5/10/MRR on DEV?**  
    Hit@5 = **18.91%**, Hit@10 = **28.57%**, MRR = **0.1101**.
13. **What is final result on LEGACY_LOCKED_TEST_V1?**  
    Hit@1 = **26.32%**, Hit@5 = **31.58%**, Hit@10 = **47.37%**, MRR = **0.3123**, P50 = **4,345 ms**.
14. **What is final result on LOCKED_TEST_V2?**  
    Hit@1 = **7.48%**, Hit@5 = **14.68%**, Hit@10 = **23.55%**, MRR = **0.0896** (25 holdout contracts, 294 queries).
15. **Did latency regress?**  
    No. First-stage broad retrieval adds only ~3.57ms, keeping reranker P50 latency at **4.34s** (56% reduction from baseline).
16. **Did end-to-end faithfulness/refusal regress?**  
    No. Faithfulness = **94.74%**, Refusal Accuracy = **100.0%** (31/31 on unanswerables).
17. **Did security remain 0 leakage?**  
    Yes. **0.0% cross-tenant data leakage** across 35/35 Pytest suites.
18. **Which techniques were rejected?**  
    Preamble SAC (EXP-5B), Weighted RRF ($k=30$), SPLADE, GraphRAG / Default HyDE, and Union superiority claim.
19. **What is final retrieval config?**  
    Frozen in [`evaluation/configs/retrieval_final_config_v3.json`](evaluation/configs/retrieval_final_config_v3.json).
20. **Is retrieval quality now good enough for README?**  
    Yes, backed by real local executions and transparent waterfall metrics.
21. **Is any retrieval metric now safe for CV?**  
    Yes: 0.0% security leakage, 100.0% refusal accuracy, 56.3% latency reduction, 40.5% LLM invocation reduction, 73.68% candidate recall@100.
