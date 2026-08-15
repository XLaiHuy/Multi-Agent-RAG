# Post-Pruning Recall Waterfall Audit

**Evaluation Dataset:** CUAD DEV Split (20 Contracts, 238 Evaluated Answerable Queries)  
**Pipeline Architecture:** Two-Stage Broad Retrieval ($k=100$) $\rightarrow$ Parent Dedup $\rightarrow$ Top-20 Pruning $\rightarrow$ CrossEncoder Rerank  
**Timestamp:** August 2026

---

## 1. Step-by-Step Recall Waterfall Funnel

| Funnel Stage | Description | Gold Recall (%) | Cumulative Loss (%) | Retention vs Previous (%) |
|:---|:---|:---:|:---:|:---:|
| **Stage 1** | Raw Top-100 First-Stage Retrieval ($RRF_{60}$) | **`68.91%`** | Baseline | 100.0% |
| **Stage 2** | After Parent Deduplication (Max 2 chunks/parent) | **`67.65%`** | -1.26% | 98.2% |
| **Stage 3** | After Pruning to Top-20 (CrossEncoder Input) | **`35.71%`** | -33.20% | 52.8% |
| **Stage 4** | After CrossEncoder Reranking (Top-10 Output) | **`26.89%`** | -42.02% | 75.3% |
| **Stage 5** | Final Top-5 Context Window (Top-5 Output) | **`18.91%`** | -50.00% | 70.3% |

---

## 2. Key Diagnostic Takeaways

1. **Parent Deduplication Loss (-1.26%):**  
   Capping child chunks at 2 per parent context block incurs minimal recall loss while preventing repetitive boilerplate clauses from crowding the pool.
2. **Top-20 Pruning Bottleneck (-31.94%):**  
   Pruning from 100 down to 20 drops recall from 67.65% to 35.71%. This demonstrates that when scaling to large document corpora, increasing the reranker candidate pool from 20 to 30–50 is the single most direct lever for higher downstream HitRate.
3. **CrossEncoder Ranking Fidelity:**  
   The CrossEncoder retains 75.3% of the available gold candidates in its Top-10 and 70.3% in its Top-5.
