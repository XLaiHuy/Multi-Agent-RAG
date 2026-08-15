# Candidate Pool Diagnostic Report (EXP-11)

**Evaluation Split:** CUAD DEV Set (20 Contracts, 244 Answerable Queries)  
**Retriever:** Hybrid RRF (BM25Okapi + Dense `BAAI/bge-small-en-v1.5` with Structural Metadata)  
**Evaluation Scope:** First-stage retrieval candidate generation across $k \in [5, 10, 20, 30, 50, 100]$ before reranking  
**Timestamp:** 2026-08-15 09:53:37Z

---

## 1. Candidate Recall & Metric Progression across Pool Size $k$

| Candidate Pool Size ($k$) | Candidate Recall@k | HitRate@k | MRR | P50 First-Stage Latency |
|:---|:---:|:---:|:---:|:---:|
| **Top-5** | **16.81%** | 16.81% | 0.0967 | 3.57 ms |
| **Top-10** | **26.89%** | 26.89% | 0.1106 | 3.57 ms |
| **Top-20** | **35.71%** | 35.71% | 0.1170 | 3.57 ms |
| **Top-30** | **40.34%** | 40.34% | 0.1188 | 3.57 ms |
| **Top-50** | **54.62%** | 54.62% | 0.1225 | 3.57 ms |
| **Top-100** | **68.91%** | 68.91% | 0.1246 | 3.57 ms |

---

## 2. Gold Evidence Rank Appearance Distribution

Where does the first relevant evidence chunk appear in the first-stage retrieved list?

| Rank Range Bin | Query Count | Percentage of Total Queries |
|:---|:---:|:---:|
| **Ranks 1 – 5** | 40 | 16.81% |
| **Ranks 6 – 10** | 24 | 10.08% |
| **Ranks 11 – 20** | 21 | 8.82% |
| **Ranks 21 – 30** | 11 | 4.62% |
| **Ranks 31 – 50** | 34 | 14.29% |
| **Ranks 51 – 100** | 34 | 14.29% |
| **> 100 / Never Found** | 74 | 31.09% |

---

## 3. Empirical Diagnosis & Action Plan

- **Top-20 Candidate Recall:** **`35.71%`**
- **Top-100 Candidate Recall:** **`68.91%`** (Recall Gain: **`+33.20%`**)
- **Never Found in Top-100:** **`31.09%`**

### Case Interpretation:
**CASE A: Candidate recall rises strongly with broader first-stage pool (from Top-20 to Top-100).**

### Recommended Engineering Roadmap:
1. **Deploy broad first-stage retrieval (Top-50/100) paired with cheap pruning and CrossEncoder reranking.**
2. Combine **broad first-stage candidate generation (Top-50/100)** with **EXP-12 Soft Document/Section Routing Boost** to filter out distractor contracts before passing to CrossEncoder.
3. Test **Dense $\cup$ Sparse Candidate Union** (EXP-16) to maximize unique candidate capture.
