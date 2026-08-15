# Fair Comparison: Candidate Union vs Equal RRF at Identical Candidate Budgets

**Evaluation Dataset:** CUAD DEV Split (20 Contracts, 238 Evaluated Answerable Queries)  
**Dense Model:** `BAAI/bge-m3` (1024-d)  
**Sparse Model:** `BM25Okapi`  
**Evaluation Protocol:** Strict identical candidate budgets $k \in [20, 50, 100]$.  
**Timestamp:** 2026-08-15 21:13:34Z  
**Runtime:** 1914.50s  

---

## 1. CandidateHitRate at Identical Candidate Budgets

| Candidate Strategy | CandidateHitRate @20 | CandidateHitRate @50 | CandidateHitRate @100 | First-Stage MRR @100 |
|:---|:---:|:---:|:---:|:---:|
| **Dense_Only** | 43.70% | 59.24% | 71.01% | 0.1207 |
| **BM25_Only** | 38.24% | 56.30% | 73.53% | 0.1120 |
| **Interleaved_Union** | 40.76% | 62.18% | 78.15% | 0.1210 |
| **Equal_RRF** | 42.44% | 62.18% | 77.31% | 0.1251 |

---

## 2. Key Scientific Findings & Analysis

1. **At Identical Budget $k=20$:** Equal RRF achieves **`42.44%`** vs **`40.76%`** for Interleaved Union.
2. **At Identical Budget $k=50$:** Equal RRF achieves **`62.18%`** vs **`62.18%`** for Interleaved Union.
3. **At Identical Budget $k=100$:** Equal RRF achieves **`77.31%`** vs **`78.15%`** for Interleaved Union.
