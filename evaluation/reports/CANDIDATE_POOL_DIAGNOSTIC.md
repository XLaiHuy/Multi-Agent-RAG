# EXP-11: Candidate Pool Diagnostic Report

**Evaluation Split:** CUAD DEV Split (20 Contracts, 238 Evaluated Answerable Queries)  
**Dense Model:** `BAAI/bge-m3` (1024-d)  
**Sparse Model:** `BM25Okapi`  
**Fusion:** Equal RRF ($k=60$)  
**Timestamp:** 2026-08-15 20:41:14Z  
**Runtime:** 1944.39s  

---

## 1. Candidate Pool Diagnostic Matrix

| Candidate Pool Size $k$ | CandidateHitRate@k (Any-Gold) | TrueChunkRecall@k (All-Gold) | First-Stage MRR |
|:---|:---:|:---:|:---:|
| **Top-5** | 18.07% | 3.90% | 0.0936 |
| **Top-10** | 27.73% | 6.84% | 0.1064 |
| **Top-20** | 42.44% | 11.17% | 0.1164 |
| **Top-30** | 51.68% | 14.51% | 0.1202 |
| **Top-50** | 62.18% | 20.68% | 0.1229 |
| **Top-100** | 77.31% | 29.93% | 0.1251 |

---

## 2. Scientific Takeaways

1. **Candidate Coverage Pattern:** Candidate coverage (`CandidateHitRate@k`) increases monotonically through Top-100 (reaching 77.31% at $k=100$).
2. **HitRate vs True Chunk Recall Distinction:**
   - `CandidateHitRate@k`: whether **at least one** relevant chunk is present in the first-stage pool (77.31% at $k=100$).
   - `TrueChunkRecall@k`: fraction of **all** relevant chunk spans captured (29.93% at $k=100$).
