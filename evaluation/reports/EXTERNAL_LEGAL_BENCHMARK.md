# External Standard Benchmark: LegalBench-RAG Evaluation Report

**Benchmark:** LegalBench-RAG (Stanford RegLab / Guha et al. 2023 - CUAD Component)  
**Corpus Scope:** 25 Multi-Domain Legal Contracts (Zero-Shot Multi-Contract Index, 1221 chunks)  
**Evaluation Set:** 293 Answerable Standard Legal Queries  
**Evaluation Protocol:** REAL_LOCAL Deterministic Vector & Sparse Retrieval + CrossEncoder Reranking  
**Timestamp:** 2026-08-15 08:47:19Z

---

## 1. Overall Benchmark Performance

| Metric | Measured Result | Standard Baseline (BM25 Only) | Published Reference (BGE Baseline) |
|:---|:---:|:---:|:---:|
| **HitRate@1 (Top-1 Accuracy)** | **4.10%** | 12.4% | 18.2% |
| **HitRate@5** | **14.68%** | 22.8% | 29.5% |
| **HitRate@10** | **23.55%** | 31.2% | 38.1% |
| **Mean Reciprocal Rank (MRR)** | **0.0896** | 0.1420 | 0.2010 |
| **nDCG@5** | **0.0341** | 0.1650 | 0.2240 |
| **P50 Query Latency (CPU)** | **4122.3 ms** | 35.0 ms | 4,200 ms |

---

## 2. Clause-Category Breakdown

| Clause Category | Query Count | HitRate@5 | MRR |
|:---|:---:|:---:|:---:|
| Agreement Date                 |  23 |    4.3% | 0.0435 |
| Anti-Assignment                |  12 |   25.0% | 0.1927 |
| Audit Rights                   |   8 |   50.0% | 0.2646 |
| Cap On Liability               |  14 |   28.6% | 0.1488 |
| Change Of Control              |   3 |   33.3% | 0.3333 |
| Covenant Not To Sue            |   5 |    0.0% | 0.0000 |
| Document Name                  |  25 |    0.0% | 0.0067 |
| Effective Date                 |  19 |    0.0% | 0.0000 |
| Expiration Date                |  19 |   15.8% | 0.1127 |
| General                        |  30 |    6.7% | 0.0276 |
| Governing Law                  |  21 |   14.3% | 0.0926 |
| IP Ownership Assignment        |   4 |   75.0% | 0.2583 |
| Insurance                      |   8 |   50.0% | 0.2542 |
| Minimum Commitment             |   5 |    0.0% | 0.0000 |
| No-Solicit Of Customers        |   2 |    0.0% | 0.0556 |
| No-Solicit Of Employees        |   5 |   40.0% | 0.2306 |
| Non-Compete                    |  11 |    0.0% | 0.0000 |
| Notice Period To Terminate Renewal |   4 |   50.0% | 0.3000 |
| Parties                        |  32 |    9.4% | 0.0665 |
| Post-Termination Services      |   7 |    0.0% | 0.0238 |
| Renewal Term                   |   5 |   40.0% | 0.3333 |
| Revenue/Profit Sharing         |   8 |   37.5% | 0.2125 |
| Right Of First Refusal         |   1 |    0.0% | 0.0000 |
| Termination For Convenience    |  10 |   10.0% | 0.0625 |
| Uncapped Liability             |   4 |   50.0% | 0.1125 |
| Volume Restriction             |   3 |    0.0% | 0.0370 |
| Warranty Duration              |   5 |    0.0% | 0.0583 |

---

## 3. Benchmark Defensibility & Integrity Statement

1. **Zero Data Contamination:** Evaluated exclusively on holdout contracts disjoint from all development/tuning sets.
2. **Standard Evaluation:** Exact character-level substring matching and token-level boundary verification.
3. **Reproducibility:** Frozen manifest with SHA256 checksums preserved in `evaluation/manifests/cuad_locked_test_v2_manifest.json`.
