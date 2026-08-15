# Phase 4 Fast Evaluation Harness, Candidate-Budget Recovery & Final Report

**Gate Status**: **PASSED**  
**Repository HEAD**: `f6533536fb4e8af274c978aae634ba982aa70a38`  
**Evaluation Harness**: Single-Pass Unified Runner with Reusable Intermediate Representation Caching  

---

## 1. Evaluation Harness Acceleration & Cache Benchmark

| Evaluation Stage | Cold Run (No Cache) | Warm Cache (Intermediate Hit) | Acceleration Ratio |
| :--- | :---: | :---: | :---: |
| Corpus Parsing & Chunking | 4.8s | 0.0s (Loaded from disk) | Instantaneous |
| BM25 Corpus Indexing | 1.2s | 0.0s | Instantaneous |
| Dense Corpus Embedding (BGE-M3, 1034 chunks) | ~2380.0s | 0.0s (Loaded `.npy`) | Instantaneous |
| Query Embedding (238 queries) | ~35.0s | 0.0s (Loaded `.npy`) | Instantaneous |
| First-Stage Retrieval (Dense + BM25 + RRF) | ~8.0s | 0.0s (Loaded candidates) | Instantaneous |
| CrossEncoder Reranking & Metrics Computation | ~25.0s | ~25.0s | 1.0x (Always recomputed) |
| **Total Evaluation Suite Wall-Clock Time** | **~2443.2s (~40.7 min)** | **~25.8s (~0.4 min)** | **>90x Acceleration** |

---

## 2. Candidate Budget Sweep & Pareto Analysis (GLOBAL Workflow)

| Candidate Budget $k$ | Candidate HitRate (Pre-CE) | True Chunk Recall (Pre-CE) | Post-CE Hit@1 | Post-CE Hit@5 | Post-CE Hit@10 | Post-CE MRR | Post-CE nDCG@5 | CE P50 (ms) | CE P95 (ms) | $\Delta$Hit@10 vs k=20 | Classification |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **$k=10$** | 27.73% | 6.58% | 5.04% | 18.91% | 27.73% | 0.1100 | 0.0463 | 72.1 | 131.3 | -3.36% | `FAST` |
| **$k=20$** | 43.28% | 10.64% | 5.46% | 20.17% | 31.09% | 0.1173 | 0.0477 | 158.3 | 270.7 | Baseline | **`BASELINE / PARETO_DEFAULT`** |
| **$k=30$** | 52.94% | 13.70% | 5.88% | 19.75% | 29.41% | 0.1177 | 0.0478 | 230.4 | 388.1 | -1.68% | `PARETO_EXPANDED` |
| **$k=40$** | 59.66% | 16.59% | 5.88% | 19.75% | 28.99% | 0.1166 | 0.0477 | 282.2 | 438.4 | -2.10% | `PARETO_EXPANDED` |
| **$k=50$** | 64.29% | 18.91% | 5.46% | 19.33% | 28.57% | 0.1139 | 0.0466 | 392.4 | 581.0 | -2.52% | `HIGH_ACCURACY` |
| **$k=75$** | 72.27% | 22.53% | 5.46% | 18.49% | 28.99% | 0.1123 | 0.0451 | 453.2 | 613.0 | -2.10% | `HIGH_ACCURACY` |

---

## 3. Final V4 Frozen Configuration

We freeze the Pareto-optimal configuration into `evaluation/configs/retrieval_final_config_v4.json`:
- `dense_model`: `BAAI/bge-m3` (`EVALUATION_SELECTED`)
- `sparse_retriever`: `BM25Okapi`
- `fusion_method`: `RRF` ($k=60$)
- `broad_candidate_pool_size`: `100`
- `reranker_input_budget`: `20` (Optimal balance of HitRate@10=31.09% and CE P50=158.3ms)
- `reranker_model`: `cross-encoder/ms-marco-TinyBERT-L-2-v2` (`FAST_DEFAULT`)
- `reranker_max_seq_length`: `512`
