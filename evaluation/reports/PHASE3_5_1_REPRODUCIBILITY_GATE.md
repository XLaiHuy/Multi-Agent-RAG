# Phase 3.5.1 Reproducibility Gate Report

**Gate Status**: **PASSED**  
**Execution Type**: Standalone Committed Repository Scripts (PyTorch CPU Inference)  
**Quality Verification**: `compileall` = 0 (100% clean), `pytest` = 42/42 PASS  
**Canonical Configuration**: Neutral shared defaults module `backend/app/core/retrieval_defaults.py`  

---

## 1. Executive Summary & Verification Matrix

All 5 core evaluation benchmarks have been executed from the committed scripts with zero mock data, zero simulated API calls, strict tokenizer-aware inference, and complete machine-readable JSON logging.

| Benchmark Script | Dataset / Split | Queries Evaluated | Exit Code | Wall Clock Runtime | JSON Artifact | Markdown Report |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| `evaluate_locked_test_v1.py` | `LEGACY_LOCKED_TEST_V1` (10 contracts) | 19 answerable | **0** | 1093.39s | [`locked_test_v1.json`](evaluation/results/phase3_5_1/locked_test_v1.json) | [`RETRIEVAL_BENCHMARK_LOCKED_TEST.md`](evaluation/reports/RETRIEVAL_BENCHMARK_LOCKED_TEST.md) |
| `run_external_legalbench.py` | `CUSTOM_CUAD_HOLDOUT_V2` (25 contracts) | 293 answerable | **0** | 2359.18s | [`custom_cuad_holdout_v2.json`](evaluation/results/phase3_5_1/custom_cuad_holdout_v2.json) | [`EXTERNAL_LEGAL_BENCHMARK.md`](evaluation/reports/EXTERNAL_LEGAL_BENCHMARK.md) |
| `run_exp11_candidate_pool_diagnostic.py` | `CUAD_DEV_SPLIT` (20 contracts) | 238 answerable | **0** | 1944.39s | [`candidate_pool.json`](evaluation/results/phase3_5_1/candidate_pool.json) | [`CANDIDATE_POOL_DIAGNOSTIC.md`](evaluation/reports/CANDIDATE_POOL_DIAGNOSTIC.md) |
| `candidate_union_vs_rrf.py` | `CUAD_DEV_SPLIT` (20 contracts) | 238 answerable | **0** | 1914.50s | [`rrf_vs_union.json`](evaluation/results/phase3_5_1/rrf_vs_union.json) | [`CANDIDATE_UNION_VS_RRF_FAIR_COMPARISON.md`](evaluation/reports/CANDIDATE_UNION_VS_RRF_FAIR_COMPARISON.md) |
| `two_stage_pruning_audit.py` | `CUAD_DEV_SPLIT` (20 contracts) | 238 answerable | **0** | 1965.18s | [`waterfall.json`](evaluation/results/phase3_5_1/waterfall.json) | [`POST_PRUNING_RECALL_WATERFALL.md`](evaluation/reports/POST_PRUNING_RECALL_WATERFALL.md) |

---

## 2. Measured Retrieval Results

### A. Legacy Locked Test V1 (N = 19 Answerable Queries)
- **Manifest Hash**: `98ff98d586d4de15f246bad8c2492b22b5fe5d058d4b801d6ebd740d530e36a5`
- **Candidate HitRate @100**: `78.95%`
- **HitRate @1**: `15.79%`
- **HitRate @5**: `26.32%`
- **HitRate @10**: `47.37%`
- **MRR**: `0.2241`
- **Classification**: `README_SAFE` (Informative point metric; NOT `CV_SAFE` due to N=19 sample size).

### B. Custom CUAD Holdout V2 (N = 293 Answerable Queries)
- **Manifest Hash**: `1f0116f95bf24f19a0eb0750a12f889ed8b80da3c4b7644140d8a8477fd71205`
- **Candidate HitRate @100**: `74.06%`
- **HitRate @1**: `4.78%`
- **HitRate @5**: `19.11%`
- **HitRate @10**: `28.67%`
- **MRR**: `0.1078`
- **Classification**: Custom holdout validation split (`CUSTOM_CUAD_HOLDOUT_V2`), NOT official LegalBench-RAG.

### C. EXP-11 Candidate Pool Diagnostic (DEV Split, N = 238 Queries)
| Candidate Budget k | Candidate HitRate @k | True Chunk Recall @k | MRR @k |
| :--- | :---: | :---: | :---: |
| Top-5 | 18.07% | 3.90% | 0.0936 |
| Top-10 | 27.73% | 6.84% | 0.1064 |
| Top-20 | 42.44% | 11.17% | 0.1164 |
| Top-30 | 51.68% | 14.51% | 0.1202 |
| Top-50 | 62.18% | 20.68% | 0.1229 |
| Top-100 | 77.31% | 29.93% | 0.1251 |

### D. Fair Candidate Union vs Equal RRF Comparison (DEV Split, N = 238 Queries)
| Strategy | HitRate @20 | HitRate @50 | HitRate @100 | MRR @100 |
| :--- | :---: | :---: | :---: | :---: |
| Dense Only (BGE-M3) | 43.70% | 59.24% | 71.01% | 0.1207 |
| BM25 Only (Okapi) | 38.24% | 56.30% | 73.53% | 0.1120 |
| Interleaved Union | 40.76% | 62.18% | 78.15% | 0.1210 |
| Equal RRF (k=60) | 42.44% | 62.18% | 77.31% | 0.1251 |

**Finding**: At identical candidate budgets (@20), Equal RRF achieves **42.44%** HitRate vs **40.76%** for Union, and higher MRR (0.1251 vs 0.1210). The claim that Union strictly dominates RRF is scientifically refuted when candidate budgets are held constant.

### E. Post-Pruning Recall & HitRate Waterfall Funnel (DEV Split, N = 238 Queries)
| Stage | Candidate HitRate | True Chunk Recall | Cumulative Hit Loss |
| :--- | :---: | :---: | :---: |
| Stage 1: Raw Top-100 First-Stage Retrieval | 77.31% | 29.93% | Baseline (0.00%) |
| Stage 2: After Parent Deduplication (Max 2 child/parent) | 77.31% | 24.66% | **+0.00%** (Hit preserved 100%) |
| Stage 3: After Top-20 Truncation (Reranker budget) | 43.28% | 10.64% | **+34.03%** (Bottleneck) |
| Stage 4: After CrossEncoder Top-10 | 31.09% | 7.08% | **+12.18%** |
| Stage 5: Final Top-5 Context Window | 20.17% | 4.23% | **+10.92%** |

- **CrossEncoder Latency**: P50 = `157.9ms`, P95 = `253.5ms`, Inference Failures = `0`.

---

## 3. Scientific Validity & Claim Downgrades

1. **BGE-M3 Status**: Classified as `EVALUATION_SELECTED` / `OPTIONAL_HIGH_ACCURACY`. Default production embedding remains `BAAI/bge-small-en-v1.5` (384-d) for low-latency CPU operation until production trade-offs justify migration.
2. **Locked Test V1**: Classified as `README_SAFE`. It is an informative historical point metric and must NOT be cited as a rigorous CV metric due to sample size $N=19$.
3. **Agent Real API Benchmark**: Formally categorized as `REAL_API_NOT_RUN`. No simulated token or cost figures are published.
4. **Official LegalBench-RAG**: Categorized as `NOT_RUN`. All holdout evaluations are transparently labeled as `CUSTOM_CUAD_HOLDOUT_V2`.
5. **Soft Routing EXP-12**: Downgraded to `REJECTED_NEGLIGIBLE_GAIN` (+0.42% on DEV not statistically reproducible across holdout).

---

## 4. Reproducibility Manifest

```bash
# To independently reproduce all benchmarks from repository root:
python -m compileall backend evaluation tests scripts
pytest tests/
python evaluation/scripts/evaluate_locked_test_v1.py
python evaluation/scripts/run_external_legalbench.py
python evaluation/scripts/run_exp11_candidate_pool_diagnostic.py
python evaluation/scripts/candidate_union_vs_rrf.py
python evaluation/scripts/two_stage_pruning_audit.py
```
