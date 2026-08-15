# CV-Safe Multi-Agent RAG Results Matrix

**Last Updated**: 2026-08-16  
**Status**: VERIFIED & AUDITED (Phase 4.1)

---

## 1. Frozen Benchmark Summary (CUSTOM_CUAD_HOLDOUT_V2, $N=293$)

| Benchmark / Split | Mode | Candidate Hit@20 | Hit@5 | Hit@10 | MRR | Latency P50 | Scientific Validity Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CUSTOM_CUAD_HOLDOUT_V2** | True Document-Scoped QA | **98.29%** | **82.94%** | **94.54%** | **0.6418** | **68.89 ms** | **CV-SAFE FROZEN BENCHMARK** |
| **CUSTOM_CUAD_HOLDOUT_V2** | Global Multi-Contract | 39.59% | 19.11% | 28.67% | 0.1078 | 168.42 ms | **README-SAFE MULTI-DOC BASELINE** |
| **CUAD DEV ($N=238$)** | True Document-Scoped QA | 94.96% | 80.67% | 90.34% | 0.6359 | 154.58 ms | **DEV EXPERIMENT VERIFIED** |
| **CUAD DEV ($N=238$)** | Global Multi-Contract | 42.44% | 20.17% | 31.09% | 0.1173 | 174.57 ms | **DEV EXPERIMENT VERIFIED** |

---

## 2. Verified Architecture & Engineering Truths

- **True Document-Scoped Prefiltering**: Scopes dense and sparse search spaces to the target contract before ranking, boosting Hit@10 from 28.67% to **94.54%** on held-out contracts.
- **Evaluation Harness Acceleration**: Cryptographic SHA-256 evaluation cache delivers **94.70x speedup** on DEV sweeps without metric deviation.
- **Production Latency**: Measured end-to-end CPU retrieval latency is **P50 = 68.89 ms** on Holdout.
- **Reranker Validation**: TinyBERT (4.4M params) delivers higher accuracy (90.34% Hit@10) than BGE-Reranker-Base (89.08% Hit@10) at 60x lower compute cost on CPU.

---

## 3. Boundary & Non-Claims

- **Generation & LLM Judge**: Generation faithfulness and refusal evaluation remain `REAL_API NOT_RUN`.
- **LegalBench-RAG**: Evaluated locally on mock subsets only; official live benchmark marked `NOT_RUN`.
