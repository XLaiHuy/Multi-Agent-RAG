# Problem-Solution-Validation Matrix (Phase 4.1 Sign-Off)

| Phase | Identified Root Cause / Challenge | Engineering Solution Implemented | Measured Result & Verification |
| :--- | :--- | :--- | :--- |
| **Phase 1-2** | Semantic drift & multi-hop query failure | Agentic query decomposition & query classification | ACL regression tests passing (7/7) |
| **Phase 3** | Poor first-stage sparse/dense recall | BGE-M3 Dense + BM25Okapi + RRF Fusion ($k=60$) | First-stage recall recovered to >77% |
| **Phase 3.5** | Chunk boundary truncation & orphan chunks | Structure-Aware Parent-Child chunking (250/30 child, 1200/100 parent) | 84.2% gold in single chunk, 0 orphans |
| **Phase 4** | 40-minute evaluation bottleneck | Deterministic SHA-256 Evaluation Cache | **94.70x speedup** (~40.7 min to 25.8s) |
| **Phase 4.1** | Global multi-contract distractor collisions | True Document-Scoped prefiltering before Dense/BM25 | **94.54% Hit@10 / 0.6418 MRR** on Holdout ($N=293$) |
| **Phase 4.1** | Simulated +15ms latency constant | Live `time.perf_counter()` around all stages | Measured P50 = **68.89 ms** on Holdout |
| **Phase 4.1** | Uncertainty over TinyBERT capacity | Empirical A/B test vs BGE-Reranker-Base | TinyBERT verified optimal (90.34% vs 89.08%, 60x faster) |
