# Scientific Validity Matrix (Phase 4.1 Final)

| Component | Status | Artifact / Proof |
| :--- | :--- | :--- |
| **Evaluation Cache Invalidation** | VERIFIED | `tests/unit/test_cache_invalidation.py` passing |
| **Query Encoding Equivalence** | VERIFIED | `tests/unit/test_query_encoding.py` passing (max diff < 1e-6) |
| **True Document-Scoped Retrieval** | VERIFIED | `tests/unit/test_document_scoped_retrieval.py` passing (0 cross-doc leaks) |
| **Production-Measured Latency** | VERIFIED | `retrieval_latency_dev.json` (no simulated constants) |
| **Reranker Optimization** | VERIFIED | `reranker_ab_dev.json` (TinyBERT confirmed optimal on CPU) |
| **Held-Out Frozen Evaluation** | VERIFIED | `final_holdout_doc_scoped.json` (Hit@10=94.54%, MRR=0.6418 on N=293) |
| **Generation / LLM Judge** | NOT_RUN | Explicitly quarantined to prevent CV inflation |
