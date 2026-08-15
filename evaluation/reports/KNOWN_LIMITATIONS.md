# Known Limitations & Engineering Boundaries

**Date**: August 14, 2026  
**Auditor Mode**: STRICT BENCHMARK-INTEGRITY REPAIR MODE  

---

## 1. Documented System Limitations

### 1.1 Local CPU Reranker Latency
- **Observation**: Running `cross-encoder/ms-marco-TinyBERT-L-2-v2` on standard CPU without dedicated GPU acceleration exhibits a P50 latency of **~3.8 seconds per query** for candidate pools of 5–10 chunks.
- **Impact**: Real-time interactive UI queries experience a 2–4s turnaround when neural reranking is triggered.
- **Mitigation / Next Step**: Deploy ONNX Runtime with quantized int8 weights or execute on GPU instances with batched inference.

### 1.2 Multi-Column & Cross-Page PDF Pagination
- **Observation**: `MasterDocumentParser` parses PDF documents using PyMuPDF block layout grouping. Clauses spanning page breaks or split across table columns may be partitioned across separate blocks, resulting in lower initial chunk recall (0.000 R@5 on synthetic multi-page test blocks).
- **Impact**: Complex multi-page PDF agreements require sliding-window chunk boundary stitching or layout-aware OCR.
- **Mitigation / Next Step**: Implement layout-aware document reconstruction with reading order detection.

### 1.3 LLM Quota Boundaries on Public Free Tiers
- **Observation**: Live Gemini API endpoints on free trial tiers enforce a **15 Requests-Per-Minute (RPM)** limit.
- **Impact**: Full 510-contract CUAD batch evaluation across all 41 categories (20,000+ API calls) requires exponential backoff pacing or dedicated Enterprise tier quotas.
- **Mitigation / Next Step**: Evaluate frozen subsets (5–50 queries) in CI and utilize batch inference APIs for large offline audits.

### 1.4 Dataset Scale Qualification
- **Observation**: The current frozen TEST evaluation split comprises **10 official CUAD contracts (585 child chunks, 110 parent blocks, 50 queries)**.
- **Qualification**: While full 510-contract dataset extraction scripts (`evaluation/prepare_cuad.py`) are fully functional, reported ablation numbers are strictly bounded to the 10-contract test set.
