# Benchmark Integrity Audit & Diagnostic Report

**Audit Date**: August 14, 2026  
**Auditor Mode**: STRICT BENCHMARK-INTEGRITY REPAIR MODE  
**Repository**: [XLaiHuy/Multi-Agent-RAG](https://github.com/XLaiHuy/Multi-Agent-RAG)  

---

## 1. Inventory of Synthetic, Estimated & Unsubstantiated Metrics

| Artifact / File | Specific Problem Identified | Classification | Required Repair Action | Status |
| :--- | :--- | :--- | :--- | :---: |
| `tests/fixtures/cuad_small/` | Synthetic CI fixture labeled as official CUAD data | Synthetic CI Smoke Fixture | Relabel as `SYNTHETIC CI FIXTURE - NOT OFFICIAL CUAD DATA`. Build real official CUAD downloader & separate datasets. | **PENDING REPAIR** |
| `evaluation/manifests/cuad_manifest.json` | 8 manually written queries labeled with CUAD provenance | Synthetic CI Manifest | Programmatically generate `cuad_official_manifest.json` from official CUAD dataset. | **PENDING REPAIR** |
| `evaluation/reports/ocr_and_multiformat_audit.json` | Hardcoded OCR degradation CER/WER (0.012, 0.038, 0.089...) and downstream recall | Synthetic / Hardcoded | Delete synthetic dictionary. Implement executable OCR rendering, degradation transforms, real OCR execution, and Levenshtein CER/WER calculation. | **PENDING REPAIR** |
| `evaluation/benchmarks/eval_ablation.py` | Ternary hardcoded values for Faithfulness (0.96 vs 0.88), Citation Prec (0.95 vs 0.85), and fixed 380 tokens | Synthetic Estimation | Compute actual Faithfulness and Citation Precision per query from raw text and ground truth blocks. Log raw traces. | **PENDING REPAIR** |
| `evaluation/reports/CLAIM_VERIFICATION.md` | Verification status fields statically written | Static Text | Create executable test runner/script that asserts and outputs programmatic verification results. | **PENDING REPAIR** |
| `README.md` & Reports | Terminology like "100% complete", "fully verified", "6100 platform RPS", "62.5% cost reduction" | Over-claimed / Imprecise Language | Downgrade and qualify language: "Local retrieval microbenchmark", "LLM invocation reduction", "CI smoke benchmark (N=8)". | **PENDING REPAIR** |

---

## 2. Integrity Repair Plan

1. **Phase 1-2**: Document inventory & clearly relabel synthetic CI smoke fixtures in `tests/fixtures/README.md` and `tests/fixtures/cuad_small/`.
2. **Phase 3-4**: Implement `evaluation/download_cuad.py` and `evaluation/prepare_cuad.py` to download and parse official CUAD (Atticus Project v1 / HuggingFace CUAD) with strict SHA-256 validation.
3. **Phase 5-6**: Purge all hardcoded OCR CER/WER and build real OCR degradation pipeline (clean, 200/150/100 DPI, 2°/5° skew, blur, noise) measuring real Levenshtein CER/WER and downstream DeltaRecall.
4. **Phase 7-11**: Execute real multi-format parsing, real 7-variant retrieval ablation, real Adaptive vs Fixed trace logging (`adaptive_trace.jsonl`), real Gemini API runs, and corpus-scale local retrieval microbenchmarks (10, 50, 100 contracts).
5. **Phase 12-15**: Add benchmark integrity regression tests in `tests/test_benchmark_integrity.py`, regenerate all markdown reports programmatically from raw outputs, and update `README.md` with qualified wording.
