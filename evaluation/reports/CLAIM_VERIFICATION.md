# Claim Verification & Legacy Audit Ledger

**Date**: August 14, 2026  
**Auditor Mode**: STRICT BENCHMARK-INTEGRITY REPAIR MODE  
**Principle**: "A lower real score is better than a perfect fake score."

---

## 1. Claim Verification Matrix

| # | Legacy / Unsubstantiated Claim | Audit Evidence | Real Measured Value | Final Audit Verdict | Required Action Taken |
| :---: | :--- | :--- | :--- | :---: | :--- |
| **1** | *"Evaluated on official CUAD benchmark"* | Tests were originally running against `tests/fixtures/cuad_small/` (8 synthetic queries). | Official CUAD v1 downloaded (Atticus Project `data.zip`, SHA-256 verified). 10 official contracts, 50 leak-free queries evaluated. | **REPAIRED & QUALIFIED** | Relabeled synthetic fixtures. Ran real benchmark against official dataset. |
| **2** | *"Achieved 62.5% cost reduction via adaptive routing"* | Number was derived from synthetic cost multiplier estimates without raw query trace logs. | Fixed Pipeline: 4.00 LLM calls/Q. Adaptive Multi-Agent: 2.36 LLM calls/Q. **Measured reduction: 41.0%**. | **CORRECTED TO MEASURED** | Replaced "62.5% cost reduction" with measured "41.0% LLM invocation reduction". |
| **3** | *"100% OCR accuracy on all degraded scans"* | Hardcoded dictionary in `ocr_and_multiformat_audit.json` with synthetic numbers. | Live Tesseract OCR execution measured Levenshtein CER: Clean (0.000), 100 DPI (0.007), Med Gaussian Noise (0.089). | **REPLACED WITH REAL METRICS** | Replaced synthetic numbers with executable degradation runner. |
| **4** | *"Parent-Child + Reranker achieves 100% Precision"* | Metric came from a 3-contract smoke fixture with exact string matches. | Evaluated across 585 official CUAD chunks: Recall@5=0.0102, MRR=0.0847 (+92.5% over BM25), Citation Prec=0.032. | **CORRECTED TO MEASURED** | Published real empirical metrics from full 7-variant ablation. |
| **5** | *"Multi-format ingestion 100% verified"* | Untested format assertions. | Tested TXT, MD, JSON, DOCX (1.000 R@5) and PDF layout blocks (0.000 R@5 across block boundaries). | **QUALIFIED & DOCUMENTED** | Documented PDF block pagination limitation explicitly. |

---

## 2. Integrity Commitments

1. **No Mock Metrics**: No benchmark metric shall ever be injected as a constant or hardcoded dictionary.
2. **Deterministic Reproducibility**: All evaluation runs produce traceable `.jsonl` files under `evaluation/runs/<run_id>/`.
3. **Qualified Terminology**: Marketing claims ("SOTA", "production-ready 100%") are replaced with empirical engineering statements.
