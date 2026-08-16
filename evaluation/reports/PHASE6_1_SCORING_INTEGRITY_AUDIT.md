# Phase 6.1: Scoring Integrity & Scorer Repair Audit

## 1. Objectives & Corrective Actions
Phase 6.1 audited and repaired three scoring mechanisms:
1. **Citation Fallback Removal**: Eliminated automatic assignment of rank-1 retrieved chunk when model answers without brackets. All citations must be explicitly present in answer text.
2. **Deterministic Refusal Classification**: Classified refusals into strict sentinel (`INSUFFICIENT_EVIDENCE:`), explanatory prose refusal, and standard answers.
3. **Evidence-Grounded Judge Separation**: Separated Groundedness (judged strictly against retrieved chunks supplied to generator) from Semantic Correctness (judged against gold annotations).

## 2. Measured Metric Reconciliations

| Metric Area | Raw Evaluator (Phase 6) | Strict Scorer (Phase 6.1) | Delta / Notes |
|---|---|---|---|
| **Citation Fallback** | Top-1 Chunk Fallback Allowed | **Zero Fallback (Regex Only)** | Scientifically strict |
| **Explicit Citation Compliance** | Not explicitly reported | **98.51%** (84/85) | Model emitted citations in text |
| **Child Citation Hit (Accepted)** | 86.57% (58/67) | **85.07%** (58/67) | Exact gold clause match |
| **Child Citation Coverage (All)** | Not isolated | **62.00%** (58/100) | Full denominator transparency |
| **Citation Precision (Macro)** | 82.84% | **80.97%** | Strict regex span evaluation |
| **Citation Precision (Micro)** | Not reported | **73.53%** | Strict regex span evaluation |
| **Citation Recall (Macro)** | 63.50% | **63.00%** | Strict gold overlap |
| **Grounded Claim Rate** | 100.0% (vs Gold) | **97.93% (vs Retrieved)** | True grounding evaluation |
| **Semantic Correctness** | Conflated with grounding | **92.54% (1.85 / 2.0)** | Independent gold evaluation |
| **Balanced Accuracy** | 74.50% | **74.50%** | Invariant under strict parsing |
| **Unanswerable Refusal** | 82.00% | **82.00%** | Invariant under strict parsing |
| **Answerable Acceptance** | 67.00% | **67.00%** | Invariant under strict parsing |
