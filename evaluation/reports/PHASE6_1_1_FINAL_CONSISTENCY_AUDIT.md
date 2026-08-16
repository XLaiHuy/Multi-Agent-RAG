# Phase 6.1.1: Final Metric Consistency & Reporting Audit

## 1. Executive Summary
Phase 6.1.1 completed a final consistency patch across all machine-readable JSON artifacts, master claim registries, public documentation, and test assertions with **zero new production API calls** and **zero new judge calls**.

## 2. Refusal Accounting Breakdown ($N=200$)

| Category | Query Count | Percentage |
|---|---|---|
| **Answerable Accepted** | 67 / 100 | 67.00% |
| **Answerable Strict Refusal** (`INSUFFICIENT_EVIDENCE:`) | 27 / 100 | 27.00% |
| **Answerable Ambiguous Prose Refusal** | 6 / 100 | 6.00% |
| **Unanswerable Strict Refusal** (`INSUFFICIENT_EVIDENCE:`) | 78 / 100 | 78.00% |
| **Unanswerable Ambiguous Prose Refusal** | 4 / 100 | 4.00% |
| **Unanswerable Answered (False Answer)** | 18 / 100 | 18.00% |
| **Observed System / Runtime Errors** | 0 / 200 | 0.00% |

### Dual Refusal View
1. **Strict Conservative Evaluation** (Sentinel-only prefix):
   - Strict Balanced Accuracy: **72.50%**
   - Strict Unanswerable Refusal: **78.00%** (78 / 100)
   - Strict False Refusal: **27.00%** (27 / 100)
2. **Inclusive Prose-Aware Evaluation** (Strict Sentinel + Explanatory Prose Refusals):
   - Inclusive Balanced Accuracy: **74.50%**
   - Inclusive Unanswerable Refusal: **82.00%** (82 / 100)
   - Inclusive False Refusal: **33.00%** (33 / 100)

## 3. Dynamically Derived Agent Action Counts

| Agent Component | Invocation Count | Action Breakdown |
|---|---|---|
| **Planner Agent** | 200 calls | 200 structured plans generated, 0 retrieval query modifications |
| **Critic Agent** | 200 calls | 200 proceed decisions, 0 query expansions |
| **Verifier Agent** | 85 calls | 85 passed decisions, 0 regenerations, 0 refusals |

Causal classification: **`PLANNER_PRESENT_NO_ISOLATED_CAUSAL_EFFECT`**.

## 4. Public Documentation & Test Verification
- Zero duplicate Phase 6 blocks in `README.md`.
- Zero broken local machine file URLs.
- Single machine-readable source of truth: `evaluation/results/phase6_1/final_public_metrics.json`.
- 67 / 67 passing unit, security, and metric consistency tests.
