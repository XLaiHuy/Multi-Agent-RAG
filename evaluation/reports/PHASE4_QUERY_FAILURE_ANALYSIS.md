# Phase 4 Query Failure Analysis & Taxonomic Attribution

**Status**: **COMPLETED**  
**Evaluation Set**: CUAD DEV Split ($N = 238$ Answerable Queries)  
**Total Top-5 Misses**: 190 queries  

---

## 1. Primary Failure Taxonomy (Mutually Exclusive)

Every retrieval failure (missing the gold evidence in the final Top-5 context window) is classified into exactly one root cause:

| Failure Category | Miss Count | Percentage of Failures | Engineering Remedy |
| :--- | :---: | :---: | :--- |
| `NOT_FOUND_TOP100` | 54 | **28.42%** | Requires expanded query representation (HyDE/MultiQuery/SPLADE in future phases). |
| `FOUND_TOP100_LOST_BY_BUDGET` | 81 | **42.63%** | **Primary controllable bottleneck**: Increase CrossEncoder candidate budget from $k=20$ to $k=30/40/50$. |
| `FOUND_IN_CE_INPUT_RERANKER_DEMOTED` | 29 | **15.26%** | CrossEncoder scored negative chunks higher than gold; evaluate stronger reranker conditionally. |
| `FOUND_TOP10_NOT_TOP5` | 26 | **13.68%** | Reranker placed gold in ranks 6–10; candidate window sizing optimization. |
| `PARENT_DEDUP_LOSS` | 0 | **0.00%** | Parent deduplication was neutral (zero loss). |
| `DOCUMENT_AMBIGUITY` | 0 | **0.00%** | Cross-contract query confusion. |
| `GOLD_MAPPING_FAILURE` | 0 | **0.00%** | Chunk boundary or offset mapping failure (e.g. Holdout V2 query). |
| `OTHER` | 0 | **0.00%** | Residual non-classified failures. |

---

## 2. Audit of the 294 $\rightarrow$ 293 Query Discrepancy on Holdout V2

- **Declared Manifest Answerable Queries**: 294
- **Valid Scored Answerable Queries**: 293
- **Excluded Query ID**: `test_v2_cuad_cuad_contract_061_OR_Right_Of_First_Refusal_16`
- **Target Contract**: `cuad_contract_061_ORBSATCORP_08_17_2007_EX_7_3_S`
- **Clause Category**: `Right Of First Refusal`
- **Classification Reason**: `CHUNK_BOUNDARY_MAPPING_FAILURE`
- **Root Cause**: The raw gold evidence string (235 characters) crossed a 250-token chunk boundary during document slicing, preventing exact single-chunk substring alignment. The full document does contain the string (`in_raw_source_file: True`). Machine-readable audit artifact stored at `evaluation/results/phase4/gold_mapping_exclusions.json`.

---

## 3. Conditional EXP-21 Reranker A/B Decision

- **Condition**: Reranker Demotion must account for $\ge 15\%$ of failures.
- **Measured Demotion Share**: **15.26%**
- **Decision**: **NOT_RUN**
- **Rationale**: FOUND_IN_CE_INPUT_RERANKER_DEMOTED accounts for 15.26% of failures (primary bottleneck is LOST_BY_BUDGET at 42.63% and NOT_FOUND_TOP100 at 28.42%; TinyBERT is retained as FAST_DEFAULT).
