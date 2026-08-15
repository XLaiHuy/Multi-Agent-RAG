# Phase 4 Task Formulation Audit: Global Search vs Document-Scoped QA

**Status**: **COMPLETED**  
**Evaluation Split**: CUAD DEV Split (20 Contracts, 238 valid answerable queries)  
**Dense Model**: `BAAI/bge-m3` (1024-d) | **Sparse**: `BM25Okapi` | **Fusion**: `RRF (k=60)` | **Reranker**: `TinyBERT`  

---

## 1. Executive Summary & Core Finding

This audit tests the hypothesis that the low recall of global multi-contract legal retrieval is driven by **inherent cross-contract document ambiguity** rather than weakness in clause-level semantic ranking.

### Product Workflow Definitions:
- **Workflow A — GLOBAL MULTI-CONTRACT SEARCH**:
  - *Input*: Query text only (e.g., *"Which jurisdiction governs this agreement?"*).
  - *Search Scope*: All 20 authorized contracts (1034 total candidate chunks).
  - *Constraint*: Document identity is NOT provided; the system must disambiguate across all contracts.
- **Workflow B — DOCUMENT-SCOPED CONTRACT QA**:
  - *Input*: Query text + `selected_document_id` (explicit user session / active document context).
  - *Search Scope*: Only the active target contract (~61.6 chunks on average).
  - *Constraint*: Simulates standard product UX where a lawyer opens a specific contract and queries it.

---

## 2. Comparative Retrieval Matrix

| Metric | GLOBAL MULTI-CONTRACT | DOCUMENT-SCOPED QA | Absolute Delta ($\Delta$) | Relative Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Searchable Chunk Count** | 1034 chunks | 61.6 chunks | -972.4 | -95.1% corpus noise |
| **Candidate HitRate @5** | 18.07% | 70.17% | **+52.10%** | +288.4% |
| **Candidate HitRate @10** | 27.73% | 73.11% | **+45.38%** | +163.6% |
| **Candidate HitRate @20** | 42.44% | 76.05% | **+33.61%** | +79.2% |
| **Candidate HitRate @50** | 62.18% | 77.31% | **+15.13%** | +24.3% |
| **Candidate HitRate @100** | 77.31% | 77.31% | **+0.00%** | +0.0% |
| **True Chunk Recall @20** | 11.17% | 29.20% | **+18.03%** | +161.4% |
| **Post-Rerank HitRate @1** | 5.46% | 44.96% | **+39.50%** | +723.1% |
| **Post-Rerank HitRate @5** | 20.17% | 70.59% | **+50.42%** | +250.0% |
| **Post-Rerank HitRate @10** | 31.09% | 75.21% | **+44.12%** | +141.9% |
| **Post-Rerank MRR** | 0.1173 | 0.5529 | **+0.4356** | +371.3% |
| **Post-Rerank nDCG@5** | 0.0477 | 0.2856 | **+0.2379** | +498.5% |

---

## 3. Query Ambiguity Breakdown

Query categories were bucketed based on how many contracts in the corpus possess positive annotations for that clause category:

| Ambiguity Bucket | Description | Query Count | Global HitRate @5 | Global HitRate @10 | Global MRR |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `UNIQUE_DOCUMENT` | Clause category exists in exactly 1 contract | 3 | 66.67% | 66.67% | 0.5000 |
| `2_TO_3_POSSIBLE_DOCUMENTS` | Clause category exists in 2–3 contracts | 7 | 57.14% | 71.43% | 0.3299 |
| `4_PLUS_POSSIBLE_DOCUMENTS` | Clause category exists in $\ge 4$ contracts (e.g. Governing Law, Termination) | 228 | 18.42% | 29.39% | 0.1057 |

---

## 4. Scientific Conclusion & Product Interpretation

> [!IMPORTANT]
> **Interpretation Rule**: We do **NOT** claim that our retriever was modified or artificially improved.
> **Scientific Finding**: Global multi-contract retrieval across generic legal questions suffers from substantial cross-contract ambiguity. When the product workflow explicitly supplies the active `document_id` (Document-Scoped QA), the retrieval HitRate @10 jumps from **31.09%** to **75.21%** (+44.12 percentage points) and MRR from **0.1173** to **0.5529**.
