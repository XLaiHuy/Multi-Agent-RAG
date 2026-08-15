# RETRIEVAL_BENCHMARK_REAL_LOCAL

**Run ID**: `retrieval_real_20260814_181902_f1eea1`  
**Evaluation Date**: 2026-08-14 18:31 UTC+7  
**Mode**: REAL LOCAL — No LLM API calls. Pure retrieval metrics only.  
**Dataset**: Official CUAD v1 Frozen TEST Split (10 contracts)  
**Answerable Queries Evaluated**: 19  
**Unanswerable Queries Excluded**: 31  

> [!IMPORTANT]
> This report covers ONLY retrieval-stage metrics.
> Faithfulness and Citation metrics are EXCLUDED — they require LLM generation and are reported in `RAG_BENCHMARK_REAL_API.md`.

---

## Audit Corrections Applied

| # | Bug Found | Fix Applied |
|---|-----------|-------------|
| 1 | Variant A (`Dense_Only`) was using BM25 results — not dense at all | Fixed: Uses real `InMemoryDenseRetriever` with BAAI/bge-small-en-v1.5 embeddings + cosine similarity |
| 2 | Hybrid RRF fused `[bm25_ids, reversed(bm25_ids[:10])]` — BM25 vs itself | Fixed: Fuses real BM25 ranked list + real Dense ranked list |
| 3 | Gold evidence mapping used 80-char word overlap with fallback to first chunk | Fixed: Uses exact substring search in chunk text + parent_text, then 60% word overlap |
| 4 | HitRate@5/10 not computed | Fixed: Added `compute_hit_rate_at_k` to all variants |
| 5 | Faithfulness/Citation in local benchmark were not from LLM generation | Fixed: Removed from local retrieval report |
| 6 | Unanswerable queries included in retrieval metrics | Fixed: Excluded; only answerable queries evaluated |
| 7 | No OCR downstream retrieval degradation test | Fixed: Added BM25 retrieval on simulated CER-degraded corpus |

---

## 7-Variant Retrieval Ablation Results

**Corpus**: 10 Official CUAD Contracts, ~585 child chunks (~250 tokens each)  
**Answerable Queries**: 19  
**Embeddings**: BAAI/bge-small-en-v1.5 (local CPU)  
**Reranker**: cross-encoder/ms-marco-TinyBERT-L-2-v2 (local CPU)  

| Variant | Recall@5 | Recall@10 | HitRate@5 | HitRate@10 | MRR | nDCG@5 | P50 (ms) | P95 (ms) | Avg LLM Calls/Q |
|---------|----------|-----------|-----------|------------|-----|--------|----------|----------|-----------------|
| **A_Dense_Only** | 0.0526 | 0.0848 | 0.2632 | 0.3684 | 0.1330 | 0.0591 | 33.1 | 50.7 | 1.0 |
| **B_BM25_Only** | 0.0453 | 0.0576 | 0.2632 | 0.3684 | 0.1048 | 0.0458 | 34.9 | 43.9 | 1.0 |
| **C_Hybrid_RRF** | 0.0439 | 0.0673 | 0.2105 | 0.3684 | 0.1500 | 0.0585 | 32.2 | 58.6 | 1.0 |
| **D_Hybrid_ParentChild** | 0.0439 | 0.0673 | 0.2105 | 0.3684 | 0.1500 | 0.0585 | 32.2 | 36.1 | 1.0 |
| **E_Hybrid_ParentChild_Reranker** | 0.0576 | 0.1073 | 0.3684 | 0.4737 | 0.2527 | 0.0930 | 9952.0 | 17197.3 | 1.0 |
| **F_Fixed_Full_Pipeline** | 0.0576 | 0.1073 | 0.3684 | 0.4737 | 0.2527 | 0.0930 | 12319.9 | 18903.5 | 4.0 |
| **G_Adaptive_MultiAgent** | 0.0576 | 0.1073 | 0.3684 | 0.4737 | 0.2527 | 0.0930 | 14142.4 | 22655.5 | 2.16 |

---

## Dense vs BM25 Independence Verification

| Metric | Dense Only (A) | BM25 Only (B) | Identical? |
|--------|----------------|----------------|------------|
| Recall@5 | 0.0526 | 0.0453 | ✅ Different |
| HitRate@5 | 0.2632 | 0.2632 |  |
| MRR | 0.133 | 0.1048 |  |

> Note: If Dense and BM25 produce identical aggregate metrics, the per-query ranked lists below will show whether they are truly retrieving different documents.

---

## OCR Downstream Retrieval Degradation

Simulates OCR character substitution errors at empirically measured CER levels from the live Tesseract run.  
Tests BM25 retrieval quality on artificially degraded contract corpus.  

| Degradation Condition | CER | Queries | Mean Recall@5 | Mean HitRate@5 | Mean MRR |
|----------------------|-----|---------|---------------|----------------|----------|
| clean | 0.000 | 10 | 0.3861 | 0.9000 | 0.8643 |
| mild_noise_cer_0.007 | 0.007 | 10 | 0.4028 | 0.9000 | 0.9143 |
| medium_noise_cer_0.089 | 0.089 | 10 | 0.3861 | 0.9000 | 0.8458 |

---

## Per-Query Manual Inspection (First 10 Answerable Queries)

### Query: `eval_cuad_contract_00_Document_Name`

**Question**: What is the official title and full document name of this agreement?  
**Source Contract**: `cuad_contract_003_CENTRACKINTERNATIONALINC_10_29`  
**Gold Evidence**: `WEB SITE HOSTING AGREEMENT`  
**Gold Answer Start (offset)**: 225  

**Gold Chunk IDs**: `['cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p0_c0', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p0_c1', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p0_c2', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p0_c3', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p0_c4', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p1_c0', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p1_c1', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p1_c2', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p1_c3', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p1_c4', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p1_c5', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p2_c0', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p2_c1', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p2_c2', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p2_c3']`  
**Gold ID Count**: 15  
**Gold Mapping Methods**: ['exact_substring', 'exact_substring', 'exact_substring', 'exact_substring', 'exact_substring', 'word_overlap', 'word_overlap', 'word_overlap', 'word_overlap', 'word_overlap', 'word_overlap', 'word_overlap', 'word_overlap', 'word_overlap', 'word_overlap']  

| Rank | Dense chunk_id | BM25 chunk_id | Hybrid chunk_id | Reranked chunk_id | Is Gold? |
|------|---------------|---------------|-----------------|-------------------|----------|
| 1 | `OPHARMA_INC_05_11_202_v1_p2_c5` | `KS_INC_02_18_2016_EX__v1_p7_c3` | `KS_INC_02_18_2016_EX__v1_p7_c3` | `RMACEUTICALS__INC____v1_p19_c4` |  |
| 2 | `OPHARMA_INC_05_11_202_v1_p2_c0` | `nc____Remarketing_Ag_v1_p18_c5` | `OPHARMA_INC_05_11_202_v1_p2_c5` | `ceuticalsInc_2018110_v1_p29_c4` |  |
| 3 | `ARMACEUTICALS__INC____v1_p0_c2` | `NC_02_21_2020_EX_10_1_v1_p1_c2` | `RMACEUTICALS__INC____v1_p13_c2` | `NTERNATIONALINC_10_29_v1_p2_c3`✓ | ✓ |
| 4 | `RMACEUTICALS__INC____v1_p19_c4` | `nc____Remarketing_Ag_v1_p18_c2` | `KS_INC_02_18_2016_EX__v1_p6_c3` | `nc____Remarketing_Ag_v1_p18_c2` |  |
| 5 | `KS_INC_02_18_2016_EX__v1_p7_c3` | `NC_02_21_2020_EX_10_1_v1_p1_c3` | `nc____Remarketing_Ag_v1_p18_c5` | `OPHARMA_INC_05_11_202_v1_p2_c0` |  |
| 6 | `RMACEUTICALS__INC____v1_p13_c2` | `nc____Remarketing_Ag_v1_p18_c4` | `OPHARMA_INC_05_11_202_v1_p2_c0` | `OPHARMA_INC_05_11_202_v1_p2_c5` |  |
| 7 | `ARMACEUTICALS__INC____v1_p2_c2` | `ceuticalsInc_2018110_v1_p29_c4` | `NC_02_21_2020_EX_10_1_v1_p1_c2` | `ARMACEUTICALS__INC____v1_p2_c2` |  |
| 8 | `ARMACEUTICALS__INC____v1_p2_c1` | `NTERNATIONALINC_10_29_v1_p2_c3`✓ | `ARMACEUTICALS__INC____v1_p0_c2` | `RMACEUTICALS__INC____v1_p13_c2` | ✓ |
| 9 | `KS_INC_02_18_2016_EX__v1_p6_c5` | `nc____Remarketing_Ag_v1_p18_c3` | `nc____Remarketing_Ag_v1_p18_c2` | `NC_02_21_2020_EX_10_1_v1_p1_c2` |  |
| 10 | `KS_INC_02_18_2016_EX__v1_p6_c3` | `OPHARMA_INC_05_11_202_v1_p2_c5` | `RMACEUTICALS__INC____v1_p19_c4` | `nc____Remarketing_Ag_v1_p18_c4` |  |

**Per-Query Metrics Across Variants:**

| Variant | Recall@5 | HitRate@5 | MRR | First Relevant Rank |
|---------|----------|-----------|-----|---------------------|
| A_Dense_Only | 0.0000 | 0.0000 | 0.0000 | not found |
| B_BM25_Only | 0.0000 | 0.0000 | 0.1250 | 8 |
| C_Hybrid_RRF | 0.0000 | 0.0000 | 0.0000 | not found |
| D_Hybrid_ParentChild | 0.0000 | 0.0000 | 0.0000 | not found |
| E_Hybrid_ParentChild_Reranker | 0.0667 | 1.0000 | 0.3333 | 3 |
| F_Fixed_Full_Pipeline | 0.0667 | 1.0000 | 0.3333 | 3 |
| G_Adaptive_MultiAgent | 0.0667 | 1.0000 | 0.3333 | 3 |

### Query: `eval_cuad_contract_00_Parties`

**Question**: Who are the named parties entering into this agreement?  
**Source Contract**: `cuad_contract_003_CENTRACKINTERNATIONALINC_10_29`  
**Gold Evidence**: `Centrack International`  
**Gold Answer Start (offset)**: 330  

**Gold Chunk IDs**: `['cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p0_c0', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p0_c1', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p0_c2', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p0_c3', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p0_c4', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p2_c0', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p2_c1', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p2_c2', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p2_c3']`  
**Gold ID Count**: 9  
**Gold Mapping Methods**: ['exact_substring', 'exact_substring', 'exact_substring', 'exact_substring', 'exact_substring', 'exact_substring', 'exact_substring', 'exact_substring', 'exact_substring']  

| Rank | Dense chunk_id | BM25 chunk_id | Hybrid chunk_id | Reranked chunk_id | Is Gold? |
|------|---------------|---------------|-----------------|-------------------|----------|
| 1 | `RMACEUTICALS__INC____v1_p22_c4` | `nc____Remarketing_Ag_v1_p16_c0` | `NC_02_21_2020_EX_10_1_v1_p0_c1` | `aceuticalsInc_2018110_v1_p2_c2` |  |
| 2 | `ARMACEUTICALS__INC____v1_p0_c2` | `ARMACEUTICALS__INC____v1_p2_c1` | `RMACEUTICALS__INC____v1_p22_c4` | `NTERNATIONALINC_10_29_v1_p2_c3`✓ | ✓ |
| 3 | `RMACEUTICALS__INC____v1_p13_c2` | `OPHARMA_INC_05_11_202_v1_p2_c4` | `RMACEUTICALS__INC____v1_p13_c2` | `aceuticalsInc_2018110_v1_p0_c3` |  |
| 4 | `RMACEUTICALS__INC____v1_p23_c2` | `NTERNATIONALINC_10_29_v1_p2_c3`✓ | `nc____Remarketing_Ag_v1_p16_c0` | `RMACEUTICALS__INC____v1_p13_c2` | ✓ |
| 5 | `NC_02_21_2020_EX_10_1_v1_p0_c1` | `nc____Remarketing_Ag_v1_p17_c5` | `ARMACEUTICALS__INC____v1_p2_c1` | `KS_INC_02_18_2016_EX__v1_p0_c0` |  |
| 6 | `OPHARMA_INC_05_11_202_v1_p2_c5` | `aceuticalsInc_2018110_v1_p2_c2` | `ARMACEUTICALS__INC____v1_p0_c2` | `nc____Remarketing_Ag_v1_p17_c5` |  |
| 7 | `RMACEUTICALS__INC____v1_p22_c3` | `NC_02_21_2020_EX_10_1_v1_p0_c1` | `OPHARMA_INC_05_11_202_v1_p2_c4` | `RMACEUTICALS__INC____v1_p22_c4` |  |
| 8 | `KS_INC_02_18_2016_EX__v1_p6_c5` | `aceuticalsInc_2018110_v1_p0_c3` | `NTERNATIONALINC_10_29_v1_p2_c3`✓ | `NTERNATIONALINC_10_29_v1_p2_c2`✓ | ✓ |
| 9 | `NTERNATIONALINC_10_29_v1_p2_c2`✓ | `KS_INC_02_18_2016_EX__v1_p0_c0` | `RMACEUTICALS__INC____v1_p23_c2` | `ARMACEUTICALS__INC____v1_p2_c1` | ✓ |
| 10 | `NC_02_21_2020_EX_10_1_v1_p0_c4` | `aceuticalsInc_2018110_v1_p2_c5` | `nc____Remarketing_Ag_v1_p17_c5` | `NC_02_21_2020_EX_10_1_v1_p0_c1` |  |

**Per-Query Metrics Across Variants:**

| Variant | Recall@5 | HitRate@5 | MRR | First Relevant Rank |
|---------|----------|-----------|-----|---------------------|
| A_Dense_Only | 0.0000 | 0.0000 | 0.1111 | 9 |
| B_BM25_Only | 0.1111 | 1.0000 | 0.2500 | 4 |
| C_Hybrid_RRF | 0.0000 | 0.0000 | 0.1250 | 8 |
| D_Hybrid_ParentChild | 0.0000 | 0.0000 | 0.1250 | 8 |
| E_Hybrid_ParentChild_Reranker | 0.1111 | 1.0000 | 0.5000 | 2 |
| F_Fixed_Full_Pipeline | 0.1111 | 1.0000 | 0.5000 | 2 |
| G_Adaptive_MultiAgent | 0.1111 | 1.0000 | 0.5000 | 2 |

### Query: `eval_cuad_contract_00_Agreement_Date`

**Question**: What is the effective execution date of this agreement?  
**Source Contract**: `cuad_contract_003_CENTRACKINTERNATIONALINC_10_29`  
**Gold Evidence**: `6th day of April, 1999`  
**Gold Answer Start (offset)**: 292  

**Gold Chunk IDs**: `['cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p0_c0', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p0_c1', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p0_c2', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p0_c3', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p0_c4']`  
**Gold ID Count**: 5  
**Gold Mapping Methods**: ['exact_substring', 'exact_substring', 'exact_substring', 'exact_substring', 'exact_substring']  

| Rank | Dense chunk_id | BM25 chunk_id | Hybrid chunk_id | Reranked chunk_id | Is Gold? |
|------|---------------|---------------|-----------------|-------------------|----------|
| 1 | `NTERNATIONALINC_10_29_v1_p1_c5` | `KS_INC_02_18_2016_EX__v1_p6_c3` | `RMACEUTICALS__INC____v1_p13_c2` | `NC_02_21_2020_EX_10_1_v1_p0_c0` |  |
| 2 | `RMACEUTICALS__INC____v1_p23_c3` | `ceuticalsInc_2018110_v1_p23_c4` | `INC_03_21_2005_EX_10__v1_p3_c2` | `NTERNATIONALINC_10_29_v1_p1_c5` |  |
| 3 | `RMACEUTICALS__INC____v1_p18_c3` | `ceuticalsInc_2018110_v1_p22_c2` | `ARMACEUTICALS__INC____v1_p5_c3` | `ceuticalsInc_2018110_v1_p22_c2` |  |
| 4 | `ARMACEUTICALS__INC____v1_p5_c0` | `INC_03_21_2005_EX_10__v1_p3_c2` | `Inc____Remarketing_Ag_v1_p4_c3` | `OPHARMA_INC_05_11_202_v1_p1_c1` |  |
| 5 | `RMACEUTICALS__INC____v1_p13_c2` | `RMACEUTICALS__INC____v1_p13_c2` | `OPHARMA_INC_05_11_202_v1_p2_c5` | `RMACEUTICALS__INC____v1_p23_c2` |  |
| 6 | `Inc____Remarketing_Ag_v1_p9_c2` | `RMACEUTICALS__INC____v1_p21_c1` | `KS_INC_02_18_2016_EX__v1_p6_c3` | `OPHARMA_INC_05_11_202_v1_p2_c5` |  |
| 7 | `NC_02_21_2020_EX_10_1_v1_p0_c0` | `Inc____Remarketing_Ag_v1_p4_c3` | `NTERNATIONALINC_10_29_v1_p1_c5` | `ceuticalsInc_2018110_v1_p23_c4` |  |
| 8 | `KS_INC_02_18_2016_EX__v1_p0_c2` | `RMACEUTICALS__INC____v1_p22_c0` | `ceuticalsInc_2018110_v1_p23_c4` | `INC_03_21_2005_EX_10__v1_p3_c2` |  |
| 9 | `INC_03_21_2005_EX_10__v1_p3_c2` | `RMACEUTICALS__INC____v1_p23_c2` | `RMACEUTICALS__INC____v1_p23_c3` | `RMACEUTICALS__INC____v1_p23_c3` |  |
| 10 | `OPHARMA_INC_05_11_202_v1_p1_c1` | `ARMACEUTICALS__INC____v1_p2_c1` | `ceuticalsInc_2018110_v1_p22_c2` | `RMACEUTICALS__INC____v1_p18_c3` |  |

**Per-Query Metrics Across Variants:**

| Variant | Recall@5 | HitRate@5 | MRR | First Relevant Rank |
|---------|----------|-----------|-----|---------------------|
| A_Dense_Only | 0.0000 | 0.0000 | 0.0000 | not found |
| B_BM25_Only | 0.0000 | 0.0000 | 0.0000 | not found |
| C_Hybrid_RRF | 0.0000 | 0.0000 | 0.0000 | not found |
| D_Hybrid_ParentChild | 0.0000 | 0.0000 | 0.0000 | not found |
| E_Hybrid_ParentChild_Reranker | 0.0000 | 0.0000 | 0.0000 | not found |
| F_Fixed_Full_Pipeline | 0.0000 | 0.0000 | 0.0000 | not found |
| G_Adaptive_MultiAgent | 0.0000 | 0.0000 | 0.0000 | not found |

### Query: `eval_cuad_contract_00_Effective_Date`

**Question**: What is the designated effective start date specified in the agreement?  
**Source Contract**: `cuad_contract_003_CENTRACKINTERNATIONALINC_10_29`  
**Gold Evidence**: `The term of this Agreement for the Hosted Site shall commence upon April 1, 1999 and shall continue for a period of six (6) months, unless earlier terminated in accordance with provisions hereof.`  
**Gold Answer Start (offset)**: 10363  

**Gold Chunk IDs**: `['cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p1_c0', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p1_c1', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p1_c2', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p1_c3', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p1_c4', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p1_c5']`  
**Gold ID Count**: 6  
**Gold Mapping Methods**: ['exact_substring', 'exact_substring', 'exact_substring', 'exact_substring', 'exact_substring', 'exact_substring']  

| Rank | Dense chunk_id | BM25 chunk_id | Hybrid chunk_id | Reranked chunk_id | Is Gold? |
|------|---------------|---------------|-----------------|-------------------|----------|
| 1 | `RMACEUTICALS__INC____v1_p23_c3` | `aceuticalsInc_2018110_v1_p9_c5` | `ARMACEUTICALS__INC____v1_p5_c0` | `ceuticalsInc_2018110_v1_p29_c4` |  |
| 2 | `RMACEUTICALS__INC____v1_p18_c3` | `aceuticalsInc_2018110_v1_p9_c2` | `RMACEUTICALS__INC____v1_p14_c5` | `aceuticalsInc_2018110_v1_p1_c4` |  |
| 3 | `aceuticalsInc_2018110_v1_p1_c4` | `RMACEUTICALS__INC____v1_p21_c1` | `aceuticalsInc_2018110_v1_p9_c5` | `RMACEUTICALS__INC____v1_p18_c3` |  |
| 4 | `NTERNATIONALINC_10_29_v1_p1_c5`✓ | `ARMACEUTICALS__INC____v1_p2_c1` | `RMACEUTICALS__INC____v1_p23_c3` | `RMACEUTICALS__INC____v1_p14_c5` | ✓ |
| 5 | `aceuticalsInc_2018110_v1_p1_c3` | `RMACEUTICALS__INC____v1_p15_c0` | `aceuticalsInc_2018110_v1_p9_c2` | `aceuticalsInc_2018110_v1_p1_c3` |  |
| 6 | `ARMACEUTICALS__INC____v1_p2_c2` | `aceuticalsInc_2018110_v1_p2_c1` | `RMACEUTICALS__INC____v1_p18_c3` | `RMACEUTICALS__INC____v1_p21_c2` |  |
| 7 | `ARMACEUTICALS__INC____v1_p5_c0` | `RMACEUTICALS__INC____v1_p14_c5` | `RMACEUTICALS__INC____v1_p21_c1` | `RMACEUTICALS__INC____v1_p23_c3` |  |
| 8 | `RMACEUTICALS__INC____v1_p21_c2` | `Inc____Remarketing_Ag_v1_p1_c4` | `aceuticalsInc_2018110_v1_p1_c4` | `NTERNATIONALINC_10_29_v1_p1_c5`✓ | ✓ |
| 9 | `OPHARMA_INC_05_11_202_v1_p2_c0` | `ceuticalsInc_2018110_v1_p29_c4` | `ARMACEUTICALS__INC____v1_p2_c1` | `RMACEUTICALS__INC____v1_p15_c0` |  |
| 10 | `ceuticalsInc_2018110_v1_p26_c1` | `KS_INC_02_18_2016_EX__v1_p6_c3` | `NTERNATIONALINC_10_29_v1_p1_c5`✓ | `ARMACEUTICALS__INC____v1_p2_c2` | ✓ |

**Per-Query Metrics Across Variants:**

| Variant | Recall@5 | HitRate@5 | MRR | First Relevant Rank |
|---------|----------|-----------|-----|---------------------|
| A_Dense_Only | 0.1667 | 1.0000 | 0.2500 | 4 |
| B_BM25_Only | 0.0000 | 0.0000 | 0.0000 | not found |
| C_Hybrid_RRF | 0.0000 | 0.0000 | 0.1000 | 10 |
| D_Hybrid_ParentChild | 0.0000 | 0.0000 | 0.1000 | 10 |
| E_Hybrid_ParentChild_Reranker | 0.0000 | 0.0000 | 0.1250 | 8 |
| F_Fixed_Full_Pipeline | 0.0000 | 0.0000 | 0.1250 | 8 |
| G_Adaptive_MultiAgent | 0.0000 | 0.0000 | 0.1250 | 8 |

### Query: `eval_cuad_contract_00_Expiration_Date`

**Question**: When does the initial term of this agreement expire or terminate?  
**Source Contract**: `cuad_contract_003_CENTRACKINTERNATIONALINC_10_29`  
**Gold Evidence**: `The term of this Agreement for the Hosted Site shall commence upon April 1, 1999 and shall continue for a period of six (6) months, unless earlier terminated in accordance with provisions hereof.`  
**Gold Answer Start (offset)**: 10363  

**Gold Chunk IDs**: `['cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p1_c0', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p1_c1', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p1_c2', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p1_c3', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p1_c4', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p1_c5']`  
**Gold ID Count**: 6  
**Gold Mapping Methods**: ['exact_substring', 'exact_substring', 'exact_substring', 'exact_substring', 'exact_substring', 'exact_substring']  

| Rank | Dense chunk_id | BM25 chunk_id | Hybrid chunk_id | Reranked chunk_id | Is Gold? |
|------|---------------|---------------|-----------------|-------------------|----------|
| 1 | `RMACEUTICALS__INC____v1_p18_c3` | `RMACEUTICALS__INC____v1_p18_c3` | `RMACEUTICALS__INC____v1_p18_c3` | `NTERNATIONALINC_10_29_v1_p1_c5`✓ | ✓ |
| 2 | `NTERNATIONALINC_10_29_v1_p1_c5`✓ | `NTERNATIONALINC_10_29_v1_p1_c5`✓ | `NTERNATIONALINC_10_29_v1_p1_c5`✓ | `RMACEUTICALS__INC____v1_p18_c3` | ✓ |
| 3 | `ARMACEUTICALS__INC____v1_p5_c4` | `KS_INC_02_18_2016_EX__v1_p0_c1` | `ARMACEUTICALS__INC____v1_p5_c4` | `KS_INC_02_18_2016_EX__v1_p0_c0` |  |
| 4 | `RMACEUTICALS__INC____v1_p18_c4` | `RMACEUTICALS__INC____v1_p20_c3` | `RMACEUTICALS__INC____v1_p18_c4` | `KS_INC_02_18_2016_EX__v1_p0_c1` |  |
| 5 | `ceuticalsInc_2018110_v1_p27_c2` | `RMACEUTICALS__INC____v1_p18_c4` | `KS_INC_02_18_2016_EX__v1_p0_c1` | `ceuticalsInc_2018110_v1_p27_c0` |  |
| 6 | `ceuticalsInc_2018110_v1_p26_c1` | `ARMACEUTICALS__INC____v1_p5_c4` | `RMACEUTICALS__INC____v1_p21_c2` | `ARMACEUTICALS__INC____v1_p5_c4` |  |
| 7 | `KS_INC_02_18_2016_EX__v1_p0_c1` | `RMACEUTICALS__INC____v1_p21_c2` | `ceuticalsInc_2018110_v1_p27_c0` | `RMACEUTICALS__INC____v1_p19_c3` |  |
| 8 | `ceuticalsInc_2018110_v1_p27_c0` | `ceuticalsInc_2018110_v1_p27_c0` | `ceuticalsInc_2018110_v1_p26_c4` | `ceuticalsInc_2018110_v1_p26_c1` |  |
| 9 | `RMACEUTICALS__INC____v1_p21_c2` | `ceuticalsInc_2018110_v1_p26_c0` | `KS_INC_02_18_2016_EX__v1_p0_c2` | `RMACEUTICALS__INC____v1_p18_c4` |  |
| 10 | `RMACEUTICALS__INC____v1_p19_c3` | `KS_INC_02_18_2016_EX__v1_p0_c0` | `RMACEUTICALS__INC____v1_p20_c3` | `ceuticalsInc_2018110_v1_p26_c4` |  |

**Per-Query Metrics Across Variants:**

| Variant | Recall@5 | HitRate@5 | MRR | First Relevant Rank |
|---------|----------|-----------|-----|---------------------|
| A_Dense_Only | 0.1667 | 1.0000 | 0.5000 | 2 |
| B_BM25_Only | 0.1667 | 1.0000 | 0.5000 | 2 |
| C_Hybrid_RRF | 0.1667 | 1.0000 | 0.5000 | 2 |
| D_Hybrid_ParentChild | 0.1667 | 1.0000 | 0.5000 | 2 |
| E_Hybrid_ParentChild_Reranker | 0.1667 | 1.0000 | 1.0000 | 1 |
| F_Fixed_Full_Pipeline | 0.1667 | 1.0000 | 1.0000 | 1 |
| G_Adaptive_MultiAgent | 0.1667 | 1.0000 | 1.0000 | 1 |

### Query: `eval_cuad_contract_00_Renewal_Term`

**Question**: What are the renewal terms, auto-renewal mechanisms, or extension conditions?  
**Source Contract**: `cuad_contract_003_CENTRACKINTERNATIONALINC_10_29`  
**Gold Evidence**: `This Agreement shall automatically be renewed for one (1) or more one (1) month periods unless either the Customer or i-on gives notice to the other party of its intention not to renew the`  
**Gold Answer Start (offset)**: 10559  

**Gold Chunk IDs**: `['cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p1_c0', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p1_c1', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p1_c2', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p1_c3', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p1_c4', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p1_c5']`  
**Gold ID Count**: 6  
**Gold Mapping Methods**: ['exact_substring', 'exact_substring', 'exact_substring', 'exact_substring', 'exact_substring', 'exact_substring']  

| Rank | Dense chunk_id | BM25 chunk_id | Hybrid chunk_id | Reranked chunk_id | Is Gold? |
|------|---------------|---------------|-----------------|-------------------|----------|
| 1 | `ARMACEUTICALS__INC____v1_p5_c4` | `RMACEUTICALS__INC____v1_p13_c4` | `RMACEUTICALS__INC____v1_p18_c3` | `RMACEUTICALS__INC____v1_p18_c3` |  |
| 2 | `ceuticalsInc_2018110_v1_p27_c1` | `RMACEUTICALS__INC____v1_p18_c3` | `NTERNATIONALINC_10_29_v1_p1_c5`✓ | `NTERNATIONALINC_10_29_v1_p1_c5`✓ | ✓ |
| 3 | `KS_INC_02_18_2016_EX__v1_p4_c2` | `KS_INC_02_18_2016_EX__v1_p3_c1` | `ARMACEUTICALS__INC____v1_p5_c3` | `ARMACEUTICALS__INC____v1_p5_c4` |  |
| 4 | `NTERNATIONALINC_10_29_v1_p1_c2`✓ | `NTERNATIONALINC_10_29_v1_p1_c5`✓ | `KS_INC_02_18_2016_EX__v1_p0_c2` | `ARMACEUTICALS__INC____v1_p5_c3` | ✓ |
| 5 | `NTERNATIONALINC_10_29_v1_p1_c4`✓ | `ARMACEUTICALS__INC____v1_p7_c0` | `NTERNATIONALINC_10_29_v1_p1_c3`✓ | `ARMACEUTICALS__INC____v1_p7_c0` | ✓ |
| 6 | `KS_INC_02_18_2016_EX__v1_p4_c3` | `KS_INC_02_18_2016_EX__v1_p6_c3` | `RMACEUTICALS__INC____v1_p13_c4` | `KS_INC_02_18_2016_EX__v1_p3_c1` |  |
| 7 | `RMACEUTICALS__INC____v1_p18_c3` | `ARMACEUTICALS__INC____v1_p5_c3` | `ARMACEUTICALS__INC____v1_p5_c4` | `NTERNATIONALINC_10_29_v1_p1_c4`✓ | ✓ |
| 8 | `KS_INC_02_18_2016_EX__v1_p0_c2` | `RMACEUTICALS__INC____v1_p21_c1` | `ceuticalsInc_2018110_v1_p27_c1` | `KS_INC_02_18_2016_EX__v1_p4_c2` |  |
| 9 | `NTERNATIONALINC_10_29_v1_p1_c5`✓ | `KS_INC_02_18_2016_EX__v1_p0_c1` | `KS_INC_02_18_2016_EX__v1_p3_c1` | `KS_INC_02_18_2016_EX__v1_p2_c2` | ✓ |
| 10 | `KS_INC_02_18_2016_EX__v1_p2_c2` | `KS_INC_02_18_2016_EX__v1_p0_c0` | `KS_INC_02_18_2016_EX__v1_p4_c2` | `NTERNATIONALINC_10_29_v1_p1_c2`✓ | ✓ |

**Per-Query Metrics Across Variants:**

| Variant | Recall@5 | HitRate@5 | MRR | First Relevant Rank |
|---------|----------|-----------|-----|---------------------|
| A_Dense_Only | 0.3333 | 1.0000 | 0.2500 | 4 |
| B_BM25_Only | 0.1667 | 1.0000 | 0.2500 | 4 |
| C_Hybrid_RRF | 0.3333 | 1.0000 | 0.5000 | 2 |
| D_Hybrid_ParentChild | 0.3333 | 1.0000 | 0.5000 | 2 |
| E_Hybrid_ParentChild_Reranker | 0.1667 | 1.0000 | 0.5000 | 2 |
| F_Fixed_Full_Pipeline | 0.1667 | 1.0000 | 0.5000 | 2 |
| G_Adaptive_MultiAgent | 0.1667 | 1.0000 | 0.5000 | 2 |

### Query: `eval_cuad_contract_00_Notice_Period_T`

**Question**: What is the notice period required to cancel or prevent automatic renewal?  
**Source Contract**: `cuad_contract_003_CENTRACKINTERNATIONALINC_10_29`  
**Gold Evidence**: `This Agreement shall automatically be renewed for one (1) or more one (1) month periods unless either the Customer or i-on gives notice to the other party of its intention not to renew the`  
**Gold Answer Start (offset)**: 10559  

**Gold Chunk IDs**: `['cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p1_c0', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p1_c1', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p1_c2', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p1_c3', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p1_c4', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p1_c5']`  
**Gold ID Count**: 6  
**Gold Mapping Methods**: ['exact_substring', 'exact_substring', 'exact_substring', 'exact_substring', 'exact_substring', 'exact_substring']  

| Rank | Dense chunk_id | BM25 chunk_id | Hybrid chunk_id | Reranked chunk_id | Is Gold? |
|------|---------------|---------------|-----------------|-------------------|----------|
| 1 | `NTERNATIONALINC_10_29_v1_p1_c5`✓ | `Inc____Remarketing_Ag_v1_p4_c4` | `RMACEUTICALS__INC____v1_p18_c3` | `NTERNATIONALINC_10_29_v1_p1_c5`✓ | ✓ |
| 2 | `ARMACEUTICALS__INC____v1_p5_c4` | `RMACEUTICALS__INC____v1_p21_c1` | `NTERNATIONALINC_10_29_v1_p1_c5`✓ | `ARMACEUTICALS__INC____v1_p5_c4` | ✓ |
| 3 | `RMACEUTICALS__INC____v1_p18_c3` | `RMACEUTICALS__INC____v1_p18_c3` | `RMACEUTICALS__INC____v1_p21_c2` | `RMACEUTICALS__INC____v1_p21_c2` |  |
| 4 | `ARMACEUTICALS__INC____v1_p5_c5` | `ceuticalsInc_2018110_v1_p27_c1` | `KS_INC_02_18_2016_EX__v1_p4_c2` | `Inc____Remarketing_Ag_v1_p4_c4` |  |
| 5 | `RMACEUTICALS__INC____v1_p21_c2` | `RMACEUTICALS__INC____v1_p13_c4` | `Inc____Remarketing_Ag_v1_p4_c4` | `RMACEUTICALS__INC____v1_p18_c3` |  |
| 6 | `ceuticalsInc_2018110_v1_p26_c1` | `NTERNATIONALINC_10_29_v1_p1_c5`✓ | `RMACEUTICALS__INC____v1_p21_c1` | `NTERNATIONALINC_10_29_v1_p1_c4`✓ | ✓ |
| 7 | `ceuticalsInc_2018110_v1_p26_c4` | `INC_03_21_2005_EX_10__v1_p3_c0` | `ARMACEUTICALS__INC____v1_p5_c4` | `ARMACEUTICALS__INC____v1_p5_c5` |  |
| 8 | `ARMACEUTICALS__INC____v1_p8_c1` | `KS_INC_02_18_2016_EX__v1_p6_c3` | `ceuticalsInc_2018110_v1_p27_c1` | `ceuticalsInc_2018110_v1_p26_c4` |  |
| 9 | `RMACEUTICALS__INC____v1_p23_c4` | `Inc____Remarketing_Ag_v1_p9_c5` | `ARMACEUTICALS__INC____v1_p5_c5` | `ceuticalsInc_2018110_v1_p26_c1` |  |
| 10 | `NTERNATIONALINC_10_29_v1_p1_c4`✓ | `aceuticalsInc_2018110_v1_p9_c2` | `RMACEUTICALS__INC____v1_p13_c4` | `Inc____Remarketing_Ag_v1_p9_c5` | ✓ |

**Per-Query Metrics Across Variants:**

| Variant | Recall@5 | HitRate@5 | MRR | First Relevant Rank |
|---------|----------|-----------|-----|---------------------|
| A_Dense_Only | 0.1667 | 1.0000 | 1.0000 | 1 |
| B_BM25_Only | 0.0000 | 0.0000 | 0.1667 | 6 |
| C_Hybrid_RRF | 0.1667 | 1.0000 | 0.5000 | 2 |
| D_Hybrid_ParentChild | 0.1667 | 1.0000 | 0.5000 | 2 |
| E_Hybrid_ParentChild_Reranker | 0.1667 | 1.0000 | 1.0000 | 1 |
| F_Fixed_Full_Pipeline | 0.1667 | 1.0000 | 1.0000 | 1 |
| G_Adaptive_MultiAgent | 0.1667 | 1.0000 | 1.0000 | 1 |

### Query: `eval_cuad_contract_00_Governing_Law`

**Question**: Which jurisdiction or state laws govern the interpretation of this agreement?  
**Source Contract**: `cuad_contract_003_CENTRACKINTERNATIONALINC_10_29`  
**Gold Evidence**: `This Agreement was entered into in the State of Florida, and its validity, construction, interpretation, and legal effect shall be governed by the laws and judicial decisions of the State of Florida a`  
**Gold Answer Start (offset)**: 14093  

**Gold Chunk IDs**: `['cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p2_c0', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p2_c1', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p2_c2', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p2_c3']`  
**Gold ID Count**: 4  
**Gold Mapping Methods**: ['exact_substring', 'exact_substring', 'exact_substring', 'exact_substring']  

| Rank | Dense chunk_id | BM25 chunk_id | Hybrid chunk_id | Reranked chunk_id | Is Gold? |
|------|---------------|---------------|-----------------|-------------------|----------|
| 1 | `nc____Remarketing_Ag_v1_p17_c4` | `RMACEUTICALS__INC____v1_p27_c5` | `RMACEUTICALS__INC____v1_p27_c5` | `KS_INC_02_18_2016_EX__v1_p6_c0` |  |
| 2 | `OPHARMA_INC_05_11_202_v1_p2_c0` | `KS_INC_02_18_2016_EX__v1_p6_c0` | `nc____Remarketing_Ag_v1_p17_c3` | `RMACEUTICALS__INC____v1_p23_c2` |  |
| 3 | `nc____Remarketing_Ag_v1_p17_c3` | `RMACEUTICALS__INC____v1_p26_c5` | `RMACEUTICALS__INC____v1_p26_c5` | `RMACEUTICALS__INC____v1_p27_c5` |  |
| 4 | `RMACEUTICALS__INC____v1_p27_c5` | `nc____Remarketing_Ag_v1_p17_c3` | `nc____Remarketing_Ag_v1_p17_c4` | `OPHARMA_INC_05_11_202_v1_p2_c0` |  |
| 5 | `ARMACEUTICALS__INC____v1_p0_c2` | `NTERNATIONALINC_10_29_v1_p2_c3`✓ | `RMACEUTICALS__INC____v1_p23_c2` | `NTERNATIONALINC_10_29_v1_p2_c3`✓ | ✓ |
| 6 | `INC_03_21_2005_EX_10__v1_p3_c2` | `ceuticalsInc_2018110_v1_p23_c4` | `KS_INC_02_18_2016_EX__v1_p6_c0` | `ISORTRUST_02_18_2005__v1_p2_c0` |  |
| 7 | `RMACEUTICALS__INC____v1_p26_c5` | `KS_INC_02_18_2016_EX__v1_p6_c1` | `OPHARMA_INC_05_11_202_v1_p2_c0` | `INC_03_21_2005_EX_10__v1_p3_c2` |  |
| 8 | `RMACEUTICALS__INC____v1_p23_c2` | `RMACEUTICALS__INC____v1_p23_c2` | `INC_03_21_2005_EX_10__v1_p3_c2` | `RMACEUTICALS__INC____v1_p26_c5` |  |
| 9 | `KS_INC_02_18_2016_EX__v1_p6_c5` | `ceuticalsInc_2018110_v1_p22_c2` | `KS_INC_02_18_2016_EX__v1_p6_c4` | `nc____Remarketing_Ag_v1_p17_c3` |  |
| 10 | `ceuticalsInc_2018110_v1_p28_c4` | `nc____Remarketing_Ag_v1_p17_c4` | `RMACEUTICALS__INC____v1_p26_c4` | `ceuticalsInc_2018110_v1_p23_c4` |  |

**Per-Query Metrics Across Variants:**

| Variant | Recall@5 | HitRate@5 | MRR | First Relevant Rank |
|---------|----------|-----------|-----|---------------------|
| A_Dense_Only | 0.0000 | 0.0000 | 0.0000 | not found |
| B_BM25_Only | 0.2500 | 1.0000 | 0.2000 | 5 |
| C_Hybrid_RRF | 0.0000 | 0.0000 | 0.0000 | not found |
| D_Hybrid_ParentChild | 0.0000 | 0.0000 | 0.0000 | not found |
| E_Hybrid_ParentChild_Reranker | 0.2500 | 1.0000 | 0.2000 | 5 |
| F_Fixed_Full_Pipeline | 0.2500 | 1.0000 | 0.2000 | 5 |
| G_Adaptive_MultiAgent | 0.2500 | 1.0000 | 0.2000 | 5 |

### Query: `eval_cuad_contract_00_Termination_For`

**Question**: Does the contract permit either party to terminate for convenience without cause, and what notice is required?  
**Source Contract**: `cuad_contract_003_CENTRACKINTERNATIONALINC_10_29`  
**Gold Evidence**: `Either party may terminate this Agreement without cause at any time effective upon thirty (30) days' written notice.`  
**Gold Answer Start (offset)**: 10880  

**Gold Chunk IDs**: `['cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p1_c0', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p1_c1', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p1_c2', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p1_c3', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p1_c4', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p1_c5']`  
**Gold ID Count**: 6  
**Gold Mapping Methods**: ['exact_substring', 'exact_substring', 'exact_substring', 'exact_substring', 'exact_substring', 'exact_substring']  

| Rank | Dense chunk_id | BM25 chunk_id | Hybrid chunk_id | Reranked chunk_id | Is Gold? |
|------|---------------|---------------|-----------------|-------------------|----------|
| 1 | `RMACEUTICALS__INC____v1_p18_c4` | `INC_03_21_2005_EX_10__v1_p3_c0` | `NTERNATIONALINC_10_29_v1_p1_c5`✓ | `NTERNATIONALINC_10_29_v1_p1_c5`✓ | ✓ |
| 2 | `ceuticalsInc_2018110_v1_p26_c1` | `NTERNATIONALINC_10_29_v1_p1_c5`✓ | `ceuticalsInc_2018110_v1_p26_c3` | `RMACEUTICALS__INC____v1_p21_c2` | ✓ |
| 3 | `ceuticalsInc_2018110_v1_p26_c3` | `ceuticalsInc_2018110_v1_p26_c3` | `ceuticalsInc_2018110_v1_p26_c1` | `RMACEUTICALS__INC____v1_p18_c4` |  |
| 4 | `NTERNATIONALINC_10_29_v1_p1_c5`✓ | `RMACEUTICALS__INC____v1_p21_c1` | `RMACEUTICALS__INC____v1_p18_c4` | `NTERNATIONALINC_10_29_v1_p2_c2` | ✓ |
| 5 | `ceuticalsInc_2018110_v1_p26_c4` | `RMACEUTICALS__INC____v1_p19_c2` | `ceuticalsInc_2018110_v1_p26_c4` | `ceuticalsInc_2018110_v1_p26_c4` |  |
| 6 | `RMACEUTICALS__INC____v1_p22_c4` | `KS_INC_02_18_2016_EX__v1_p6_c3` | `INC_03_21_2005_EX_10__v1_p3_c0` | `INC_03_21_2005_EX_10__v1_p3_c0` |  |
| 7 | `RMACEUTICALS__INC____v1_p21_c2` | `ISORTRUST_02_18_2005__v1_p2_c0` | `RMACEUTICALS__INC____v1_p21_c1` | `ceuticalsInc_2018110_v1_p26_c3` |  |
| 8 | `RMACEUTICALS__INC____v1_p18_c2` | `ARMACEUTICALS__INC____v1_p8_c3` | `RMACEUTICALS__INC____v1_p19_c2` | `ISORTRUST_02_18_2005__v1_p2_c0` |  |
| 9 | `NTERNATIONALINC_10_29_v1_p2_c2` | `ceuticalsInc_2018110_v1_p26_c1` | `KS_INC_02_18_2016_EX__v1_p6_c3` | `KS_INC_02_18_2016_EX__v1_p5_c2` |  |
| 10 | `KS_INC_02_18_2016_EX__v1_p5_c2` | `ceuticalsInc_2018110_v1_p26_c4` | `RMACEUTICALS__INC____v1_p22_c4` | `ceuticalsInc_2018110_v1_p26_c1` |  |

**Per-Query Metrics Across Variants:**

| Variant | Recall@5 | HitRate@5 | MRR | First Relevant Rank |
|---------|----------|-----------|-----|---------------------|
| A_Dense_Only | 0.1667 | 1.0000 | 0.2500 | 4 |
| B_BM25_Only | 0.1667 | 1.0000 | 0.5000 | 2 |
| C_Hybrid_RRF | 0.1667 | 1.0000 | 1.0000 | 1 |
| D_Hybrid_ParentChild | 0.1667 | 1.0000 | 1.0000 | 1 |
| E_Hybrid_ParentChild_Reranker | 0.1667 | 1.0000 | 1.0000 | 1 |
| F_Fixed_Full_Pipeline | 0.1667 | 1.0000 | 1.0000 | 1 |
| G_Adaptive_MultiAgent | 0.1667 | 1.0000 | 1.0000 | 1 |

### Query: `eval_cuad_contract_00_Cap_On_Liabilit`

**Question**: What is the maximum aggregate liability cap specified under the contract?  
**Source Contract**: `cuad_contract_003_CENTRACKINTERNATIONALINC_10_29`  
**Gold Evidence**: `i-on will not be liable under any circumstances for any lost profits or other consequential damages, even if i-on has been advised as to the possibility of such damages. i-on's liability for damages t`  
**Gold Answer Start (offset)**: 11273  

**Gold Chunk IDs**: `['cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p2_c0', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p2_c1', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p2_c2', 'cuad_contract_003_CENTRACKINTERNATIONALINC_10_29_v1_p2_c3']`  
**Gold ID Count**: 4  
**Gold Mapping Methods**: ['exact_substring', 'exact_substring', 'exact_substring', 'exact_substring']  

| Rank | Dense chunk_id | BM25 chunk_id | Hybrid chunk_id | Reranked chunk_id | Is Gold? |
|------|---------------|---------------|-----------------|-------------------|----------|
| 1 | `KS_INC_02_18_2016_EX__v1_p3_c1` | `Inc____Remarketing_Ag_v1_p1_c4` | `KS_INC_02_18_2016_EX__v1_p3_c0` | `RMACEUTICALS__INC____v1_p16_c1` |  |
| 2 | `KS_INC_02_18_2016_EX__v1_p3_c0` | `RMACEUTICALS__INC____v1_p16_c2` | `RMACEUTICALS__INC____v1_p16_c2` | `KS_INC_02_18_2016_EX__v1_p3_c0` |  |
| 3 | `INC_03_21_2005_EX_10__v1_p2_c3` | `KS_INC_02_18_2016_EX__v1_p6_c3` | `RMACEUTICALS__INC____v1_p16_c1` | `KS_INC_02_18_2016_EX__v1_p3_c1` |  |
| 4 | `RMACEUTICALS__INC____v1_p16_c0` | `nc____Remarketing_Ag_v1_p20_c0` | `KS_INC_02_18_2016_EX__v1_p0_c4` | `nc____Remarketing_Ag_v1_p11_c1` |  |
| 5 | `ceuticalsInc_2018110_v1_p24_c5` | `KS_INC_02_18_2016_EX__v1_p3_c0` | `Inc____Remarketing_Ag_v1_p6_c4` | `RMACEUTICALS__INC____v1_p16_c2` |  |
| 6 | `KS_INC_02_18_2016_EX__v1_p1_c0` | `aceuticalsInc_2018110_v1_p9_c3` | `Inc____Remarketing_Ag_v1_p1_c4` | `ceuticalsInc_2018110_v1_p24_c5` |  |
| 7 | `RMACEUTICALS__INC____v1_p16_c2` | `ISORTRUST_02_18_2005__v1_p1_c2` | `KS_INC_02_18_2016_EX__v1_p3_c1` | `Inc____Remarketing_Ag_v1_p1_c4` |  |
| 8 | `RMACEUTICALS__INC____v1_p23_c0` | `aceuticalsInc_2018110_v1_p1_c0` | `KS_INC_02_18_2016_EX__v1_p6_c3` | `aceuticalsInc_2018110_v1_p9_c3` |  |
| 9 | `ceuticalsInc_2018110_v1_p26_c0` | `RMACEUTICALS__INC____v1_p16_c1` | `INC_03_21_2005_EX_10__v1_p2_c3` | `aceuticalsInc_2018110_v1_p1_c0` |  |
| 10 | `ARMACEUTICALS__INC____v1_p6_c0` | `nc____Remarketing_Ag_v1_p11_c1` | `nc____Remarketing_Ag_v1_p20_c0` | `RMACEUTICALS__INC____v1_p23_c0` |  |

**Per-Query Metrics Across Variants:**

| Variant | Recall@5 | HitRate@5 | MRR | First Relevant Rank |
|---------|----------|-----------|-----|---------------------|
| A_Dense_Only | 0.0000 | 0.0000 | 0.0000 | not found |
| B_BM25_Only | 0.0000 | 0.0000 | 0.0000 | not found |
| C_Hybrid_RRF | 0.0000 | 0.0000 | 0.0000 | not found |
| D_Hybrid_ParentChild | 0.0000 | 0.0000 | 0.0000 | not found |
| E_Hybrid_ParentChild_Reranker | 0.0000 | 0.0000 | 0.0000 | not found |
| F_Fixed_Full_Pipeline | 0.0000 | 0.0000 | 0.0000 | not found |
| G_Adaptive_MultiAgent | 0.0000 | 0.0000 | 0.0000 | not found |


---

## Raw Data Provenance

- **Run directory**: `evaluation/runs/retrieval_real_20260814_181902_f1eea1/`
- **Per-variant traces**: `trace_<variant>.jsonl` (one JSON line per query)
- **Per-query diagnostics**: `per_query_diagnostics.json`
- **OCR degradation**: `evaluation/reports/ocr_retrieval_degradation.json`

*Benchmark integrity repair: all metrics above come from real indexed document retrieval with real embedding and BM25 computation. No metric is hardcoded.*