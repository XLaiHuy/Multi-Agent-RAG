# Dataset Integrity & Provenance Report

**Date**: August 14, 2026  
**Auditor Mode**: STRICT BENCHMARK-INTEGRITY REPAIR MODE  
**Provenance Status**: 100% Verified Official CUAD v1 + Isolated Synthetic CI Fixtures  

---

## 1. Dataset Partitioning & Provenance Overview

The evaluation suite strictly partitions contract data into three distinct tiers to prevent data contamination and misattribution:

```
Multi-Agent-RAG Evaluation Datasets
├── [Tier 1: Official CUAD v1] -> evaluation/datasets/cuad/
│   ├── raw/CUADv1.json (Official 510 contracts, SHA-256 verified)
│   └── processed/contracts/ (Extracted test contracts: .txt and .md)
├── [Tier 2: Synthetic CI Fixture] -> tests/fixtures/cuad_small/
│   └── Explicitly labeled: SYNTHETIC CI FIXTURE - NOT OFFICIAL CUAD DATA
└── [Tier 3: Multi-Format & OCR Stress] -> evaluation/datasets/
    ├── multiformat/ (Format variants: TXT, MD, JSON, DOCX, PDF)
    └── ocr_stress/ (Rasterized 100-300 DPI, skewed, blurred, noisy images)
```

---

## 2. Official CUAD v1 Test Set Contracts

**Source**: The Atticus Project ([GitHub Repository](https://github.com/TheAtticusProject/cuad))  
**Archive**: `data.zip` (SHA-256: `f8161d18bea4e9c05e78fa6dda61c19c846fb8087ea969c172753bc2f45b999a`)  
**Split**: Frozen TEST Split (Seed 42)  

| # | Contract Safe ID | Title / Type | Chars | SHA-256 (Raw Text Content) | Valid Annotations |
| :---: | :--- | :--- | :---: | :--- | :---: |
| 1 | `cuad_contract_003_CENTRACKINTERNATIONAL` | WEB SITE HOSTING AGREEMENT | 15,176 | `8532356d811b76bd7ae536cf22910742ba61815dc39741fd9b295d7bc80c99f1` | 10 |
| 2 | `cuad_contract_004_NELNETINC_04_08_2020` | JOINT FILING AGREEMENT | 2,752 | `32a76f2d5c3f3ca76d97c5553e1eb8498877bc95484aa3a7bfadbf5d0c29f648` | 1 |
| 3 | `cuad_contract_005_MATINASBIOPHARMA` | COLLABORATION AGREEMENT | 58,477 | `cf79ec0ebba4839cf9e5ea7724285b736ca2cfbbd8e6fa3eefd6b9ea0b15e478` | 14 |
| 4 | `cuad_contract_006_GRIFFINLANDANDNUR` | LEASE AGREEMENT | 39,268 | `295a0a38b1f51dd719a6917637ca8d8102dbd8719f561ee515e1975e5331e808` | 12 |
| 5 | `cuad_contract_007_QUICKLOGICCORP_11` | CO-DEVELOPMENT AGREEMENT | 29,864 | `30a6184a56a6428e519aaee381c81dc0ec16df8d0426b3be373b5030248ddb16` | 12 |
| 6 | `cuad_contract_008_DUKEENERGYCORP_04` | AMENDMENT TO CREDIT AGREEMENT | 28,131 | `a79058b8f2d561219fcda9f381fbc03dfb3a72d3e4293f0b2f5d7cfbbd591fe5` | 3 |
| 7 | `cuad_contract_009_DIAMONDRESORTSINT` | EMPLOYMENT AGREEMENT | 57,605 | `d559c5d713c72bdf3310cb233b8a1c97efef59635fc9385d033320c29ceb1022` | 21 |
| 8 | `cuad_contract_010_CYTOMXTHERAPEUTI` | EXCLUSIVE RESEARCH & LICENSE | 86,608 | `784d00cf05d15c7e090fce04b50d53c7a0c006ee7ef8ec619427e5e3328e93ae` | 20 |
| 9 | `cuad_contract_011_COMSCOREINC_05_0` | SPONSORSHIP AGREEMENT | 16,929 | `bb33ae5c328db300fc48e1a1415df8a5a54f67ff4f009e56314f85e49c7f1236` | 7 |
| 10 | `cuad_contract_012_BLACKPEBBLEACQUI` | PROMISSORY NOTE AGREEMENT | 12,028 | `9944fc6bda193f4864455848bb440fc7a5eb8347895e638d21b44ec2f6eb8b14` | 7 |

---

## 3. Lexical Overlap & Data Contamination Audit

**Audit File**: `evaluation/reports/cuad_leakage_audit.json`  
**Total Queries Audited**: 210 official candidate questions  
**Leak-Free Queries**: 210 (100.0%)  
**Flagged Direct Answer Overlap**: 0  

### Audit Methodology:
- **Title Overlap Test**: Checked word jaccard between query text and contract titles (mean overlap: **0.052**).
- **Answer Span Leakage Test**: Verified that query text does not contain verbatim target answer text strings (all queries below 0.70 word-level overlap threshold).
- **Programmatic Manifest**: Created `evaluation/manifests/cuad_official_manifest.json` containing 50 curated leak-free queries across 10 official contracts.

---

## 4. Synthetic CI Fixture Isolation

**Directory**: `tests/fixtures/cuad_small/`  
**Disclaimer File**: `tests/fixtures/README.md`  

### Mandatory Disclaimer:
> `SYNTHETIC CI FIXTURE — NOT OFFICIAL CUAD DATA`  
> Contracts and Q&A manifests located in `tests/fixtures/cuad_small/` are synthetic, miniaturized test fixtures created solely for fast, offline unit testing and continuous integration smoke tests. They do NOT represent the official CUAD dataset and must not be used to publish performance claims.
