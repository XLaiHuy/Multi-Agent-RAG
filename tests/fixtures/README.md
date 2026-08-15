# CI Test Fixtures (Synthetic Data)

> [!IMPORTANT]
> **DATASET CLASSIFICATION: SYNTHETIC CI FIXTURE — NOT OFFICIAL CUAD DATA**
> The files in this directory (`tests/fixtures/cuad_small/`) are small, manually constructed synthetic contract fixtures designed strictly for fast, deterministic CI smoke testing, unit tests, parser validation, chunker AST boundary testing, and security regression suites.
> They are **NOT** the official Atticus Project CUAD benchmark dataset. Official CUAD evaluation datasets and manifests are located under `evaluation/datasets/cuad/` and `evaluation/manifests/`.

---

## 1. Directory Structure & Assets

### `tests/fixtures/cuad_small/`

| File Name | Format | Purpose | Classification |
| :--- | :--- | :--- | :--- |
| `contract_cuad_01.md` | Markdown AST | Master canonical NDA & Services Agreement fixture. | **SYNTHETIC CI FIXTURE** |
| `contract_cuad_01.docx` | Microsoft Word (DOCX) | Format invariance & DOCX parser evaluation. | **SYNTHETIC CI FIXTURE** |
| `contract_cuad_01.json` | Structured JSON Blocks | Structured block parser evaluation. | **SYNTHETIC CI FIXTURE** |
| `contract_cuad_02.md` | Markdown AST | Master canonical License & Commercial Agreement. | **SYNTHETIC CI FIXTURE** |
| `contract_cuad_02.docx` | Microsoft Word (DOCX) | Multi-contract comparison & parser testing. | **SYNTHETIC CI FIXTURE** |
| `contract_cuad_02.json` | Structured JSON Blocks | Structured block comparison testing. | **SYNTHETIC CI FIXTURE** |
| `cuad_qa_manifest.json` | JSON Manifest | 8 synthetic QA pairs for unit and smoke tests. | **SYNTHETIC CI MANIFEST** |

---

## 2. Usage Policy
1. Automated CI pipelines (`pytest tests/`) may use these fixtures for rapid, deterministic verification.
2. Full benchmark reports must NEVER report metrics on these fixtures as "CUAD Benchmark Results".
3. Any benchmark run using these fixtures must be explicitly tagged with `run_type: CI_SMOKE`.
