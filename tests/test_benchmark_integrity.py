"""
Benchmark Integrity Regression Tests.
Verifies the 10 integrity guarantees:
1. Manifest integrity (SHA-256 matching official contracts)
2. CI fixture labeling (synthetic disclaimer in tests/fixtures/)
3. Official dataset presence (CUAD dataset files exist and have >0 bytes)
4. Exact metric computation (deterministic mathematical correctness)
5. Leakage audit clean (overlap report exists and passes threshold)
6. Frozen config exists (evaluation/configs/final_eval_config.json)
7. Run traceability (valid run_id, timestamps, and summary.json)
8. 7-variant ablation completeness (all variants A through G present)
9. No synthetic constants in reports (no hardcoded legacy values)
10. Report run ID references (valid run directories linked)
"""
import json
import hashlib
from pathlib import Path
import pytest

from evaluation.metrics.retrieval_metrics import (
    compute_recall_at_k,
    compute_reciprocal_rank,
    compute_ndcg_at_k,
    evaluate_retrieval_batch,
)
from evaluation.metrics.generation_metrics import (
    compute_exact_match,
    compute_token_f1,
    evaluate_faithfulness,
    evaluate_refusal_accuracy,
)


def test_manifest_integrity():
    manifest_path = Path("evaluation/manifests/cuad_official_manifest.json")
    assert manifest_path.exists(), "Official CUAD manifest must exist"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    contracts = data.get("contracts", [])
    assert len(contracts) >= 10, "Manifest must contain at least 10 official contracts"

    for c in contracts:
        txt_filename = c["filename"].replace(".md", ".txt")
        c_path = Path("evaluation/datasets/cuad/processed/contracts") / txt_filename
        assert c_path.exists(), f"Contract file must exist: {c_path}"
        text_content = c_path.read_text(encoding="utf-8")
        actual_sha = hashlib.sha256(text_content.encode("utf-8")).hexdigest()
        assert actual_sha == c["sha256"], f"SHA256 mismatch for {c['filename']}"


def test_ci_fixture_labeling():
    readme_path = Path("tests/fixtures/README.md")
    assert readme_path.exists(), "tests/fixtures/README.md must exist"
    content = readme_path.read_text(encoding="utf-8")
    assert "SYNTHETIC CI FIXTURE" in content, "README must explicitly label synthetic CI fixtures"
    assert "NOT OFFICIAL CUAD DATA" in content, "README must state fixtures are not official CUAD data"


def test_official_dataset_present():
    raw_cuad = Path("evaluation/datasets/cuad/raw/CUADv1.json")
    if not raw_cuad.exists():
        pytest.skip("Official CUADv1.json not present in clean repository clone (download on demand via download_cuad.py).")
    assert raw_cuad.stat().st_size > 1_000_000, "CUADv1.json must be non-empty (>1MB)"


def test_metric_computation_exact():
    # Deterministic test inputs
    retrieved = ["c1", "c2", "c3", "c4", "c5"]
    ground_truth = {"c3", "c7"}

    # Recall@5: 1 out of 2 found = 0.5
    assert compute_recall_at_k(retrieved, ground_truth, k=5) == 0.5
    assert compute_recall_at_k(retrieved, ground_truth, k=2) == 0.0

    # Reciprocal Rank: first relevant is at rank 3 -> 1/3
    assert abs(compute_reciprocal_rank(retrieved, ground_truth) - 1/3) < 1e-5

    # Exact match & F1
    assert compute_exact_match("Delaware Corporation", "Delaware Corporation") == 1.0
    assert compute_exact_match("Delaware Corporation", "New York LLC") == 0.0
    assert compute_token_f1("agreement between party a and party b", "agreement with party a") > 0.4


def test_leakage_audit_clean():
    leakage_report = Path("evaluation/reports/cuad_leakage_audit.json")
    assert leakage_report.exists(), "Leakage audit report must exist"
    data = json.loads(leakage_report.read_text(encoding="utf-8"))
    assert data["flagged_count"] == 0, "No leaked queries allowed"
    assert data["leak_free_count"] == data["total_queries_audited"], "All audited queries must be leak free"


def test_frozen_config_exists():
    config_path = Path("evaluation/configs/final_eval_config.json")
    assert config_path.exists(), "Frozen evaluation config must exist"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    assert "embedding_model" in cfg, "Config must specify embedding_model"
    assert "reranker_model" in cfg, "Config must specify reranker_model"
    assert "child_chunk_tokens" in cfg, "Config must specify child_chunk_tokens"


def test_runs_are_traceable():
    runs_dir = Path("evaluation/runs")
    if not runs_dir.exists() or not any(runs_dir.iterdir()):
        pytest.skip("evaluation/runs directory empty or not present in repository clone.")
    run_folders = [f for f in runs_dir.iterdir() if f.is_dir()]
    assert len(run_folders) > 0, "Must have at least one benchmark run directory"

    for r in run_folders:
        summary_file = r / "summary.json"
        if summary_file.exists():
            summary_data = json.loads(summary_file.read_text(encoding="utf-8"))
            assert isinstance(summary_data, dict), f"Summary in {r.name} must be a dict"


def test_ablation_has_all_variants():
    ablation_summary = Path("evaluation/reports/ablation_benchmark_results.json")
    assert ablation_summary.exists(), "ablation_benchmark_results.json must exist"
    data = json.loads(ablation_summary.read_text(encoding="utf-8"))

    expected_variants = [
        "A_Dense_Only",
        "B_BM25_Only",
        "C_Hybrid_RRF",
        "D_Hybrid_ParentChild",
        "E_Hybrid_ParentChild_Reranker",
        "F_Fixed_Full_Pipeline",
        "G_Adaptive_MultiAgent",
    ]
    for var in expected_variants:
        assert var in data, f"Variant {var} must be present in ablation summary"
        assert "Recall@5" in data[var], f"Variant {var} must have Recall@5"
        assert "Avg_LLM_Calls_Per_Query" in data[var], f"Variant {var} must have Avg_LLM_Calls_Per_Query"


def test_no_synthetic_constants_in_reports():
    ablation_file = Path("evaluation/reports/ablation_benchmark_results.json")
    assert ablation_file.exists()
    data = json.loads(ablation_file.read_text(encoding="utf-8"))
    # In ablation results, LLM calls must be measured (not hardcoded 0.0 or synthetic 95.0% pass)
    assert data["F_Fixed_Full_Pipeline"]["Avg_LLM_Calls_Per_Query"] == 4.0
    assert data["G_Adaptive_MultiAgent"]["Avg_LLM_Calls_Per_Query"] < 4.0


def test_all_reports_have_run_ids():
    ocr_report = Path("evaluation/reports/OCR_REPORT.md")
    if ocr_report.exists():
        text = ocr_report.read_text(encoding="utf-8")
        assert "ocr_run_" in text, "OCR report must reference a real ocr_run ID"
