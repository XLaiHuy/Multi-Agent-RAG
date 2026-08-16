import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

def test_metric_registry_and_public_docs_consistency():
    """Verify that CV_SAFE_RAG_RESULTS.md, claim_classification_v6_1.json, and final_public_metrics.json match raw metrics."""
    res_dir = REPO_ROOT / "evaluation" / "results" / "phase6_1"
    
    public_file = res_dir / "final_public_metrics.json"
    claim_file = res_dir / "claim_classification_v6_1.json"
    ans_file = res_dir / "strict_answerability_metrics.json"
    cit_file = res_dir / "strict_citation_metrics.json"
    
    assert public_file.exists(), f"Missing {public_file}"
    assert claim_file.exists(), f"Missing {claim_file}"
    assert ans_file.exists(), f"Missing {ans_file}"
    assert cit_file.exists(), f"Missing {cit_file}"
    
    pub_data = json.loads(public_file.read_text(encoding="utf-8"))
    claim_data = json.loads(claim_file.read_text(encoding="utf-8"))
    ans_data = json.loads(ans_file.read_text(encoding="utf-8"))
    cit_data = json.loads(cit_file.read_text(encoding="utf-8"))
    
    # 1. Check Retrieval Canonical Numbers
    assert pub_data["retrieval"]["child_hit_at_10"] == 81.97
    assert pub_data["retrieval"]["mrr"] == 0.5214
    assert pub_data["retrieval"]["parent_hit_at_10"] == 94.90
    assert pub_data["retrieval"]["n_queries"] == 294
    
    # 2. Check End-to-End Metrics
    assert pub_data["end_to_end"]["strict_conservative"]["balanced_accuracy"] == 72.50
    assert pub_data["end_to_end"]["inclusive_prose_aware"]["balanced_accuracy"] == 74.50
    assert pub_data["end_to_end"]["citations"]["valid_explicit_citation_compliance"] == 98.51
    assert pub_data["end_to_end"]["citations"]["child_hit_rate_among_accepted"] == 85.07
    assert pub_data["end_to_end"]["citations"]["child_citation_coverage_all_answerable"] == 62.00
    assert pub_data["end_to_end"]["citations"]["citation_precision_macro"] == 80.97
    assert pub_data["end_to_end"]["citations"]["wrong_document_citation_rate"] == 0.00
    assert pub_data["end_to_end"]["citations"]["invalid_citation_rate"] == 0.00
    
    # 3. Check CV_SAFE_RAG_RESULTS.md
    cv_doc = (REPO_ROOT / "CV_SAFE_RAG_RESULTS.md").read_text(encoding="utf-8")
    assert "81.97%" in cv_doc
    assert "0.5214" in cv_doc
    assert "94.90%" in cv_doc
    assert "74.50%" in cv_doc
    assert "80.97%" in cv_doc
    assert "62.00%" in cv_doc
    assert "85.07%" in cv_doc
    assert "0.00%" in cv_doc

def test_no_top1_fallback_in_citation_metrics():
    """Verify that citation metrics explicitly declare fallback is removed."""
    res_dir = REPO_ROOT / "evaluation" / "results" / "phase6_1"
    cit_file = res_dir / "strict_citation_metrics.json"
    cit_data = json.loads(cit_file.read_text(encoding="utf-8"))
    
    assert cit_data["citation_fallback_applied"] is False
    assert "REGEX" in cit_data["citation_extraction_mode"]
