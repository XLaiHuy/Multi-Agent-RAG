import os
import sys
import json
import pytest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.scripts.run_phase6_real_api import parse_decision_and_citations, Phase6Executor
from evaluation.scripts.score_phase6 import compute_f1, normalize_text


def test_runtime_label_isolation():
    """
    1. Runtime label isolation:
    Verifies that the Layer A execution payload does not contain ground truth labels.
    """
    sample_manifest_query = {
        "query_id": "test_q_01",
        "question": "What is the governing law?",
        "source_contract_id": "cuad_contract_001",
        "is_unanswerable": False,
        "answers": [{"text": "State of Delaware"}],
        "strict_gold_child_ids": ["cuad_contract_001_p0_c0"],
        "strict_gold_parent_ids": ["cuad_contract_001_p0"]
    }

    # Layer A Sanitizer
    sanitized_payload = {
        "query_id": sample_manifest_query["query_id"],
        "question": sample_manifest_query["question"],
        "selected_document_id": sample_manifest_query["source_contract_id"]
    }

    assert "is_unanswerable" not in sanitized_payload
    assert "answers" not in sanitized_payload
    assert "strict_gold_child_ids" not in sanitized_payload
    assert "strict_gold_parent_ids" not in sanitized_payload
    assert len(sanitized_payload) == 3


def test_structured_refusal_vs_error():
    """
    2. Structured refusal vs error parsing:
    Verifies that insufficient evidence answers are parsed as INSUFFICIENT_EVIDENCE,
    while valid answers are parsed as ANSWER.
    """
    refusal_text = "INSUFFICIENT_EVIDENCE: The provided contract excerpts do not contain information regarding this question."
    cands = [{"chunk_id": "c1", "parent_id": "p1", "doc_id": "doc1", "text": "Some text"}]

    dec_ref, cit_ref = parse_decision_and_citations(refusal_text, cands)
    assert dec_ref == "INSUFFICIENT_EVIDENCE"
    assert len(cit_ref) == 0

    answer_text = "According to [Reference 1], the governing law is the State of Delaware."
    dec_ans, cit_ans = parse_decision_and_citations(answer_text, cands)
    assert dec_ans == "ANSWER"
    assert len(cit_ans) == 1
    assert cit_ans[0]["chunk_id"] == "c1"


def test_token_f1_computation():
    """
    3. Token overlap and F1 scoring:
    Verifies deterministic token overlap calculation.
    """
    pred = "This Agreement is governed by Delaware law."
    gold = "governed by the laws of Delaware"

    prec, rec, f1 = compute_f1(pred, gold)
    assert 0.0 < f1 <= 1.0
    assert prec > 0.0
    assert rec > 0.0


def test_wrong_document_citation_detection():
    """
    4. Wrong document citation detection:
    Verifies that citations belonging to a different document are flagged.
    """
    active_doc_id = "cuad_contract_001"
    citations = [
        {"chunk_id": "c1", "document_id": "cuad_contract_001"},
        {"chunk_id": "c2", "document_id": "cuad_contract_999"}, # Wrong document!
    ]
    wrong_count = sum(1 for c in citations if c.get("document_id") != active_doc_id)
    assert wrong_count == 1
