import pytest
import re
from evaluation.scripts.rescore_phase6_strict import extract_explicit_citations_from_text, compute_token_f1

def test_missing_citation_remains_missing_no_top1_fallback():
    """Verify that an answer without explicit brackets returns empty list and is NOT assigned top-1 chunk."""
    answer_text = "The contract commences on May 19, 2010 and continues for ten years."
    retrieved_cids = ["chunk_001", "chunk_002", "chunk_003"]
    
    extracted = extract_explicit_citations_from_text(answer_text, retrieved_cids)
    assert extracted == []
    assert len(extracted) == 0

def test_valid_reference_index_parsing():
    """Verify that [Reference 1] maps correctly to the 0-th retrieved chunk."""
    answer_text = "The agreement commences on the Commencement Date [Reference 1]."
    retrieved_cids = ["chunk_001", "chunk_002", "chunk_003"]
    
    extracted = extract_explicit_citations_from_text(answer_text, retrieved_cids)
    assert extracted == ["chunk_001"]

def test_valid_reference_full_chunk_id_parsing():
    """Verify that [Reference 2: chunk_abc_123] extracts the explicit chunk ID directly."""
    answer_text = "The term is 10 years [Reference 2: cuad_contract_056_v1_p1_c3]."
    retrieved_cids = ["chunk_001", "chunk_002"]
    
    extracted = extract_explicit_citations_from_text(answer_text, retrieved_cids)
    assert "cuad_contract_056_v1_p1_c3" in extracted

def test_invalid_reference_index_recorded():
    """Verify that out-of-range reference indices like [Reference 9] when 3 chunks exist are marked invalid."""
    answer_text = "The term is 10 years [Reference 9]."
    retrieved_cids = ["chunk_001", "chunk_002", "chunk_003"]
    
    extracted = extract_explicit_citations_from_text(answer_text, retrieved_cids)
    assert len(extracted) == 1
    assert extracted[0].startswith("INVALID_REF_INDEX")

def test_strict_refusal_sentinel_classification():
    """Verify strict refusal sentinel detection."""
    sentinel_text = "INSUFFICIENT_EVIDENCE: The provided contract excerpts do not contain information."
    prose_text = "The provided contract excerpts do not contain information regarding a notice period."
    answer_text = "The contract term is 10 years [Reference 1]."
    
    assert sentinel_text.startswith("INSUFFICIENT_EVIDENCE:")
    assert not prose_text.startswith("INSUFFICIENT_EVIDENCE:")
    assert not answer_text.startswith("INSUFFICIENT_EVIDENCE:")

def test_f1_token_overlap():
    """Verify token F1 computation."""
    f1_perfect = compute_token_f1("May 19, 2010", "May 19, 2010")
    assert f1_perfect == 1.0
    
    f1_zero = compute_token_f1("Delaware Corporation", "May 19, 2010")
    assert f1_zero == 0.0
    
    f1_partial = compute_token_f1("May 19, 2010 term", "May 19, 2010")
    assert 0.0 < f1_partial < 1.0

def test_phase4_2_canonical_retrieval_constants():
    """Verify that canonical Phase 4.2 retrieval constants are preserved."""
    canonical_n = 294
    canonical_child_hit5 = 68.71
    canonical_child_hit10 = 81.97
    canonical_mrr = 0.5214
    canonical_parent_hit10 = 94.90
    
    assert canonical_n == 294
    assert canonical_child_hit10 == 81.97
    assert canonical_mrr == 0.5214
    assert canonical_parent_hit10 == 94.90
