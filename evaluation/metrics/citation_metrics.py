"""
Citation & Evidence Evaluation Metrics.
Implements Citation Precision, Citation Recall, and Evidence Coverage.
"""
from typing import List, Dict, Any, Set
from backend.app.domain.schemas import CitationItem


def compute_citation_precision(
    predicted_citations: List[CitationItem], ground_truth_block_ids: Set[str]
) -> float:
    """Fraction of citations pointing to valid ground truth evidence blocks."""
    if not predicted_citations:
        return 1.0 if not ground_truth_block_ids else 0.0

    valid_count = 0
    for cit in predicted_citations:
        if cit.block_id in ground_truth_block_ids or cit.document_id in ground_truth_block_ids:
            valid_count += 1

    return valid_count / len(predicted_citations)


def compute_citation_recall(
    predicted_citations: List[CitationItem], ground_truth_block_ids: Set[str]
) -> float:
    """Fraction of ground truth evidence blocks that were cited."""
    if not ground_truth_block_ids:
        return 1.0

    cited_ids = {cit.block_id for cit in predicted_citations}.union({cit.document_id for cit in predicted_citations})
    retrieved_truth = cited_ids.intersection(ground_truth_block_ids)
    return len(retrieved_truth) / len(ground_truth_block_ids)


def compute_evidence_coverage(
    predicted_citations: List[CitationItem], required_facets: List[str]
) -> float:
    """Fraction of required facets addressed by at least one citation."""
    if not required_facets:
        return 1.0

    combined_cited_text = " ".join(cit.supporting_text.lower() for cit in predicted_citations)
    covered = 0
    for facet in required_facets:
        words = [w for w in facet.lower().split() if len(w) > 3]
        if words and any(w in combined_cited_text for w in words):
            covered += 1

    return covered / len(required_facets)
