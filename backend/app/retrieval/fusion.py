"""
Reciprocal Rank Fusion (RRF) & Hierarchical Parent Expansion Module.
Combines multiple ranked retrieval lists deterministically and resolves child search hits to parent context blocks.
"""
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class RetrievedCandidate:
    chunk_id: str
    doc_id: str
    doc_version: int
    text: str # Parent or Child text
    is_parent_expanded: bool
    parent_id: Optional[str]
    page_number: int
    section_path: List[str]
    block_id: str
    bbox: Optional[Dict[str, float]]
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: Optional[float] = None
    metadata: Dict[str, Any] = None


def reciprocal_rank_fusion(
    ranked_lists: List[List[str]], k: int = 60
) -> List[Tuple[str, float]]:
    """
    Combines multiple ranked lists of chunk_ids using Reciprocal Rank Fusion (RRF).
    Formula: RRF_score(d) = sum_{l in lists} 1 / (k + rank_l(d))
    """
    scores: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked):
            # 1-based rank
            scores[chunk_id] = scores.get(chunk_id, 0.0) + (1.0 / (k + rank + 1))

    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results


class HierarchicalParentExpander:
    """
    Expands retrieved child chunks to their enclosing Parent context blocks (~1000-1500 tokens).
    Ensures the LLM generator receives complete clauses, headings, and surrounding paragraph context.
    """

    @staticmethod
    def expand_candidates(
        candidates: List[RetrievedCandidate],
        use_parent_expansion: bool = True,
    ) -> List[RetrievedCandidate]:
        """
        Deduplicates by parent_id (if multiple child chunks from the same parent are retrieved)
        and swaps text with parent_text for context synthesis.
        """
        if not use_parent_expansion:
            return candidates

        seen_parents = set()
        expanded_list: List[RetrievedCandidate] = []

        for cand in candidates:
            parent_id = cand.parent_id or cand.chunk_id
            parent_text = cand.metadata.get("parent_text") if cand.metadata else None

            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)

            if parent_text:
                # Expand to parent context
                cand.text = parent_text
                cand.is_parent_expanded = True

            expanded_list.append(cand)

        return expanded_list
