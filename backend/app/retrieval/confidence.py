"""
Deterministic Multi-Signal Retrieval Confidence Engine.
Calculates mathematical retrieval confidence without relying on LLM self-estimation.
Combines:
1. BM25 & Dense rank consensus (overlap @ top-k)
2. Top-1 vs Top-2 score margin (separation distance)
3. RRF top-1 score magnitude
4. CrossEncoder reranker confidence score (if reranked)
5. Metadata & Section path agreement
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from backend.app.core.config import get_settings


@dataclass
class ConfidenceSignals:
    bm25_dense_rank_agreement: float
    top_score_margin: float
    rrf_consensus: float
    reranker_score: float
    metadata_match: float
    final_confidence: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "bm25_dense_rank_agreement": round(self.bm25_dense_rank_agreement, 3),
            "top_score_margin": round(self.top_score_margin, 3),
            "rrf_consensus": round(self.rrf_consensus, 3),
            "reranker_score": round(self.reranker_score, 3),
            "metadata_match": round(self.metadata_match, 3),
            "final_confidence": round(self.final_confidence, 3),
        }


class RetrievalConfidenceEngine:
    """
    Computes a normalized confidence score in [0.0, 1.0] from physical retrieval signals.
    """

    def __init__(self):
        self.settings = get_settings()
        self.w_rank = self.settings.confidence_weight_bm25_dense_rank
        self.w_rrf = self.settings.confidence_weight_rrf_top
        self.w_margin = self.settings.confidence_weight_score_margin
        self.w_rerank = self.settings.confidence_weight_rerank_score
        self.w_meta = self.settings.confidence_weight_metadata_match

    def compute_confidence(
        self,
        dense_ranked_ids: List[str],
        bm25_ranked_ids: List[str],
        fused_scores: List[float],
        rerank_scores: Optional[List[float]] = None,
        query: str = "",
        top_candidates_meta: Optional[List[Dict[str, Any]]] = None,
    ) -> ConfidenceSignals:
        # 1. BM25 / Dense Rank Agreement (Jaccard / Rank Overlap @ top-5)
        top5_dense = set(dense_ranked_ids[:5])
        top5_bm25 = set(bm25_ranked_ids[:5])
        if top5_dense or top5_bm25:
            intersection = top5_dense.intersection(top5_bm25)
            rank_agreement = len(intersection) / max(1, len(top5_dense.union(top5_bm25)))
        else:
            rank_agreement = 0.0

        # 2. Top-1 vs Top-2 Score Margin
        if len(fused_scores) >= 2:
            s1 = fused_scores[0]
            s2 = fused_scores[1]
            margin = max(0.0, min(1.0, (s1 - s2) / max(0.001, s1)))
        elif len(fused_scores) == 1:
            margin = 1.0
        else:
            margin = 0.0

        # 3. RRF Consensus (RRF Top score scaled relative to theoretical maximum 2 * (1/61) ~ 0.0328)
        if fused_scores:
            top_rrf = fused_scores[0]
            rrf_norm = max(0.0, min(1.0, top_rrf / 0.0328))
        else:
            rrf_norm = 0.0

        # 4. Reranker Score
        if rerank_scores and len(rerank_scores) > 0:
            rerank_conf = float(rerank_scores[0])
        else:
            # If reranking was not used, mirror dense/RRF confidence
            rerank_conf = rrf_norm

        # 5. Metadata Match (query keywords appearing in section_path / title)
        metadata_score = 0.0
        if query and top_candidates_meta:
            query_words = set(query.lower().split())
            matched_count = 0
            for meta in top_candidates_meta[:3]:
                sec = " ".join(meta.get("section_path", [])).lower()
                title = str(meta.get("title", "")).lower()
                if any(w in sec or w in title for w in query_words if len(w) > 3):
                    matched_count += 1
            metadata_score = matched_count / max(1, min(3, len(top_candidates_meta)))

        # Weighted composite confidence
        final_score = (
            self.w_rank * rank_agreement +
            self.w_margin * margin +
            self.w_rrf * rrf_norm +
            self.w_rerank * rerank_conf +
            self.w_meta * metadata_score
        )
        final_score = max(0.0, min(1.0, final_score))

        return ConfidenceSignals(
            bm25_dense_rank_agreement=rank_agreement,
            top_score_margin=margin,
            rrf_consensus=rrf_norm,
            reranker_score=rerank_conf,
            metadata_match=metadata_score,
            final_confidence=final_score,
        )


_confidence_engine_instance: Optional[RetrievalConfidenceEngine] = None


def get_confidence_engine() -> RetrievalConfidenceEngine:
    global _confidence_engine_instance
    if _confidence_engine_instance is None:
        _confidence_engine_instance = RetrievalConfidenceEngine()
    return _confidence_engine_instance
