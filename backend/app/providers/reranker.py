"""
Local CrossEncoder Reranker Provider.
Uses HuggingFace / SentenceTransformers CrossEncoder (e.g. BAAI/bge-reranker-base)
with score normalization and batch scoring.
"""
import logging
from typing import List, Tuple, Optional
from backend.app.core.config import get_settings
from backend.app.providers.interfaces import RerankerProvider

logger = logging.getLogger("reranker")


class LocalCrossEncoderReranker(RerankerProvider):
    """Local CrossEncoder reranker with lazy loading and sigmoid score normalization."""

    def __init__(self, model_name: Optional[str] = None):
        settings = get_settings()
        self.model_name = model_name or settings.reranker_model
        self._model = None

    def _get_model(self):
        if self._model is None:
            logger.info(f"[Reranker] Loading CrossEncoder model '{self.model_name}'...")
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self, query: str, candidate_texts: List[str], top_n: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Reranks candidates against the query.
        Returns sorted list of (original_index, normalized_score) descending.
        """
        if not candidate_texts or not query.strip():
            return [(i, 0.0) for i in range(min(top_n, len(candidate_texts)))]

        try:
            import torch
            model = self._get_model()
            pairs = [(query, text) for text in candidate_texts]
            with torch.no_grad():
                raw_scores = model.predict(pairs, batch_size=32, show_progress_bar=False)

            indexed_scores = []
            for idx, score in enumerate(raw_scores):
                # Sigmoid normalization: 1 / (1 + exp(-score))
                import math
                try:
                    norm_score = 1.0 / (1.0 + math.exp(-float(score)))
                except OverflowError:
                    norm_score = 1.0 if score > 0 else 0.0
                indexed_scores.append((idx, norm_score))

            indexed_scores.sort(key=lambda x: x[1], reverse=True)
            return indexed_scores[:top_n]
        except Exception as e:
            logger.warning(f"[Reranker] Model prediction skipped ({e}), using default rank order.")
            return [(i, max(0.1, 1.0 - (i * 0.1))) for i in range(min(top_n, len(candidate_texts)))]


_reranker_instance: Optional[RerankerProvider] = None


def get_reranker() -> RerankerProvider:
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = LocalCrossEncoderReranker()
    return _reranker_instance
