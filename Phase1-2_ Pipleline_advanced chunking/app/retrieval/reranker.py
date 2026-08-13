import os
# pyrefly: ignore [missing-import]
from sentence_transformers import CrossEncoder
from app.retrieval.vector_retriever import SearchResult


class Reranker:
    """
    Reranker using SentenceTransformers CrossEncoder to re-score candidate SearchResults.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base").strip()
        print(f"[Reranker] Loading CrossEncoder model '{self.model_name}'...", flush=True)
        self.model = CrossEncoder(self.model_name)

    def rerank(self, query: str, candidates: list[SearchResult], top_n: int = 5) -> list[SearchResult]:
        """
        Re-ranks a list of candidate SearchResults using CrossEncoder.
        
        Args:
            query: User's query string.
            candidates: List of candidate SearchResults (e.g. top-20 from RRF).
            top_n: Number of top candidates to return after re-ranking.
            
        Returns:
            List of SearchResult objects sorted by rerank_score descending, sliced to top_n.
        """
        if not candidates or not query.strip():
            return candidates[:top_n]

        # Prepare (query, text) pairs for the CrossEncoder
        pairs = [(query, c.text) for c in candidates]
        scores = self.model.predict(pairs)

        # Attach rerank_score to metadata and sort
        for c, s in zip(candidates, scores):
            if c.metadata is None:
                c.metadata = {}
            c.metadata["rerank_score"] = float(s)

        ranked = sorted(candidates, key=lambda c: c.metadata["rerank_score"], reverse=True)
        return ranked[:top_n]
