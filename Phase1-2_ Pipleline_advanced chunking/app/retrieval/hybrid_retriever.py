from dataclasses import dataclass, field
import chromadb
from rank_bm25 import BM25Okapi
from app.core.config import Settings, get_settings
from app.retrieval.vector_retriever import VectorRetriever, SearchResult
from app.retrieval.reranker import Reranker


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k: int = 60
) -> list[tuple[str, float]]:
    """
    Combines multiple ranked lists of chunk_ids using Reciprocal Rank Fusion (RRF).
    
    Args:
        ranked_lists: A list of lists, where each inner list contains chunk_ids sorted 
                     by relevance (highest first).
        k: Constant parameter for RRF (default 60).
        
    Returns:
        A list of tuples (chunk_id, rrf_score) sorted in descending order of rrf_score.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked):
            # rank is 0-indexed in python, so we use (rank + 1) for 1-based rank
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
            
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class BM25Retriever:
    """
    A sparse retriever implementing the BM25 algorithm using the rank_bm25 library.
    It dynamically builds its index from the documents stored in Chroma DB.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.chroma_client = chromadb.PersistentClient(path=self.settings.chroma_path)
        self.collection = self.chroma_client.get_collection(name=self.settings.collection_name)
        self._build_index()
        
    def _build_index(self):
        # Pull all documents from Chroma collection to build BM25 index
        all_docs = self.collection.get(include=["documents", "metadatas"])
        
        self.chunk_ids = all_docs.get("ids", [])
        self.documents = all_docs.get("documents", [])
        self.metadatas = all_docs.get("metadatas", []) or [{}] * len(self.chunk_ids)
        
        if not self.chunk_ids:
            self.bm25 = None
        else:
            # Tokenize corpus (simple whitespace and lowercase split)
            tokenized_corpus = [doc.lower().split() for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_corpus)

    def reload_index(self):
        """Re-fetches collection and rebuilds BM25 index."""
        self._build_index()

    def search(self, query: str, top_k: int = 20, where: dict | None = None) -> list[tuple[str, float]]:
        """
        Search documents matching the query using BM25.
        
        Returns:
            A list of tuples (chunk_id, score) sorted in descending order of BM25 score.
        """
        if not self.bm25 or not query.strip():
            return []
            
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        
        ranked = []
        for i, (cid, score) in enumerate(zip(self.chunk_ids, scores)):
            meta = self.metadatas[i]
            # Simple where filter logic
            match = True
            if where:
                for k, v in where.items():
                    if meta.get(k) != v:
                        match = False
                        break
            if match:
                ranked.append((cid, score))
                
        ranked = sorted(ranked, key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


class HybridRetriever:
    """
    Hybrid retriever that runs Vector Search and BM25 Search, 
    merges the results using Reciprocal Rank Fusion (RRF),
    and returns the top candidates.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        vector_retriever: VectorRetriever | None = None,
        reranker: Reranker | None = None,
    ):
        self.settings = settings or get_settings()
        self.vector_retriever = vector_retriever or VectorRetriever(self.settings)
        self.bm25_retriever = BM25Retriever(self.settings)
        self._reranker = reranker
        if self._reranker is None and self.settings.enable_reranker:
            self._reranker = Reranker()

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = Reranker()
        return self._reranker

    def reload_index(self):
        """Reloads underlying BM25 index after upload."""
        self.bm25_retriever.reload_index()

    def search(self, query: str, top_k: int = 10, use_rerank: bool = False, where: dict | None = None) -> list[SearchResult]:
        should_rerank = use_rerank and self.settings.enable_reranker
        if not query.strip():
            raise ValueError("Query string cannot be empty.")

        # 1. Fetch top-20 candidates from Vector Search
        vector_results = self.vector_retriever.search(query=query, top_k=20, where=where)
        vector_ranked_ids = [res.chunk_id for res in vector_results]

        # 2. Fetch top-20 candidates from BM25 Search
        bm25_results = self.bm25_retriever.search(query=query, top_k=20, where=where)
        bm25_ranked_ids = [cid for cid, _ in bm25_results]

        # 3. Merge results using Reciprocal Rank Fusion (RRF)
        fused_results = reciprocal_rank_fusion([vector_ranked_ids, bm25_ranked_ids], k=60)

        # If reranking, take top-20 candidates for reranking. Otherwise take top_k.
        candidate_count = 20 if should_rerank else top_k
        top_fused = fused_results[:candidate_count]

        # Map vector search results for fast lookup
        vector_lookup: dict[str, SearchResult] = {res.chunk_id: res for res in vector_results}

        # Map all documents in BM25 index for fast lookup
        bm25_docs = {
            cid: (doc, meta)
            for cid, doc, meta in zip(
                self.bm25_retriever.chunk_ids,
                self.bm25_retriever.documents,
                self.bm25_retriever.metadatas
            )
        }

        candidates: list[SearchResult] = []
        for chunk_id, rrf_score in top_fused:
            if chunk_id in vector_lookup:
                res = vector_lookup[chunk_id]
                if res.metadata is None:
                    res.metadata = {}
                res.metadata["rrf_score"] = rrf_score
                candidates.append(res)
            else:
                doc_text, meta = bm25_docs.get(chunk_id, ("", {}))
                source = meta.get("source", meta.get("filename", "unknown"))
                
                res = SearchResult(
                    chunk_id=chunk_id,
                    text=doc_text,
                    source=source,
                    distance=1.0,
                    similarity=0.0,
                    metadata=meta.copy() if meta else {}
                )
                res.metadata["rrf_score"] = rrf_score
                candidates.append(res)

        if should_rerank:
            return self.reranker.rerank(query=query, candidates=candidates, top_n=top_k)

        return candidates

    def multi_query_search(
        self,
        queries: list[str],
        top_k: int = 12,
        use_rerank: bool = False,
        where: dict | None = None
    ) -> list[SearchResult]:
        """
        Runs search across multiple query variations, fuses ranked results with RRF,
        and returns deduplicated, high-recall candidates.
        """
        if not queries:
            return []

        all_ranked_lists: list[list[str]] = []
        chunk_map: dict[str, SearchResult] = {}

        for q in queries:
            if not q.strip():
                continue
            res_list = self.search(query=q, top_k=top_k, use_rerank=False, where=where)
            ranked_ids = []
            for item in res_list:
                ranked_ids.append(item.chunk_id)
                if item.chunk_id not in chunk_map:
                    chunk_map[item.chunk_id] = item
                else:
                    # Keep maximum similarity score
                    if item.similarity > chunk_map[item.chunk_id].similarity:
                        chunk_map[item.chunk_id].similarity = item.similarity
            if ranked_ids:
                all_ranked_lists.append(ranked_ids)

        if not all_ranked_lists:
            return []

        fused = reciprocal_rank_fusion(all_ranked_lists, k=60)
        final_candidates: list[SearchResult] = []

        for cid, score in fused[:top_k]:
            if cid in chunk_map:
                candidate = chunk_map[cid]
                if candidate.metadata is None:
                    candidate.metadata = {}
                candidate.metadata["multi_query_score"] = score
                final_candidates.append(candidate)

        return stitch_context_chunks(final_candidates)


def stitch_context_chunks(candidates: list[SearchResult]) -> list[SearchResult]:
    """
    Context Stitching: Groups adjacent chunks (sequential chunk_index from same source)
    together to reconstruct full tables and unbroken paragraphs.
    """
    if not candidates:
        return candidates

    # Group by source
    grouped: dict[str, list[SearchResult]] = {}
    for c in candidates:
        grouped.setdefault(c.source, []).append(c)

    stitched: list[SearchResult] = []
    seen_ids = set()

    for source, chunk_list in grouped.items():
        # Try to sort by chunk_index in metadata or numeric suffix of chunk_id
        def get_index(chunk: SearchResult) -> int:
            if chunk.metadata and "chunk_index" in chunk.metadata:
                try:
                    return int(chunk.metadata["chunk_index"])
                except (ValueError, TypeError):
                    pass
            # Try to extract trailing integer from chunk_id (e.g. 'doc_12')
            parts = chunk.chunk_id.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                return int(parts[1])
            return 0

        sorted_chunks = sorted(chunk_list, key=get_index)

        i = 0
        while i < len(sorted_chunks):
            current = sorted_chunks[i]
            current_idx = get_index(current)
            merged_text = current.text
            merged_id = current.chunk_id
            best_sim = current.similarity
            best_dist = current.distance
            best_meta = dict(current.metadata) if current.metadata else {}

            j = i + 1
            # Check if subsequent chunk is adjacent (difference of 1 in index)
            while j < len(sorted_chunks):
                next_chunk = sorted_chunks[j]
                next_idx = get_index(next_chunk)
                if next_idx == current_idx + (j - i):
                    # Adjacent! Merge text
                    merged_text += f"\n\n[...liền kề...]\n{next_chunk.text}"
                    merged_id += f"+{next_chunk.chunk_id.rsplit('_', 1)[-1]}"
                    best_sim = max(best_sim, next_chunk.similarity)
                    best_dist = min(best_dist, next_chunk.distance)
                    j += 1
                else:
                    break

            stitched.append(
                SearchResult(
                    chunk_id=merged_id,
                    text=merged_text,
                    source=source,
                    distance=best_dist,
                    similarity=best_sim,
                    metadata=best_meta
                )
            )
            i = j

    # Re-sort stitched chunks by similarity descending
    return sorted(stitched, key=lambda c: c.similarity, reverse=True)
