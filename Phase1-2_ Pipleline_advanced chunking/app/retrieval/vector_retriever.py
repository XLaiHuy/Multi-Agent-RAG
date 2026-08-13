from dataclasses import dataclass, field
# pyrefly: ignore [missing-import]
import chromadb
from app.core.config import Settings, get_settings
from app.core.embedding import get_embedding_provider


@dataclass
class SearchResult:
    chunk_id: str
    text: str
    source: str
    distance: float
    similarity: float
    metadata: dict = field(default_factory=dict)


class VectorRetriever:

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.embedder = get_embedding_provider(self.settings)
        self.chroma_client = chromadb.PersistentClient(path=self.settings.chroma_path)
        self.collection = self.chroma_client.get_collection(name=self.settings.collection_name)

    def search(self, query: str, top_k: int = 3, where: dict | None = None) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("Query string cannot be empty.")

        query_vector = self.embedder.embed_query(query)
        
        query_params = {
            "query_embeddings": [query_vector],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_params["where"] = where
            
        res = self.collection.query(**query_params)

        results: list[SearchResult] = []
        if not res or not res.get("ids") or not res["ids"][0]:
            return results

        ids = res["ids"][0]
        docs = res["documents"][0] if res.get("documents") else [""] * len(ids)
        metas = res["metadatas"][0] if res.get("metadatas") else [{}] * len(ids)
        distances = res["distances"][0] if res.get("distances") else [0.0] * len(ids)

        for chunk_id, doc_text, meta, dist in zip(ids, docs, metas, distances):
            sim = max(0.0, 1.0 - dist)
            source = meta.get("source", meta.get("filename", "unknown"))
            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    text=doc_text,
                    source=source,
                    distance=dist,
                    similarity=sim,
                    metadata=meta,
                )
            )

        return results
