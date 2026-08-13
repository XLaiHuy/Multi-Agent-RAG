import os
from pathlib import Path
import chromadb
from app.core.config import Settings, get_settings
from app.core.embedding import get_embedding_provider
from app.ingestion.cleaning import clean_text
from app.ingestion.chunking import Chunk, chunk_document
from app.ingestion.loader import load_document


class IngestionPipeline:

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.embedder = get_embedding_provider(self.settings)

        # Create chroma directory if needed
        os.makedirs(self.settings.chroma_path, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=self.settings.chroma_path)

    def get_collection(self, reset: bool = False):
        coll_name = self.settings.collection_name
        if reset:
            try:
                self.chroma_client.delete_collection(name=coll_name)
                print(f"[Pipeline] Deleted existing collection: {coll_name}")
            except Exception:
                pass

        collection = self.chroma_client.get_or_create_collection(
            name=coll_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": self.settings.embedding_model,
                "embedding_dimension": self.settings.embedding_dimension,
            },
        )
        return collection

    def ingest_directory(
        self, raw_dir: Path, reset: bool = False, chunk_size: int = 500, chunk_overlap: int = 75
    ) -> dict[str, int]:
        collection = self.get_collection(reset=reset)

        raw_dir = Path(raw_dir)
        if not raw_dir.exists():
            raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

        supported_extensions = {
            ".md", ".txt", ".pdf", ".json", ".docx", ".doc",
            ".xlsx", ".xls", ".csv", ".png", ".jpg", ".jpeg", ".webp", ".bmp"
        }
        files = [
            f for f in raw_dir.iterdir()
            if f.is_file() and f.suffix.lower() in supported_extensions
        ]

        if not files:
            raise ValueError(f"No supported documents found in {raw_dir}")

        print(f"[Pipeline] Found {len(files)} files to ingest.")

        all_chunks: list[Chunk] = []
        doc_count = 0

        for file_path in files:
            try:
                raw_doc = load_document(file_path)
                cleaned_text = clean_text(raw_doc.text)
                chunks = chunk_document(
                    doc_id=raw_doc.doc_id,
                    text=cleaned_text,
                    source=raw_doc.source,
                    extra_metadata=raw_doc.metadata,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                all_chunks.extend(chunks)
                doc_count += 1
                print(f"  - Loaded & chunked: {file_path.name} -> {len(chunks)} chunks")
            except Exception as e:
                print(f"  - [WARNING] Failed to process {file_path.name}: {e}")

        chunk_ids = [chunk.chunk_id for chunk in all_chunks]
        chunk_texts = [chunk.text for chunk in all_chunks]
        chunk_metadatas = [chunk.metadata for chunk in all_chunks]

        print(f"[Pipeline] Embedding & Upserting {len(all_chunks)} chunks into Chroma DB (Provider: {self.settings.embedding_provider})...")
        doc_tuples = [
            (chunk.text, chunk.metadata.get("filename", chunk.source))
            for chunk in all_chunks
        ]
        chunk_embeddings = self.embedder.embed_documents_batch(doc_tuples)

        collection = self.get_collection(reset=False)
        collection.upsert(
            ids=chunk_ids,
            documents=chunk_texts,
            metadatas=chunk_metadatas,
            embeddings=chunk_embeddings,
        )

        print("[Pipeline] Ingestion completed successfully!")
        return {
            "documents_loaded": doc_count,
            "chunks_generated": len(all_chunks),
            "chunks_embedded": len(chunk_ids),
        }

    def ingest_file(
        self, file_path: Path, chunk_size: int = 500, chunk_overlap: int = 75
    ) -> dict:
        """Ingest a single file into ChromaDB. Used by the Upload API."""
        from app.ingestion.loader import load_document
        from app.ingestion.cleaning import clean_text
        from app.ingestion.chunking import chunk_document

        file_path = Path(file_path)
        collection = self.get_collection(reset=False)

        raw_doc = load_document(file_path)
        cleaned_text = clean_text(raw_doc.text)
        chunks = chunk_document(
            doc_id=raw_doc.doc_id,
            text=cleaned_text,
            source=raw_doc.source,
            extra_metadata=raw_doc.metadata,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        if not chunks:
            raise ValueError(f"No chunks generated from {file_path.name}")

        print(f"[Pipeline] Ingesting single file: {file_path.name} -> {len(chunks)} chunks")

        doc_tuples = [(c.text, c.metadata.get("filename", c.source)) for c in chunks]
        embeddings = self.embedder.embed_documents_batch(doc_tuples)

        collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
            embeddings=embeddings,
        )
        print(f"[Pipeline] Done. {len(chunks)} chunks indexed.")
        return {
            "documents_loaded": 1,
            "chunks_generated": len(chunks),
            "chunks_embedded": len(chunks),
        }
