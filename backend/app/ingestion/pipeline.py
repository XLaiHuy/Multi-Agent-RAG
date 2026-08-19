"""
Document Ingestion Pipeline.
Executes the sequential, observable, idempotent stages:
QUEUED -> PARSING -> OCR if required -> NORMALIZING -> CHUNKING -> EMBEDDING -> INDEXING -> READY / FAILED.
"""
import os
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

from sqlalchemy.orm import Session

from backend.app.core.config import get_settings
from backend.app.domain.canonical import CanonicalDocument
from backend.app.ingestion.parsers import MasterDocumentParser
from backend.app.ingestion.chunker import StructureAwareParentChildChunker
from backend.app.providers.embeddings import get_embedding_provider
from backend.app.retrieval.dense import get_dense_retriever
from backend.app.retrieval.bm25 import get_bm25_retriever
from backend.app.persistence.database import JobRepository, SessionLocal

logger = logging.getLogger("ingestion_pipeline")


class IngestionPipeline:
    """
    Orchestrates the multi-stage ingestion process for uploaded contracts.
    """

    def __init__(self):
        self.settings = get_settings()
        self.embedder = get_embedding_provider()
        self.dense = get_dense_retriever()
        self.bm25 = get_bm25_retriever()
        self.chunker = StructureAwareParentChildChunker(
            child_target_tokens=self.settings.child_chunk_size,
            child_overlap_tokens=self.settings.child_chunk_overlap,
            parent_target_tokens=self.settings.parent_chunk_size,
            parent_overlap_tokens=self.settings.parent_chunk_overlap,
        )

    def process_job(self, job_id: str, document_id: str, file_path: Path, tenant_id: str = "default_tenant") -> bool:
        """
        Executes all stages of document ingestion with progress tracking.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            with SessionLocal() as db:
                JobRepository.update_job_status(db, job_id, "FAILED", 0, f"File not found: {file_path}")
            return False

        try:
            # Stage 1: PARSING
            with SessionLocal() as db:
                JobRepository.update_job_status(db, job_id, "PARSING", 15)

            logger.info(f"[Ingestion] Job {job_id}: Parsing {file_path.name}...")
            canonical_doc: CanonicalDocument = MasterDocumentParser.parse(file_path, document_id)

            # Stage 2: NORMALIZING & CHUNKING
            with SessionLocal() as db:
                JobRepository.update_job_status(db, job_id, "CHUNKING", 40)

            logger.info(f"[Ingestion] Job {job_id}: Chunking into token-aware Parent-Child blocks...")
            child_chunks, parent_chunks = self.chunker.chunk_canonical_document(
                canonical_doc, doc_version=1, tenant_id=tenant_id
            )

            if not child_chunks:
                raise ValueError("Zero searchable chunks generated from document.")

            # Stage 3: EMBEDDING
            with SessionLocal() as db:
                JobRepository.update_job_status(db, job_id, "EMBEDDING", 65)

            logger.info(f"[Ingestion] Job {job_id}: Embedding {len(child_chunks)} child chunks...")
            child_texts = [c.text for c in child_chunks]
            embeddings = self.embedder.embed_documents_batch(child_texts, batch_size=64)

            # Stage 4: INDEXING (Chroma Vector DB + BM25 Sparse Index)
            with SessionLocal() as db:
                JobRepository.update_job_status(db, job_id, "INDEXING", 85)

            logger.info(f"[Ingestion] Job {job_id}: Upserting into Dense & BM25 indices...")
            child_ids = [c.chunk_id for c in child_chunks]
            child_metas = [c.metadata for c in child_chunks]

            # Upsert into Chroma
            self.dense.upsert_chunks(
                chunk_ids=child_ids,
                texts=child_texts,
                embeddings=embeddings,
                metadatas=child_metas,
            )

            # Append to BM25 Index
            self.bm25.add_chunks(
                chunk_ids=child_ids,
                documents=child_texts,
                metadatas=child_metas,
            )

            # Stage 5: READY
            with SessionLocal() as db:
                JobRepository.update_job_status(
                    db, job_id, "READY", 100,
                    meta_info={
                        "child_chunks_indexed": len(child_chunks),
                        "parent_chunks": len(parent_chunks),
                        "page_count": len(canonical_doc.pages),
                    }
                )

            logger.info(f"[Ingestion] Job {job_id} successfully completed for {file_path.name}!")
            return True

        except Exception as e:
            err_msg = f"{str(e)}\n{traceback.format_exc()}"
            logger.error(f"[Ingestion] Job {job_id} failed: {err_msg}")
            with SessionLocal() as db:
                JobRepository.update_job_status(db, job_id, "FAILED", 0, error_message=str(e))
            return False


_ingestion_pipeline_instance: Optional[IngestionPipeline] = None


def get_ingestion_pipeline() -> IngestionPipeline:
    global _ingestion_pipeline_instance
    if _ingestion_pipeline_instance is None:
        _ingestion_pipeline_instance = IngestionPipeline()
    return _ingestion_pipeline_instance
