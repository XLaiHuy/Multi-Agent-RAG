"""
Hierarchical Parent-Child Text Splitter for Legal RAG:
Splits large legal contracts/statutes into large Parent Chunks (entire Section/Article ~1200-1500 tokens)
and smaller Child Chunks (~200 tokens).
Returns Child Chunks for indexing with metadata referencing their Parent Chunk ID and parent text.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any
from pathlib import Path

from app.ingestion.loader import RawDocument
from app.ingestion.chunking import Chunk

def clean_text(text: str) -> str:
    return text.strip() if text else ""


@dataclass
class ParentChildChunk:
    child_chunk: Chunk
    parent_id: str
    parent_text: str


class ParentChildChunker:
    """
    Hierarchical chunker that generates:
    1. Parent Chunks (Section/Article level, ~1200-1500 chars) for complete LLM context.
    2. Child Chunks (~250-300 chars) for fine-grained BM25 & Dense similarity search.
    """

    def __init__(self, parent_size: int = 1500, child_size: int = 300, child_overlap: int = 50):
        self.parent_size = parent_size
        self.child_size = child_size
        self.child_overlap = child_overlap

    def chunk_document(self, doc: RawDocument) -> List[Chunk]:
        raw_text = clean_text(doc.text)
        if not raw_text:
            return []

        # Step 1: Split into Parent Chunks
        parent_texts = self._split_text(raw_text, max_size=self.parent_size, overlap=100)
        chunks: List[Chunk] = []

        for p_idx, p_text in enumerate(parent_texts):
            parent_id = f"{doc.doc_id}_parent_{p_idx}"
            
            # Step 2: Split each Parent into Child Chunks
            child_texts = self._split_text(p_text, max_size=self.child_size, overlap=self.child_overlap)
            
            for c_idx, c_text in enumerate(child_texts):
                child_id = f"{doc.doc_id}_parent_{p_idx}_child_{c_idx}"

                meta = dict(doc.metadata)
                meta.update({
                    "doc_id": doc.doc_id,
                    "source": doc.source,
                    "parent_id": parent_id,
                    "parent_text": p_text,
                    "is_child": True,
                })

                chunk = Chunk(
                    chunk_id=child_id,
                    doc_id=doc.doc_id,
                    text=c_text,
                    source=doc.source,
                    chunk_index=c_idx,
                    metadata=meta,
                )
                chunks.append(chunk)

        return chunks

    def _split_text(self, text: str, max_size: int, overlap: int) -> List[str]:
        if len(text) <= max_size:
            return [text]

        paragraphs = text.split("\n\n")
        chunks = []
        current = ""

        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            if len(current) + len(p) + 2 <= max_size:
                current = f"{current}\n\n{p}".strip()
            else:
                if current:
                    chunks.append(current)
                # If paragraph itself exceeds max_size, slice by sentences or words
                if len(p) > max_size:
                    for i in range(0, len(p), max_size - overlap):
                        chunks.append(p[i:i + max_size])
                    current = ""
                else:
                    current = p

        if current:
            chunks.append(current)

        return chunks
