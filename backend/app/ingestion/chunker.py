"""
Token-Aware and Structure-Aware Hierarchical Parent-Child Chunker.
Splits CanonicalDocuments along legal structural boundaries:
Contract/Article -> Section -> Clause -> Paragraph -> Sentence -> Token Boundary.
Produces:
- Child Chunks: ~200-300 tokens (indexed in BM25 and Vector Search)
- Parent Chunks: ~1000-1500 tokens (retrieved for LLM synthesis context)
- Child Overlap: ~40-60 tokens
Preserves document_id, version, page_number, section_path, parent_id, block_id, bbox, and offsets.
"""
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from backend.app.domain.canonical import CanonicalDocument, CanonicalPage, CanonicalBlock, BoundingBox
from backend.app.core.config import get_settings


@dataclass
class IndexedChunk:
    chunk_id: str
    doc_id: str
    doc_version: int
    text: str
    is_parent: bool
    parent_id: Optional[str]
    page_number: int
    section_path: List[str]
    block_id: str
    bbox: Optional[Dict[str, float]]
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


def estimate_token_count(text: str) -> int:
    """Fast, reliable token estimator (~1.3 tokens per word for English/Technical/Legal text)."""
    words = text.split()
    return max(1, int(len(words) * 1.33))


def split_text_into_sentences(text: str) -> List[str]:
    """Splits text into sentences respecting abbreviations and clause markers."""
    # Split on periods/semicolons followed by whitespace and uppercase/number
    sentences = re.split(r"(?<=[.;:])\s+(?=[A-Z0-9\(\[])", text)
    return [s.strip() for s in sentences if s.strip()]


class StructureAwareParentChildChunker:
    """
    Hierarchical chunker that respects document structure and token budgets.
    """

    def __init__(
        self,
        child_target_tokens: int = 250,
        child_overlap_tokens: int = 50,
        parent_target_tokens: int = 1200,
        parent_overlap_tokens: int = 100,
    ):
        self.child_target = child_target_tokens
        self.child_overlap = child_overlap_tokens
        self.parent_target = parent_target_tokens
        self.parent_overlap = parent_overlap_tokens

    def chunk_canonical_document(
        self, doc: CanonicalDocument, doc_version: int = 1, tenant_id: str = "default_tenant"
    ) -> Tuple[List[IndexedChunk], List[IndexedChunk]]:
        """
        Returns (child_chunks, parent_chunks).
        Child chunks are indexed for search; parent chunks are stored for context expansion.
        """
        child_chunks: List[IndexedChunk] = []
        parent_chunks: List[IndexedChunk] = []

        all_blocks = doc.get_all_blocks()
        if not all_blocks:
            return [], []

        # 1. Group contiguous blocks into Parent Chunks (~1000-1500 tokens)
        current_parent_blocks: List[CanonicalBlock] = []
        current_parent_tokens = 0
        parent_idx = 0

        grouped_parents: List[Tuple[str, List[CanonicalBlock]]] = []

        for block in all_blocks:
            b_tokens = estimate_token_count(block.text)
            
            # If section path changes completely or parent size reached, flush parent
            path_changed = False
            if current_parent_blocks and block.section_path:
                if current_parent_blocks[-1].section_path != block.section_path:
                    # If major section (level 1) changed, break parent
                    if len(block.section_path) > 0 and len(current_parent_blocks[-1].section_path) > 0:
                        if block.section_path[0] != current_parent_blocks[-1].section_path[0]:
                            path_changed = True

            if (current_parent_tokens + b_tokens > self.parent_target or path_changed) and current_parent_blocks:
                parent_id = f"{doc.doc_id}_v{doc_version}_p{parent_idx}"
                grouped_parents.append((parent_id, current_parent_blocks))
                parent_idx += 1
                current_parent_blocks = [block]
                current_parent_tokens = b_tokens
            else:
                current_parent_blocks.append(block)
                current_parent_tokens += b_tokens

        if current_parent_blocks:
            parent_id = f"{doc.doc_id}_v{doc_version}_p{parent_idx}"
            grouped_parents.append((parent_id, current_parent_blocks))

        # 2. For each parent, create Parent Chunk record and split into Child Chunks (~200-300 tokens)
        for parent_id, p_blocks in grouped_parents:
            parent_full_text = "\n\n".join(b.text for b in p_blocks)
            parent_page = p_blocks[0].page_number if p_blocks else 1
            parent_section = p_blocks[0].section_path if p_blocks else []
            parent_bbox = p_blocks[0].bbox.to_dict() if (p_blocks and p_blocks[0].bbox) else None

            p_chunk = IndexedChunk(
                chunk_id=parent_id,
                doc_id=doc.doc_id,
                doc_version=doc_version,
                text=parent_full_text,
                is_parent=True,
                parent_id=None,
                page_number=parent_page,
                section_path=parent_section,
                block_id=p_blocks[0].block_id if p_blocks else "",
                bbox=parent_bbox,
                token_count=estimate_token_count(parent_full_text),
                metadata={"tenant_id": tenant_id, "title": doc.title, "is_parent": True},
            )
            parent_chunks.append(p_chunk)

            # Split parent into child chunks
            child_idx = 0
            current_child_sentences: List[str] = []
            current_child_tokens = 0
            current_child_page = parent_page
            current_child_bbox = parent_bbox
            current_child_block_id = p_blocks[0].block_id if p_blocks else ""

            for b in p_blocks:
                sentences = split_text_into_sentences(b.text)
                for sent in sentences:
                    s_tokens = estimate_token_count(sent)
                    if current_child_tokens + s_tokens > self.child_target and current_child_sentences:
                        c_text = " ".join(current_child_sentences)
                        c_id = f"{parent_id}_c{child_idx}"
                        c_chunk = IndexedChunk(
                            chunk_id=c_id,
                            doc_id=doc.doc_id,
                            doc_version=doc_version,
                            text=c_text,
                            is_parent=False,
                            parent_id=parent_id,
                            page_number=b.page_number,
                            section_path=b.section_path,
                            block_id=b.block_id,
                            bbox=b.bbox.to_dict() if b.bbox else None,
                            token_count=estimate_token_count(c_text),
                            metadata={
                                "tenant_id": tenant_id,
                                "parent_id": parent_id,
                                "parent_text": parent_full_text,
                                "title": doc.title,
                                "filename": doc.title,
                                "page_number": b.page_number,
                                "section_path": b.section_path,
                                "block_id": b.block_id,
                                "is_child": True,
                            },
                        )
                        child_chunks.append(c_chunk)
                        child_idx += 1

                        # Keep overlap sentences
                        overlap_sents = []
                        overlap_toks = 0
                        for prev_s in reversed(current_child_sentences):
                            prev_t = estimate_token_count(prev_s)
                            if overlap_toks + prev_t <= self.child_overlap:
                                overlap_sents.insert(0, prev_s)
                                overlap_toks += prev_t
                            else:
                                break

                        current_child_sentences = overlap_sents + [sent]
                        current_child_tokens = overlap_toks + s_tokens
                    else:
                        current_child_sentences.append(sent)
                        current_child_tokens += s_tokens

            if current_child_sentences:
                c_text = " ".join(current_child_sentences)
                c_id = f"{parent_id}_c{child_idx}"
                last_block = p_blocks[-1]
                c_chunk = IndexedChunk(
                    chunk_id=c_id,
                    doc_id=doc.doc_id,
                    doc_version=doc_version,
                    text=c_text,
                    is_parent=False,
                    parent_id=parent_id,
                    page_number=last_block.page_number,
                    section_path=last_block.section_path,
                    block_id=last_block.block_id,
                    bbox=last_block.bbox.to_dict() if last_block.bbox else None,
                    token_count=estimate_token_count(c_text),
                    metadata={
                        "tenant_id": tenant_id,
                        "parent_id": parent_id,
                        "parent_text": parent_full_text,
                        "title": doc.title,
                        "filename": doc.title,
                        "page_number": last_block.page_number,
                        "section_path": last_block.section_path,
                        "block_id": last_block.block_id,
                        "is_child": True,
                    },
                )
                child_chunks.append(c_chunk)

        return child_chunks, parent_chunks
