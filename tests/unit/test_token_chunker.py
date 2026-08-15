"""
Unit Tests for Structure-Aware & Token-Aware Parent-Child Chunker.
"""
import pytest
from backend.app.domain.canonical import CanonicalDocument, CanonicalPage, CanonicalBlock, BlockType
from backend.app.ingestion.chunker import StructureAwareParentChildChunker, estimate_token_count


def test_estimate_token_count():
    text = "The quick brown fox jumps over the lazy dog." # 9 words
    tokens = estimate_token_count(text)
    assert 9 <= tokens <= 15


def test_parent_child_chunker_structure_preservation():
    blocks = [
        CanonicalBlock(
            block_id="b0",
            block_type=BlockType.HEADING,
            text="Article 1: Definitions and Scope",
            page_number=1,
            section_path=["Article 1"],
        ),
        CanonicalBlock(
            block_id="b1",
            block_type=BlockType.PARAGRAPH,
            text="This agreement is entered into by and between Party A and Party B for software services. " * 15,
            page_number=1,
            section_path=["Article 1", "1.1 Scope"],
        ),
        CanonicalBlock(
            block_id="b2",
            block_type=BlockType.HEADING,
            text="Article 2: Term and Termination",
            page_number=2,
            section_path=["Article 2"],
        ),
        CanonicalBlock(
            block_id="b3",
            block_type=BlockType.PARAGRAPH,
            text="Either party may terminate this agreement with sixty days written notice to the other party. " * 15,
            page_number=2,
            section_path=["Article 2", "2.1 Termination"],
        ),
    ]

    doc = CanonicalDocument(
        doc_id="contract_test_001",
        title="Master Services Agreement.pdf",
        doc_type="pdf",
        pages=[CanonicalPage(page_number=1, blocks=blocks[:2]), CanonicalPage(page_number=2, blocks=blocks[2:])],
    )

    chunker = StructureAwareParentChildChunker(
        child_target_tokens=100,
        child_overlap_tokens=20,
        parent_target_tokens=400,
        parent_overlap_tokens=50,
    )

    child_chunks, parent_chunks = chunker.chunk_canonical_document(doc, doc_version=1, tenant_id="tenant_alpha")

    # Verify parent and child generations
    assert len(parent_chunks) >= 2
    assert len(child_chunks) >= 2

    # Verify child chunks have parent references
    for c in child_chunks:
        assert c.parent_id is not None
        assert c.metadata.get("parent_text") is not None
        assert c.metadata.get("tenant_id") == "tenant_alpha"
        assert c.doc_id == "contract_test_001"
        assert c.page_number in [1, 2]
        assert len(c.section_path) > 0
