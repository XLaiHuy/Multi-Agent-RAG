import pytest
from app.retrieval.hybrid_retriever import reciprocal_rank_fusion
from app.retrieval.vector_retriever import SearchResult

def test_reciprocal_rank_fusion():
    list1 = ["doc_a", "doc_b", "doc_c"]
    list2 = ["doc_b", "doc_a", "doc_d"]
    fused = reciprocal_rank_fusion([list1, list2], k=60)
    
    assert len(fused) == 4
    # Highest ranked chunk across both lists should be doc_b or doc_a
    top_chunk, score = fused[0]
    assert top_chunk in ["doc_a", "doc_b"]
    assert score > 0.0

def test_search_result_dataclass():
    res = SearchResult(
        chunk_id="chunk_101",
        text="Sample text",
        source="sample.pdf",
        distance=0.1,
        similarity=0.9,
        metadata={"filename": "sample.pdf"}
    )
    assert res.chunk_id == "chunk_101"
    assert res.similarity == 0.9
    assert res.metadata["filename"] == "sample.pdf"

def test_stitch_context_chunks():
    from app.retrieval.hybrid_retriever import stitch_context_chunks
    chunk1 = SearchResult(
        chunk_id="doc_1", text="Table Part 1", source="paper.pdf",
        distance=0.1, similarity=0.9, metadata={"chunk_index": 0}
    )
    chunk2 = SearchResult(
        chunk_id="doc_2", text="Table Part 2", source="paper.pdf",
        distance=0.15, similarity=0.85, metadata={"chunk_index": 1}
    )
    chunk_other = SearchResult(
        chunk_id="other_9", text="Unrelated text", source="other.pdf",
        distance=0.3, similarity=0.7, metadata={"chunk_index": 9}
    )

    stitched = stitch_context_chunks([chunk1, chunk2, chunk_other])
    assert len(stitched) == 2
    # The two adjacent chunks from paper.pdf should be stitched together
    paper_chunk = [c for c in stitched if c.source == "paper.pdf"][0]
    assert "Table Part 1" in paper_chunk.text
    assert "Table Part 2" in paper_chunk.text
    assert paper_chunk.similarity == 0.9
