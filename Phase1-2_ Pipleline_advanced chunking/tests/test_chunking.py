import pytest
from app.ingestion.cleaning import clean_text
from app.ingestion.chunking import chunk_document

def test_clean_text():
    raw_text = "   Xin chào   \n\n\n  Thế giới  RAG   "
    cleaned = clean_text(raw_text)
    assert "Xin chào" in cleaned
    assert "Thế giới RAG" in cleaned

def test_chunk_document():
    doc_id = "doc_test_1"
    text = "Đây là văn bản thử nghiệm nhằm mục đích kiểm tra chức năng phân đoạn dữ liệu (chunking). " * 10
    chunks = chunk_document(
        doc_id=doc_id,
        text=text,
        source="test_source.txt",
        chunk_size=100,
        chunk_overlap=20
    )
    assert len(chunks) > 0
    assert chunks[0].doc_id == doc_id
    assert chunks[0].source == "test_source.txt"
    assert len(chunks[0].text) > 0
