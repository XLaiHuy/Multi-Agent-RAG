"""
Unit Tests for Docling Configuration and OCR Fallback Invariants.
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from backend.app.ingestion.parsers import MasterDocumentParser, DoclingPDFParserAdapter
from backend.app.domain.canonical import CanonicalDocument, CanonicalPage, CanonicalBlock, BlockType
from backend.app.core.config import Settings


def test_docling_disabled_uses_native_parser(tmp_path, monkeypatch):
    """When USE_DOCLING_PARSER=false, native PyMuPDF path is executed."""
    monkeypatch.setattr("backend.app.ingestion.parsers.get_settings", lambda: Settings(USE_DOCLING_PARSER=False))

    # Mock NativePDFParser
    dummy_doc = CanonicalDocument(
        doc_id="d1", title="test.pdf", doc_type="pdf",
        pages=[CanonicalPage(page_number=1, blocks=[
            CanonicalBlock(block_id="b1", block_type=BlockType.PARAGRAPH, text="Sample text over 100 characters " * 5, page_number=1)
        ])]
    )
    with patch("backend.app.ingestion.parsers.NativePDFParser.parse_pdf", return_value=dummy_doc) as mock_native:
        with patch.object(DoclingPDFParserAdapter, "is_available", return_value=True) as mock_avail:
            res = MasterDocumentParser.parse(Path("dummy.pdf"), "d1")
            assert res == dummy_doc
            mock_native.assert_called_once()
            # is_available not called because use_docling_parser is False
            mock_avail.assert_not_called()


def test_docling_enabled_attempts_adapter(tmp_path, monkeypatch):
    """When USE_DOCLING_PARSER=true and Docling is available, Docling adapter is attempted."""
    monkeypatch.setattr("backend.app.ingestion.parsers.get_settings", lambda: Settings(USE_DOCLING_PARSER=True))

    docling_result = CanonicalDocument(
        doc_id="d_docling", title="test.pdf", doc_type="pdf",
        pages=[CanonicalPage(page_number=1, blocks=[
            CanonicalBlock(block_id="bd1", block_type=BlockType.PARAGRAPH, text="Docling parsed content.", page_number=1)
        ])]
    )
    with patch.object(DoclingPDFParserAdapter, "is_available", return_value=True):
        with patch.object(DoclingPDFParserAdapter, "parse_pdf", return_value=docling_result) as mock_parse:
            res = MasterDocumentParser.parse(Path("dummy.pdf"), "d_docling")
            assert res == docling_result
            mock_parse.assert_called_once()


def test_docling_missing_falls_back_per_policy(tmp_path, monkeypatch):
    """When USE_DOCLING_PARSER=true but Docling is not installed, falls back to native parser."""
    monkeypatch.setattr("backend.app.ingestion.parsers.get_settings", lambda: Settings(USE_DOCLING_PARSER=True))

    native_doc = CanonicalDocument(
        doc_id="d_fallback", title="test.pdf", doc_type="pdf",
        pages=[CanonicalPage(page_number=1, blocks=[
            CanonicalBlock(block_id="bn1", block_type=BlockType.PARAGRAPH, text="Native content text " * 10, page_number=1)
        ])]
    )
    with patch.object(DoclingPDFParserAdapter, "is_available", return_value=False):
        with patch("backend.app.ingestion.parsers.NativePDFParser.parse_pdf", return_value=native_doc) as mock_native:
            res = MasterDocumentParser.parse(Path("dummy.pdf"), "d_fallback")
            assert res == native_doc
            mock_native.assert_called_once()


def test_empty_scanned_document_not_silently_indexed(tmp_path, monkeypatch):
    """When a PDF has 0 extractable characters and no OCR provider is configured, raises ValueError."""
    monkeypatch.setattr("backend.app.ingestion.parsers.get_settings", lambda: Settings(USE_DOCLING_PARSER=False))

    empty_doc = CanonicalDocument(
        doc_id="d_empty", title="empty.pdf", doc_type="pdf",
        pages=[CanonicalPage(page_number=1, blocks=[])]
    )
    with patch("backend.app.ingestion.parsers.NativePDFParser.parse_pdf", return_value=empty_doc):
        with pytest.raises(ValueError, match="contains zero extractable text and no OCR provider is configured"):
            MasterDocumentParser.parse(Path("empty.pdf"), "d_empty", ocr_provider=None)
