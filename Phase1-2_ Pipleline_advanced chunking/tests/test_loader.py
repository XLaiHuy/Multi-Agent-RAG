import pytest
from pathlib import Path
from PIL import Image, ImageDraw
import io

from app.ingestion.loader import load_document, load_text_file, load_image_document


def test_load_text_file(tmp_path: Path):
    sample_file = tmp_path / "sample.md"
    sample_file.write_text("# Test Title\nThis is a sample markdown content.", encoding="utf-8")

    doc = load_document(sample_file)
    assert doc.doc_id == "sample"
    assert doc.source == "sample.md"
    assert "Test Title" in doc.text
    assert doc.metadata["file_type"] == ".md"


def test_load_image_synthetic(tmp_path: Path):
    """Test creating an image with text and running loader on it."""
    img_path = tmp_path / "test_chart.png"
    img = Image.new("RGB", (300, 100), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((10, 10), "Doanh thu Q1: 100 trieu", fill=(0, 0, 0))
    d.text((10, 30), "Doanh thu Q2: 250 trieu", fill=(0, 0, 0))
    img.save(img_path)

    doc = load_document(img_path)
    assert doc.doc_id == "test_chart"
    assert doc.source == "test_chart.png"
    assert doc.metadata["file_type"] == "image"
    assert len(doc.text) > 0
    print("OCR extracted from synthetic image:\n", doc.text)
