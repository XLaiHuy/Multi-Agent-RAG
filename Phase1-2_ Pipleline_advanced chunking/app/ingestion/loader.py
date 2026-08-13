"""
Document Loader tích hợp:
- Text/Markdown files (.md, .txt)
- Text-based PDF & Scanned PDF với Smart OCR (pdf-inspector, PyMuPDF, Gemini Vision OCR)
- Image files (.png, .jpg, .jpeg, .webp, .bmp, .tiff) qua Gemini Multimodal Vision OCR
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class RawDocument:
    doc_id: str
    source: str
    text: str
    metadata: dict = field(default_factory=dict)


def load_text_file(path: Path) -> RawDocument:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    text = path.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        raise ValueError(f"File is empty: {path}")

    return RawDocument(
        doc_id=path.stem,
        source=path.name,
        text=text,
        metadata={
            "filename": path.name,
            "source": path.name,
            "file_type": path.suffix.lower(),
        },
    )


def load_image_document(path: Path) -> RawDocument:
    """
    Load file ảnh (.png, .jpg, .jpeg, .webp, .bmp) bằng Gemini Multimodal Vision OCR.
    Trích xuất toàn bộ text, bảng biểu và mô tả sơ đồ/biểu đồ.
    """
    from app.ingestion.ocr import get_ocr_engine

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    engine = get_ocr_engine()
    print(f"  [Loader] Running Multimodal OCR for image: {path.name}...")
    markdown_text = engine.extract_from_image_file(path)

    if not markdown_text or markdown_text.startswith("[OCR Error"):
        raise ValueError(f"Không thể trích xuất nội dung từ ảnh {path.name}: {markdown_text}")

    return RawDocument(
        doc_id=path.stem,
        source=path.name,
        text=markdown_text,
        metadata={
            "filename": path.name,
            "source": path.name,
            "file_type": "image",
            "image_format": path.suffix.lower(),
            "char_count": len(markdown_text),
            "ocr_engine": "gemini_multimodal_vision",
        },
    )


def load_pdf_with_inspector(path: Path) -> RawDocument:
    """
    Dùng pdf-inspector (Firecrawl) để load PDF.
    - Tự phân loại: text_based / scanned / image_based / mixed
    - Trả về Markdown có cấu trúc (giữ heading, bảng, code block)
    """
    import pdf_inspector

    result = pdf_inspector.process_pdf(str(path))
    markdown_text = result.markdown or ""
    pdf_type = getattr(result, "pdf_type", "unknown")

    if not markdown_text.strip():
        raise ValueError(
            f"pdf-inspector không trích xuất được nội dung từ '{path.name}'. "
            f"PDF type: {pdf_type}"
        )

    print(f"  [pdf-inspector] {path.name}: type={pdf_type}, chars={len(markdown_text)}")

    return RawDocument(
        doc_id=path.stem,
        source=path.name,
        text=markdown_text,
        metadata={
            "filename": path.name,
            "source": path.name,
            "file_type": ".pdf",
            "pdf_type": str(pdf_type),
            "char_count": len(markdown_text),
        },
    )


def load_pdf_with_ocr_engine(path: Path) -> RawDocument:
    """
    Fallback mạnh mẽ: dùng PyMuPDF + Gemini Vision OCR để phân tích từng trang PDF.
    Trang có text thì lấy text, trang scanned/ảnh thì render và OCR.
    """
    from app.ingestion.ocr import get_ocr_engine

    engine = get_ocr_engine()
    markdown_text = engine.extract_scanned_pdf(path)

    if not markdown_text.strip():
        raise ValueError(f"Không thể trích xuất văn bản từ PDF '{path.name}'.")

    return RawDocument(
        doc_id=path.stem,
        source=path.name,
        text=markdown_text,
        metadata={
            "filename": path.name,
            "source": path.name,
            "file_type": ".pdf",
            "pdf_type": "scanned_or_hybrid_ocr",
            "char_count": len(markdown_text),
            "ocr_engine": "gemini_vision_pymupdf",
        },
    )


def load_pdf_fallback_pypdf(path: Path) -> RawDocument:
    """
    Fallback cuối cùng: dùng pypdf cơ bản.
    """
    from pypdf import PdfReader

    reader = PdfReader(path)
    page_texts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            page_texts.append(page_text)

    full_text = "\n\n".join(page_texts).strip()
    if not full_text:
        raise ValueError(f"PDF '{path.name}' không có text native.")

    return RawDocument(
        doc_id=path.stem,
        source=path.name,
        text=full_text,
        metadata={
            "filename": path.name,
            "source": path.name,
            "file_type": ".pdf",
            "pdf_type": "text_based_fallback",
            "page_count": len(reader.pages),
        },
    )


def load_pdf_file(path: Path) -> RawDocument:
    """
    Load PDF:
    1. Thử pdf-inspector trước
    2. Nếu thất bại hoặc tài liệu scanned -> Dùng Vision OCR Engine (PyMuPDF + Gemini Vision)
    3. Cuối cùng fallback về pypdf
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    # Bước 1: Thử pdf-inspector
    try:
        doc = load_pdf_with_inspector(path)
        # Nếu trích xuất được hơn 100 ký tự -> thành công
        if len(doc.text.strip()) >= 100:
            return doc
    except Exception as e:
        print(f"  [Loader] pdf-inspector could not process '{path.name}' ({e}), switching to Vision OCR...")

    # Bước 2: Thử Vision OCR Engine (xử lý scanned PDF & biểu đồ)
    try:
        return load_pdf_with_ocr_engine(path)
    except Exception as e:
        print(f"  [Loader] Vision OCR encountered error ({e}), switching to fallback pypdf...")

    # Bước 3: Fallback pypdf
    return load_pdf_fallback_pypdf(path)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


def load_document(path: Path) -> RawDocument:
    path = Path(path)
    ext = path.suffix.lower()
    if ext in [".md", ".txt"]:
        return load_text_file(path)
    elif ext == ".pdf":
        return load_pdf_file(path)
    elif ext in IMAGE_EXTENSIONS:
        return load_image_document(path)
    else:
        raise ValueError(f"Unsupported file format '{ext}' for file: {path}")
