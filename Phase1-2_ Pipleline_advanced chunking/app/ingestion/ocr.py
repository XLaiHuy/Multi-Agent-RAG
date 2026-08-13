"""
Multimodal Vision OCR Module:
- Trích xuất văn bản, bảng biểu Markdown, mô tả biểu đồ/sơ đồ từ hình ảnh (.png, .jpg, .jpeg, .webp)
- Tự động nhận diện và OCR các trang PDF dạng scanned/chứa ảnh bằng Gemini Vision + PyMuPDF
"""
import os
import io
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


class VisionOCREngine:
    """
    Sử dụng Gemini Multimodal Vision để trích xuất văn bản, bảng số liệu và sơ đồ từ ảnh.
    """

    def __init__(self, model_name: str = "gemini-flash-latest"):
        self.model_name = os.getenv("LLM_MODEL", model_name)
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("[VisionOCR] WARNING: GEMINI_API_KEY is not set. Multimodal OCR disabled.")
            self.client = None
        else:
            self.client = genai.Client(api_key=api_key)

    def extract_from_image_bytes(self, image_bytes: bytes, mime_type: str = "image/png") -> str:
        """
        Gửi image bytes lên Gemini Vision để bóc tách Markdown text, Table và Diagram info.
        """
        if not self.client:
            return "[OCR Error: GEMINI_API_KEY is not configured]"

        prompt = """Bạn là một chuyên gia Document Parser và OCR đa phương thức cao cấp.
Nhiệm vụ của bạn là phân tích hình ảnh tài liệu/sơ đồ và chuyển đổi thành định dạng Markdown chuẩn xác:
1. Trích xuất toàn bộ chữ (Tiếng Việt và Tiếng Anh) giữ nguyên cấu trúc tiêu đề (#, ##, ###), bullet point.
2. Nếu có BẢNG BIỂU: chuyển thành bảng Markdown (| Cột 1 | Cột 2 |).
3. Nếu có BIỂU ĐỒ / SƠ ĐỒ / HÌNH MINH HỌA: tạo một mục `### [Mô tả hình ảnh/Sơ đồ]` tóm tắt rõ các luồng dữ liệu, giá trị trục X/Y, xu hướng và chú thích chính trong hình.
4. Giữ nguyên các công thức toán/kỹ thuật nếu có.
5. Chỉ trả về nội dung Markdown kết quả, không thêm lời chào, không bọc trong ```markdown block."""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    prompt,
                ],
            )
            return response.text.strip() if response and response.text else ""
        except Exception as e:
            print(f"[VisionOCR] Extraction error: {e}")
            return f"[OCR Extraction Failed: {e}]"

    def extract_from_image_file(self, image_path: Path) -> str:
        """Đọc file ảnh từ ổ đĩa và chạy OCR."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        ext = image_path.suffix.lower().replace(".", "")
        if ext == "jpg":
            mime_type = "image/jpeg"
        elif ext in ["png", "webp", "jpeg"]:
            mime_type = f"image/{ext}"
        else:
            mime_type = "image/png"

        image_bytes = image_path.read_bytes()
        return self.extract_from_image_bytes(image_bytes, mime_type=mime_type)

    def extract_scanned_pdf(self, pdf_path: Path, min_char_threshold: int = 40) -> str:
        """
        Duyệt qua các trang của file PDF bằng PyMuPDF:
        - Nếu trang đã có text native tốt (>= min_char_threshold): lấy trực tiếp.
        - Nếu trang là scanned / chứa ảnh ít text: render trang thành ảnh 200 DPI và chạy Gemini Vision OCR.
        """
        import pymupdf

        doc = pymupdf.open(str(pdf_path))
        page_results = []

        print(f"  [VisionOCR] Analyzing PDF pages in {pdf_path.name} ({len(doc)} pages)...")

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1
            native_text = page.get_text().strip()

            if len(native_text) >= min_char_threshold:
                # Trang có text tốt
                page_results.append(f"<!-- Page {page_num} -->\n{native_text}")
            else:
                # Trang scanned hoặc ảnh biểu đồ -> render thành pixmap và OCR
                print(f"  [VisionOCR] Page {page_num} is scanned/image-heavy ({len(native_text)} chars) -> Triggering Multimodal OCR...")
                pix = page.get_pixmap(dpi=200)
                img_bytes = pix.tobytes(output="png")
                ocr_text = self.extract_from_image_bytes(img_bytes, mime_type="image/png")
                if ocr_text:
                    page_results.append(f"<!-- Page {page_num} [Scanned/OCR] -->\n{ocr_text}")
                elif native_text:
                    page_results.append(f"<!-- Page {page_num} -->\n{native_text}")

        doc.close()
        full_text = "\n\n---\n\n".join(page_results).strip()
        return full_text


_ocr_engine: Optional[VisionOCREngine] = None


def get_ocr_engine() -> VisionOCREngine:
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = VisionOCREngine()
    return _ocr_engine
