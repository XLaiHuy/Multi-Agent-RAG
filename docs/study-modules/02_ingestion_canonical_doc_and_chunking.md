# Module 2: Ingestion, OCR Status, CanonicalDocument & Parent–Child Chunking

---

## A. Intuition (Trực giác)
Hợp đồng pháp lý không phải là văn bản xuôi thông thường. Chúng được tổ chức theo cấu trúc phân cấp chặt chẽ: Chương $\rightarrow$ Điều khoản $\rightarrow$ Khoản nhỏ $\rightarrow$ Ngoại lệ.
- Nếu bạn dùng **Fixed-size Chunking** (ví dụ: cứ 500 ký tự cắt một phát), nhát cắt có thể rơi đúng vào giữa một con số bồi thường hoặc cắt rời câu khẳng định khỏi câu ngoại lệ (ví dụ: *"Bên A phải bồi thường 1 triệu USD..."* nằm ở chunk 1, còn *"...trừ trường hợp thiên tai"* lại rơi sang chunk 2). Khi đó, câu trả lời của RAG sẽ sai lệch hoàn toàn về mặt pháp lý.
- **Structure-Aware Parent-Child Chunking** giải quyết việc này: Ta lập chỉ mục các đoạn con nhỏ (~250 tokens) chứa các mệnh đề nguyên tử để tìm kiếm cực nhạy, nhưng khi nộp bằng chứng cho LLM, ta lấy toàn bộ đoạn cha (~1200 tokens) chứa toàn bộ điều khoản bao quanh để LLM thấy đầy đủ các điều kiện ràng buộc.

---

## B. Role in My System (Vai trò trong hệ thống)
File mã nguồn:
- `backend/app/domain/models.py`: Định nghĩa cấu trúc chuẩn hóa `CanonicalDocument`, `CanonicalPage`, `CanonicalBlock`.
- `backend/app/ingestion/parsers.py`: Bộ phân tích đa định dạng `MasterDocumentParser`, `NativePDFParser`, và bộ phân tích mật độ `OCRGatingAnalyzer`.
- `backend/app/ingestion/chunker.py`: Bộ chia đoạn phân cấp cấu trúc `StructureAwareParentChildChunker`.

---

## C. Actual Code Trace & Data Structures

```python
# Pseudocode từ backend/app/domain/models.py
class CanonicalBlock(BaseModel):
    block_id: str
    text: str
    block_type: BlockType  # HEADING, PARAGRAPH, TABLE_ROW, LIST_ITEM
    section_path: List[str]  # e.g. ["8. LIMITATION OF LIABILITY", "8.1 General Cap"]
    page_number: int
    bbox: Optional[Tuple[float, float, float, float]]  # Tọa độ bounding box

class CanonicalDocument(BaseModel):
    document_id: str
    tenant_id: str
    filename: str
    pages: List[CanonicalPage]
    metadata: Dict[str, Any]
```

### Thuật toán Parent-Child Chunking (`StructureAwareParentChildChunker`)
1. **Duyệt qua các CanonicalBlock**: Nhận diện ranh giới cấu trúc dựa trên `HEADING` và `section_path`.
2. **Xây dựng Parent Chunk (~1,200 tokens)**: Gom các block thuộc cùng một Section lớn (ví dụ toàn bộ Điều 8).
3. **Cắt Child Chunks (~250 tokens, overlap 40 tokens)**: Tách nhỏ Section thành các đoạn nguyên tử theo ranh giới câu (`sent_tokenize`), lưu `parent_id` trỏ ngược về Parent Chunk tương ứng.

---

## D. Honest Audit: Trạng thái thực tế của tính năng OCR

| Tiêu chí | Trạng thái trong Code | Chi tiết kỹ thuật |
|---|---|---|
| **Code tồn tại ở đâu?** | `backend/app/ingestion/parsers.py` | Lớp `OCRGatingAnalyzer` phân tích tỷ lệ `char_count / page_area` và mật độ hình ảnh. |
| **Gating Logic hoạt động thế nào?** | `should_trigger_ocr(page)` | Nếu mật độ ký tự số hóa $< 50$ chars/page hoặc có ảnh chiếm $> 60\%$ diện tích trang $\rightarrow$ Đánh dấu cần OCR. |
| **Wiring trong Runtime Ingestion** | **Partially Wired / Ready Interface** | Hàm parse trong `MasterDocumentParser` hiện tại gọi `NativePDFParser` (sử dụng PyMuPDF). Lớp trừu tượng `OCRProvider` đã được định nghĩa nhưng chưa được inject mặc định vào service layer. |
| **Tôi nên nói gì trong Phỏng vấn?** | **Minh bạch & Kỹ thuật cao** | *"Trong repo, tôi đã thiết kế đầy đủ Domain Model `CanonicalBlock` có `bbox` và `OCRGatingAnalyzer` để phát hiện trang scan dựa trên ngưỡng mật độ ký tự. Hiện tại pipeline mặc định chạy Native PDF để tối ưu tốc độ CPU (sub-second), và tôi đã trừu tượng hóa `OCRProvider` để sẵn sàng cắm Cloud Vision hoặc Tesseract khi mở rộng lên production."* |

---

## E. Checkpoint: 7 Câu hỏi Phỏng vấn chuyên sâu

1. *(Easy)*: Tại sao việc bảo tồn `section_path` (đường dẫn mục lục) trong metadata của chunk lại quan trọng đối với RAG pháp lý?
2. *(Easy)*: Sự khác nhau về kích thước và vai trò giữa Child Chunk (~250 tokens) và Parent Chunk (~1200 tokens) là gì?
3. *(Medium)*: Giả sử một hợp đồng có bảng biểu (Table) phức tạp chứa các mức phí phạt, `MasterDocumentParser` sẽ biểu diễn nó thành các `CanonicalBlock` như thế nào?
4. *(Medium)*: Tại sao fixed-size chunking (ví dụ LangChain RecursiveCharacterTextSplitter thuần túy) lại gây nguy hiểm cho việc trích xuất điều khoản pháp lý?
5. *(Hard)*: Trình bày cơ chế hoạt động của `OCRGatingAnalyzer`: Những yếu tố nào quyết định một trang PDF cần được gửi qua OCR thay vì trích xuất Native text?
6. *(Hard)*: Nếu hai điều khoản con (`c_1` và `c_2`) cùng thuộc về một điều khoản cha (`p_1`) được xếp hạng cao trong top retrieval, hệ thống xử lý thế nào để tránh gửi trùng lặp `p_1` nhiều lần vào context của LLM?
7. *(Deep-Dive)*: Bounding box (`bbox`) được lưu trong `CanonicalBlock` phục vụ cho mục đích gì trong UI của ứng dụng pháp lý?
