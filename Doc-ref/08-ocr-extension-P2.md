# Module 8 (P2 — chỉ làm nếu còn dư thời gian) — OCR & Scanned PDF

> Không bắt đầu module này nếu Definition of Done ở file `00-ROADMAP.md` chưa hoàn thành. Đây là phần mở rộng, không phải P0.

## 1. Text-based PDF vs Scanned PDF

Text-based PDF chứa lớp text có thể extract trực tiếp (`PyPDFLoader` đọc được ngay). Scanned PDF thực chất là **ảnh** của trang giấy — extract trực tiếp ra chuỗi rỗng hoặc ký tự lỗi. Cần OCR (Optical Character Recognition) để chuyển ảnh → text trước khi đưa vào pipeline RAG bình thường.

## 2. Pipeline OCR dự kiến

```
Scanned PDF → Page Rendering → Image Preprocessing → OCR → Layout Detection
→ Reading Order → Table/Figure Handling → Text Cleaning → Section Detection
→ Chunking → Indexing   (nối tiếp vào pipeline RAG chính từ Module 01)
```

| Bước | Giải thích |
|---|---|
| Page Rendering | Chuyển từng trang PDF thành ảnh (image) |
| Deskew | Chỉnh nghiêng trang scan bị lệch |
| Denoising | Loại nhiễu ảnh (vết mực, bụi scan) |
| Binarization | Chuyển ảnh về đen-trắng để OCR chính xác hơn |
| Layout analysis | Xác định vùng text, bảng, hình ảnh trên trang |
| Reading order | Xác định thứ tự đọc đúng (quan trọng với layout 2 cột) |
| Table extraction | Trích bảng thành dữ liệu có cấu trúc, không phải text phẳng |
| Header/footer removal | Loại nội dung lặp lại mỗi trang (không mang thông tin) |
| Page-level metadata | Gắn số trang, tên file gốc cho từng đoạn OCR |
| Confidence score | Điểm tin cậy OCR trả về cho từng từ/dòng — dùng để lọc kết quả kém |

## 3. Đo chất lượng OCR

- **Character Error Rate (CER)**: tỉ lệ ký tự sai so với ground truth.
- **Word Error Rate (WER)**: tỉ lệ từ sai so với ground truth.
- Cách kiểm tra nhanh không cần ground truth đầy đủ: lấy mẫu 3-5 trang, đọc thủ công so với output OCR, ước lượng tỉ lệ lỗi.

## 4. So sánh công cụ (mức thực dụng)

| Công cụ | Điểm mạnh | Điểm yếu | Phù hợp khi |
|---|---|---|---|
| Tesseract | Miễn phí, phổ biến, offline | Chất lượng trung bình với layout phức tạp/tiếng Việt dấu | Baseline nhanh |
| PaddleOCR | Hỗ trợ tiếng Việt khá tốt, có layout detection | Setup phụ thuộc nặng (PaddlePaddle) | Cần chất lượng tốt hơn Tesseract |
| EasyOCR | Dễ cài, hỗ trợ nhiều ngôn ngữ | Chậm hơn, độ chính xác không bằng PaddleOCR ở văn bản dày đặc | Prototype nhanh |
| DocTR | Pipeline OCR + layout hiện đại (deep learning) | Cần GPU để nhanh | Muốn chất lượng cao, có GPU |
| OCRmyPDF | Wrapper quanh Tesseract, xuất PDF có lớp text ẩn (searchable PDF) | Vẫn giới hạn bởi chất lượng Tesseract | Chỉ cần làm PDF searchable, không cần structured output |
| Unstructured | Xử lý đa định dạng, có sẵn cleaning/chunking tích hợp | Trừu tượng hóa nhiều, khó tùy biến sâu | Muốn pipeline nhanh, ít code |
| Marker | Chuyển PDF → Markdown giữ cấu trúc (heading, bảng) | Nặng, cần GPU cho tốc độ tốt | Tài liệu học thuật/kỹ thuật nhiều cấu trúc |
| Docling (IBM) | Layout + table + reading order tốt, output có cấu trúc | Còn khá mới, model lớn | Muốn pipeline document AI hiện đại |

## 5. Đề xuất pipeline OCR tiếng Việt cho khóa luận CNTT (P2)

```
PDF khóa luận (scan hoặc mixed)
  → OCRmyPDF (tạo searchable PDF nhanh, fallback an toàn)
  → Nếu cần structure tốt hơn: Docling hoặc Marker (heading, bảng)
  → Text Cleaning (loại header/footer lặp lại từng trang, số trang)
  → Section Detection theo heading (Chương/Mục)
  → Chunk theo section trước, rồi recursive splitting trong mỗi section
  → Indexing (nối vào pipeline chính)
```

**Ghi chú:** module này chỉ nên bắt đầu sau khi toàn bộ Definition of Done P0 đã đạt. Đừng để OCR làm trễ tiến độ 2 ngày.

Đi tiếp: mở file `09-capstone-cv.md`.
