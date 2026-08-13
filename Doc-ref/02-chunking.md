# Module 2 — Chunking nâng cao

## 1. So sánh các chiến lược chunking

| Chiến lược | Cách hoạt động | Ưu điểm | Nhược điểm | Dùng khi nào |
|---|---|---|---|---|
| Fixed-size | Cắt theo N ký tự cố định | Đơn giản, nhanh | Cắt ngang câu/ý | Chỉ để baseline so sánh |
| Recursive character splitting | Cắt theo thứ tự separator (\n\n → \n → . → " ") tới khi đạt size | Giữ ranh giới tự nhiên tốt hơn fixed-size | Vẫn có thể cắt ngang ý nếu văn bản không có separator rõ | **Mặc định P0 cho MVP** |
| Sentence-based | Cắt theo câu, gộp câu tới khi đủ size | Không cắt ngang câu | Câu quá ngắn/dài gây lệch size | Văn bản văn phong ngắn gọn |
| Semantic chunking | Cắt tại điểm embedding similarity giữa 2 câu liên tiếp giảm mạnh | Giữ chunk mạch lạc theo ý nghĩa | Tốn thêm 1 lượt embedding, chậm hơn | Khi có thời gian tối ưu chất lượng (P1) |
| Parent-child chunking | Chunk nhỏ để search, nhưng trả về chunk cha (đoạn lớn hơn) làm context | Search chính xác + context đủ rộng | Cần lưu 2 tầng dữ liệu | Tài liệu dài, nhiều đoạn liên quan nhau |
| Small-to-big retrieval | Tương tự parent-child, index câu nhỏ, retrieve rồi mở rộng ra đoạn quanh nó | Cân bằng precision/recall | Phức tạp hơn khi implement | P1, sau khi baseline ổn |
| Sliding window + overlap | Chunk cố định nhưng chồng lấn (overlap) | Giảm mất ngữ cảnh ở ranh giới chunk | Dư thừa dữ liệu, tăng index size | Kết hợp với recursive splitting |
| Chunk theo heading/section | Dựa vào cấu trúc Markdown/heading | Giữ nguyên cấu trúc logic tài liệu | Cần tài liệu có heading rõ | Tài liệu kỹ thuật, báo cáo có mục lục |

## 2. Tác động của tham số

- **Chunk size nhỏ** (200-300 token): retrieval chính xác hơn (ít nhiễu) nhưng context có thể thiếu, dễ mất liên kết ý.
- **Chunk size lớn** (800-1200 token): giữ đủ ngữ cảnh nhưng dễ lẫn nhiều ý trong 1 chunk → similarity search kém chính xác hơn, tốn token khi generate.
- **Overlap** (thường 10-20% chunk size): giảm nguy cơ một ý bị cắt đúng ranh giới 2 chunk, nhưng tăng dung lượng index và có thể gây trùng lặp trong context.
- **Embedding context length**: nếu chunk vượt quá giới hạn của embedding model, phần dư bị cắt/mất — luôn kiểm tra giới hạn model bạn dùng.
- **Retrieval noise**: chunk quá lớn hoặc quá tổng quát làm similarity score giữa các chunk khác nhau gần bằng nhau → retriever khó phân biệt chunk nào thực sự liên quan.

## 3. Cấu hình mặc định đề xuất cho project 2 ngày

```
chunk_size = 500 token (~700-800 ký tự tiếng Việt)
chunk_overlap = 75 token (15%)
splitter = RecursiveCharacterTextSplitter (separators ưu tiên: "\n\n", "\n", ". ", " ")
```

Đây là điểm khởi đầu an toàn — không tối ưu ngay, chỉ tối ưu sau khi có baseline metric (xem Module 06).

## 4. Code skeleton

```python
# app/ingestion/chunking.py
from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter

@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    chunk_index: int

def chunk_document(doc_id: str, text: str, source: str,
                    chunk_size: int = 500, chunk_overlap: int = 75) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    pieces = splitter.split_text(text)
    if not pieces:
        raise ValueError(f"Chunking không sinh ra chunk nào cho doc {doc_id}")
    return [
        Chunk(chunk_id=f"{doc_id}_{i}", text=p, source=source, chunk_index=i)
        for i, p in enumerate(pieces)
    ]
```

## 5. Bài tập 2 — Chunking experiment (P0)

**Mục tiêu:** hiểu chunk size ảnh hưởng thế nào tới retrieval bằng số liệu, không phải cảm tính.
**Yêu cầu:** thử `chunk_size=300` và `chunk_size=800` (cùng overlap 15%) trên cùng bộ tài liệu, đo Recall@5 trên ≥10 câu hỏi mẫu (có thể dùng tạm bộ câu hỏi rút gọn trước khi làm bộ 20-30 câu đầy đủ ở Module 06).
**Output mong đợi:** bảng 2 dòng so sánh `chunk_size | Recall@5 | avg_chunks_retrieved`.
**Tiêu chí hoàn thành:** có số liệu thật, không phải "cảm thấy tốt hơn".
**Lỗi thường gặp:** so sánh nhưng quên giữ cố định các biến khác (embedding model, top-k) → không so sánh công bằng.

## 6. Validation Module 2

- [ ] Chunking chạy không lỗi trên toàn bộ tài liệu ingest
- [ ] Có ít nhất 2 cấu hình chunk size được thử và ghi lại số liệu
- [ ] Mỗi chunk giữ được `source` + `chunk_index` để trace ngược
- [ ] Không có chunk rỗng hoặc chunk chỉ toàn khoảng trắng lọt vào index

## 7. Quiz kiểm tra hiểu biết

1. Vì sao chunk quá lớn làm giảm chất lượng similarity search?
   *Đáp: Chunk lớn chứa nhiều ý khác nhau, vector trung bình hóa mất đi tính đặc trưng, khó phân biệt với chunk khác.*
2. Overlap giải quyết vấn đề gì?
   *Đáp: Giảm nguy cơ một ý quan trọng bị cắt đúng ranh giới giữa 2 chunk, khiến cả 2 chunk đều không đủ ngữ cảnh.*
3. Parent-child chunking khác gì recursive splitting thông thường?
   *Đáp: Parent-child search trên chunk nhỏ (chính xác) nhưng trả về đoạn cha lớn hơn làm context (đủ thông tin), tách vai trò "tìm" và "đọc".*
4. Vì sao nên chốt một cấu hình chunking mặc định trước rồi mới thử nghiệm thay vì tối ưu ngay từ đầu?
   *Đáp: Cần baseline để so sánh; tối ưu không có baseline là tối ưu mù, không biết có thực sự tốt hơn không.*
5. Semantic chunking có nên làm trong MVP 2 ngày không? Vì sao?
   *Đáp: Không bắt buộc (P1) — tốn thêm 1 lượt embedding cho việc chia chunk, ưu tiên có baseline chạy trước.*

Đi tiếp: mở file `03-advanced-retrieval.md`.
