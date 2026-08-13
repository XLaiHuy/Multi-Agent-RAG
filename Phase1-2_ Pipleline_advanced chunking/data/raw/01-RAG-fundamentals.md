# Module 1 — RAG Fundamentals & Pipeline hoàn chỉnh

## 1. RAG giải quyết vấn đề gì

LLM có 3 giới hạn cố hữu: (1) kiến thức đóng băng tại thời điểm train, (2) không biết dữ liệu riêng/nội bộ của bạn, (3) dễ "bịa" (hallucinate) khi bị hỏi thứ nó không chắc. RAG (Retrieval-Augmented Generation) giải quyết bằng cách: **trước khi trả lời, đi tìm đoạn văn bản liên quan trong kho dữ liệu riêng, rồi đưa đoạn đó vào prompt làm bằng chứng cho LLM**. LLM không "học" dữ liệu mới — nó chỉ đọc và tổng hợp cái được đưa vào context tại thời điểm hỏi.

Vì sao không prompt thẳng LLM với toàn bộ tài liệu? Vì tài liệu lớn hơn context window, tốn cost theo token, và LLM có xu hướng "lost in the middle" (bỏ sót thông tin ở giữa văn bản dài). Retrieval giải quyết bằng cách chỉ đưa vào phần thực sự liên quan.

## 2. Hai pipeline: Offline (Ingestion) và Online (Query)

```
OFFLINE — chạy 1 lần khi có tài liệu mới
Documents → Load → Clean → Metadata → Chunk → Embed → Lưu vào Vector DB

ONLINE — chạy mỗi lần user hỏi
Query → (Rewrite) → Retrieve (Vector+BM25) → Rerank → Context Construction
      → Prompt → LLM Generate → Citation → Log/Cache → Trả kết quả
```

Nhầm lẫn phổ biến nhất của người mới: nhét cả 2 pipeline vào chung một hàm. Hãy tách rõ — ingestion là batch job, query là request-response.

## 3. Từng thành phần

| Thành phần | Chức năng | Input | Output | Lựa chọn phổ biến | Lỗi thường gặp | Cách debug |
|---|---|---|---|---|---|---|
| Document Loader | Đọc file → text thô | .pdf/.txt/.docx | raw text + page no. | `PyPDFLoader`, `unstructured` | Mất format bảng, PDF scan ra chuỗi rỗng | In độ dài text theo từng file, phát hiện file rỗng |
| Cleaning | Loại bỏ header/footer/noise | raw text | clean text | Regex, `unstructured.clean` | Xóa nhầm nội dung thật | So sánh trước/sau trên 3 file mẫu |
| Metadata Extraction | Gắn nguồn, trang, tiêu đề | clean text | text + metadata dict | Rule-based, LLM extract | Thiếu `source` → mất khả năng citation | Assert mọi chunk có `source` |
| Chunking | Chia nhỏ để embed | clean text | list[chunk] | Xem Module 02 | Chunk quá to/nhỏ | Đếm token/chunk, histogram |
| Embedding | Text → vector | chunk text | vector (float[]) | `text-embedding-3-small`, `bge-small` | Model mismatch giữa ingest và query | Log dimension, so khớp model ID |
| Vector Database | Lưu & tìm vector gần nhất | vectors + metadata | top-k candidates | Chroma, Qdrant | Quên persist → mất data khi restart | Test load lại sau khi restart container |
| Retriever | Truy vấn semantic (+keyword) | query | ranked chunks | Xem Module 03 | Trả về chunk không liên quan | Log score, xem thủ công top-5 |
| Context Construction | Ghép chunk thành context | ranked chunks | context string | Truncate theo token budget | Context vượt limit → lỗi API | Đếm token trước khi gọi LLM |
| Prompt Generation | Ghép instruction + context + câu hỏi | context + query | final prompt | Template có placeholder rõ | Prompt injection từ tài liệu | Escape/nhắc nhở LLM "chỉ dùng context" |
| Answer Generation | LLM sinh câu trả lời | prompt | answer text | Claude/GPT | Hallucination ngoài context | So câu trả lời với chunk nguồn |
| Citation | Gắn nguồn cho câu trả lời | answer + chunks | answer + [source] | ID hóa chunk, yêu cầu LLM trích dẫn ID | LLM bịa citation | Validate citation ID có tồn tại trong context đã gửi |
| Cache | Tránh gọi lại LLM/embedding trùng | query/text | cached result | Redis/dict in-memory | Cache stale khi data đổi | Invalidate cache theo doc version |
| Logging | Ghi lại mọi bước để debug | mọi input/output trung gian | log record | `structlog`, JSON log | Log quá ít hoặc lộ secret | Log query, top-k ids, latency mỗi bước |
| Evaluation | Đo chất lượng hệ thống | eval dataset | metrics | Xem Module 06 | Chỉ test cảm tính | Có bộ câu hỏi cố định, chạy lại được |

## 4. Code skeleton — Ingestion tối thiểu (P0)

```python
# app/ingestion/loader.py
from dataclasses import dataclass
from pathlib import Path

@dataclass
class RawDocument:
    doc_id: str
    source: str
    text: str
    metadata: dict

def load_text_file(path: Path) -> RawDocument:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    text = path.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        raise ValueError(f"File rỗng sau khi đọc: {path}")
    return RawDocument(
        doc_id=path.stem,
        source=str(path),
        text=text,
        metadata={"filename": path.name},
    )
```

```python
# app/ingestion/cleaning.py
import re

def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)      # thu gọn dòng trống dư thừa
    text = re.sub(r"[ \t]{2,}", " ", text)        # thu gọn khoảng trắng
    text = text.strip()
    if len(text) < 20:
        raise ValueError("Text sau khi clean quá ngắn, nghi ngờ lỗi extract")
    return text
```

## 5. Bài tập 1 — Basic retrieval (P0)

**Mục tiêu:** chứng minh vòng ingest → embed → search chạy được end-to-end.
**Yêu cầu:** index 5–10 tài liệu text thật (không phải data giả), truy vấn top-k=3, in ra score + metadata (`source`) cho từng kết quả.
**Gợi ý:** dùng Chroma local trước, đừng setup Qdrant server ngay ngày 1.
**Output mong đợi:** console log dạng `[score=0.82] source=doc3.txt: "..."`
**Tiêu chí hoàn thành:** chạy lại được nhiều lần, không lỗi, kết quả top-1 thực sự liên quan tới câu hỏi thử.
**Lỗi thường gặp:** quên set embedding model giống nhau lúc ingest và lúc query → search trả về rác.

## 6. Validation Module 1

- [ ] Load được ≥5 tài liệu thật không lỗi
- [ ] Mỗi chunk có `source` trong metadata
- [ ] Truy vấn thử 3 câu hỏi, top-1 liên quan bằng mắt thường
- [ ] Log thể hiện rõ từng bước (load → clean → chunk → embed → store)

## 7. Quiz kiểm tra hiểu biết

1. Vì sao không nhét toàn bộ tài liệu vào prompt thay vì dùng retrieval?
   *Đáp: Context window giới hạn, cost tăng theo token, và LLM dễ bỏ sót thông tin giữa văn bản dài ("lost in the middle").*
2. Metadata `source` dùng để làm gì trong pipeline?
   *Đáp: Để sinh citation và để debug/trace chunk nào tạo ra câu trả lời nào.*
3. Điều gì xảy ra nếu embedding model lúc ingest khác lúc query?
   *Đáp: Vector không cùng không gian → similarity search vô nghĩa, kết quả rác.*
4. Vì sao cần tách rõ offline pipeline và online pipeline?
   *Đáp: Ingestion là batch, có thể chậm; query cần nhanh, real-time — gộp chung sẽ làm chậm trải nghiệm hỏi-đáp.*
5. Citation dựa trên gì để không bị LLM bịa nguồn?
   *Đáp: Gắn ID cho từng chunk gửi vào context, yêu cầu LLM chỉ trích dẫn ID có trong context, rồi validate ID đó tồn tại.*

Đi tiếp: mở file `02-chunking.md`.
