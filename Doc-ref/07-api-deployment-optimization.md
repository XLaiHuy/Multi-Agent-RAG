# Module 7 — Optimization, API, Deployment

## 1. Optimization — chỉ làm SAU khi có baseline metric (Module 06)

Thứ tự tối ưu đề xuất (từ tác động lớn/rẻ → tác động nhỏ/đắt):

1. **Chunking** (chunk size/overlap) — rẻ để thử, tác động lớn tới Recall
2. **Retriever top-k** — thử k=10/20/30 trước reranker
3. **Reranker top-n** — số lượng chunk cuối đưa vào context
4. **Prompt** — instruction rõ ràng hơn, ví dụ few-shot nếu cần
5. **Query rewrite** — chỉ khi thấy nhiều case query mơ hồ trong eval
6. **Context length / token budget** — cân bằng đủ thông tin vs chi phí
7. **Caching** — cache embedding của chunk (không đổi) và cache câu hỏi lặp lại
8. **Batch embedding** — ingest nhiều chunk cùng lúc thay vì gọi API từng chunk
9. **Async processing** — retrieval và các I/O nên async trong FastAPI
10. **Model selection** (LLM/embedding model) — đổi model tốn công test lại toàn bộ eval, để cuối cùng
11. **Deduplication / Timeout / Retry / Rate limit** — thuộc về độ ổn định vận hành, làm trước khi deploy

**Quy tắc:** mỗi lần tối ưu 1 biến, đo lại bằng bộ eval, so sánh với baseline — không đổi nhiều biến cùng lúc vì sẽ không biết cái nào thực sự tạo ra khác biệt.

## 2. Kiến trúc API

```
Frontend/Streamlit (tùy chọn)
        ↓
     FastAPI
        ↓
  LangGraph Application
        ↓
 Retriever / Vector DB
        ↓
    LLM Provider
```

| Endpoint | Method | Chức năng |
|---|---|---|
| `/documents/upload` | POST | Nhận file, lưu raw |
| `/documents/ingest` | POST | Chạy pipeline ingestion cho file đã upload |
| `/documents` | GET | Danh sách tài liệu đã ingest |
| `/chat` | POST | Nhận câu hỏi, chạy LangGraph, trả answer + citation |
| `/evaluate` | POST | Chạy evaluation trên eval set, trả metrics |
| `/health` | GET | Health check cho container/orchestrator |

## 3. Schema mẫu (Pydantic)

```python
# app/schemas/chat.py
from pydantic import BaseModel

class ChatRequest(BaseModel):
    query: str
    top_k: int = 5

class Citation(BaseModel):
    chunk_id: str
    source: str
    snippet: str

class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    latency_ms: float
    is_refusal: bool = False
```

```python
# app/api/routes_chat.py
from fastapi import APIRouter, HTTPException

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống")
    try:
        result = await run_rag_graph(request.query, top_k=request.top_k)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {exc}") from exc
    return result
```

## 4. Dockerfile tối thiểu

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY data/ ./data/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
services:
  rag-api:
    build: .
    ports:
      - "8000:8000"
    env_file: .env
    volumes:
      - ./data:/app/data     # persist vector DB + eval data
```

```
# .env.example
LLM_PROVIDER=anthropic
LLM_API_KEY=your_key_here
EMBEDDING_MODEL=text-embedding-3-small
VECTOR_DB_PATH=./data/processed/chroma
LOG_LEVEL=INFO
```

## 5. Lựa chọn deploy chi phí thấp/miễn phí

- **Render / Railway / Fly.io** — deploy container trực tiếp từ Dockerfile, free tier đủ demo.
- **Hugging Face Spaces (Docker SDK)** — miễn phí, dễ chia sẻ link demo trong CV.
- Với vector DB local (Chroma embedded) — không cần server riêng, giảm chi phí và độ phức tạp deploy.

## 6. Deployment checklist

- [ ] `docker build` thành công không lỗi
- [ ] `docker run` chạy được, `/health` trả 200
- [ ] `.env.example` đầy đủ, không commit `.env` thật (secret) vào git
- [ ] Volume mount đảm bảo data không mất khi container restart
- [ ] README có hướng dẫn chạy local (không Docker) và chạy Docker
- [ ] Log không in ra API key/secret

## 7. Bài tập 9 — Deployment (P0)

**Mục tiêu:** hệ thống chạy được qua FastAPI + Docker end-to-end.
**Yêu cầu:** build image, chạy container, gọi `/chat` bằng `curl`/Postman, nhận answer có citation.
**Tiêu chí hoàn thành:** người khác (không phải bạn) có thể `docker run` theo README và hệ thống chạy đúng ngay lần đầu.

## 8. Validation Module 7

- [ ] Có bảng optimization: biến nào đã thử, kết quả trước/sau (liên kết với Module 06)
- [ ] API chạy đủ các endpoint P0, có xử lý lỗi (không trả 500 trần trụi không rõ nguyên nhân)
- [ ] Docker build + run thành công trên máy sạch (thử trên terminal mới/máy khác nếu có thể)
- [ ] README có phần "Cách chạy" rõ ràng từng bước

## 9. Quiz kiểm tra hiểu biết

1. Vì sao không nên tối ưu nhiều biến (chunk size + reranker + prompt) cùng lúc?
   *Đáp: Không biết biến nào thực sự tạo ra khác biệt về kết quả, mất khả năng so sánh có kiểm soát (A/B rõ ràng).*
2. Vì sao cần endpoint `/health` riêng?
   *Đáp: Để container orchestrator/load balancer kiểm tra hệ thống còn sống mà không cần chạy toàn bộ pipeline RAG tốn kém.*
3. Batch embedding khác gì gọi embedding API từng chunk một?
   *Đáp: Gửi nhiều chunk trong 1 request giảm số lần gọi API, giảm overhead network, thường rẻ và nhanh hơn đáng kể khi ingest nhiều tài liệu.*
4. Vì sao Chroma (embedded, local) phù hợp cho MVP hơn là setup 1 server vector DB riêng?
   *Đáp: Không cần quản lý thêm 1 service, giảm độ phức tạp deploy trong thời gian giới hạn, vẫn đủ khả năng cho demo/CV.*
5. Vì sao `.env` thật không nên commit vào git dù chỉ để "test nhanh"?
   *Đáp: Rò rỉ API key/secret là rủi ro bảo mật thật, và một khi đã commit vào lịch sử git thì rất khó xóa sạch hoàn toàn.*

Đi tiếp: mở file `08-ocr-extension-P2.md` (tùy chọn) hoặc thẳng `09-capstone-cv.md` nếu đã hết giờ ngày 2.
