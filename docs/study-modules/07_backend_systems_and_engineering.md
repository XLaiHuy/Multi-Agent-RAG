# Module 7: Backend Systems, API & Software Engineering

---

## A. Intuition (Trực giác)
Một mô hình AI xuất sắc trong Jupyter Notebook chỉ là một file nghiên cứu. Để đưa nó vào doanh nghiệp, bạn cần một **hệ thống phần mềm kỹ thuật chuẩn mực (Software Engineering)**:
- API không bị block khi xử lý tác vụ nặng (Asynchronous FastAPI).
- Kiến trúc tách bạch rõ ràng (Clean Architecture: Router $\rightarrow$ Service $\rightarrow$ Provider $\rightarrow$ Domain Model) để khi đổi model từ Gemini sang Claude hay Ollama, bạn chỉ cần đổi đúng 1 file Provider mà không làm gãy toàn bộ API.
- Cơ chế truyền phát phản hồi trực tiếp (Server-Sent Events - SSE) để người dùng không phải ngồi nhìn màn hình trắng xóa 30 giây.

---

## B. Clean Architecture trong Multi-Agent Safe-RAG

```
backend/app/
├── domain/          # Các thực thể nghiệp vụ cốt lõi (CanonicalDocument, Query, User)
├── ingestion/       # Xử lý dữ liệu đầu vào (Parsers, Token Chunker)
├── providers/       # Giao tiếp với 3rd party (Embedding, Reranker, LLM Gateway)
├── retrieval/       # Thuật toán tìm kiếm (Dense, BM25, RRF Fusion)
├── agents/          # Logic tác tử (Planner, Critic, Verifier)
├── application/     # Service Layer điều phối nghiệp vụ (DocumentService, RAGService)
└── api/v1/          # Controller / FastAPI Routes (auth.py, documents.py, rag.py)
```

### Tại sao dùng Dependency Injection trong FastAPI?
Trong `backend/app/api/deps.py`, các service và database session được inject thông qua `Depends()`:
```python
# Lợi ích: Dễ dàng mock components khi viết Unit Test mà không cần kết nối database thật
@router.post("/query")
async def query_rag_endpoint(
    request: RAGQueryRequest,
    current_user: User = Depends(get_current_user),
    rag_service: RAGService = Depends(get_rag_service)
):
    return await rag_service.execute_pipeline(request, current_user)
```

---

## C. Persistence Strategy: In-Memory vs. SQLite vs. ChromaDB

| Thành phần lưu trữ | Công nghệ sử dụng | Dữ liệu lưu trữ | Tại sao chọn giải pháp này? |
|---|---|---|---|
| **Relational Metadata** | SQLite (SQLAlchemy ORM) | User accounts, Tenant IDs, Document Metadata, ACLs | Nhẹ, không cần cài đặt server riêng trong môi trường demo/eval, hỗ trợ đầy đủ ACID. |
| **Dense Vector Index** | ChromaDB & In-Memory Slices | 1024-dim Vector embeddings của các Child Chunks | Hỗ trợ lưu trữ vector phân vùng theo `doc_id` với độ trễ tìm kiếm Cosine cực thấp. |
| **Lexical Index** | In-Memory BM25Okapi Index | Tokenized inverted index theo từng tài liệu | Truy vấn sub-millisecond trên CPU mà không cần dựng cụm Elasticsearch cồng kềnh. |

---

## D. Checkpoint: 8 Câu hỏi System Design & Software Engineering

1. *(Easy)*: Server-Sent Events (SSE) khác gì với WebSockets trong việc hiển thị câu trả lời RAG streaming?
2. *(Easy)*: Tại sao nên tách riêng `domain/models.py` độc lập với ORM Database Models?
3. *(Medium)*: Làm thế nào hệ thống xử lý các tác vụ upload tài liệu dung lượng lớn mà không làm nghẽn Event Loop của FastAPI?
4. *(Medium)*: Trình bày cơ chế mã hóa mật khẩu và xác thực JWT Bearer Token trong `backend/app/core/security.py`.
5. *(Hard)*: *(System Design)*: Giả sử hệ thống cần scale lên **100,000 hợp đồng** và **1,000 người dùng đồng thời**, bạn sẽ thay đổi kiến trúc lưu trữ và retrieval như thế nào?
6. *(Hard)*: *(System Design)*: Làm thế nào để thiết kế một hàng đợi bất đồng bộ (Async Ingestion Queue) với Celery / Redis để xử lý tài liệu khi người dùng upload hàng loạt?
7. *(Deep-Dive)*: Nếu dịch vụ Gemini API bị nghẽn (HTTP 429 / 503), cơ chế Circuit Breaker và Exponential Backoff Retry trong gateway cần được cài đặt ra sao?
8. *(Deep-Dive)*: Làm thế nào để đảm bảo tính cô lập bộ nhớ cache (Cache Invalidation) khi một tài liệu bị chỉnh sửa hoặc xóa bởi người dùng?
