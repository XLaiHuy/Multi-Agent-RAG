# Module 9 — Capstone: Ghép hệ thống hoàn chỉnh & Nội dung CV

## 1. Sơ đồ ghép nối toàn bộ các module

```
[01] Ingestion (load, clean, metadata)
        ↓
[02] Chunking (recursive, size=500/overlap=75)
        ↓
[01] Embedding → Vector DB (Chroma)
        ↓
[03] Hybrid Retrieval (Vector + BM25 → RRF → Cross-encoder Rerank)
        ↓
[04] LangGraph: analyze_query → retrieve → grade_documents
     → (rewrite_query nếu cần, tối đa 1 lần) → generate → verify
        ↓
[05] Agent node: quyết định skip/vector_only/hybrid tại bước analyze_query
        ↓
[01] Citation Verification
        ↓
[07] FastAPI (/chat, /documents, /evaluate, /health) + Docker
        ↓
[06] Evaluation dataset (20-30 câu) + bảng benchmark 3 experiment
```

Đây chính là kiến trúc bạn sẽ vẽ lên README và trình bày khi phỏng vấn.

## 2. Thứ tự implement khuyến nghị khi dùng AI hỗ trợ code (vibe coding có kiểm soát)

Đừng yêu cầu AI sinh toàn bộ project 1 lần. Theo đúng thứ tự, mỗi bước xong mới sang bước sau:

1. "Chốt kiến trúc P0 và tạo repository skeleton theo cấu trúc thư mục ở file 00."
2. "Triển khai ingestion (loader + cleaning + chunking) từ Module 01-02, kèm test."
3. "Triển khai embedding + Chroma vector store, test bài tập 1."
4. "Triển khai hybrid retrieval + RRF + reranker từ Module 03, test bài tập 3-4."
5. "Tạo evaluation dataset 20-30 câu và đo baseline (Module 06) trước khi làm LangGraph."
6. "Chuyển pipeline sang LangGraph Graph 1 rồi Graph 2 (Module 04), test bài tập 5-6."
7. "Thêm agent node quyết định retrieval strategy (Module 05), test bài tập 7."
8. "Thêm FastAPI, Docker, README (Module 07), test bài tập 9."
9. "Chạy lại evaluation đầy đủ, hoàn thiện bảng benchmark 3 experiment."
10. "Review toàn bộ repository như một senior AI engineer: tìm lỗi logic, thiếu error handling, thiếu test, tên biến không rõ nghĩa."

Ở mỗi bước, luôn tự đọc lại code AI sinh ra trước khi chạy — mục tiêu là hiểu đủ sâu để giải thích, không phải copy-paste mù.

## 3. Checklist trước khi đưa lên GitHub

- [ ] `.env` thật không nằm trong repo (kiểm tra `.gitignore`)
- [ ] `requirements.txt` cài được sạch trên môi trường mới (test trong venv trống)
- [ ] README có: mô tả, kiến trúc (ảnh/ASCII), cách chạy local, cách chạy Docker, bảng benchmark, giới hạn hiện tại (limitations)
- [ ] Có ít nhất 1-2 ảnh chụp màn hình hoặc gif demo `/chat` hoạt động
- [ ] Không có code chết (dead code)/comment TODO không rõ nghĩa nằm rải rác
- [ ] Test cơ bản (`pytest`) chạy pass

## 4. Nội dung CV

**Tên project:** Advanced Hybrid-Retrieval RAG Agent với LangGraph Self-Correction

**Mô tả 2-3 dòng:**
Hệ thống hỏi-đáp trên tài liệu riêng, kết hợp hybrid retrieval (dense + BM25 qua Reciprocal Rank Fusion), cross-encoder reranking và LangGraph để tự động đánh giá/viết lại truy vấn khi tài liệu tìm được không liên quan. Có bộ evaluation định lượng (Recall@k, Faithfulness, latency) so sánh 3 cấu hình retrieval, đóng gói Docker và expose qua FastAPI.

**3 bullet point CV:**
- Xây dựng pipeline Advanced RAG (hybrid dense+BM25 retrieval, RRF fusion, cross-encoder reranking) giúp tăng Recall@5 từ [X]% (baseline vector-only) lên [Y]% (đo bằng bộ eval 20-30 câu tự xây dựng).
- Thiết kế LangGraph state machine với self-correction (grade documents → rewrite query có giới hạn vòng lặp) và 1 agent node ra quyết định động chiến lược retrieval theo loại câu hỏi.
- Triển khai FastAPI + Docker, xây dựng evaluation pipeline định lượng (retrieval/generation/system metrics) để so sánh khách quan giữa các cấu hình thay vì đánh giá cảm tính.

*(Điền số [X]/[Y] thật từ bảng benchmark Module 06 — không để trống hoặc bịa số.)*

**Tech stack:** Python, FastAPI, LangChain, LangGraph, Chroma, BM25 (rank_bm25), Cross-encoder Reranker, Docker, pytest.

**README introduction (đoạn mở đầu):**
> Advanced RAG Agent là hệ thống hỏi-đáp dựa trên tài liệu, được xây dựng để giải quyết 2 vấn đề cốt lõi của RAG cơ bản: retrieval không chính xác khi chỉ dùng dense search, và hallucination khi tài liệu không đủ thông tin. Hệ thống kết hợp hybrid retrieval, reranking và một LangGraph workflow có khả năng tự đánh giá độ liên quan của tài liệu tìm được trước khi sinh câu trả lời, kèm citation nguồn và cơ chế từ chối khi không đủ bằng chứng.

**Giải thích 60 giây khi phỏng vấn:**
> Mình xây một hệ thống RAG để hỏi-đáp trên tài liệu riêng. Điểm khác với RAG cơ bản là mình dùng hybrid retrieval — kết hợp vector search và BM25 qua Reciprocal Rank Fusion — vì vector search một mình hay bỏ sót các câu hỏi có từ khóa/mã số chính xác. Sau đó mình rerank top-20 kết quả bằng cross-encoder để tăng độ chính xác trước khi đưa vào LLM. Phần mình tâm đắc nhất là dùng LangGraph để xây một luồng có khả năng tự sửa: sau khi retrieve, hệ thống chấm điểm xem tài liệu có thực sự liên quan không, nếu không thì viết lại câu hỏi và thử lại — tối đa 1 lần để tránh vòng lặp vô hạn. Mình cũng xây một bộ 20-30 câu hỏi evaluation để đo Recall@k, độ trung thực câu trả lời và latency, so sánh 3 cấu hình khác nhau bằng số liệu thật thay vì chỉ demo cảm tính. Hệ thống được đóng gói Docker và expose qua FastAPI.

## 5. Phân loại trung thực: Implemented / Experimented / Planned

Trước khi viết vào CV, tự điền bảng này — **không phóng đại phần chưa làm**:

| Hạng mục | Trạng thái |
|---|---|
| Hybrid retrieval (Vector+BM25+RRF) | Implemented / Experimented / Planned |
| Cross-encoder reranking | Implemented / Experimented / Planned |
| LangGraph Graph 1 (basic) | Implemented / Experimented / Planned |
| LangGraph Graph 2 (corrective, có rewrite loop) | Implemented / Experimented / Planned |
| LangGraph Graph 3 (agentic đầy đủ) | Implemented / Experimented / Planned |
| Agent node quyết định động | Implemented / Experimented / Planned |
| Multi-agent supervisor | Implemented / Experimented / Planned |
| MCP server hóa retriever | Implemented / Experimented / Planned |
| Evaluation pipeline + benchmark table | Implemented / Experimented / Planned |
| FastAPI + Docker | Implemented / Experimented / Planned |
| OCR scanned PDF | Implemented / Experimented / Planned |
| GraphRAG (knowledge graph) | Implemented / Experimented / Planned |

`Experimented` = đã thử code/chạy thử nhưng chưa ổn định hoặc chưa đo metric đầy đủ — vẫn có thể nói trong phỏng vấn nhưng phải trung thực về mức độ hoàn thiện.

## 6. Definition of Done cuối cùng (nhắc lại từ file 00)

Trước khi coi project "xong", đối chiếu lại checklist ở `00-ROADMAP.md` mục 6 — tất cả các mục P0 phải được tick.

---

Đến đây là hết bộ tài liệu. Thứ tự khuyến nghị: đọc `00` → `07` trong ngày 1-2 theo bảng lộ trình, làm bài tập + validate ngay sau mỗi module, chỉ mở `08` nếu dư thời gian, và dùng `09` để tổng hợp README + CV ở cuối.
