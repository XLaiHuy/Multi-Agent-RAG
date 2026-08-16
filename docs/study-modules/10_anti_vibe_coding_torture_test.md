# Module 10: Anti-Vibe-Coding Torture Test (30 Trap Questions)

> **Mục đích**: 30 câu hỏi bẫy cực hiểm được thiết kế để phân biệt giữa **người chỉ đọc thuộc lòng README** và **người thực sự hiểu tường tận từng dòng code và kiến trúc hệ thống**.

---

## 🎯 30 Câu hỏi Bẫy Kỹ thuật & Đáp án Chi tiết

### Nhóm 1: Data Structures & Ingestion Pipeline (Câu 1 – 6)

1. **Câu hỏi**: Trong `CanonicalBlock`, trường `section_path` có kiểu dữ liệu là gì và nó được cập nhật như thế nào khi đi qua một block có `block_type == BlockType.HEADING`?
   - **Đáp án code thật**: `section_path` là `List[str]`. Khi parser gặp một heading mới, nó dựa vào cấp độ heading (dựa trên font size hoặc numbering regex như `8.1.1`) để pop các heading cùng cấp/cấp thấp hơn ra khỏi stack và append heading mới vào.
2. **Câu hỏi**: Tại sao `StructureAwareParentChildChunker` lại cần tham số `child_overlap_tokens = 40`? Overlap này được tính theo ký tự hay theo word tokens?
   - **Đáp án code thật**: Tính theo **word tokens** (sử dụng tokenizer). Overlap 40 tokens đảm bảo các mệnh đề ở ranh giới giữa 2 child chunks không bị đứt đoạn ngữ nghĩa.
3. **Câu hỏi**: Bounding box `bbox` trong `CanonicalBlock` gồm những phần tử nào và hệ tọa độ của nó quy ước ra sao?
   - **Đáp án code thật**: Tuple `(x0, y0, x1, y1)` biểu diễn tọa độ góc trên-trái và góc dưới-phải của block trên trang PDF (đơn vị points, chuẩn PyMuPDF).
4. **Câu hỏi**: Trong `parsers.py`, `OCRGatingAnalyzer` sử dụng ngưỡng (threshold) cụ thể nào để quyết định một trang cần OCR?
   - **Đáp án code thật**: Mật độ ký tự $< 50$ ký tự trên một trang chuẩn hoặc diện tích ảnh chiếm $> 60\%$ diện tích trang (`char_density_threshold` và `image_area_ratio_threshold`).
5. **Câu hỏi**: Nếu một file DOCX được tải lên, lớp nào trong `parsers.py` sẽ chịu trách nhiệm xử lý?
   - **Đáp án code thật**: `DocxDocumentParser` (kế thừa từ `BaseDocumentParser`, sử dụng thư viện `python-docx` để duyệt qua các paragraphs và tables).
6. **Câu hỏi**: Metadata của một Child Chunk chứa những trường bắt buộc nào để có thể liên kết ngược về Parent Chunk?
   - **Đáp án code thật**: `chunk_id`, `parent_id`, `doc_id`, `section_path`, `page_number`, `token_count`.

---

### Nhóm 2: Retrieval, RRF & Cross-Encoder (Câu 7 – 12)

7. **Câu hỏi**: Hàm `reciprocal_rank_fusion` trong `rrf_fusion.py` xử lý trường hợp một chunk chỉ xuất hiện trong danh sách Dense mà không có trong BM25 như thế nào?
   - **Đáp án code thật**: Nó gán điểm BM25 bằng 0 (tương đương thứ hạng vô cùng $\infty$), chỉ cộng phần điểm từ Dense: $\frac{1}{60 + r_{\text{dense}}}$.
8. **Câu hỏi**: Model reranker `ms-marco-TinyBERT-L-2-v2` có bao nhiêu tham số và tại sao nó chạy được sub-second trên CPU?
   - **Đáp án code thật**: Có khoảng **4.4 triệu tham số** (2 Transformer layers, hidden size 128). Kích thước cực nhỏ giúp inference 20 cặp text mất chưa tới 50ms trên CPU 4 luồng.
9. **Câu hỏi**: Kích thước chiều vector (dimension) của BGE-M3 trong `backend/app/core/config.py` là bao nhiêu?
   - **Đáp án code thật**: **1024 chiều** (`DENSE_DIMENSION = 1024`).
10. **Câu hỏi**: Trong `hybrid_search.py`, Candidate Budget mặc định là bao nhiêu trước khi đưa vào Cross-Encoder?
    - **Đáp án code thật**: **20 ứng viên** (`candidate_budget = 20`).
11. **Câu hỏi**: Tại sao điểm số đầu ra của CrossEncoder lại cần được chuẩn hóa qua hàm Sigmoid trước khi xếp hạng?
    - **Đáp án code thật**: Đầu ra thô của CrossEncoder là logit $(-\infty, +\infty)$. Qua hàm Sigmoid $\frac{1}{1 + e^{-x}}$, điểm số được đưa về khoảng xác suất $[0.0, 1.0]$ để dễ đặt ngưỡng lọc (thresholding).
12. **Câu hỏi**: Parent Context Expansion diễn ra trước hay sau bước Cross-Encoder reranking?
    - **Đáp án code thật**: **Diễn ra sau**. Cross-Encoder chấm điểm trên các Child Chunks (~250 tok) để tận dụng độ phân giải cao; sau khi chọn Top-5 child tốt nhất mới tra cứu lấy Parent Chunks (~1200 tok) nạp vào LLM.

---

### Nhóm 3: Multi-Agent Logic & Verification (Câu 13 – 18)

13. **Câu hỏi**: Lớp `RetrievalPlanner` trả về output dưới dạng Pydantic Model nào?
    - **Đáp án code thật**: `PlannerOutput` gồm các trường: `intent_category`, `complexity_level`, `sub_queries`, `reasoning`.
14. **Câu hỏi**: Trong `verifier.py`, regex nào được sử dụng để trích xuất các trích dẫn `[Reference N: <chunk_id>]` từ text câu trả lời?
    - **Đáp án code thật**: `re.findall(r"\[(?:Reference|Ref)\s*(\d+)?(?::\s*([a-zA-Z0-9_\-\.]+))?\]", text)` hoặc trích xuất bracket ID trực tiếp.
15. **Câu hỏi**: Nếu Verifier phát hiện một câu trả lời có chứa trích dẫn `[Reference 9]` nhưng context chỉ có 5 references, Verifier sẽ làm gì?
    - **Đáp án code thật**: Đánh dấu trích dẫn đó là `INVALID_REFERENCE_INDEX`, trừ điểm compliance và yêu cầu `REGENERATE` hoặc chuyển thành `REFUSE`.
16. **Câu hỏi**: Biến cờ (flag) nào trong `Settings` điều khiển việc sử dụng mô hình LLM nào cho Verifier?
    - **Đáp án code thật**: `VERIFIER_MODEL` trong `Settings` (mặc định production là `gemini-flash-latest`, benchmark Phase 6 là `gemma-4-26b-a4b-it`).
17. **Câu hỏi**: Trong file nào bước `EvidenceCritic` quyết định xem có cần mở rộng truy vấn không?
    - **Đáp án code thật**: `backend/app/agents/critic.py` trong phương thức `evaluate_sufficiency()`.
18. **Câu hỏi**: Sự khác nhau giữa `status == "PASSED"` và `status == "REFUSE"` trong kết quả trả về của `AnswerVerifier` là gì?
    - **Đáp án code thật**: `PASSED`: câu trả lời được bảo vệ trọn vẹn bởi trích dẫn hợp lệ; `REFUSE`: câu trả lời chứa khẳng định không có căn cứ hoặc vi phạm trích dẫn, hệ thống trả về thông báo từ chối an toàn.

---

### Nhóm 4: Evaluation, Metrics & Protocol (Câu 19 – 24)

19. **Câu hỏi**: Tập test Held-Out trong Phase 6 gồm bao nhiêu câu hỏi và bao nhiêu hợp đồng?
    - **Đáp án code thật**: **200 câu hỏi** (100 answerable, 100 unanswerable) trên **25 hợp đồng hoàn toàn chưa từng thấy**.
20. **Câu hỏi**: Giá trị SHA-256 của file `predictions.jsonl` trong Phase 6.1 là gì?
    - **Đáp án code thật**: `5bcd34525c397daaed0ed2c2b7fd50a84e5efd259df9a94e4861e4addc0dbde3`.
21. **Câu hỏi**: Tại sao tỷ lệ Wrong-Document Citation lại đạt chính xác $0.00\%$ (0/140)?
    - **Đáp án code thật**: Vì hệ thống bắt buộc Document Scoping ở tầng truy hồi (Candidate Pool chỉ chứa chunk của document được chỉ định) và Verifier đối soát `chunk.doc_id == target_doc_id`.
22. **Câu hỏi**: Trong Phase 4.2, Canonical MRR đạt được là bao nhiêu trên 294 câu hỏi?
    - **Đáp án code thật**: **0.5214**.
23. **Câu hỏi**: Điểm số Grounded Material Claim Rate ($97.93\%$) được chấm dựa trên cơ sở dữ liệu nào (Gold Evidence hay Retrieved Context)?
    - **Đáp án code thật**: Dựa trên **Retrieved Context thực tế được cung cấp cho Generator** (đo lường tính trung thực - Faithfulness), hoàn toàn độc lập với Gold Evidence.
24. **Câu hỏi**: Tỷ lệ từ chối theo chuẩn khắt khe (Strict Sentinel Refusal Rate) trên 100 câu hỏi unanswerable là bao nhiêu phần trăm?
    - **Đáp án code thật**: **78.00%** (78/100 câu xuất chính xác tiền tố `INSUFFICIENT_EVIDENCE:`).

---

### Nhóm 5: Systems, Security & FastAPI (Câu 25 – 30)

25. **Câu hỏi**: Header nào được client gửi lên để truyền JWT Bearer Token vào FastAPI endpoint?
    - **Đáp án code thật**: `Authorization: Bearer <jwt_token>`.
26. **Câu hỏi**: Trong `backend/app/core/config.py`, biến `DATABASE_URL` mặc định trỏ về đâu?
    - **Đáp án code thật**: `sqlite+aiosqlite:///./safe_rag.db` (Async SQLite).
27. **Câu hỏi**: Làm thế nào middleware trong `rag.py` ngăn chặn việc người dùng Tenant A truy vấn tài liệu của Tenant B?
    - **Đáp án code thật**: Truy vấn DB lấy `Document` theo `doc_id`, so sánh `doc.tenant_id == current_user.tenant_id`. Nếu khác, raise `HTTPException(status_code=403, detail="Access denied")`.
28. **Câu hỏi**: File nào trong repo chứa toàn bộ 67 bài unit test đang pass?
    - **Đáp án code thật**: Nằm rải rác trong thư mục `tests/` (`tests/unit/`, `tests/agents/`, `tests/security/`).
29. **Câu hỏi**: Tốc độ tăng tốc của Evaluation Cache ($116.8\times$) được đo lường chính xác trên workload cụ thể nào?
    - **Đáp án code thật**: Trên micro-benchmark lặp lại gồm **25 câu hỏi, 3 hợp đồng, 90 chunks** (thời gian chạy giảm từ 179.7s cold xuống 1.54s warm).
30. **Câu hỏi**: Sự khác biệt giữa `requirements.txt` và `pyproject.toml` trong repository này là gì?
    - **Đáp án code thật**: `pyproject.toml` cấu hình metadata dự án, build system và cấu hình pytest/ruff; `requirements.txt` ghim chính xác phiên bản các thư viện phụ thuộc để cài đặt môi trường.
