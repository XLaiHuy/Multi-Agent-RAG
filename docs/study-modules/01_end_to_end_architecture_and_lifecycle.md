# Module 1: End-to-End Architecture & Request Lifecycle

---

## A. Intuition (Trực giác cho người mới bắt đầu)
Hãy tưởng tượng bạn là một Luật sư tập sự được giao một chồng hợp đồng 100 trang.
- **Luồng Ingestion (Nạp tài liệu)** giống như việc bạn đọc lướt qua toàn bộ hợp đồng, đánh dấu số mục lục (Mục 1, Mục 8.1, Mục 8.2), cắt nhỏ từng đoạn văn ra các thẻ ghi nhớ (child chunks) để dễ tìm, nhưng vẫn ghi chú xem thẻ đó thuộc trang nào và chương nào (parent section). Sau đó bạn xếp chúng vào 2 ngăn kéo: một ngăn kéo tìm theo từ khóa chính xác (BM25) và một ngăn kéo tìm theo ý nghĩa tương đồng (Dense Vector Index).
- **Luồng Query (Trả lời câu hỏi)** giống như khi khách hàng hỏi: *"Mức giới hạn trách nhiệm bồi thường là bao nhiêu?"*. Bạn không đọc lại từ trang 1 đến trang 100. Bạn chỉ tìm đúng trong ngăn kéo của hợp đồng đó, lấy ra 20 thẻ ghi nhớ tiềm năng nhất, nhờ một chuyên gia kiểm duyệt chọn ra 5 đoạn chính xác nhất, đọc toàn bộ điều khoản cha xung quanh đó để hiểu trọn vẹn ngữ cảnh, soạn thảo câu trả lời kèm trích dẫn số trang/điều khoản, và nhờ một luật sư cao cấp (Verifier) rà soát lại xem câu trả lời có bịa đặt không trước khi gửi cho khách hàng.

---

## B. Role in My System (Vai trò trong Multi-Agent Safe-RAG)
Trong phân tích pháp lý, rủi ro lớn nhất của RAG truyền thống là:
1. **Lẫn lộn điều khoản giữa các hợp đồng khác nhau** (Cross-contract collision).
2. **Cắt vụn ngữ cảnh** làm mất các câu loại trừ quan trọng (ví dụ: *"Ngoại trừ trường hợp vi phạm nghiêm trọng..."* nằm ở câu sau).
3. **Ảo giác hoặc trích dẫn sai điều khoản** (Hallucinated citations).

Kiến trúc 2 luồng độc lập của hệ thống giải quyết triệt để 3 vấn đề này thông qua: **Document Scoping**, **Parent-Child Chunking**, và **Evidence-Bounded Multi-Agent Verification**.

---

## C. Actual Runtime Flow (Chi tiết luồng thực thi từ Code)

```text
========================================================================================
1. INGESTION LIFECYCLE (Khi tải tài liệu lên)
========================================================================================
[Raw PDF / DOCX / TXT]
       │
       ▼  (backend/app/api/v1/endpoints/documents.py -> upload_document)
[Document Ingestion Service] (backend/app/application/document_service.py)
       │
       ▼  (backend/app/ingestion/parsers.py -> MasterDocumentParser.parse)
[CanonicalDocument Model] (backend/app/domain/models.py)
  ├── metadata: {doc_id, filename, tenant_id, page_count}
  └── pages: [CanonicalPage] -> blocks: [CanonicalBlock(text, bbox, type, section_path)]
       │
       ▼  (backend/app/ingestion/chunker.py -> StructureAwareParentChildChunker.chunk)
[Hierarchical Chunks]
  ├── Child Chunks (~250 tokens, overlap 40 tok) -> Gắn parent_id, doc_id, section_path
  └── Parent Chunks (~1200 tokens) -> Giữ nguyên vẹn toàn bộ ngữ cảnh section
       │
       ├──► [Dense Embedding Provider] (backend/app/providers/embedding_provider.py)
       │      └── BGE-M3 (1024-dim) -> Lưu vào In-Memory Slice / ChromaDB
       │
       └──► [BM25 Inverted Index] (backend/app/retrieval/sparse_search.py)
              └── Tokenize từ vựng pháp lý -> Lưu vào BM25Index scoped theo doc_id

========================================================================================
2. QUERY LIFECYCLE (Khi người dùng hỏi câu hỏi)
========================================================================================
[User Question + selected_document_id + JWT Token]
       │
       ▼  (backend/app/api/v1/endpoints/rag.py -> query_rag)
[Security & ACL Validation]
  └── Kiểm tra tenant_id và quyền sở hữu doc_id (Chặn tấn công IDOR)
       │
       ▼  (backend/app/agents/planner.py -> RetrievalPlanner.plan)
[Planner Agent Step]
  └── Phân tích độ phức tạp câu hỏi, trích xuất thực thể pháp lý, định hình intent
       │
       ▼  (backend/app/retrieval/hybrid_search.py -> ScopedHybridRetriever.retrieve)
[Document-Scoped Hybrid Search (Candidate Budget = 20)]
  ├── Dense Search: BGE-M3 Cosine Similarity trong phạm vi doc_id (Top-20 Child Chunks)
  ├── Sparse Search: BM25Okapi Keyword Matching trong phạm vi doc_id (Top-20 Child Chunks)
  └── Fusion: Reciprocal Rank Fusion (RRF k=60) kết hợp không tham số
       │
       ▼  (backend/app/providers/reranker_provider.py -> TinyBERT CrossEncoder)
[Neural Cross-Encoder Reranking]
  ├── Rerank Top-20 candidates -> Lọc ra Top-5 Child Chunks có điểm liên quan cao nhất
  └── Parent Expansion: Mở rộng 5 Child Chunks về Parent Chunks (~1200 tokens) tương ứng
       │
       ▼  (backend/app/agents/critic.py -> EvidenceCritic.evaluate)
[Critic Agent Step]
  └── Đánh giá tính đầy đủ của bằng chứng (Sufficiency Audit: PROCEED hoặc EXPAND)
       │
       ▼  (backend/app/providers/gemini_gateway.py -> generate_grounded_answer)
[Generation Step (Evidence-Bounded Synthesis)]
  └── Sinh câu trả lời với chỉ dẫn nghiêm ngặt: Bắt buộc trích dẫn [Reference N: <chunk_id>]
       hoặc xuất sentinel "INSUFFICIENT_EVIDENCE:" nếu ngữ cảnh không đủ
       │
       ▼  (backend/app/agents/verifier.py -> AnswerVerifier.verify)
[Verifier Agent Step]
  ├── Rà soát từng câu khẳng định xem có được chứng minh bởi ngữ cảnh không
  ├── Xác minh ID trích dẫn [Reference N] có hợp lệ và thuộc đúng doc_id không
  └── Ra quyết định: PASS (Chấp nhận) / REGENERATE (Yêu cầu sinh lại) / REFUSE (Từ chối)
       │
       ▼
[JSON / SSE Stream Response to Client]
```

---

## D. Concrete Example (Ví dụ thực tế từng bước)

- **User Query**: *"What is the liability cap and does it exclude gross negligence?"*
- **Target Document**: `cuad_contract_056.pdf` (Master Services Agreement).

1. **Ingestion Step**:
   - `NativePDFParser` bóc tách văn bản trang 14, phát hiện heading `"8. LIMITATION OF LIABILITY"`.
   - `StructureAwareParentChildChunker` tạo ra Parent Chunk `p_8` (chứa toàn bộ mục 8.1, 8.2, 8.3 dài 850 tokens).
   - Tạo ra 2 Child Chunks:
     - `c_8_1`: Chứa câu *"Total liability under this Agreement shall not exceed $1,000,000..."* (~220 tokens).
     - `c_8_2`: Chứa câu *"The limitations in Section 8.1 shall not apply to damages resulting from gross negligence or willful misconduct..."* (~190 tokens).
2. **Retrieval Step**:
   - BM25 bắt chính xác các từ khóa hiếm: `liability cap`, `gross negligence`.
   - Dense BGE-M3 bắt ngữ nghĩa tương đồng: `limitation of liability`, `aggregate maximum liability`.
   - RRF kết hợp 2 danh sách, đưa `c_8_1` và `c_8_2` lên vị trí #1 và #2.
   - Cross-Encoder chấm điểm tương quan cao (0.94 và 0.91).
   - Parent expansion lấy toàn bộ `p_8` đưa vào context của Generator.
3. **Generation & Verification**:
   - Generator sinh câu trả lời:
     > *"The liability cap under this agreement is limited to $1,000,000 in the aggregate [Reference 1: cuad_contract_056_p14_c1]. However, this limitation explicitly excludes damages arising from gross negligence or willful misconduct [Reference 2: cuad_contract_056_p14_c2]."*
   - Verifier kiểm tra: Cả hai mệnh đề đều có trích dẫn hợp lệ, ID trích dẫn thuộc đúng hợp đồng `cuad_contract_056` $\rightarrow$ **PASS**.

---

## E. Why This Design? (Tại sao thiết kế như vậy?)

- **Tại sao không dùng Traditional One-Shot RAG (Retrieve $\rightarrow$ Prompt $\rightarrow$ LLM)?**
  Traditional RAG không có cơ chế tự phản biện. Nếu retrieval trả về đoạn văn không liên quan, Generator vẫn sẽ cố gắng trả lời và dẫn đến ảo giác (hallucination) hoặc bịa ra một mức bồi thường sai. Bounded Agentic RAG bổ sung bước Critic và Verifier để chủ động phát hiện sự thiếu hụt bằng chứng và kích hoạt cơ chế từ chối an toàn (`INSUFFICIENT_EVIDENCE`).

---

## F. Failure Modes (Hệ thống có thể lỗi ở đâu?)

1. **Chất lượng OCR**: Nếu tài liệu scan quá mờ, Ingestion sẽ tạo ra text rác, dẫn đến embedding sai.
2. **Truy vấn đa tài liệu phức tạp**: Nếu người dùng hỏi so sánh giữa 5 hợp đồng cùng lúc mà không chỉ định phạm vi, cơ chế document-scoping hiện tại cần được mở rộng qua routing đa tài liệu.

---

## G. Interview Answer Pitches (Bài nói mẫu khi Phỏng vấn)

### Bản 30 giây (Elevator Pitch)
> *"Dự án của tôi là Multi-Agent Safe-RAG, một hệ thống chuyên sâu cho phân tích hợp đồng pháp lý. Hệ thống giải quyết 2 bài toán lớn nhất của RAG truyền thống là mất mát ngữ cảnh và ảo giác trích dẫn. Tôi kết hợp Ingestion phân cấp Parent-Child, tìm kiếm lai Hybrid BGE-M3 + BM25 với RRF và Cross-Encoder trong phạm vi hợp đồng được chỉ định, cùng một pipeline 3 tác tử Planner, Critic và Verifier để đảm bảo 100% câu trả lời có căn cứ trích dẫn rõ ràng hoặc từ chối an toàn."*

### Bản 60 giây (Standard Technical Pitch)
> *"Trong RAG pháp lý, việc tìm kiếm trên toàn bộ kho hợp đồng thường gây ra va chạm điều khoản giữa các thỏa thuận khác nhau, khiến HitRate@10 giảm từ 81.97% xuống 28.67%. Để khắc phục, kiến trúc của tôi phân tách thành 2 luồng:*
> *Ở luồng Ingestion, tài liệu được bóc tách theo cấu trúc và chia thành các đoạn con ~250 tokens để lập chỉ mục và mở rộng về đoạn cha ~1200 tokens để giữ trọn vẹn ngữ cảnh điều khoản. Ở luồng Query, câu hỏi được giới hạn trong phạm vi hợp đồng chỉ định, tìm kiếm song song qua BGE-M3 và BM25, kết hợp qua RRF và rerank bằng TinyBERT Cross-Encoder.*
> *Tầng suy luận gồm Planner phân loại câu hỏi, Critic đánh giá tính đầy đủ của bằng chứng, và Verifier rà soát tính xác thực của trích dẫn trước khi trả về người dùng, đạt 80.97% precision trích dẫn và 0/140 trích dẫn sai tài liệu trên tập kiểm thử 200 câu hỏi độc lập."*

---

## H. Checkpoint: 7 Câu hỏi Phỏng vấn thử thách

1. *(Easy)*: Tại sao hệ thống lại chia thành 2 luồng Ingestion Lifecycle và Query Lifecycle độc lập?
2. *(Easy)*: Dữ liệu thay đổi cấu trúc biểu diễn như thế nào từ khi là file PDF thô đến khi vào Vector Database?
3. *(Medium)*: Tại sao cần bước Parent Context Expansion sau khi Reranker đã chọn ra các Child Chunks?
4. *(Medium)*: Vai trò cụ thể của Critic Agent là gì? Nó khác gì với Verifier Agent?
5. *(Hard)*: Nếu một câu hỏi pháp lý yêu cầu thông tin không hề có trong hợp đồng, luồng thực thi sẽ từ chối ở bước nào và bằng cơ chế gì?
6. *(Hard)*: Phân biệt sự khác nhau giữa Document-Scoped Retrieval và Corpus-Wide Retrieval về mặt thuật toán và rủi ro va chạm dữ liệu?
7. *(Deep-Dive)*: Tại sao hệ thống sử dụng kết hợp BGE-M3 + BM25 + RRF thay vì chỉ dùng một mình Dense Embedding model hiện đại?
