# Module 0: Master Tutor Repository Audit & System Map

> **Mục tiêu**: Cung cấp bức tranh toàn cảnh trung thực về hệ thống `Multi-Agent Safe-RAG`, thiết lập ranh giới rõ ràng giữa **Runtime Code thực tế**, **Evaluation Artifacts**, và **Docs/README claims** để chuẩn bị tâm thế vững vàng nhất cho các cuộc phỏng vấn kỹ thuật.

---

## 1. Bản đồ hệ thống từ Code thật (Code-Level System Map)

```
[User / Client]
       │ (REST / JWT Bearer)
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ FastAPI Application (backend/app/api/v1/endpoints/...)                       │
│ ├── auth.py (JWT Authentication & Tenant Scope extraction)                  │
│ ├── documents.py (Upload & Ingestion triggers)                              │
│ └── rag.py (Query entrypoint & SSE streaming)                               │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼ (INGESTION PIPELINE)                        ▼ (QUERY PIPELINE)
┌──────────────────────────────────────────────┐ ┌──────────────────────────────────────────────┐
│ backend/app/ingestion/                       │ │ backend/app/retrieval/                       │
│ ├── parsers.py                               │ │ ├── hybrid_search.py                         │
│ │   ├── MasterDocumentParser                 │ │ │   ├── Document-Scoped Filter (doc_id)      │
│ │   ├── NativePDFParser (PyMuPDF / pdfplumber│ │ │   ├── dense_search.py (BGE-M3 Cosine)      │
│ │   └── OCRGatingAnalyzer (Image/Text ratio) │ │ │   ├── sparse_search.py (BM25Okapi)         │
│ ├── chunker.py                               │ │ │   ├── rrf_fusion.py (RRF k=60)             │
│ │   └── StructureAwareParentChildChunker     │ │ │   └── reranker_provider.py (TinyBERT)      │
│ │       ├── Child ~250 tok (Dense + BM25)    │ │ │       └── Parent Context Expansion         │
│ │       └── Parent ~1200 tok (Context)       │ └──────────────────────┬───────────────────────┘
│ └── persistence/                             │                        │ Top-5 Candidates
│     └── ChromaDB & In-Memory Indices         │                        ▼
└──────────────────────────────────────────────┘ ┌──────────────────────────────────────────────┐
                                                 │ Multi-Agent Reasoning Stack (backend/app/...) │
                                                 │ ├── agents/planner.py (Complexity / Intent)  │
                                                 │ ├── agents/critic.py (Sufficiency Audit)     │
                                                 │ ├── providers/gemini_gateway.py (Synthesis)  │
                                                 │ └── agents/verifier.py (Citation & Support)  │
                                                 └──────────────────────┬───────────────────────┘
                                                                        ▼
                                                 Verified Answer [Ref N] or INSUFFICIENT_EVIDENCE
```

---

## 2. Bảng đối chiếu sự thật: Docs/README Claim vs. Runtime Thực tế

Khi phỏng vấn Senior AI Engineer, việc phát hiện và giải thích trung thực các điểm chênh lệch giữa README và Code là bằng chứng mạnh nhất cho thấy bạn **thực sự hiểu code** thay vì chỉ đọc tài liệu:

| Component / Chủ đề | Docs / README Claim | Runtime Thực tế trong Code | Cách trả lời chuẩn trong Phỏng vấn |
|---|---|---|---|
| **Số lượng Agent** | README đôi lúc gọi *"Planner, Critic, Generator, Verifier (4 agents)"*. | Runtime có **3 Agent classes riêng biệt** (`Planner`, `Critic`, `Verifier`) + **1 Generation Step** (gọi qua Gateway/LLM Client). | *"Kiến trúc của tôi gồm 3 tác tử phân tích và kiểm duyệt (Planner, Critic, Verifier) xoay quanh một bước sinh câu trả lời có ràng buộc ngữ cảnh (Generation Step)."* |
| **OCR Pipeline** | Docs mô tả hỗ trợ Scanned PDF qua OCR. | Code có đầy đủ `OCRGatingAnalyzer`, logic trích xuất ảnh và interface `OCRProvider`, nhưng pipeline mặc định chạy `NativePDFParser` (digital text extraction). | *"Hệ thống đã thiết kế sẵn OCR Gating Analyzer để phát hiện trang scan dựa trên tỷ lệ mật độ text/ảnh, hiện tại đang tối ưu cho Native PDF và đã sẵn sàng interface để cắm Tesseract hoặc Cloud Vision."* |
| **Planner Causality** | Docs gợi ý Planner định hướng retrieval. | Trong Phase 6 evaluation, câu hỏi hợp đồng được truyền thẳng vào hybrid search để giữ retrieval protocol đóng băng cố định ($N=200$). | *"Planner phân tích độ phức tạp và phân loại dạng câu hỏi (QA, so sánh, rủi ro). Trong benchmark chuẩn hóa, chúng tôi cố tình cố định query để đo lường năng lực retrieval thuần túy."* |
| **Database lưu trữ** | Docs đề cập PostgreSQL & Redis. | Runtime development/eval sử dụng **In-Memory document slices + SQLite + ChromaDB**. | *"Trong môi trường microservices/cloud, kiến trúc thiết kế để cắm PostgreSQL cho metadata và Redis cho cache; hiện tại bản local runtime chạy SQLite và In-Memory index để tối ưu độ trễ CPU."* |
| **Khẳng định "Zero Hallucination"** | Một số tài liệu marketing ban đầu ghi *"Zero Hallucination"*. | Phase 6.1 đã chuẩn hóa thành: **80.97% Macro Citation Precision**, **0/140 Wrong-Document Citations**, và **97.93% Grounded Material Claim Rate** dưới sự chấm điểm độc lập của LLM Judge. | *"Không có hệ thống LLM nào dám cam kết 100% tuyệt đối không ảo giác. Chúng tôi đo lường chính xác bằng 80.97% precision trích dẫn điều khoản và 0/140 trích dẫn lẫn lộn tài liệu khác."* |

---

## 3. Khung chương trình 8 Bài học & Điều kiện tiên quyết (Prerequisites)

1. **Lesson 1 — End-to-End Lifecycle**: Nắm vững sơ đồ luồng dữ liệu từ lúc upload tài liệu đến khi trả lời câu hỏi. *(Prereq: Khái niệm cơ bản về Client-Server & REST API)*.
2. **Lesson 2 — Ingestion & Chunking**: Nắm rõ cách bóc tách cấu trúc hợp đồng và cơ chế Parent-Child chunking. *(Prereq: Cấu trúc PDF DOM, Tokenization)*.
3. **Lesson 3 — Hybrid Search, RRF & Reranking**: Nắm vững toán học của BM25, Cosine Similarity, RRF ($k=60$) và Cross-Encoder. *(Prereq: Vector Space Model, Information Retrieval basics)*.
4. **Lesson 4 — Scoped Retrieval & ACL**: Nắm vững hiện tượng Cross-Contract Collision và cơ chế bảo mật đa khách hàng (Multi-Tenant). *(Prereq: SQL Indexing, RBAC, IDOR)*.
5. **Lesson 5 — Agentic Reasoning Stack**: Nắm vững logic hoạt động của Planner, Critic, Verifier và cơ chế từ chối trả lời an toàn. *(Prereq: Prompt Engineering, Structured JSON Outputs)*.
6. **Lesson 6 — Defending CV Metrics**: Nắm vững từng công thức, tử số, mẫu số của các chỉ số trên CV ($81.97\%$ Hit@10, $72.5\%$ Balanced Accuracy, $80.97\%$ Precision). *(Prereq: Confusion Matrix, Precision/Recall, MRR/nDCG)*.
7. **Lesson 7 — Backend Architecture & Systems**: Nắm vững thiết kế FastAPI, Clean Architecture, Dependency Injection và SSE streaming. *(Prereq: Python AsyncIO, FastAPI, ORM)*.
8. **Lesson 8 — Technical Debt & Limitations**: Nắm vững các giới hạn kỹ thuật và xây dựng câu trả lời thuyết phục cho câu hỏi *"Nếu có thêm 1 tháng, bạn sẽ làm gì?"*. *(Prereq: System Design, Production Trade-offs)*.
