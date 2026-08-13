# Advanced RAG Agent — Khóa học 2 ngày (00: Roadmap & Kiến trúc)

> Cách dùng bộ tài liệu này: đọc theo thứ tự file `01` → `09`. Mỗi module có 4 phần cố định:
> **Lý thuyết → Code skeleton → Bài tập → Validation (điều kiện để coi là "xong" module đó)**.
> Không sang module tiếp theo nếu validation module hiện tại chưa pass — đây là nguyên tắc quan trọng nhất để tránh vỡ tiến độ trong 2 ngày.

## 1. Bản đồ kiến thức (Basic RAG → Production Advanced RAG)

```
Cấp 0 — LLM thuần
  Prompt → LLM → Answer  (không có dữ liệu riêng, dễ hallucinate)

Cấp 1 — RAG cơ bản (Naive RAG)
  Docs → Chunk → Embed → Vector DB → Top-k Retrieve → Stuff vào Prompt → LLM

Cấp 2 — RAG có chất lượng (Quality RAG)
  + Cleaning/metadata tử tế, + Citation, + Refusal khi thiếu info, + Eval cơ bản

Cấp 3 — Advanced Retrieval RAG
  + Hybrid (BM25 + Vector), + RRF, + Reranker, + Query rewrite/HyDE

Cấp 4 — Orchestrated RAG (LangGraph)
  + State machine, + Conditional routing, + Grading, + Self-correction (Corrective/Self-RAG)

Cấp 5 — Agentic RAG
  + Agent tự quyết định retrieval/tool, + Subagent chuyên biệt, + Multi-step reasoning

Cấp 6 — Production RAG
  + API, Docker, Observability, Evaluation pipeline, Cost/latency budget

Cấp 7 — Mở rộng (P2, sau MVP)
  + OCR/Document AI, + GraphRAG (knowledge graph), + MCP tool ecosystem, + Multi-agent phức tạp
```

Project của bạn trong 2 ngày sẽ đi từ **Cấp 1 → Cấp 6**, chạm nhẹ Cấp 7 nếu còn thời gian.

## 2. Kiến trúc MVP (P0) — chốt ngay từ đầu, không đổi giữa chừng

```
                     ┌─────────────────────────┐
                     │        FastAPI           │  P0
                     │  /documents /chat /eval   │
                     └────────────┬──────────────┘
                                  │
                     ┌────────────▼──────────────┐
                     │   LangGraph Application     │  P0
                     │  analyze → retrieve → grade │
                     │  → generate → verify        │
                     └──┬──────────────────────┬───┘
                        │                      │
             ┌──────────▼─────────┐   ┌────────▼─────────┐
             │  Hybrid Retriever   │   │   LLM Provider    │  P0
             │  Vector + BM25      │   │  (Claude/OpenAI)  │
             │  → RRF → Reranker   │   └────────────────────┘
             └──────────┬──────────┘
                        │
             ┌──────────▼──────────┐
             │  Vector DB (Chroma) │  P0
             │  + Chunk store       │
             └──────────┬───────────┘
                        │
             ┌──────────▼──────────┐
             │ Ingestion Pipeline   │  P0
             │ Load → Clean → Chunk │
             │ → Embed              │
             └───────────────────────┘

P1 (sau khi P0 chạy ổn): Subagent chuyên biệt, Query rewrite loop có giới hạn,
                          Streamlit UI, Caching, thêm metric nâng cao.
P2 (chỉ nếu còn dư thời gian): OCR PDF scan, GraphRAG, MCP server hóa retriever,
                          multi-agent supervisor đầy đủ.
```

**Nguyên tắc bất biến:** một pipeline nhỏ chạy ổn định + đo được > một pipeline to nhưng không chạy hết hoặc không đo được gì.

## 3. Tech stack đã chốt

| Thành phần | Lựa chọn | Vì sao | Thay thế | Bắt buộc MVP? |
|---|---|---|---|---|
| Ngôn ngữ | Python 3.11 | Ecosystem RAG tốt nhất | — | Có |
| API | FastAPI | Async, type hint, docs tự sinh | Flask | Có |
| Orchestration | LangGraph | State machine rõ ràng, dễ giải thích khi phỏng vấn | Chuỗi function thủ công | Có |
| Framework RAG phụ trợ | LangChain (loader/splitter/embeddings interface) | Tiết kiệm thời gian viết boilerplate | Tự viết | Có (nhưng dùng có chọn lọc) |
| Vector DB | Chroma (local, embedded) | Zero-config, chạy ngay trong Docker, đủ cho demo | Qdrant (nếu cần scale/cloud) | Có |
| Sparse search | rank_bm25 | Nhẹ, không cần server riêng | Elasticsearch | Có |
| Reranker | Cross-encoder (`bge-reranker-base` hoặc Cohere Rerank API) | Tăng Precision@k rõ rệt, cost thấp | Bỏ qua nếu hết thời gian | P1 nhưng nên có |
| Embedding | `text-embedding-3-small` (OpenAI) hoặc `bge-small-en` local | Rẻ/nhanh, đủ chất lượng cho MVP | Sentence-Transformers | Có |
| LLM | Claude Haiku/Sonnet hoặc GPT-4o-mini | Giá rẻ, đủ cho demo, có tool calling | — | Có |
| Container | Docker | Yêu cầu bắt buộc trong Definition of Done | — | Có |
| Test | pytest | Giải thích được với nhà tuyển dụng là có test | — | P1 |
| MCP | Model Context Protocol | Chuẩn hóa expose tool cho agent bên ngoài | Bỏ qua | **P2 — không cần cho MVP** |
| OCR | Docling/Marker | Nếu còn thời gian ngày 2 | Tesseract | **P2** |

## 4. Cấu trúc repository

```
advanced-rag-agent/
├── app/
│   ├── api/            # FastAPI routes
│   ├── agents/          # Agent/subagent logic
│   ├── graph/            # LangGraph StateGraph definitions
│   ├── ingestion/         # Loader, cleaning, chunking
│   ├── retrieval/          # Vector search, BM25, hybrid, rerank
│   ├── generation/          # Prompt templates, answer generation, citation
│   ├── evaluation/           # Eval dataset runner, metrics
│   ├── core/                  # Config, logging, LLM client wrapper
│   ├── schemas/                 # Pydantic models
│   └── main.py
├── data/{raw,processed,evaluation}/
├── scripts/            # ingest_cli.py, run_eval.py
├── tests/
├── notebooks/           # Thử nghiệm nhanh, không dùng cho production code
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

## 5. Lộ trình 2 ngày theo module (map với các file trong bộ tài liệu)

| Block thời gian | Module | Deliverable |
|---|---|---|
| Ngày 1 sáng | `01` RAG Fundamentals + `02` Chunking | Repo skeleton, ingestion + chunking chạy được, unit test cơ bản |
| Ngày 1 chiều | `03` Advanced Retrieval | Hybrid search + reranker chạy, so sánh trước/sau |
| Ngày 1 tối | `06` Evaluation (phần dataset) | 20-30 câu hỏi eval dataset + đo baseline (Recall@5, latency) |
| Ngày 2 sáng | `04` Graph & LangGraph | Graph 1 (basic) → Graph 2 (corrective) chạy được |
| Ngày 2 trưa | `05` Agent/Subagent (+ đọc lướt MCP) | Ít nhất 1 agent node ra quyết định động |
| Ngày 2 chiều | `07` API + Deployment | FastAPI + Docker chạy, benchmark 2 cấu hình |
| Ngày 2 tối | `09` Capstone | README, demo, CV content, review tổng |

Nếu thiếu thời gian, được phép bỏ theo thứ tự: `08` OCR → phần Subagent nâng cao trong `05` → Graph 3 (agentic) trong `04`, **không được bỏ**: hybrid retrieval, citation, evaluation, API, Docker.

## 6. Definition of Done (chốt cuối)

- [ ] Ingest ít nhất 1 bộ tài liệu thật (text/PDF extract được)
- [ ] Semantic retrieval hoạt động
- [ ] Hybrid retrieval hoặc reranker hoạt động
- [ ] Câu trả lời có citation nguồn
- [ ] Có xử lý "không đủ thông tin" khi tài liệu không có đáp án
- [ ] Có LangGraph workflow (tối thiểu Graph 2 — corrective)
- [ ] Có ≥20 câu hỏi evaluation + kết quả đo
- [ ] Có bảng so sánh ≥2 cấu hình (vd: basic vs hybrid+rerank)
- [ ] API chạy được qua FastAPI
- [ ] Dockerfile build & run được
- [ ] README đầy đủ + ảnh/gif demo
- [ ] Nội dung CV trung thực (Implemented/Experimented/Planned rõ ràng)

Đi tiếp: mở file `01-RAG-fundamentals.md`.
