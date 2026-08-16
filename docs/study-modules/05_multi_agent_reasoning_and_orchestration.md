# Module 5: Multi-Agent Layer (Planner, Critic, Generator Step, Verifier)

---

## A. Intuition (Trực giác)
Trong quy trình làm việc của một công ty luật:
1. **Planner (Người lập kế hoạch)**: Đọc câu hỏi của khách hàng, xác định đây là câu hỏi tra cứu điều khoản đơn giản, câu hỏi so sánh hay câu hỏi đánh giá rủi ro pháp lý.
2. **Retrieval**: Trợ lý đi lục hồ sơ lấy các trang tài liệu liên quan.
3. **Critic (Người thẩm định bằng chứng)**: Xem các trang tài liệu được mang về và tự hỏi: *"Tài liệu này có đủ để trả lời câu hỏi không? Có bị thiếu trang chứa định nghĩa không?"*. Nếu thiếu, yêu cầu tìm thêm.
4. **Generator (Người soạn thảo)**: Viết câu trả lời dựa trên tài liệu được cung cấp, bắt buộc ghi chú rõ từng câu trích từ trang nào.
5. **Verifier (Luật sư trưởng thẩm duyệt)**: Cầm bản dự thảo và soi lại từng câu: *"Câu này có đúng trích từ trang 5 không? Có bịa ra con số không?"*. Nếu thấy trích dẫn hợp lệ thì ký duyệt (PASS), nếu không thì hủy câu trả lời và thông báo từ chối (`INSUFFICIENT_EVIDENCE`).

---

## B. Chi tiết Implementation trong Code

Hệ thống được tổ chức thành **3 Agent classes độc lập** + **1 Generation Step**:

```
[Query + Doc ID] ──► Planner (backend/app/agents/planner.py)
                           │
                           ▼
                    Hybrid Retrieval (Top-5 Parent Chunks)
                           │
                           ▼
                      Critic (backend/app/agents/critic.py)
                           │
                     [PROCEED / EXPAND]
                           │
                           ▼
                    Generation Step (backend/app/providers/gemini_gateway.py)
                           │
                           ▼
                      Verifier (backend/app/agents/verifier.py)
                           │
                     [PASS / REFUSE]
```

### 1. Planner Agent (`RetrievalPlanner`)
- **Nhiệm vụ**: Phân loại dạng câu hỏi (`DIRECT_QA`, `COMPARISON`, `RISK_AUDIT`), trích xuất thực thể, đánh giá độ phức tạp.
- **Audit thực tế Phase 6**: Planner được gọi 200/200 lần, sinh cấu trúc JSON phân tích. Trong benchmark đông cứng, câu hỏi được truyền nguyên văn để cô lập bài test retrieval (`PLANNER_PRESENT_NO_ISOLATED_CAUSAL_EFFECT`).

### 2. Critic Agent (`EvidenceCritic`)
- **Nhiệm vụ**: Đánh giá độ đầy đủ của ngữ cảnh so với câu hỏi.
- **Audit thực tế Phase 6**: Gọi 200/200 lần. Với Top-5 parent chunks (~6000 tokens context), Critic đưa ra 200 quyết định `PROCEED`.

### 3. Generation Step (`GeminiAPIGateway`)
- **Nhiệm vụ**: Nhận context và system prompt nghiêm ngặt.
- **Quy tắc sinh**:
  - Mọi khẳng định phải có trích dẫn định dạng `[Reference N: <chunk_id>]`.
  - Nếu ngữ cảnh không có thông tin $\rightarrow$ Bắt buộc xuất tiền tố: `INSUFFICIENT_EVIDENCE: <lý do>`.

### 4. Verifier Agent (`AnswerVerifier`)
- **Nhiệm vụ**: Kiểm toán tính xác thực của câu trả lời đã sinh.
- **Audit thực tế Phase 6**: Được kích hoạt trên 85 câu trả lời được chấp nhận. Xác minh 140/140 trích dẫn điều khoản hợp lệ và thuộc đúng tài liệu.

---

## C. So sánh: Traditional RAG vs. Bounded Multi-Agent RAG

| Tiêu chí | Traditional RAG (One-Shot) | Bounded Multi-Agent Safe-RAG |
|---|---|---|
| **Quy trình** | `Retrieve -> Prompt -> LLM` | `Plan -> Retrieve -> Critique -> Generate -> Verify` |
| **Xử lý câu hỏi không có đáp án** | Dễ bịa ra câu trả lời (Hallucination) | Chủ động từ chối bằng sentinel `INSUFFICIENT_EVIDENCE` (82.00% refusal) |
| **Độ tin cậy trích dẫn** | Thường không trích dẫn hoặc bịa số chunk | **98.51%** câu trả lời có trích dẫn hợp lệ, **0/140** trích dẫn sai tài liệu |
| **Chi phí API / Độ trễ** | 1 call / ~3 giây | **3.42 calls / 3,971.9 tokens / 32.6s P50** |
| **Trade-off cốt lõi** | Rẻ, nhanh nhưng rủi ro cao | Chi phí cao hơn nhưng **đảm bảo an toàn pháp lý tối đa** |

---

## D. Checkpoint: 10 Câu hỏi Phỏng vấn Multi-Agent

1. *(Easy)*: Tại sao hệ thống lại gọi là "Bounded" Multi-Agent (Tác tử có chặn giới hạn)?
2. *(Easy)*: Verifier Agent làm thế nào để phát hiện một trích dẫn `[Reference N]` là hợp lệ hay bịa đặt?
3. *(Medium)*: Nếu Critic Agent phát hiện bằng chứng bị thiếu, cơ chế lặp lại (Retry Loop) được giới hạn tối đa bao nhiêu lần để tránh vòng lặp vô tận?
4. *(Medium)*: Tại sao trong code hệ thống lại tách biệt thành 3 lớp Agent và 1 bước Generation thay vì gộp chung vào 1 prompt khổng lồ?
5. *(Medium)*: Chi phí đánh đổi về số lượng API calls (3.42 calls/query) và độ trễ (32.6s P50) mang lại giá trị định lượng cụ thể nào cho hệ thống?
6. *(Hard)*: Giải thích cơ chế hoạt động của tiền tố `INSUFFICIENT_EVIDENCE:` và tại sao việc chuẩn hóa sentinel này lại quan trọng cho việc đánh giá tự động?
7. *(Hard)*: Trong ablation study ở môi trường DEV, hệ thống Full Multi-Agent vượt trội hơn Base RAG ở những chỉ số cụ thể nào?
8. *(Hard)*: Nếu bạn tích hợp framework LangGraph vào hệ thống này, StateGraph và các Node/Edge sẽ được tổ chức như thế nào?
9. *(Deep-Dive)*: Tại sao trong báo cáo khoa học Phase 6.1, chúng ta ghi nhận phân loại `PLANNER_PRESENT_NO_ISOLATED_CAUSAL_EFFECT` thay vì khẳng định "Planner giúp cải thiện retrieval"?
10. *(Deep-Dive)*: Thiết kế một cơ chế Circuit Breaker để tự động giáng cấp (fallback) từ Multi-Agent xuống Base RAG khi hệ thống gặp tải cao hoặc bị nghẽn API quota.
