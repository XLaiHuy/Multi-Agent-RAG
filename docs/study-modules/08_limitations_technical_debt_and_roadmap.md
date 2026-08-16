# Module 8: Limitations, Technical Debt & "Nếu có thêm 1 tháng"

---

## A. Mindset của một Senior Engineer khi nói về giới hạn
Một ứng viên non kinh nghiệm thường cố che giấu khuyết điểm hoặc tuyên bố hệ thống của mình hoàn hảo 100%.
Ngược lại, một **Senior/Staff Engineer** sẽ chủ động mổ xẻ các điểm nghẽn kỹ thuật (technical debt), giải thích rõ các thỏa hiệp (trade-offs) đã đưa ra trong từng giai đoạn và trình bày một lộ trình nâng cấp có thứ tự ưu tiên khoa học.

---

## B. Audit toàn diện 8 Giới hạn Kỹ thuật của Hệ thống

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                             TECHNICAL LIMITATIONS & ARCHITECTURAL DEBT                          │
├────────────────────┬─────────────────────────────────────────────────┬───────────┬───────────────┤
│ Phân loại          │ Mô tả hiện trạng trong Code                     │ Mức độ    │ Hướng xử lý   │
├────────────────────┼─────────────────────────────────────────────────┼───────────┼───────────────┤
│ **1. OCR Wiring**  │ Có OCRGatingAnalyzer nhưng pipeline ingestion   │ Medium    │ Wire Tesseract│
│                    │ mặc định chưa inject OCRProvider runtime.       │           │ hoặc Vision AI│
├────────────────────┼─────────────────────────────────────────────────┼───────────┼───────────────┤
│ **2. Planner**     │ Planner chạy sinh JSON nhưng trong benchmark    │ Low       │ Thêm dynamic  │
│    **Causality**   │ retrieval query được cố định để đo lường thuần. │ (Eval)    │ query rewrite │
├────────────────────┼─────────────────────────────────────────────────┼───────────┼───────────────┤
│ **3. Cross-Doc**   │ Tối ưu tuyệt đối cho 1 contract (Scoped); chưa  │ Medium    │ Xây dựng Meta-│
│    **Comparison**  │ hỗ trợ so sánh song song 10 contracts 1 lúc.    │           │ Router đa doc │
├────────────────────┼─────────────────────────────────────────────────┼───────────┼───────────────┤
│ **4. Latency**     │ P50 end-to-end là 32.6s do chạy tuần tự         │ Medium    │ Chạy song song│
│    **Overhead**    │ 3 bước LLM API calls + verification.            │           │ async agents  │
├────────────────────┼─────────────────────────────────────────────────┼───────────┼───────────────┤
│ **5. Same-Model**  │ Judge sử dụng gemma-4-26b-a4b-it giống với      │ Low       │ Dùng Claude   │
│    **Judge**       │ generator (dù đánh giá trên retrieved context). │           │ hoặc GPT-4o   │
├────────────────────┼─────────────────────────────────────────────────┼───────────┼───────────────┤
│ **6. Persistence** │ SQLite + In-Memory index phù hợp demo/eval;     │ Medium    │ Chuyển sang   │
│    **Scalability** │ chưa phân tán cụm trên Kubernetes.              │ (Prod)    │ PGVector/Redis│
└────────────────────┴─────────────────────────────────────────────────┴───────────┴───────────────┘
```

---

## C. Kịch bản trả lời xuất sắc: "Nếu có thêm 1 tháng, bạn sẽ cải tiến điều gì?"

> *"Nếu có thêm 1 tháng phát triển toàn thời gian, tôi sẽ triển khai lộ trình 4 giai đoạn theo thứ tự ưu tiên kỹ thuật rõ ràng:*
>
> 1. **P0 — Hoàn thiện OCR Pipeline Runtime & Async Ingestion (Tuần 1)**:
>    *Wire trực tiếp `OCRProvider` sử dụng Tesseract/Cloud Vision vào `MasterDocumentParser`, kích hoạt tự động qua `OCRGatingAnalyzer` khi mật độ ký tự $<50$ chars/page. Đưa toàn bộ tác vụ nạp tài liệu vào hàng đợi bất đồng bộ Celery + Redis.*
>
> 2. **P1 — Tối ưu hóa độ trễ Multi-Agent & Streaming (Tuần 2)**:
>    *Chuyển đổi quy trình gọi Agent tuần tự sang cơ chế song song hóa (Parallel Asynchronous Orchestration). Tích hợp token streaming trực tiếp từ Generator qua SSE giúp Time-to-First-Token (TTFT) giảm từ 32s xuống dưới 2s.*
>
> 3. **P2 — Mở rộng Multi-Document Comparative Reasoning (Tuần 3)**:
>    *Phát triển lớp Meta-Routing cho phép Planner tự động phân rã câu hỏi so sánh (ví dụ: So sánh điều khoản bảo mật giữa Hợp đồng A và Hợp đồng B) thành các sub-queries độc lập trên từng scoped index, sau đó tổng hợp ma trận so sánh.*
>
> 4. **P3 — Độc lập hóa Evaluation Judge & Benchmarking mở rộng (Tuần 4)**:
>    *Chạy benchmark đối sánh độc lập với mô hình thẩm định khác họ (Cross-Family Judge như Claude-3.5-Sonnet hoặc GPT-4o) để kiểm chứng thêm tính khách quan của chỉ số Groundedness."*

---

## D. Checkpoint: 10 Câu hỏi Phỏng vấn Phản biện (Adversarial Questions)

1. *"Tại sao hệ thống của bạn mất tới 32 giây để trả lời một câu hỏi? Khách hàng thực tế có chấp nhận độ trễ này không?"*
2. *"Nếu một tài liệu là PDF dạng bảng biểu phức tạp scan nghiêng, hệ thống của bạn sẽ gãy ở bước nào?"*
3. *"Tại sao trong báo cáo bạn thừa nhận Planner không làm thay đổi câu truy vấn retrieval trong Phase 6?"*
4. *"Nếu một nhân viên vô tình upload một hợp đồng chứa mã độc hoặc prompt injection ẩn, hệ thống phòng thủ ra sao?"*
5. *"Tại sao bạn không dùng LangGraph hay CrewAI mà lại tự viết orchestration bằng code thuần?"*
6. *"Sự khác biệt giữa việc đo lường trên CUAD Dataset và việc áp dụng thực tế trên hợp đồng doanh nghiệp Việt Nam là gì?"*
7. *"Làm thế nào bạn chứng minh chỉ số 97.93% Groundedness của LLM Judge không bị hiện tượng 'model tự khen chính mình' (Self-Evaluation Bias)?"*
8. *"Nếu chi phí gọi API tăng gấp 10 lần, bạn sẽ cắt giảm component nào đầu tiên trong pipeline mà ít ảnh hưởng đến độ chính xác nhất?"*
9. *"Tại sao hệ thống không sử dụng GraphRAG (Knowledge Graph) để liên kết các thực thể pháp lý?"*
10. *"Điểm yếu lớn nhất trong toàn bộ codebase hiện tại mà bạn muốn đập đi xây lại ngay lập tức là gì?"*
