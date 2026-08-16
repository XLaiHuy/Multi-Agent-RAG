# Module 4: Document-Scoped Retrieval, Multi-Tenant ACL & Collision Drop

---

## A. Intuition (Trực giác)
Khi bạn dùng Google Drive và mở hợp đồng của Công ty A để hỏi: *"Hạn thanh toán là bao nhiêu ngày?"*, bạn kỳ vọng hệ thống tìm câu trả lời **duy nhất trong hợp đồng của Công ty A**.
Nếu hệ thống tìm kiếm trên toàn bộ hàng nghìn hợp đồng trong kho của bạn (Corpus-wide), nó sẽ tìm thấy 50 điều khoản thanh toán khác nhau của Công ty B, C, D (vốn có câu chữ y hệt nhau vì đều dùng mẫu hợp đồng chuẩn). Khi đó, RAG sẽ lấy nhầm điều khoản của Công ty B trả lời cho Công ty A.
Đây gọi là **Cross-Contract Collision (Va chạm điều khoản chéo)**.

---

## B. Benchmark Chứng minh: Document-Scoped vs. Corpus-Wide

Hệ thống đã thực hiện thí nghiệm đối đầu trực tiếp trên cùng tập dữ liệu 25 hợp đồng CUAD ($N=294$ câu hỏi):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HIỆU NĂNG TÌM KIẾM THEO PHẠM VI                         │
├────────────────────────────────────────┬────────────────────────────────────┤
│ Document-Scoped Retrieval (Có Scope)   │ 81.97% Strict Child HitRate@10     │
├────────────────────────────────────────┼────────────────────────────────────┤
│ Corpus-Wide Retrieval (Không Scope)   │ 28.67% Strict Child HitRate@10     │
├────────────────────────────────────────┴────────────────────────────────────┤
│ 💥 SỤT GIẢM HIỆU NĂNG DO VA CHẠM: -53.30% (Chỉ còn 1/3 độ chính xác!)      │
└─────────────────────────────────────────────────────────────────────────────┘
```

> **Ý nghĩa khoa học**: Thí nghiệm chứng minh rằng trong bài toán Legal QA trên hợp đồng, việc đóng khung không gian tìm kiếm vào tài liệu mục tiêu (**Document Scoping**) là điều kiện tiên quyết để đạt độ chính xác thực tế, chứ không thể chỉ dựa vào việc tăng kích thước model.

---

## C. Filtering Trước Retrieval vs. Filter Sau Retrieval

```
❌ Cách làm sai (Retrieve-Global-Then-Filter):
Query ──► Tìm Top-20 trên toàn bộ 25 hợp đồng ──► Lọc lại các chunk có doc_id == "contract_A"
Hậu quả: 20 kết quả trả về bị chiếm hết bởi contract B, C, D -> Sau khi lọc chỉ còn 0 hoặc 1 chunk của contract A -> FAIL!

✅ Cách làm đúng trong Safe-RAG (Pre-Retrieval Scoped Boundary):
Query + selected_document_id ──► Chỉ tìm kiếm trong Vector Slice & BM25 Index của contract_A ──► Trả về Top-20 tinh hoa của contract_A -> SUCCESS!
```

---

## D. Multi-Tenant ACL & Bảo mật Anti-IDOR

File mã nguồn: `backend/app/api/v1/endpoints/rag.py`, `backend/app/retrieval/document_scoped_retrieval.py`.

1. **Authentication Layer**: Giải mã JWT Token $\rightarrow$ Trích xuất `user_id` và `tenant_id`.
2. **Anti-IDOR (Insecure Direct Object References) Check**:
   - Trước khi thực hiện bất kỳ phép toán retrieval nào, truy vấn cơ sở dữ liệu xác thực:
     $$\text{Document.tenant\_id} == \text{Request.tenant\_id} \quad \text{AND} \quad \text{User có quyền READ trên Document}$$
   - Nếu không khớp $\rightarrow$ Trả về mã lỗi `HTTP 403 Forbidden` hoặc `HTTP 404 Not Found` ngay lập tức, ngăn chặn hoàn toàn việc rò rỉ dữ liệu qua kênh tìm kiếm.
3. **Kết quả kiểm thử bảo mật**: **7/7 Security & ACL Test Suites Passed** với **0 trường hợp rò rỉ chéo tenant**.

---

## E. Checkpoint: 7 Câu hỏi Phỏng vấn Architecture & Security

1. *(Easy)*: Cross-contract collision là gì và tại sao văn bản pháp lý lại dễ bị hiện tượng này hơn văn bản tin tức thông thường?
2. *(Easy)*: Sự khác nhau giữa Authentication (Xác thực) và Authorization (Phân quyền) trong hệ thống Safe-RAG là gì?
3. *(Medium)*: Tại sao kiến trúc Pre-retrieval filtering lại ưu việt hơn Post-retrieval filtering khi số lượng tài liệu tăng lên hàng nghìn?
4. *(Medium)*: Tấn công IDOR trong hệ thống RAG xảy ra như thế nào và Safe-RAG phòng chống nó ở tầng nào?
5. *(Hard)*: Về mặt cấu trúc dữ liệu, BM25 Index được lưu trữ và scoped theo từng `doc_id` như thế nào để đảm bảo không bị ô nhiễm từ vựng giữa các hợp đồng?
6. *(Hard)*: Phân tích sự khác biệt giữa **Ranh giới bảo mật (Security Boundary)** và **Ranh giới độ liên quan (Relevance Boundary)** trong kiến trúc RAG.
7. *(Deep-Dive)*: Nếu khách hàng có nhu cầu hỏi so sánh giữa 2 hợp đồng cụ thể (ví dụ Contract A vs Contract B), bạn sẽ mở rộng cơ chế Scoped Retrieval hiện tại như thế nào mà vẫn đảm bảo ngăn chặn va chạm với các hợp đồng còn lại?
