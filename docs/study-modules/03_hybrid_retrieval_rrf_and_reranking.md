# Module 3: Hybrid Retrieval, RRF & Cross-Encoder Reranking

---

## A. Intuition (Trực giác)
- **Dense Retrieval (BGE-M3)**: Giống như một chuyên gia hiểu ngữ nghĩa tổng thể. Họ biết rằng *"bồi thường thiệt hại"* và *"indemnification liability"* nói về cùng một khái niệm, dù không trùng bất kỳ chữ cái nào. Nhưng họ hay bị nhầm lẫn giữa các mã số hiệu (ví dụ: *"Điều 8.1(a)"* và *"Điều 8.1(b)"* nhìn rất giống nhau về vector).
- **Sparse Retrieval (BM25Okapi)**: Giống như một cỗ máy soi từ khóa hoàn hảo. Nó tìm chính xác từng số hiệu hợp đồng, tên riêng công ty, hoặc thuật ngữ hiếm như *"gross negligence"*. Nhưng nó mù tịt nếu người dùng dùng từ đồng nghĩa.
- **Reciprocal Rank Fusion (RRF $k=60$)**: Là vị thẩm phán công tâm. Thay vì cố cộng điểm số cosine (từ 0 đến 1) với điểm số BM25 (từ 0 đến 50) vốn có thang đo hoàn toàn khác nhau, RRF chỉ nhìn vào **thứ hạng (rank)** của từng văn bản trong 2 danh sách để chấm điểm tổng hợp.
- **Cross-Encoder Reranker (TinyBERT)**: Giống như một chuyên gia đọc kỹ từng từ một. Họ ghép cặp `(Câu hỏi, Đoạn văn)` lại và cho chạy qua các lớp Self-Attention để bắt được mối quan hệ logic tinh tế mà Dense Bi-Encoder bỏ sót.

---

## B. Toán học cốt lõi & Ví dụ tính số cụ thể

### 1. Công thức Reciprocal Rank Fusion (RRF)
$$\text{RRF\_Score}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
- $M$: Tập các phương pháp tìm kiếm (ở đây $M = \{\text{Dense}, \text{Sparse}\}$).
- $r_m(d)$: Thứ hạng của tài liệu $d$ trong danh sách $m$ ($1, 2, 3, \dots$).
- $k$: Hằng số làm mượt (Smoothing constant), chuẩn công nghiệp và thực nghiệm là $k = 60$.

### 2. Ví dụ tính toán số học chi tiết với 5 Chunks:
Giả sử có 5 đoạn văn $C_1, C_2, C_3, C_4, C_5$:

| Chunk ID | Thứ hạng Dense ($r_{\text{dense}}$) | Thứ hạng BM25 ($r_{\text{sparse}}$) | Điểm Dense RRF: $\frac{1}{60 + r_D}$ | Điểm BM25 RRF: $\frac{1}{60 + r_S}$ | **Tổng điểm RRF** | Thứ hạng sau Fusion |
|---|---|---|---|---|---|---|
| **$C_1$** | Rank 1 | Rank 4 | $\frac{1}{61} \approx 0.01639$ | $\frac{1}{64} \approx 0.01563$ | **$0.03202$** | **#1** |
| **$C_2$** | Rank 2 | Rank 2 | $\frac{1}{62} \approx 0.01613$ | $\frac{1}{62} \approx 0.01613$ | **$0.03226$** | **#1 (Top)** |
| **$C_3$** | Rank 10 | Rank 1 | $\frac{1}{70} \approx 0.01429$ | $\frac{1}{61} \approx 0.01639$ | **$0.03068$** | **#3** |
| **$C_4$** | Rank 3 | Không xuất hiện ($\infty$) | $\frac{1}{63} \approx 0.01587$ | $0.00000$ | **$0.01587$** | **#4** |
| **$C_5$** | Không xuất hiện | Rank 3 | $0.00000$ | $\frac{1}{63} \approx 0.01587$ | **$0.01587$** | **#4** |

> **Nhận xét quan trọng**: $C_2$ xuất hiện ở vị trí #2 ở cả hai danh sách sẽ có tổng điểm cao nhất ($0.03226$), vượt qua $C_1$ (chỉ đứng đầu 1 bên nhưng bên kia tụt xuống #4). Đây chính là sức mạnh cân bằng sự đồng thuận của RRF!

---

## C. Bi-Encoder vs. Cross-Encoder: Tại sao phải 2-Stage Retrieval?

```
Bi-Encoder (BGE-M3):
  Vector(Query)     ──┐
                      ├──► Cosine Similarity (Cực nhanh, tính dot product qua vector index)
  Vector(Document)  ──┘

Cross-Encoder (TinyBERT):
  [CLS] + Query + [SEP] + Document + [SEP] ──► Full Cross-Attention Layers ──► Score (0.0 đến 1.0)
  (Rất chính xác vì từng từ trong Query nhìn thấy từng từ trong Document, nhưng chậm hơn 50x)
```

**Trade-off trong hệ thống**:
- Retrieve nhiều (Candidate Budget = 20): Dùng Bi-Encoder + BM25 quét nhanh toàn bộ document.
- Rerank ít (Top-5): Dùng Cross-Encoder chỉ trên 20 ứng viên để giữ độ trễ CPU sub-second (**586 ms P50**).

---

## D. Checkpoint: 10 Câu hỏi Phỏng vấn Retrieval

1. *(Easy)*: Dense Retrieval và Sparse Retrieval khác nhau căn bản ở điểm nào?
2. *(Easy)*: Tại sao $k=60$ lại là giá trị mặc định phổ biến trong công thức RRF?
3. *(Medium)*: Tại sao chúng ta không chuẩn hóa Min-Max điểm số Cosine và điểm BM25 rồi cộng lại theo trọng số $\alpha \cdot \text{Dense} + (1-\alpha) \cdot \text{BM25}$?
4. *(Medium)*: Bi-Encoder khác Cross-Encoder ở kiến trúc Attention như thế nào?
5. *(Medium)*: Tại sao không dùng Cross-Encoder để tìm kiếm trực tiếp trên toàn bộ kho tài liệu ngay từ đầu?
6. *(Hard)*: Trong BM25, hai tham số $k_1$ và $b$ đại diện cho điều gì? Tại sao chiều dài văn bản (document length normalization) lại ảnh hưởng đến điểm BM25?
7. *(Hard)*: Hiện tượng "Term Saturation" trong BM25 là gì và tại sao nó ưu việt hơn TF-IDF cổ điển?
8. *(Hard)*: Trình bày quy trình Dedup và Parent Context Expansion sau khi Cross-Encoder hoàn tất xếp hạng top-5 child chunks.
9. *(Deep-Dive)*: BGE-M3 có hỗ trợ Multi-Vector (ColBERT) và Sparse Lexical weights; tại sao hệ thống lại chọn kết hợp BGE-M3 Dense với BM25 truyền thống thay vì chỉ dùng riêng BGE-M3?
10. *(Deep-Dive)*: Nếu một câu hỏi chứa từ khóa hoàn toàn không có trong từ điển BM25 (Out-of-Vocabulary / OOV), pipeline xử lý như thế nào để không làm hỏng điểm RRF?
