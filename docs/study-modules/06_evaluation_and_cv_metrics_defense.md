# Module 6: Evaluation & CV Metrics Defense

---

## A. Intuition (Trực giác)
Khi bạn ghi lên CV: *"Đạt 81.97% HitRate@10 và 72.5% Balanced Accuracy"*, một Senior Interviewer sẽ không bao giờ chỉ gật đầu khen ngợi. Họ sẽ hỏi:
- *"HitRate tính trên bao nhiêu câu hỏi? Tập dữ liệu nào? Đã kiểm tra rò rỉ dữ liệu (data leakage) chưa?"*
- *"Balanced Accuracy có nghĩa là gì? Tại sao không dùng Accuracy thông thường?"*
- *"80.97% Citation Precision tính theo Macro hay Micro? Mẫu số là tất cả câu hỏi hay chỉ những câu trả lời?"*

Module này trang bị cho bạn **toàn bộ công thức, tử số, mẫu số và lý do khoa học** để bảo vệ từng con số trên CV một cách bất khả xâm phạm.

---

## B. Bảng tổng hợp các chỉ số trên CV (Master CV Registry)

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               MASTER CANONICAL EVALUATION MATRIX                                 │
├───────────────────────┬──────────────┬───────────────────────────────┬───────────────────────────┤
│ Phân nhóm Metric      │ Tên chỉ số   │ Giá trị đo lường              │ Tập dữ liệu & Phạm vi     │
├───────────────────────┼──────────────┼───────────────────────────────┼───────────────────────────┤
│ **Retrieval**         │ HitRate@10   │ **81.97%**                    │ N=294 câu hỏi CUAD        │
│ (Phase 4.2 Canonical) │ HitRate@5    │ **68.71%**                    │ N=294 câu hỏi CUAD        │
│                       │ MRR          │ **0.5214**                    │ N=294 câu hỏi CUAD        │
│                       │ Parent Hit   │ **94.90%**                    │ N=294 câu hỏi CUAD        │
│                       │ CPU Latency  │ **586 ms P50** / 820 ms P95   │ BGE-M3 + BM25 + TinyBERT  │
├───────────────────────┼──────────────┼───────────────────────────────┼───────────────────────────┤
│ **End-to-End Gen**    │ Strict Acc   │ **72.50%** (Sentinel-only)    │ N=200 (100 Ans, 100 Unans)│
│ (Phase 6.1.1 Frozen)  │ Inclus. Acc  │ **74.50%** (Prose-aware)      │ N=200 (100 Ans, 100 Unans)│
│                       │ Refusal Rate │ **78.00%** (82% inclusive)    │ 78 / 100 unanswerable     │
│                       │ Acceptance   │ **67.00%**                    │ 67 / 100 answerable       │
│                       │ Citation P   │ **80.97%** (Macro) / 73.53%   │ Exact clause chunk match  │
│                       │ Citation R   │ **63.00%** (Macro)            │ Gold reference span match │
│                       │ Child Cover  │ **62.00%** (58/100)           │ All answerable queries    │
│                       │ Child Hit    │ **85.07%** (58/67)            │ Accepted answerable only  │
│                       │ Wrong Doc    │ **0 / 140 observed (0.00%)**  │ Cross-document citations  │
├───────────────────────┼──────────────┼───────────────────────────────┼───────────────────────────┤
│ **LLM Judge**         │ Groundedness │ **97.93%** (142/145 claims)   │ Judged vs Retrieved cont. │
│ (gemma-4-26b-a4b-it)  │ Semantic Cor │ **92.54%** (Mean 1.85 / 2.0)  │ Judged vs Gold reference  │
└───────────────────────┴──────────────┴───────────────────────────────┴───────────────────────────┘
```

---

## C. Chi tiết từng Metric: Công thức, Ý nghĩa & Phản biện phỏng vấn

### 1. Strict Child HitRate@10 (81.97%)
- **Định nghĩa**: Tỷ lệ các câu hỏi mà trong Top-10 child chunks (~250 tokens) lấy về có chứa ít nhất một chunk trùng khớp trực tiếp với bằng chứng vàng (gold evidence).
- **Tử số / Mẫu số**: $\frac{241}{294} \approx 81.97\%$.
- **Tại sao gọi là "Strict"?**: Một child chunk chỉ được tính là hit nếu text của nó thực sự chứa annotation. Không được tính "ké" điểm nếu chỉ có chunk anh em (sibling chunk) trong cùng section trúng.
- **Interviewer hỏi**: *"Tại sao không đo trên Parent chunk ngay từ đầu?"*
  $\rightarrow$ **Trả lời**: *"Nếu đo trên Parent chunk (~1200 tokens), HitRate sẽ cao ảo (94.90%) vì cửa sổ văn bản quá rộng. Đo trên Strict Child (~250 tokens) đảm bảo độ phân giải retrieval thực sự sắc bén ở cấp độ từng điều khoản cụ thể."*

### 2. Mean Reciprocal Rank - MRR (0.5214)
- **Công thức**: $\text{MRR} = \frac{1}{N} \sum_{i=1}^N \frac{1}{\text{rank}_i}$, trong đó $\text{rank}_i$ là thứ hạng của chunk liên quan đầu tiên.
- **Ý nghĩa**: Đo lường xem bằng chứng vàng xuất hiện cao hay thấp trong danh sách xếp hạng. MRR = 0.5214 tương đương với việc chunk đúng trung bình nằm ở **vị trí Top-1 hoặc Top-2**.

### 3. Balanced Answerability Accuracy (72.50% Strict / 74.50% Inclusive)
- **Công thức**:
  $$\text{Balanced Accuracy} = \frac{\text{Answerable Acceptance Rate} + \text{Unanswerable Refusal Rate}}{2} = \frac{67.0\% + 78.0\%}{2} = 72.50\%$$
- **Tại sao cần Balanced Accuracy?**: Trong tập test $N=200$, có đúng 100 câu có đáp án và 100 câu không có đáp án. Nếu model "học vẹt" luôn luôn trả lời, nó sẽ đúng 50%. Balanced Accuracy phạt nặng các model bịa đặt câu trả lời trên câu hỏi không có đáp án (False Answer).

### 4. Citation Precision Macro (80.97%) vs. End-to-End Coverage (62.00%)
- **Citation Precision Macro (80.97%)**: Trong số các trích dẫn mà model đưa ra trong câu trả lời, trung bình **80.97% trích dẫn trỏ đúng vào chunk chứa bằng chứng vàng**.
- **Child Citation Coverage (62.00%)**: Tính trên **toàn bộ 100 câu hỏi có đáp án**, có **58 câu** model vừa trả lời được, vừa trích dẫn đúng chính xác điều khoản con ($58/100 = 62.00\%$ khi tính cả parent overlap liên quan).
- **Phân biệt mẫu số**: $85.07\%$ là tính trên mẫu số 67 câu được chấp nhận ($58/67$), còn $62.00\%$ là tính trên toàn bộ mẫu số 100 câu hỏi.

---

## D. Checkpoint: 8 Câu hỏi Phỏng vấn Độc hiểm về Metrics

1. *(Easy)*: Dataset CUAD là gì và tại sao bạn chọn nó làm benchmark cho bài toán hợp đồng?
2. *(Easy)*: Tập Held-Out 25 hợp đồng có ý nghĩa gì trong việc chứng minh tính tổng quát hóa của model?
3. *(Medium)*: Giải thích sự khác biệt giữa Citation Precision Macro (80.97%) và Citation Precision Micro (73.53%).
4. *(Medium)*: Tại sao trong Phase 6.1 bạn lại loại bỏ hoàn toàn cơ chế "Top-1 Fallback Citation"?
5. *(Hard)*: Phân tích sự khác biệt giữa Groundedness (đánh giá dựa trên context được cấp) và Semantic Correctness (đánh giá dựa trên gold evidence).
6. *(Hard)*: Nếu interviewer nói: *"Balanced Accuracy 72.5% có vẻ chưa ấn tượng bằng các benchmark tổng quát 90%"*, bạn sẽ bảo vệ kết quả này như thế nào?
7. *(Deep-Dive)*: Trình bày sự khác biệt trong protocol phân loại giữa Strict Sentinel Refusal (78.0%) và Inclusive Prose-Aware Refusal (82.0%).
8. *(Deep-Dive)*: Làm thế nào bạn chứng minh được hệ thống đạt $0/140$ Wrong-Document Citations mà không phải do overfitting?
