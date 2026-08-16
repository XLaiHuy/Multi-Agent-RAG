# Module 9: Full Mock Technical Interview Simulation

> **Hướng dẫn sử dụng**: Đây là kịch bản phỏng vấn thực tế với Senior AI/ML Engineer. Hãy đọc kỹ từng câu hỏi, tự trả lời bằng lời trước, sau đó so sánh với câu trả lời mẫu đạt điểm 10/10 và các tiêu chí chấm điểm kỹ thuật.

---

## 🎙️ Mock Interview Session

### Question 1 (Opening Walkthrough):
**Interviewer**: *"Chào bạn, hãy giới thiệu tổng quan về dự án Multi-Agent Safe-RAG của bạn trong vòng 1 phút."*

**Rubric chấm điểm (Score 0 - 10)**:
- *Điểm 3/10 (Yếu)*: Kể lể chung chung: *"Em làm RAG có LangChain, ChromaDB và gọi API Gemini để trả lời hợp đồng..."*
- *Điểm 7/10 (Khá)*: Nêu được cấu trúc RAG, tên các models (BGE-M3, BM25, Gemini) và một vài con số.
- *Điểm 10/10 (Xuất sắc)*: Nêu rõ **Bài toán thực tế $\rightarrow$ Giải pháp cốt lõi $\rightarrow$ Kiến trúc 2 luồng $\rightarrow$ Kết quả định lượng ấn tượng**.

> **Câu trả lời mẫu đạt 10/10**:
> *"Dự án của tôi là Multi-Agent Safe-RAG, một hệ thống RAG pháp lý chuyên sâu giải quyết 2 bài toán lớn nhất của phân tích hợp đồng: mất mát ngữ cảnh điều khoản và va chạm dữ liệu giữa các hợp đồng.*
> *Về giải pháp, tôi xây dựng kiến trúc 2 luồng độc lập:*
> *Ở luồng Ingestion, tài liệu được phân tích theo cấu trúc và chia thành các đoạn con ~250 tokens để lập chỉ mục và đoạn cha ~1200 tokens để giữ trọn vẹn điều khoản bao quanh.*
> *Ở luồng Query, tìm kiếm được đóng khung trong phạm vi hợp đồng chỉ định (Document-Scoped), kết hợp BGE-M3 dense embeddings và BM25Okapi qua Reciprocal Rank Fusion ($k=60$) và TinyBERT Cross-Encoder, giúp tăng HitRate@10 từ 28.67% lên 81.97% trên 294 câu hỏi CUAD.*
> *Tầng suy luận sử dụng bộ 3 tác tử Planner, Critic và Verifier, đạt 72.5% Balanced Accuracy và 80.97% Citation Precision trên 200 câu hỏi test độc lập với 0/140 trích dẫn lẫn lộn tài liệu."*

---

### Question 2 (Retrieval Deep-Dive):
**Interviewer**: *"Tại sao bạn lại chọn kết hợp BGE-M3 và BM25 qua RRF? Tại sao không chỉ dùng mỗi BGE-M3 vốn đã là mô hình embedding rất mạnh?"*

> **Câu trả lời mẫu đạt 10/10**:
> *"BGE-M3 rất xuất sắc trong việc nắm bắt ngữ nghĩa tương đồng (semantic matching), nhưng trong văn bản pháp lý có những thực thể đặc thù mà dense embeddings dễ bị bão hòa hoặc nhầm lẫn, ví dụ như mã số hợp đồng, ngày tháng hiệu lực, hoặc các thuật ngữ hiếm mang tính ràng buộc như 'indemnification' hay 'gross negligence'. BM25 bù đắp hoàn hảo điểm yếu này bằng cơ chế so khớp từ khóa chính xác.*
> *Chúng tôi dùng RRF ($k=60$) thay vì tổ hợp tuyến tính (Linear Weighted Sum) vì điểm số Cosine (0-1) và điểm BM25 (thang điểm mở không chặn trên) có phân phối và độ lệch chuẩn hoàn toàn khác biệt. RRF chuẩn hóa dựa trên thứ hạng (Rank-based), đảm bảo tính ổn định phi tham số mà không cần phải tinh chỉnh trọng số cho từng loại hợp đồng."*

---

### Question 3 (Parent-Child Strategy):
**Interviewer**: *"Bạn giải thích cơ chế Parent-Child Chunking hoạt động như thế nào và tại sao nó lại cải thiện hiệu năng?"*

> **Câu trả lời mẫu đạt 10/10**:
> *"Trong RAG có một sự đánh đổi kinh điển giữa Retrieval Resolution (độ phân giải khi tìm kiếm) và Context Completeness (độ đầy đủ ngữ cảnh khi sinh).*
> *Nếu chunk quá lớn (~1200 tokens), vector embedding bị pha loãng thông tin, làm giảm độ nhạy tìm kiếm. Nếu chunk quá nhỏ (~250 tokens), khi đưa vào prompt của LLM, mô hình sẽ bị mất các câu ngoại lệ hoặc điều kiện tiên quyết nằm ở các câu tiếp theo của điều khoản.*
> *Parent-Child giải quyết triệt để vấn đề này bằng cách: Lập chỉ mục các Child Chunks (~250 tokens) cho Dense và BM25 search. Sau khi Cross-Encoder chọn ra top-5 Child Chunks tốt nhất, hệ thống tự động tra cứu `parent_id` trong metadata để mở rộng về Parent Chunks (~1200 tokens) tương ứng trước khi nạp vào context của LLM. Kết quả là HitRate tăng từ 81.97% ở cấp độ con lên 94.90% ở cấp độ đoạn cha."*

---

### Question 4 (Handling Unanswerable Queries & Refusal):
**Interviewer**: *"Làm thế nào hệ thống của bạn biết khi nào nên từ chối trả lời một câu hỏi mà hợp đồng không hề đề cập?"*

> **Câu trả lời mẫu đạt 10/10**:
> *"Hệ thống sử dụng cơ chế bảo vệ 2 lớp:*
> *Lớp 1 là Prompt Constraints có cấu trúc ở bước sinh, yêu cầu LLM bắt buộc phải gắn trích dẫn `[Reference N: <chunk_id>]` cho từng mệnh đề; nếu ngữ cảnh không chứa thông tin, LLM bắt buộc phải xuất tiền tố sentinel `INSUFFICIENT_EVIDENCE:`.*
> *Lớp 2 là Answer Verifier Agent. Verifier quét lại câu trả lời: nếu phát hiện câu trả lời không có trích dẫn hợp lệ hoặc trích dẫn vào các chunk không tồn tại, Verifier sẽ phủ quyết và chuyển trạng thái thành từ chối.*
> *Nhờ cơ chế này, trên 100 câu hỏi unanswerable của tập test, hệ thống đạt tỷ lệ từ chối đúng 78.00% theo chuẩn sentinel khắt khe và 82.00% theo chuẩn ngôn ngữ tự nhiên."*

---

### Question 5 (System Design & Limitations):
**Interviewer**: *"Độ trễ P50 của bạn là 32.6 giây. Nếu sếp yêu cầu bạn giảm độ trễ xuống dưới 3 giây cho môi trường Production, bạn sẽ làm những gì?"*

> **Câu trả lời mẫu đạt 10/10**:
> *"32.6 giây P50 trong benchmark phản ánh toàn bộ thời gian chạy qua 3 cuộc gọi LLM API tuần tự (Planner $\rightarrow$ Generator $\rightarrow$ Verifier) cùng thời gian mạng. Để đưa xuống dưới 3 giây cho Production, tôi sẽ thực hiện 4 giải pháp kỹ thuật:*
> *1. **Parallel Fast-Path Routing**: Với các câu hỏi tra cứu đơn giản (Direct QA), bypass Planner và gửi thẳng vào Hybrid Search, giảm ngay 1 lượt gọi LLM.*
> *2. **Server-Sent Events (SSE) Streaming**: Stream trực tiếp các token từ Generator về UI, giúp người dùng nhìn thấy chữ đầu tiên (Time-to-First-Token) trong vòng 1-2 giây.*
> *3. **Asynchronous Verification**: Trả về câu trả lời đã stream cho người dùng kèm trạng thái 'Đang kiểm duyệt', trong khi Verifier chạy nền bất đồng bộ để gắn nhãn tick xanh xác thực sau đó 1 giây.*
> *4. **Local Speculative Small Model**: Sử dụng một mô hình local nhỏ (như Gemma-2-2B hoặc Qwen-2.5-7B) trên GPU cho bước Verifier thay vì gọi qua Cloud API."*
