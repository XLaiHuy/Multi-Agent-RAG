# Module 10 (Bổ sung) — Chạy local trên máy yếu + Gemini Free Tier

> Đọc file này song song với Module 01, 03, 07 — nó chỉ **điều chỉnh lựa chọn công nghệ**, không đổi kiến trúc/pipeline đã học.

## 1. Đánh giá phần cứng — ThinkPad X1 Nano

CPU tiết kiệm điện, không GPU rời, RAM thường 8-16GB. Điều này **không ảnh hưởng nhiều** vì kiến trúc MVP của khóa học vốn thiết kế để LLM chạy qua API (không train/infer LLM tại chỗ). Hai điểm cần đổi so với stack gốc ở file `00`:

| Thành phần | Stack gốc (file 00) | Đổi cho máy yếu | Lý do |
|---|---|---|---|
| Embedding | `bge-small` local hoặc `text-embedding-3-small` | **Gemini Embedding API** (`gemini-embedding-001` hoặc tương đương hiện hành) | Free tier rộng (~10M token/phút), không tốn CPU máy bạn |
| Reranker | `bge-reranker-base` (~280MB) | `cross-encoder/ms-marco-MiniLM-L-6-v2` (~90MB) | Nhẹ hơn 3 lần, đủ chất lượng cho demo, chạy CPU chấp nhận được (top-20 rerank trong 1-2s) |
| LLM (chat/grade/generate/verify) | Bất kỳ | Gemini Flash qua API (đã chọn) | Không đổi, chỉ cần quản lý rate limit (mục 2) |
| Vector DB | Chroma | Không đổi | Chroma nhẹ, chạy tốt trên CPU thường |

**Không cần đổi gì khác** — FastAPI, LangGraph, BM25, Docker đều nhẹ, chạy bình thường trên X1 Nano.

## 2. Gemini Free Tier — con số thật và ảnh hưởng tới pipeline

Theo dữ liệu cập nhật: model Flash trên free tier hiện giới hạn khoảng **10-15 requests/phút (RPM)** và **~1.000-1.500 requests/ngày (RPD)**; riêng **embedding API rộng hơn hẳn** (hàng triệu token/phút) — nên **embedding không phải nút thắt, LLM generation mới là nút thắt**. Lưu ý: số liệu rate limit có thể thay đổi theo thời gian, nên kiểm tra lại tại `https://ai.google.dev/gemini-api/docs/rate-limits` trước khi build.

**Vấn đề cụ thể với kiến trúc Module 04:** Graph 2 (Corrective RAG) có thể gọi LLM **3-5 lần cho 1 câu hỏi** (analyze_query → grade_documents → [rewrite_query] → generate → verify). Với RPM=10-15, bạn chỉ xử lý được ~2-3 câu hỏi/phút khi chạy tuần tự — đủ cho demo trực tiếp nhưng **sẽ gãy khi chạy evaluation 20-30 câu liên tiếp** nếu không có delay.

## 3. Chiến lược đối phó (áp dụng theo thứ tự ưu tiên)

1. **Cắt bớt số lượt gọi LLM/câu hỏi trước, đừng chỉ thêm delay.** Gợi ý:
   - Gộp `analyze_query` (agent quyết định retrieval strategy — Module 05) và `grade_documents` thành **1 lệnh gọi LLM duy nhất** (prompt yêu cầu trả JSON có cả 2 quyết định), thay vì 2 lệnh riêng.
   - `verify` (Module 04) chỉ chạy khi cần — có thể bỏ qua trong 80% trường hợp và chỉ bật khi eval, không bật khi demo trực tiếp.
   - Kết quả: 5 lượt gọi/câu hỏi → 2-3 lượt/câu hỏi.
2. **Rate-limited client với retry + exponential backoff** (bắt buộc, tránh crash khi 429).
3. **Cache** — cache theo `hash(prompt)`, tránh gọi lại LLM cho câu hỏi trùng khi test đi test lại (rất hay xảy ra khi debug).
4. **Chạy evaluation (20-30 câu) với delay cố định giữa các câu**, không chạy song song — chấp nhận eval chạy chậm (vài phút) đổi lấy không bị chặn giữa chừng.
5. Nếu vẫn thiếu quota lúc cao điểm: cân nhắc tạo thêm 1 API key phụ (project Google Cloud khác) chỉ dùng riêng cho evaluation, tách khỏi key dùng cho demo — nhưng đây là giải pháp tình thế, không phải thiết kế chuẩn.

## 4. Code skeleton — Gemini client có rate limit + backoff

```python
# app/core/llm_client.py
import time
import random
import hashlib
import functools

class RateLimitError(Exception):
    pass

_cache: dict[str, str] = {}

def _cache_key(prompt: str, model: str) -> str:
    return hashlib.sha256(f"{model}::{prompt}".encode()).hexdigest()

def call_gemini(
    prompt: str,
    model: str = "gemini-2.5-flash",
    max_retries: int = 5,
    base_delay_s: float = 4.0,      # ~15 RPM => an toàn khi giãn cách ~4-6s/lệnh gọi
    use_cache: bool = True,
) -> str:
    """Gọi Gemini với cache + exponential backoff cho 429.
    base_delay_s=4 tương ứng ~15 request/phút để không chạm limit khi chạy tuần tự."""
    key = _cache_key(prompt, model)
    if use_cache and key in _cache:
        return _cache[key]

    for attempt in range(max_retries):
        try:
            time.sleep(base_delay_s)          # giãn cách cố định giữa các lệnh gọi
            response_text = _raw_call_gemini(prompt, model)  # wrap SDK thật ở đây
            if use_cache:
                _cache[key] = response_text
            return response_text
        except RateLimitError:
            backoff = (2 ** attempt) + random.uniform(0, 1)
            print(f"[rate limit] thử lại sau {backoff:.1f}s (lần {attempt+1}/{max_retries})")
            time.sleep(backoff)
    raise RuntimeError("Vượt quá số lần retry do rate limit — kiểm tra lại RPM/RPD quota")
```

```python
# app/evaluation/runner_ratelimited.py
import time

def run_evaluation_slow(eval_set: list[dict], rag_pipeline, delay_between_questions_s: float = 8.0):
    """Chạy eval tuần tự có delay cố định — dùng thay cho vòng lặp chạy dồn dập ở Module 06."""
    results = []
    for i, item in enumerate(eval_set):
        result = rag_pipeline(item["question"])
        results.append(result)
        print(f"[{i+1}/{len(eval_set)}] xong, nghỉ {delay_between_questions_s}s trước câu tiếp theo")
        if i < len(eval_set) - 1:
            time.sleep(delay_between_questions_s)
    return results
```

## 5. Ước tính thời gian chạy thực tế trên setup này

- Demo trực tiếp (1 câu hỏi, 2-3 lượt LLM sau khi gộp bước): **~10-15 giây/câu** — chấp nhận được để demo phỏng vấn.
- Chạy evaluation 20-30 câu với delay 8s giữa các câu + 2-3 lượt LLM/câu (mỗi lượt cũng cách nhau `base_delay_s`): **ước tính 10-20 phút cho cả bộ eval** — chạy 1 lần, lưu kết quả, không cần chạy lại liên tục.
- Ingestion (embedding qua API): nhanh, vì embedding free tier rộng — không cần delay đáng kể, có thể batch.

## 6. Validation bổ sung

- [ ] Có wrapper gọi LLM với cache + backoff, không gọi thẳng SDK ở nhiều nơi rải rác trong code
- [ ] Đã gộp bớt số lượt gọi LLM/câu hỏi trong Graph 2 (đo lại: bao nhiêu lượt/câu trước và sau khi gộp)
- [ ] Chạy thử evaluation 20-30 câu không bị 429 crash giữa chừng
- [ ] README ghi rõ giới hạn: "hệ thống demo dùng Gemini free tier, có delay ~Ns/lượt gọi do rate limit" — đây là điểm trung thực nên có, nhà tuyển dụng đánh giá cao việc bạn hiểu rõ giới hạn hệ thống của mình chứ không giấu.

## 7. Có nên đổi sang API khác (OpenAI/Anthropic trả phí thấp) nếu Gemini free quá chật?

Không bắt buộc cho MVP — nhưng nếu bạn có ngân sách nhỏ (vài đô), dùng key trả phí mức thấp nhất (rate limit cao hơn hẳn free tier) sẽ tiết kiệm rất nhiều thời gian debug rate-limit trong 2 ngày. Đây là đánh đổi thời gian vs chi phí, tùy bạn quyết định — nếu giữ free tier, chấp nhận pipeline chạy chậm hơn nhưng vẫn đúng kiến trúc.

---

## Trả lời câu hỏi 2: các file hiện tại đã đủ để học chưa (đã biết basic RAG + LangChain)?

Đủ về mặt lý thuyết + pipeline cho mục tiêu 2 ngày. Vì bạn đã có nền basic RAG/LangChain, cách đọc hiệu quả:

- **`01` (RAG fundamentals), `02` (chunking):** đọc lướt — phần lớn là ôn lại, chỉ dừng ở bảng so sánh chunking (mục 1) vì đó là phần hay bị hỏi sâu khi phỏng vấn.
- **`03` (Advanced Retrieval), `04` (LangGraph/GraphRAG), `05` (Agent/Subagent/MCP), `06` (Evaluation):** đây là phần **giá trị thật với bạn** — kiến thức bạn nói là chưa nắm chắc. Đọc kỹ, làm đủ bài tập, đừng bỏ qua quiz.
- **`07` (API/Deploy), `09` (Capstone/CV):** tham khảo lúc code, không cần học thuộc lý thuyết.
- **`10` (file này):** áp dụng ngay từ lúc code Module 01 (embedding) và Module 04/06 (số lượt gọi LLM), đừng đợi đến lúc deploy mới sửa — sửa kiến trúc sau khi code xong sẽ mất thời gian hơn nhiều.

Những gì bộ tài liệu **chưa đi sâu** (vì nằm ngoài scope 2 ngày, không phải vì thiếu sót): production-scale rate limiting/queueing thực thụ (Redis-based), fine-tuning embedding model, GraphRAG implementation chi tiết, MCP server implementation chi tiết. Nếu sau 2 ngày muốn mở rộng CV, đó là các hướng đi tiếp theo hợp lý.
