# Module 6 — Evaluation & Metrics

## 1. Vì sao evaluation là phần bắt buộc, không phải "nice to have"

Không có metric, bạn không thể trả lời câu phỏng vấn kinh điển: *"làm sao bạn biết hệ thống của bạn tốt?"*. Đây cũng là điểm phân biệt rõ nhất giữa 1 project demo và 1 project thể hiện tư duy kỹ sư.

## 2. Thiết kế bộ eval dataset (20–30 câu)

Chia theo loại câu hỏi để bộ test bao phủ đủ tình huống thực tế:

| Loại câu hỏi | Số lượng đề xuất | Mục đích |
|---|---|---|
| Có đáp án trực tiếp (factual, 1 chunk) | 6–8 | Test retrieval + generation cơ bản |
| Cần tổng hợp nhiều đoạn | 4–6 | Test khả năng gộp context, top-k đủ lớn |
| Không có đáp án trong tài liệu | 3–5 | Test khả năng **từ chối** thay vì bịa |
| Câu hỏi mơ hồ | 2–3 | Test rewrite/clarification |
| Dùng từ khác với tài liệu (đồng nghĩa) | 3–4 | Test dense retrieval có bắt được ý nghĩa không |
| Câu hỏi có nhiễu (thông tin thừa, câu dài) | 2–3 | Test độ ổn định khi query không "sạch" |

Mỗi câu hỏi trong dataset cần: `question`, `expected_answer` (hoặc `expected_chunk_ids` cho retrieval eval), `category`.

```json
// data/evaluation/eval_set.json (ví dụ 1 entry)
{
  "id": "q001",
  "question": "Chunk overlap dùng để làm gì?",
  "category": "factual",
  "expected_chunk_ids": ["doc02_chunking_3"],
  "expected_answer_contains": ["giảm mất ngữ cảnh", "ranh giới chunk"]
}
```

## 3. Ba nhóm metric

### Retrieval metrics — đo chất lượng "tìm đúng chunk chưa"
| Metric | Ý nghĩa |
|---|---|
| Hit Rate | Tỉ lệ câu hỏi có ít nhất 1 chunk đúng nằm trong top-k |
| Recall@k | Tỉ lệ chunk đúng được tìm thấy trong top-k trên tổng chunk đúng |
| Precision@k | Tỉ lệ chunk trong top-k thực sự liên quan |
| Mean Reciprocal Rank (MRR) | Trung bình nghịch đảo vị trí của chunk đúng đầu tiên |
| nDCG | Đo chất lượng thứ hạng có trọng số theo độ liên quan |
| Context Precision/Recall | Biến thể Precision/Recall áp cho toàn bộ context đưa vào LLM (không chỉ chunk riêng lẻ) |

### Generation metrics — đo chất lượng câu trả lời
| Metric | Ý nghĩa |
|---|---|
| Faithfulness | Câu trả lời có bám sát context được cung cấp không (không tự bịa thêm) |
| Answer Relevance | Câu trả lời có thực sự trả lời đúng câu hỏi không |
| Correctness | So với `expected_answer`, đúng về nội dung |
| Citation Accuracy | Citation trích đúng chunk đã thực sự dùng |
| Hallucination Rate | Tỉ lệ câu trả lời chứa thông tin không có trong context |
| Refusal Accuracy | Với câu hỏi "không có đáp án", hệ thống có từ chối đúng không (không bịa) |

### System metrics — đo hiệu năng vận hành
| Metric | Ý nghĩa |
|---|---|
| End-to-end latency | Tổng thời gian từ query đến answer |
| Retrieval latency / Generation latency | Tách riêng để biết bottleneck ở đâu |
| Token usage / Cost per query | Quan trọng khi demo trong phỏng vấn — thể hiện tư duy production |
| Error rate | Tỉ lệ request lỗi |
| Throughput | Số request xử lý được / đơn vị thời gian |

**Nguyên tắc quan trọng: không dùng LLM-as-a-judge làm nguồn đánh giá duy nhất.** Kết hợp: (1) metric retrieval tính toán được (Recall@k không cần LLM), (2) kiểm tra thủ công một mẫu nhỏ (5-10 câu) để xác nhận LLM-as-judge không tự tin quá mức, (3) LLM-as-judge cho faithfulness/relevance ở phần còn lại.

## 4. Bảng benchmark

```
Experiment              | Chunking      | Retriever         | Reranker | Recall@5 | Faithfulness | Avg Latency
-------------------------|---------------|--------------------|----------|----------|--------------|------------
1. Basic vector RAG       | size=500,o=75 | vector-only         | none     |   ...    |     ...      |    ...
2. Hybrid retrieval        | size=500,o=75 | vector+BM25 (RRF)    | none     |   ...    |     ...      |    ...
3. Hybrid + reranker        | size=500,o=75 | vector+BM25 (RRF)     | cross-enc|   ...    |     ...      |    ...
```
Đây chính là bảng đưa thẳng vào README + CV — chứng minh bằng số liệu rằng advanced retrieval thực sự cải thiện chất lượng, không chỉ là "có vẻ tốt hơn".

## 5. Code skeleton — evaluation runner

```python
# app/evaluation/runner.py
import json
import time
from dataclasses import dataclass, field

@dataclass
class EvalResult:
    question_id: str
    recall_at_5: float
    latency_ms: float
    faithfulness_flag: bool  # True nếu pass check thủ công/LLM-judge
    errors: list[str] = field(default_factory=list)

def compute_recall_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int = 5) -> float:
    if not expected_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hit = len(top_k & set(expected_ids))
    return hit / len(expected_ids)

def run_evaluation(eval_set_path: str, rag_pipeline) -> list[EvalResult]:
    with open(eval_set_path, encoding="utf-8") as f:
        eval_set = json.load(f)

    results: list[EvalResult] = []
    for item in eval_set:
        start = time.perf_counter()
        try:
            output = rag_pipeline(item["question"])
        except Exception as e:  # noqa: BLE001 — muốn eval tiếp dù 1 câu lỗi
            results.append(EvalResult(item["id"], 0.0, 0.0, False, errors=[str(e)]))
            continue
        latency_ms = (time.perf_counter() - start) * 1000
        recall = compute_recall_at_k(
            [c["chunk_id"] for c in output["chunks"]],
            item.get("expected_chunk_ids", []),
        )
        results.append(EvalResult(item["id"], recall, latency_ms, faithfulness_flag=True))
    return results
```

## 6. Bài tập 8 — Evaluation (P0)

**Mục tiêu:** chạy benchmark thật trên bộ dữ liệu evaluation.
**Yêu cầu:** chạy `run_evaluation` trên ≥20 câu hỏi, xuất kết quả CSV/JSON, tính Recall@5 trung bình + latency trung bình. Chạy lại cho ít nhất 2 cấu hình (vd: vector-only vs hybrid+rerank) để có bảng so sánh thật ở mục 4.
**Tiêu chí hoàn thành:** file kết quả có thể mở lại, số liệu khác nhau rõ ràng giữa 2 cấu hình (nếu bằng nhau tuyệt đối, nghi ngờ có lỗi đo).

## 7. Validation Module 6

- [ ] Có file eval dataset ≥20 câu, đủ 6 loại câu hỏi ở mục 2
- [ ] Chạy được script eval, xuất kết quả có thể tái tạo (chạy lại ra số tương tự)
- [ ] Có bảng so sánh ≥2 cấu hình với số liệu thật (không phải placeholder)
- [ ] Refusal Accuracy được đo riêng cho nhóm câu hỏi "không có đáp án trong tài liệu"

## 8. Quiz kiểm tra hiểu biết

1. Vì sao không nên dùng LLM-as-a-judge làm nguồn đánh giá duy nhất?
   *Đáp: LLM-judge có thể tự tin sai, thiên vị theo cách diễn đạt; cần kết hợp với metric tính toán được (Recall@k) và kiểm tra thủ công để đối chiếu.*
2. Recall@k và Precision@k khác nhau ở điểm nào?
   *Đáp: Recall@k đo tỉ lệ chunk đúng được tìm thấy trong top-k trên tổng số chunk đúng; Precision@k đo tỉ lệ chunk trong top-k thực sự liên quan.*
3. Refusal Accuracy đo cái gì và vì sao quan trọng cho RAG?
   *Đáp: Đo tỉ lệ hệ thống từ chối đúng khi tài liệu không chứa đáp án — quan trọng vì hallucination khi "không biết" là rủi ro lớn nhất của RAG trong thực tế.*
4. Vì sao cần tách latency thành retrieval latency và generation latency riêng?
   *Đáp: Để xác định bottleneck thực sự nằm ở đâu (tìm kiếm hay sinh câu trả lời) và tối ưu đúng chỗ thay vì đoán mò.*
5. Bảng benchmark 3 experiment (basic/hybrid/hybrid+rerank) dùng để chứng minh điều gì trong CV/phỏng vấn?
   *Đáp: Chứng minh bằng số liệu cụ thể rằng các kỹ thuật advanced retrieval thực sự cải thiện chất lượng, thay vì chỉ khẳng định cảm tính "đã dùng advanced RAG".*

Đi tiếp: mở file `07-api-deployment-optimization.md`.
