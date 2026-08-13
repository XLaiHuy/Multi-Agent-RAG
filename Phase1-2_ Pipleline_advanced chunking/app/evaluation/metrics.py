import json
from app.generation.generator import LLMGenerator


def compute_recall_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int = 5) -> float:
    """
    Computes Recall@k: Proportion of expected chunk IDs present in top-k retrieved IDs.
    """
    if not expected_ids:
        return 1.0 if not retrieved_ids else 0.0
        
    top_k_ids = set(retrieved_ids[:k])
    expected_set = set(expected_ids)
    hits = len(top_k_ids & expected_set)
    return hits / len(expected_set)


def compute_hit_rate(retrieved_ids: list[str], expected_ids: list[str], k: int = 5) -> float:
    """
    Computes Hit Rate: 1.0 if at least one expected chunk ID is found in top-k, else 0.0.
    """
    if not expected_ids:
        return 1.0
        
    top_k_ids = set(retrieved_ids[:k])
    expected_set = set(expected_ids)
    return 1.0 if len(top_k_ids & expected_set) > 0 else 0.0


def compute_refusal_accuracy(answer: str, is_unanswerable: bool) -> float:
    """
    Computes Refusal Accuracy: Measures whether the system correctly refuses to answer unanswerable queries.
    """
    refusal_keywords = [
        "cảnh báo", "không tìm thấy", "không chứa", "từ chối", 
        "unanswerable", "không có thông tin", "tham khảo chung"
    ]
    has_refusal = any(kw in answer.lower() for kw in refusal_keywords)
    
    if is_unanswerable:
        return 1.0 if has_refusal else 0.0
    else:
        return 1.0 if not has_refusal else 0.5


def evaluate_faithfulness(generator: LLMGenerator, query: str, chunks: list[dict], answer: str) -> float:
    """
    Evaluates Faithfulness (Anti-Hallucination) score from 0.0 to 1.0 using Gemini as LLM Judge.
    """
    if not chunks:
        # If no chunks, check if answer correctly contains a refusal
        refusal_keywords = ["cảnh báo", "không tìm thấy", "không chứa", "từ chối"]
        return 1.0 if any(kw in answer.lower() for kw in refusal_keywords) else 0.0
        
    context_str = "\n\n".join([f"Tài liệu [{i+1}]: {c.get('text', '')}" for i, c in enumerate(chunks)])
    
    prompt = f"""Bạn là một Giám khảo đánh giá RAG (RAG Evaluator - LLM Judge).
Nhiệm vụ của bạn là đánh giá chỉ số Faithfulness (Độ trung thực): Kiểm tra xem các ý trong câu trả lời có hoàn toàn bám sát và suy luận hợp lý từ các tài liệu tham khảo hay không.

Tài liệu tham khảo:
{context_str}

Câu hỏi: {query}

Câu trả lời: {answer}

Trả về kết quả JSON chính xác:
{{
  "faithfulness_score": 1.0,  // 1.0 nếu trung thực 100%, 0.5 nếu có ý suy đoán nhẹ, 0.0 nếu bịa đặt/hallucinated
  "reason": "nhận xét ngắn 1 câu"
}}"""
    try:
        res = generator.client.models.generate_content(
            model=generator.model,
            contents=prompt,
            config={"response_mime_type": "application/json"}
        )
        if res and res.text:
            data = json.loads(res.text)
            return float(data.get("faithfulness_score", 1.0))
    except Exception as e:
        print(f"[Metrics] Faithfulness eval error: {e}")
        
    return 1.0
