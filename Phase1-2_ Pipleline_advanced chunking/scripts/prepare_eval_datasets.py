import sys
import os
import json
import time
from pathlib import Path

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.generation.generator import LLMGenerator
from app.retrieval.hybrid_retriever import HybridRetriever


def get_curated_benchmark_items() -> list[dict]:
    """
    Returns curated, high-quality benchmark items covering UIT-ViQuAD factual QA,
    SQuAD v2.0 Unanswerable/Refusal queries, and project-specific categories.
    """
    return [
        # 1. Factual (Single Chunk / UIT-ViQuAD style)
        {
            "id": "eval_01",
            "question": "Chunking là gì và tại sao lại cần thiết trong RAG?",
            "category": "factual",
            "expected_answer": "Chunking là kỹ thuật chia nhỏ tài liệu văn bản dài thành các đoạn nhỏ để vừa với context window của LLM và tối ưu hóa việc nhúng vector.",
            "expected_chunk_ids": ["02-chunking_chunk_0"],
            "source_type": "UIT-ViQuAD_Factual",
            "is_unanswerable": False
        },
        {
            "id": "eval_02",
            "question": "DoRA viết tắt của từ gì?",
            "category": "factual",
            "expected_answer": "DoRA viết tắt của Weight-Decomposed Low-Rank Adaptation.",
            "expected_chunk_ids": ["DoRa_chunk_9"],
            "source_type": "UIT-ViQuAD_Factual",
            "is_unanswerable": False
        },
        {
            "id": "eval_03",
            "question": "Tham số chunk_overlap mang lại tác dụng gì?",
            "category": "factual",
            "expected_answer": "Chunk overlap giúp giữ lại ngữ cảnh ở ranh giới giữa hai đoạn văn bản kề nhau không bị cắt đứt.",
            "expected_chunk_ids": ["02-chunking_chunk_3"],
            "source_type": "Factual",
            "is_unanswerable": False
        },
        {
            "id": "eval_04",
            "question": "Mô hình Embedding được dùng trong dự án này tạo ra vector bao nhiêu chiều?",
            "category": "factual",
            "expected_answer": "Vector nhúng có độ dài 768 chiều.",
            "expected_chunk_ids": ["01-RAG-fundamentals_chunk_5"],
            "source_type": "Factual",
            "is_unanswerable": False
        },

        # 2. Multi-Context (Requires combining information across chunks)
        {
            "id": "eval_05",
            "question": "So sánh sự khác biệt cốt lõi giữa DoRA và LoRA?",
            "category": "multi_context",
            "expected_answer": "DoRA phân rã trọng số thành magnitude và direction, đạt hiệu năng gần với Full Fine-Tuning hơn LoRA mà không làm tăng độ trễ suy luận.",
            "expected_chunk_ids": ["DoRa_chunk_109", "DoRa_chunk_69"],
            "source_type": "Multi_Context",
            "is_unanswerable": False
        },
        {
            "id": "eval_06",
            "question": "Sự khác biệt giữa Dense Retrieval và Sparse Retrieval BM25 là gì?",
            "category": "multi_context",
            "expected_answer": "Dense Retrieval dựa vào ý nghĩa ngữ nghĩa (vector similarity), trong khi BM25 dựa vào tần suất khớp từ khóa chính xác (lexical matching).",
            "expected_chunk_ids": ["03-advanced-retrieval_chunk_2"],
            "source_type": "Multi_Context",
            "is_unanswerable": False
        },

        # 3. Unanswerable / Refusal Queries (SQuAD v2.0 style)
        {
            "id": "eval_07",
            "question": "Cách nấu món lẩu thái chua cay chuẩn vị Tom Yum tại nhà?",
            "category": "not_in_doc",
            "expected_answer": "Unanswerable / Refusal",
            "expected_chunk_ids": [],
            "source_type": "SQuAD_v2.0_Unanswerable",
            "is_unanswerable": True
        },
        {
            "id": "eval_08",
            "question": "Ai là người chiến thắng trong giải bóng đá World Cup 1998?",
            "category": "not_in_doc",
            "expected_answer": "Unanswerable / Refusal",
            "expected_chunk_ids": [],
            "source_type": "SQuAD_v2.0_Unanswerable",
            "is_unanswerable": True
        },
        {
            "id": "eval_09",
            "question": "Công thức tính diện tích hình tròn có bán kính r?",
            "category": "not_in_doc",
            "expected_answer": "Unanswerable / Refusal",
            "expected_chunk_ids": [],
            "source_type": "SQuAD_v2.0_Unanswerable",
            "is_unanswerable": True
        },

        # 4. Synonym (Rephrased / Paraphrased queries)
        {
            "id": "eval_10",
            "question": "Tổ chức dữ liệu dạng phân đoạn có ý nghĩa gì đối với việc nhúng vector?",
            "category": "synonym",
            "expected_answer": "Phân đoạn văn bản giúp kích thước vừa với cửa sổ ngữ cảnh và đảm bảo chất lượng biểu diễn vector.",
            "expected_chunk_ids": ["02-chunking_chunk_0"],
            "source_type": "Synonym_Paraphrased",
            "is_unanswerable": False
        },
        {
            "id": "eval_11",
            "question": "Phương pháp sắp xếp lại thứ hạng Reranking hoạt động như thế nào?",
            "category": "synonym",
            "expected_answer": "Reranker sử dụng mô hình Cross-Encoder để chấm điểm lại sự tương quan giữa câu hỏi và danh sách ứng viên.",
            "expected_chunk_ids": ["03-advanced-retrieval_chunk_10"],
            "source_type": "Synonym_Paraphrased",
            "is_unanswerable": False
        },

        # 5. Ambiguous (Short / Vague queries)
        {
            "id": "eval_12",
            "question": "Overlapping?",
            "category": "ambiguous",
            "expected_answer": "Lặp lại một phần văn bản giữa các chunk để giữ liên tục ngữ cảnh.",
            "expected_chunk_ids": ["02-chunking_chunk_3"],
            "source_type": "Ambiguous",
            "is_unanswerable": False
        },
        {
            "id": "eval_13",
            "question": "RRF?",
            "category": "ambiguous",
            "expected_answer": "Reciprocal Rank Fusion là thuật toán trộn thứ hạng kết quả từ nhiều nguồn tìm kiếm.",
            "expected_chunk_ids": ["03-advanced-retrieval_chunk_8"],
            "source_type": "Ambiguous",
            "is_unanswerable": False
        },

        # 6. Noisy (Queries with long / conversational noise)
        {
            "id": "eval_14",
            "question": "Chào em, anh đang tìm hiểu về thuật toán Reciprocal Rank Fusion trong dự án RAG nâng cao, em giải thích giúp anh RRF dùng làm gì nhé?",
            "category": "noisy",
            "expected_answer": "RRF giúp kết hợp thứ hạng từ tìm kiếm Vector và BM25 mà không bị ảnh hưởng bởi khác biệt thang điểm.",
            "expected_chunk_ids": ["03-advanced-retrieval_chunk_8"],
            "source_type": "Noisy",
            "is_unanswerable": False
        }
    ]


def generate_synthetic_samples(generator: LLMGenerator, retriever: HybridRetriever, num_samples: int = 6) -> list[dict]:
    """
    Generates synthetic QA testset entries directly from project document chunks using Gemini with rate limit pauses.
    """
    print(f"[Dataset Prep] Generating {num_samples} Synthetic QA samples using Gemini API...", flush=True)
    items = []
    
    all_chunk_ids = retriever.bm25_retriever.chunk_ids
    all_documents = retriever.bm25_retriever.documents
    
    if not all_chunk_ids:
        return items

    selected_indices = [1, 5, 9, 13, 17, 21]
    
    for idx_pos, i in enumerate(selected_indices[:num_samples]):
        if i >= len(all_chunk_ids):
            break
        chunk_id = all_chunk_ids[i]
        text = all_documents[i]
        
        prompt = f"""Bạn là một chuyên gia tạo dữ liệu đánh giá RAG.
Dựa trên đoạn văn bản dưới đây, hãy tạo 1 câu hỏi ngắn và 1 câu trả lời chuẩn (Ground Truth Answer).

Đoạn văn bản (Chunk ID: {chunk_id}):
{text[:500]}

Trả về JSON chính xác:
{{
  "question": "câu hỏi",
  "expected_answer": "câu trả lời chuẩn ngắn gọn 1-2 câu"
}}"""
        try:
            # Respect Gemini Free Tier Rate Limit (5 RPM)
            time.sleep(3)
            res = generator.client.models.generate_content(
                model=generator.model,
                contents=prompt,
                config={"response_mime_type": "application/json"}
            )
            if res and res.text:
                data = json.loads(res.text)
                items.append({
                    "id": f"syn_{idx_pos+1:02d}",
                    "question": data.get("question", "").strip(),
                    "category": "synthetic_factual",
                    "expected_answer": data.get("expected_answer", "").strip(),
                    "expected_chunk_ids": [chunk_id],
                    "source_type": "Synthetic_DB_Chunk",
                    "is_unanswerable": False
                })
        except Exception as e:
            print(f"[Dataset Prep] Synthetic generation pause/skip for chunk {chunk_id}: {e}")

    return items


def main():
    print("=" * 80)
    print(" 🛠️ TẠO BỘ DỮ LIỆU ĐÁNH GIÁ (EVALUATION TESTSET PREPARATION)")
    print("=" * 80)
    
    out_dir = Path("data/evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "eval_dataset.json"

    generator = LLMGenerator()
    retriever = HybridRetriever()

    curated_items = get_curated_benchmark_items()
    synthetic_items = generate_synthetic_samples(generator, retriever, num_samples=6)

    combined_dataset = curated_items + synthetic_items

    print(f"\n[Dataset Prep] Summary:")
    print(f"  • Curated UIT-ViQuAD / SQuAD v2.0 Items : {len(curated_items)} items")
    print(f"  • Synthetic DB-based Generated Items   : {len(synthetic_items)} items")
    print(f"  • TOTAL COMBINED EVALUATION ITEMS      : {len(combined_dataset)} items")

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(combined_dataset, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Đã lưu bộ dữ liệu đánh giá thành công tại: {out_file.resolve()}\n")


if __name__ == "__main__":
    main()
