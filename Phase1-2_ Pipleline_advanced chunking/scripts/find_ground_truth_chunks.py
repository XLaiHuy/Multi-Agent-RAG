import sys
import os
import json

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.retrieval.hybrid_retriever import HybridRetriever

def main():
    retriever = HybridRetriever()
    
    queries = [
        "Chunking là gì và tại sao lại cần thiết trong RAG?",
        "DoRA viết tắt của từ gì?",
        "Tham số chunk_overlap mang lại tác dụng gì?",
        "Mô hình Embedding được dùng trong dự án này tạo ra vector bao nhiêu chiều?",
        "So sánh sự khác biệt cốt lõi giữa DoRA và LoRA?",
        "Sự khác biệt giữa Dense Retrieval và Sparse Retrieval BM25 là gì?",
        "Cách nấu món lẩu thái chua cay chuẩn vị Tom Yum tại nhà?",
        "Ai là người chiến thắng trong giải bóng đá World Cup 1998?",
        "Công thức tính diện tích hình tròn có bán kính r?",
        "Tổ chức dữ liệu dạng phân đoạn có ý nghĩa gì đối với việc nhúng vector?",
        "Phương pháp sắp xếp lại thứ hạng Reranking hoạt động như thế nào?",
        "Overlapping?",
        "RRF?",
        "Chào em, anh đang tìm hiểu về thuật toán Reciprocal Rank Fusion trong dự án RAG nâng cao, em giải thích giúp anh RRF dùng làm gì nhé?"
    ]
    
    print("Finding ground truth chunks for queries...\n")
    for q in queries:
        print(f"Query: {q}")
        results = retriever.search(q, top_k=3, use_rerank=False)
        chunk_ids = [res.chunk_id for res in results]
        print(f"Top 3 Chunk IDs: {chunk_ids}\n")

if __name__ == "__main__":
    main()
