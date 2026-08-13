# Module 3 — Advanced Retrieval

## 1. Nền tảng: Dense vs Sparse

- **Dense retrieval**: query và chunk đều được embed thành vector, so sánh bằng cosine similarity. Bắt được **ý nghĩa** kể cả khi từ ngữ khác nhau ("giá vé" ~ "chi phí đi lại"). Yếu khi cần khớp chính xác mã số, tên riêng, thuật ngữ hiếm.
- **Sparse retrieval (BM25)**: dựa trên tần suất từ khóa (TF-IDF cải tiến). Mạnh khi câu hỏi dùng đúng từ trong tài liệu (mã lỗi, tên hàm, số hiệu). Yếu khi người dùng dùng từ đồng nghĩa.
- **Hybrid search**: chạy song song dense + sparse, hợp nhất kết quả. Bù trừ điểm yếu của nhau — **đây là lý do hybrid gần như luôn tốt hơn một trong hai riêng lẻ**.

## 2. Bảng kỹ thuật Advanced Retrieval

| Kỹ thuật | Giải quyết lỗi gì | Khi nào dùng | Khi nào không dùng | Độ khó | Cost latency | Cần cho MVP? |
|---|---|---|---|---|---|---|
| BM25 | Dense bỏ sót khớp từ khóa chính xác | Tài liệu kỹ thuật, có mã/số hiệu | Câu hỏi diễn giải hoàn toàn khác từ ngữ | Thấp | Rất thấp | **P0** |
| Hybrid + RRF (Reciprocal Rank Fusion) | Hợp nhất dense+sparse công bằng không cần tune trọng số | Gần như luôn nên dùng | — | Trung bình | Thấp | **P0** |
| Cross-encoder Reranker | Similarity vector chỉ là "ước lượng thô", reranker đọc kỹ query+chunk cùng lúc để chấm điểm chính xác hơn | Sau khi có top-20 ứng viên, cần lọc còn top-5 | Khi latency cực kỳ quan trọng và top-k thô đã đủ tốt | Trung bình | Trung bình (thêm 1 lượt inference) | **P0/P1 — nên có** |
| Metadata filtering | Query chỉ nên tìm trong 1 phần dữ liệu (vd: theo năm, theo phòng ban) | Có metadata rõ ràng | Dữ liệu không có metadata hữu ích | Thấp | Không đáng kể | P1 |
| Multi-query retrieval | 1 câu hỏi có thể diễn đạt nhiều cách, retrieval 1 lần dễ miss | Câu hỏi mơ hồ | Query đã rất rõ ràng, cụ thể | Trung bình | Cao (N lần retrieval) | P1 |
| Query rewriting | Câu hỏi user viết tắt/lỗi ngữ pháp/thiếu ngữ cảnh | Query ngắn, thiếu chủ ngữ | Query đã đầy đủ, rõ nghĩa | Trung bình | +1 lượt gọi LLM | P1 |
| HyDE (Hypothetical Document Embeddings) | Query và answer document có thể lệch không gian ngữ nghĩa | Câu hỏi trừu tượng, khó truy vấn trực tiếp | Domain hẹp, câu hỏi factual đơn giản (dễ sinh HyDE sai lệch) | Cao | +1 lượt gọi LLM | P2 |
| Parent document retrieval | Chunk nhỏ tốt cho search nhưng thiếu ngữ cảnh khi generate | Tài liệu dài, có cấu trúc rõ | Tài liệu ngắn | Trung bình | Thấp | P1 |
| Contextual compression | Context sau retrieval vẫn còn dư thừa, gây nhiễu cho LLM | Chunk lớn, nhiều câu không liên quan | Chunk đã ngắn gọn | Trung bình | +1 lượt gọi LLM | P2 |
| Self-query retriever | Query chứa cả điều kiện lọc (vd: "báo cáo năm 2023") | Metadata có cấu trúc | Không có metadata | Cao | Trung bình | P2 |
| Maximal Marginal Relevance (MMR) | Top-k bị trùng lặp nội dung (nhiều chunk nói cùng 1 ý) | Cần đa dạng hóa kết quả | Cần độ chính xác tuyệt đối, không quan tâm đa dạng | Thấp | Thấp | P1 |
| Corrective RAG | Retrieval trả về tài liệu không liên quan mà hệ thống vẫn generate | Cần độ tin cậy cao | MVP đơn giản, chưa cần | Cao | +1-2 lượt gọi LLM | P1 (đưa vào LangGraph, xem Module 04) |
| Self-RAG | Hệ thống tự đánh giá câu trả lời của chính nó trước khi trả về | Cần giảm hallucination mạnh | Latency-sensitive | Cao | Cao | P2 |
| Adaptive retrieval | Không phải câu hỏi nào cũng cần retrieval (vd: "chào bạn") | Chatbot đa mục đích | RAG chuyên biệt 1 domain | Trung bình | — | P2 (làm bằng agent, xem Module 05) |

## 3. Pipeline Advanced RAG thực tế cho project này (không over-engineering)

```
Query
  → Query Rewrite (chỉ khi cần, giới hạn 1 lần — xem Module 04)
  → Hybrid Retrieval (Vector top-20 + BM25 top-20)
  → Merge & Deduplicate (theo chunk_id)
  → Reciprocal Rank Fusion → top-20 hợp nhất
  → Cross-encoder Reranker → top-5
  → Context Construction (token budget check)
  → Answer Generation
  → Citation Verification
```

**Cảnh báo over-engineering:** đừng làm cả HyDE + Multi-query + Self-query + Contextual compression cùng lúc. Mỗi kỹ thuật thêm vào là +1 lượt gọi LLM (tiền + latency) — chỉ thêm khi baseline đã đo cho thấy nó thực sự cần.

## 4. Reciprocal Rank Fusion — công thức

```
RRF_score(doc) = Σ  1 / (k + rank_i(doc))     với k = 60 (giá trị mặc định phổ biến)
```
Với mỗi hệ thống retrieval (vector, BM25), `rank_i(doc)` là thứ hạng của doc trong hệ đó. Doc xuất hiện tốt ở cả 2 hệ sẽ có RRF score cao — không cần tune trọng số thủ công như weighted sum.

## 5. Code skeleton

```python
# app/retrieval/hybrid.py
from rank_bm25 import BM25Okapi

def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k: int = 60
) -> list[tuple[str, float]]:
    """ranked_lists: mỗi phần tử là 1 list chunk_id đã sắp xếp theo độ liên quan giảm dần."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class BM25Retriever:
    def __init__(self, corpus: list[str], chunk_ids: list[str]):
        if len(corpus) != len(chunk_ids):
            raise ValueError("corpus và chunk_ids phải cùng độ dài")
        tokenized = [doc.lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized)
        self.chunk_ids = chunk_ids

    def search(self, query: str, top_k: int = 20) -> list[str]:
        scores = self.bm25.get_scores(query.lower().split())
        ranked = sorted(zip(self.chunk_ids, scores), key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in ranked[:top_k]]
```

```python
# app/retrieval/rerank.py
from sentence_transformers import CrossEncoder

_MODEL = None

def get_reranker() -> CrossEncoder:
    global _MODEL
    if _MODEL is None:
        _MODEL = CrossEncoder("BAAI/bge-reranker-base")
    return _MODEL

def rerank(query: str, candidates: list[dict], top_n: int = 5) -> list[dict]:
    """candidates: [{"chunk_id": ..., "text": ...}, ...]"""
    if not candidates:
        return []
    pairs = [(query, c["text"]) for c in candidates]
    scores = get_reranker().predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_n]
```

## 6. Bài tập 3 — Hybrid search (P0)

**Mục tiêu:** kết hợp BM25 + vector search bằng RRF.
**Yêu cầu:** cùng 1 query, lấy top-20 từ vector search và top-20 từ BM25, merge bằng RRF, in ra top-10 kết quả cuối.
**Tiêu chí hoàn thành:** với câu hỏi chứa từ khóa/thuật ngữ đúng chính xác trong tài liệu, kết quả BM25 phải kéo được chunk đó lên dù vector search xếp thấp.

## 7. Bài tập 4 — Reranking (P0/P1)

**Mục tiêu:** đo tác động thật của reranker.
**Yêu cầu:** rerank top-20 (từ RRF) xuống top-5, so sánh thứ tự trước/sau bằng mắt và bằng Recall@5 trên bộ câu hỏi thử.
**Lỗi thường gặp:** quên rằng cross-encoder chậm hơn nhiều so với vector search — chỉ rerank top-20, không rerank toàn bộ corpus.

## 8. Validation Module 3

- [ ] Hybrid retrieval trả về kết quả khác (và tốt hơn ở ít nhất 1 case) so với chỉ dùng vector search
- [ ] RRF merge chạy đúng, không lỗi khi 1 trong 2 danh sách rỗng
- [ ] Reranker chạy được và top-5 sau rerank có Recall@5 ≥ top-5 trước rerank trên bộ câu hỏi thử
- [ ] Có ghi log so sánh before/after để dùng cho README sau này

## 9. Quiz kiểm tra hiểu biết

1. Vì sao hybrid search thường tốt hơn chỉ dùng dense hoặc chỉ dùng sparse?
   *Đáp: Chúng bù trừ điểm yếu cho nhau — dense bắt ý nghĩa, sparse bắt từ khóa chính xác.*
2. RRF khác gì so với việc cộng điểm similarity trực tiếp giữa 2 hệ thống?
   *Đáp: RRF dùng rank (thứ hạng) chứ không dùng score thô, nên không cần chuẩn hóa/tune trọng số giữa 2 thang điểm khác nhau (cosine similarity vs BM25 score).*
3. Reranker giải quyết vấn đề gì mà vector similarity không giải quyết được?
   *Đáp: Vector similarity so 2 vector độc lập (embed riêng), còn cross-encoder đọc query và chunk cùng lúc, ước lượng độ liên quan chính xác hơn nhưng chậm hơn — vì vậy chỉ áp dụng cho top-k nhỏ.*
4. HyDE có nên đưa vào MVP 2 ngày không?
   *Đáp: Không bắt buộc (P2) — thêm 1 lượt gọi LLM, chỉ nên thêm sau khi baseline đã đo và cho thấy cần cải thiện truy vấn trừu tượng.*
5. Vì sao cần giới hạn reranker chỉ chạy trên top-20 thay vì toàn bộ corpus?
   *Đáp: Cross-encoder tốn chi phí tính toán cho mỗi cặp (query, chunk) — chạy trên toàn corpus sẽ quá chậm, không khả thi cho query real-time.*

Đi tiếp: mở file `04-langgraph-graphrag.md`.
