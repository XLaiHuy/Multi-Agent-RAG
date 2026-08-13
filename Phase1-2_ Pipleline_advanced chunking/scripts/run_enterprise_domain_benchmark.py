"""
Benchmark Đánh giá Bài toán Thực tế Doanh nghiệp (Enterprise HR & Finance RAG Benchmark):
Đánh giá 15 câu hỏi thực tế của doanh nghiệp trên dữ liệu đa định dạng (Word, Excel, MD, Image Scan)
thông qua các chiến lược Retrieval nâng cao: BM25, Dense, Hybrid RRF, và Multi-Query Expansion.
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.graph.agentic_rag import get_hybrid_retriever, get_vector_retriever
from app.core.db import store_semantic_cache, lookup_semantic_cache

sys.stdout.reconfigure(encoding="utf-8")

EVAL_QUERIES = [
    {
        "id": "HR_01",
        "question": "Nhân viên làm việc sau 22:00 được trợ cấp bao nhiêu tiền ăn đêm?",
        "expected_kw": ["100.000", "22:00", "ăn đêm"],
        "category": "HR & OT Policy (Word)"
    },
    {
        "id": "HR_02",
        "question": "Số ngày phép năm tiêu chuẩn của nhân viên chính thức là bao nhiêu?",
        "expected_kw": ["12 ngày phép", "hưởng nguyên lương"],
        "category": "HR & OT Policy (Word)"
    },
    {
        "id": "HR_03",
        "question": "Mức thưởng KPI tối đa cho cá nhân đạt xếp loại Xuất sắc A+ là bao nhiêu?",
        "expected_kw": ["3 tháng lương", "Xuất sắc A+"],
        "category": "HR & OT Policy (Word)"
    },
    {
        "id": "HR_04",
        "question": "Mức phụ cấp OT ngày nghỉ hàng tuần Chủ nhật được tính thế nào?",
        "expected_kw": ["200%", "Chủ nhật"],
        "category": "HR & OT Policy (Word)"
    },
    {
        "id": "FIN_01",
        "question": "Phụ cấp tiền ăn công tác phí một ngày của Trưởng phòng là bao nhiêu?",
        "expected_kw": ["350.000", "Trưởng phòng"],
        "category": "Allowance Table (Excel)"
    },
    {
        "id": "FIN_02",
        "question": "Hạn mức khách sạn tối đa một đêm cho Chuyên viên nhân viên là bao nhiêu?",
        "expected_kw": ["900.000", "Chuyên viên"],
        "category": "Allowance Table (Excel)"
    },
    {
        "id": "FIN_03",
        "question": "Chức danh nào được đi máy bay Hạng Thương gia (Business)?",
        "expected_kw": ["Giám đốc", "Thương gia", "Business"],
        "category": "Allowance Table (Excel)"
    },
    {
        "id": "FIN_04",
        "question": "Hóa đơn giá trị từ bao nhiêu tiền trở lên bắt buộc phải chuyển khoản qua ngân hàng?",
        "expected_kw": ["20.000.000", "chuyển khoản"],
        "category": "VAT Invoice Rule (Markdown)"
    },
    {
        "id": "FIN_05",
        "question": "Kế toán viên có bao nhiêu ngày làm việc để kiểm tra Tờ trình thanh toán?",
        "expected_kw": ["2 ngày làm việc", "Kế toán viên"],
        "category": "VAT Invoice Rule (Markdown)"
    },
    {
        "id": "FIN_06",
        "question": "Hạn mức tạm ứng công tác phí tối đa cho mỗi chuyến công tác là bao nhiêu?",
        "expected_kw": ["30.000.000", "tạm ứng"],
        "category": "VAT Invoice Rule (Markdown)"
    },
    {
        "id": "SEC_01",
        "question": "Vi phạm quy định bảo mật dữ liệu khách hàng bị phạt tối thiểu bao nhiêu?",
        "expected_kw": ["50.000.000", "sa thải"],
        "category": "Security Rule (Word)"
    },
    {
        "id": "FIN_07",
        "question": "Vào những ngày nào trong tuần bộ phận Ngân hàng thực hiện lệnh chuyển khoản?",
        "expected_kw": ["thứ Ba", "thứ Sáu"],
        "category": "VAT Invoice Rule (Markdown)"
    },
    {
        "id": "FIN_08",
        "question": "Trong vòng mấy ngày sau khi kết thúc công tác nhân viên phải nộp hồ sơ hoàn ứng?",
        "expected_kw": ["5 ngày làm việc", "hoàn ứng"],
        "category": "VAT Invoice Rule (Markdown)"
    },
    {
        "id": "HR_05",
        "question": "Thời gian làm việc tiêu chuẩn một ngày của nhân viên là mấy giờ?",
        "expected_kw": ["8 giờ/ngày", "8:00 - 17:00"],
        "category": "HR & OT Policy (Word)"
    },
    {
        "id": "FIN_09",
        "question": "Hóa đơn VAT từ 5.000.000 trở lên có phải tra cứu trên cổng Tổng cục Thuế không?",
        "expected_kw": ["5.000.000", "Tổng cục Thuế"],
        "category": "VAT Invoice Rule (Markdown)"
    }
]


def run_enterprise_benchmark():
    print("========================================================")
    print("🏢 BENCHMARK BÀI TOÁN DOANH NGHIỆP: HR & FINANCE AI")
    print("========================================================")
    retriever = get_hybrid_retriever()
    vec_retriever = get_vector_retriever()

    modes = [
        ("Sparse BM25-only (k=3)", "sparse", 3),
        ("Dense Vector-only (k=3)", "dense", 3),
        ("Hybrid RRF (k=3)", "hybrid", 3),
        ("Hybrid RRF + Context Stitching (k=8)", "hybrid_stitch", 8),
        ("Multi-Query Expansion RRF (k=10)", "multi_query", 10),
    ]

    report = {}

    for label, mode, top_k in modes:
        hits = 0
        latencies = []
        t0 = time.perf_counter()

        for item in EVAL_QUERIES:
            q = item["question"]
            kws = item["expected_kw"]
            q_t0 = time.perf_counter()

            if mode == "sparse":
                bm25_res = retriever.bm25_retriever.search(q, top_k=top_k)
                texts = []
                for cid, score in bm25_res:
                    if cid in retriever.bm25_retriever.chunk_ids:
                        idx = retriever.bm25_retriever.chunk_ids.index(cid)
                        texts.append(retriever.bm25_retriever.documents[idx])
                retrieved_text = " ".join(texts)

            elif mode == "dense":
                dense_res = retriever.vector_retriever.search(q, top_k=top_k)
                retrieved_text = " ".join([r.text for r in dense_res])

            elif mode == "hybrid":
                hybrid_res = retriever.search(q, top_k=top_k)
                retrieved_text = " ".join([r.text for r in hybrid_res])

            elif mode == "hybrid_stitch":
                hybrid_res = retriever.search(q, top_k=top_k)
                retrieved_text = " ".join([r.text for r in hybrid_res])

            elif mode == "multi_query":
                # Multi-Query Expansion simulation
                sub_queries = [
                    q,
                    q.replace("bao nhiêu", "mức trợ cấp cụ thể"),
                    q.replace("thế nào", "quy định quy chế 2026")
                ]
                res = retriever.multi_query_search(sub_queries, top_k=top_k)
                retrieved_text = " ".join([r.text for r in res])

            q_dur = (time.perf_counter() - q_t0) * 1000
            latencies.append(q_dur)

            # Check if any keyword matches retrieved text
            hit = any(kw.lower() in retrieved_text.lower() for kw in kws)
            if hit:
                hits += 1

        tot_dur = time.perf_counter() - t0
        N = len(EVAL_QUERIES)
        hit_rate = (hits / N) * 100
        p50 = float(sum(latencies)/N)
        qps = N / tot_dur if tot_dur > 0 else 0

        report[label] = {
            "hit_rate_pct": round(hit_rate, 2),
            "hits": hits,
            "total": N,
            "avg_latency_ms": round(p50, 2),
            "qps": round(qps, 1)
        }
        print(f"  [{label}] -> Hit Rate: {hit_rate:.1f}% ({hits}/{N}) | Avg Latency: {p50:.2f}ms | QPS: {qps:.1f}")

    # Test Semantic Cache Latency
    print("\n⚡ Test Semantic Cache Latency (<10ms Check)...")
    first_q = EVAL_QUERIES[0]["question"]
    q_emb = vec_retriever.embedder.embed_query(first_q)
    store_semantic_cache(query=first_q, query_embedding=q_emb, answer="Test cached answer 100.000 VNĐ")

    cache_t0 = time.perf_counter()
    cache_res = lookup_semantic_cache(q_emb, similarity_threshold=0.96)
    cache_dur_ms = (time.perf_counter() - cache_t0) * 1000
    print(f"  [Semantic Cache] Hit latency: {cache_dur_ms:.2f}ms (Tương đồng: {cache_res['similarity']*100:.1f}%)")

    # Save benchmark report
    out_path = Path("data/evaluation/enterprise_domain_benchmark_report.json")
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[Done] Saved enterprise benchmark report to {out_path}")


if __name__ == "__main__":
    run_enterprise_benchmark()
