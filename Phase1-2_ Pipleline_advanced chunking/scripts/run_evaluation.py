import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.evaluation.runner import EvaluationRunner

def print_result(res):
    print(f"\n--- Kết quả: {res.experiment_name} ---")
    print(f"Tổng số mẫu: {res.total_samples}")
    print(f"Hit Rate @5: {res.avg_hit_rate:.2f}")
    print(f"Recall @5: {res.avg_recall_at_5:.2f}")
    print(f"Faithfulness: {res.avg_faithfulness:.2f}")
    print(f"Refusal Accuracy: {res.avg_refusal_accuracy:.2f}")
    print(f"Latency trung bình: {res.avg_latency_ms:.2f} ms")

def main():
    print("================================================================================")
    print(" 📊 CHẠY ĐÁNH GIÁ (EVALUATION) VỚI OLLAMA LOCAL")
    print("================================================================================")
    
    runner = EvaluationRunner()
    items = runner.load_dataset()
    print(f"Loaded {len(items)} items from dataset.")
    
    res1 = runner.eval_basic_vector_rag(items)
    print_result(res1)
    
    res2 = runner.eval_hybrid_rag(items)
    print_result(res2)
    
    res3 = runner.eval_agentic_rag(items)
    print_result(res3)

if __name__ == "__main__":
    main()
