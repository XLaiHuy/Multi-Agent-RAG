import time
import json
import logging
from dataclasses import dataclass, field

from app.evaluation.metrics import (
    compute_recall_at_k,
    compute_hit_rate,
    compute_refusal_accuracy,
    evaluate_faithfulness,
)
from app.generation.generator import LLMGenerator
from app.retrieval.vector_retriever import VectorRetriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.graph.basic_rag import app as basic_rag_app
from app.graph.agentic_rag import app as agentic_rag_app


@dataclass
class ItemEvalResult:
    item_id: str
    category: str
    question: str
    retrieved_chunk_ids: list[str]
    answer: str
    recall_at_5: float
    hit_rate: float
    refusal_accuracy: float
    faithfulness_score: float
    latency_ms: float
    errors: list[str] = field(default_factory=list)


@dataclass
class ExperimentResult:
    experiment_name: str
    total_samples: int
    avg_recall_at_5: float
    avg_hit_rate: float
    avg_refusal_accuracy: float
    avg_faithfulness: float
    avg_latency_ms: float
    item_results: list[ItemEvalResult] = field(default_factory=list)


class EvaluationRunner:
    """
    Runner that evaluates RAG pipelines over benchmark datasets.
    """

    def __init__(self, dataset_path: str = "data/evaluation/eval_dataset.json"):
        self.dataset_path = dataset_path
        self.generator = LLMGenerator()

    def load_dataset(self) -> list[dict]:
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def eval_basic_vector_rag(self, items: list[dict]) -> ExperimentResult:
        print("\n[Runner] Running Evaluation: Experiment 1 (Basic Vector RAG)...", flush=True)
        results = []

        for idx, item in enumerate(items):
            print(f"  • [{idx+1}/{len(items)}] Eval item {item['id']}: '{item['question'][:35]}...'", flush=True)
            start_t = time.perf_counter()
            errors = []
            retrieved_chunk_ids = []
            answer = ""
            chunks = []

            try:
                state_input = {"query": item["question"]}
                output = basic_rag_app.invoke(state_input)
                answer = output.get("answer", "")
                chunks = output.get("chunks", [])
                retrieved_chunk_ids = [c.get("chunk_id", "") for c in chunks if isinstance(c, dict)]
            except Exception as e:
                errors.append(str(e))
                answer = f"Error during execution: {e}"

            latency_ms = (time.perf_counter() - start_t) * 1000

            recall = compute_recall_at_k(retrieved_chunk_ids, item.get("expected_chunk_ids", []), k=5)
            hit_rate = compute_hit_rate(retrieved_chunk_ids, item.get("expected_chunk_ids", []), k=5)
            refusal_acc = compute_refusal_accuracy(answer, item.get("is_unanswerable", False))
            faithfulness = evaluate_faithfulness(self.generator, item["question"], chunks, answer)

            results.append(ItemEvalResult(
                item_id=item["id"],
                category=item.get("category", "unknown"),
                question=item["question"],
                retrieved_chunk_ids=retrieved_chunk_ids,
                answer=answer,
                recall_at_5=recall,
                hit_rate=hit_rate,
                refusal_accuracy=refusal_acc,
                faithfulness_score=faithfulness,
                latency_ms=latency_ms,
                errors=errors
            ))
            # Sleep to respect Gemini API rate limit (15 requests/minute)
            if idx < len(items) - 1:
                time.sleep(4.5)

        return self._aggregate_results("1. Basic Vector RAG", results)

    def eval_hybrid_rag(self, items: list[dict]) -> ExperimentResult:
        print("\n[Runner] Running Evaluation: Experiment 2 (Hybrid RAG)...", flush=True)
        hybrid_retriever = HybridRetriever()
        results = []

        for idx, item in enumerate(items):
            print(f"  • [{idx+1}/{len(items)}] Eval item {item['id']}: '{item['question'][:35]}...'", flush=True)
            start_t = time.perf_counter()
            errors = []
            retrieved_chunk_ids = []
            answer = ""

            try:
                search_res = hybrid_retriever.search(query=item["question"], top_k=5, use_rerank=False)
                retrieved_chunk_ids = [r.chunk_id for r in search_res]
                chunks_dict = [{"text": r.text, "chunk_id": r.chunk_id} for r in search_res]
                answer = self.generator.generate_answer(query=item["question"], chunks=chunks_dict)
            except Exception as e:
                errors.append(str(e))
                answer = f"Error during execution: {e}"

            latency_ms = (time.perf_counter() - start_t) * 1000

            recall = compute_recall_at_k(retrieved_chunk_ids, item.get("expected_chunk_ids", []), k=5)
            hit_rate = compute_hit_rate(retrieved_chunk_ids, item.get("expected_chunk_ids", []), k=5)
            refusal_acc = compute_refusal_accuracy(answer, item.get("is_unanswerable", False))
            faithfulness = evaluate_faithfulness(self.generator, item["question"], chunks_dict, answer)

            results.append(ItemEvalResult(
                item_id=item["id"],
                category=item.get("category", "unknown"),
                question=item["question"],
                retrieved_chunk_ids=retrieved_chunk_ids,
                answer=answer,
                recall_at_5=recall,
                hit_rate=hit_rate,
                refusal_accuracy=refusal_acc,
                faithfulness_score=faithfulness,
                latency_ms=latency_ms,
                errors=errors
            ))
            # Sleep to respect Gemini API rate limit
            if idx < len(items) - 1:
                time.sleep(4.5)

        return self._aggregate_results("2. Hybrid RAG (Vector+BM25)", results)

    def eval_agentic_rag(self, items: list[dict]) -> ExperimentResult:
        print("\n[Runner] Running Evaluation: Experiment 3 (Agentic RAG - Graph 3)...", flush=True)
        results = []

        for idx, item in enumerate(items):
            print(f"  • [{idx+1}/{len(items)}] Eval item {item['id']}: '{item['question'][:35]}...'", flush=True)
            start_t = time.perf_counter()
            errors = []
            retrieved_chunk_ids = []
            answer = ""
            output = {}

            try:
                state_input = {"query": item["question"]}
                output = agentic_rag_app.invoke(state_input)
                answer = output.get("answer", "")
                chunks = output.get("chunks", [])
                retrieved_chunk_ids = [c.get("chunk_id", "") for c in chunks if isinstance(c, dict)]
            except Exception as e:
                errors.append(str(e))
                answer = f"Error during execution: {e}"

            latency_ms = (time.perf_counter() - start_t) * 1000

            recall = compute_recall_at_k(retrieved_chunk_ids, item.get("expected_chunk_ids", []), k=5)
            hit_rate = compute_hit_rate(retrieved_chunk_ids, item.get("expected_chunk_ids", []), k=5)
            refusal_acc = compute_refusal_accuracy(answer, item.get("is_unanswerable", False))
            
            ver_status = output.get("verification_status", "grounded")
            faithfulness = 1.0 if ver_status == "grounded" else 0.5

            results.append(ItemEvalResult(
                item_id=item["id"],
                category=item.get("category", "unknown"),
                question=item["question"],
                retrieved_chunk_ids=retrieved_chunk_ids,
                answer=answer,
                recall_at_5=recall,
                hit_rate=hit_rate,
                refusal_accuracy=refusal_acc,
                faithfulness_score=faithfulness,
                latency_ms=latency_ms,
                errors=errors
            ))
            # Sleep to respect Gemini API rate limit
            if idx < len(items) - 1:
                time.sleep(4.5)

        return self._aggregate_results("3. Agentic RAG (Graph 3)", results)

    def _aggregate_results(self, exp_name: str, item_results: list[ItemEvalResult]) -> ExperimentResult:
        n = len(item_results)
        if n == 0:
            return ExperimentResult(exp_name, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

        avg_recall = sum(r.recall_at_5 for r in item_results) / n
        avg_hit = sum(r.hit_rate for r in item_results) / n
        avg_refusal = sum(r.refusal_accuracy for r in item_results) / n
        avg_faith = sum(r.faithfulness_score for r in item_results) / n
        avg_lat = sum(r.latency_ms for r in item_results) / n

        return ExperimentResult(
            experiment_name=exp_name,
            total_samples=n,
            avg_recall_at_5=avg_recall,
            avg_hit_rate=avg_hit,
            avg_refusal_accuracy=avg_refusal,
            avg_faithfulness=avg_faith,
            avg_latency_ms=avg_lat,
            item_results=item_results
        )
