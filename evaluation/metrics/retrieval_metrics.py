"""
Retrieval Evaluation Metrics.
Implements mathematically rigorous Recall@k, Hit Rate@k, Mean Reciprocal Rank (MRR), and nDCG@k.
"""
import math
from typing import List, Set, Dict, Any


def compute_hit_rate_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int = 5) -> float:
    """Hit Rate @ k: 1.0 if at least one relevant document is in top-k, else 0.0."""
    top_k = retrieved_ids[:k]
    for cid in top_k:
        if cid in ground_truth_ids:
            return 1.0
    return 0.0


def compute_recall_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int = 5) -> float:
    """Recall @ k: Proportion of ground truth relevant items retrieved in top-k."""
    if not ground_truth_ids:
        return 1.0
    top_k = set(retrieved_ids[:k])
    relevant_retrieved = top_k.intersection(ground_truth_ids)
    return len(relevant_retrieved) / len(ground_truth_ids)


def compute_reciprocal_rank(retrieved_ids: List[str], ground_truth_ids: Set[str]) -> float:
    """Reciprocal Rank (RR): 1 / (rank of first relevant item)."""
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in ground_truth_ids:
            return 1.0 / rank
    return 0.0


def compute_ndcg_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int = 5) -> float:
    """Normalized Discounted Cumulative Gain (nDCG @ k)."""
    if not ground_truth_ids:
        return 1.0

    dcg = 0.0
    for rank, cid in enumerate(retrieved_ids[:k], start=1):
        rel = 1.0 if cid in ground_truth_ids else 0.0
        dcg += (2**rel - 1) / math.log2(rank + 1)

    # Ideal DCG
    idcg = 0.0
    ideal_hits = min(len(ground_truth_ids), k)
    for rank in range(1, ideal_hits + 1):
        idcg += 1.0 / math.log2(rank + 1)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def evaluate_retrieval_batch(
    predictions: List[List[str]],
    ground_truths: List[Set[str]],
    k_values: List[int] = [5, 10],
) -> Dict[str, float]:
    """Computes mean retrieval metrics over an evaluation dataset."""
    total_samples = len(predictions)
    if total_samples == 0:
        return {}

    metrics: Dict[str, float] = {}
    for k in k_values:
        recalls = [compute_recall_at_k(preds, gts, k=k) for preds, gts in zip(predictions, ground_truths)]
        hit_rates = [compute_hit_rate_at_k(preds, gts, k=k) for preds, gts in zip(predictions, ground_truths)]
        ndcgs = [compute_ndcg_at_k(preds, gts, k=k) for preds, gts in zip(predictions, ground_truths)]
        
        metrics[f"Recall@{k}"] = sum(recalls) / total_samples
        metrics[f"HitRate@{k}"] = sum(hit_rates) / total_samples
        metrics[f"nDCG@{k}"] = sum(ndcgs) / total_samples

    mrr_list = [compute_reciprocal_rank(preds, gts) for preds, gts in zip(predictions, ground_truths)]
    metrics["MRR"] = sum(mrr_list) / total_samples

    return metrics
