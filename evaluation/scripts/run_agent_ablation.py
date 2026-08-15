#!/usr/bin/env python3
"""
Agent Ablation Benchmark Framework with Strict Label Leakage Isolation.

ARCHITECTURE:
1. execute_query_without_gold(query, corpus_id, retriever, ...):
   Pure runtime execution. Consumes ONLY query string and document index.
   ZERO benchmark metadata (is_unanswerable, gold_evidence, gold_contract_id) enters execution!

2. score_execution_against_gold(result, gold_benchmark_item):
   Pure offline evaluation layer. Evaluates answer correctness, refusal, and attribution.
"""
import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

os.environ["TOKENIZERS_PARALLELISM"] = "false"

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.config_loader import get_retrieval_config

def execute_query_without_gold(
    query: str,
    retriever_fn,
    verifier_fn=None,
) -> Dict[str, Any]:
    """
    Executes contract RAG pipeline without access to any ground truth benchmark fields.
    """
    start_time = time.perf_counter()
    retrieved_chunks = retriever_fn(query)
    
    # Verifier operates purely on query + retrieved chunk contents
    if verifier_fn:
        decision = verifier_fn(query, retrieved_chunks)
    else:
        decision = {"is_refusal": False, "answer": "Generated from retrieved context"}

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    return {
        "query": query,
        "retrieved_chunk_ids": [c["chunk_id"] for c in retrieved_chunks],
        "is_refusal": decision.get("is_refusal", False),
        "answer": decision.get("answer", ""),
        "latency_ms": latency_ms,
    }


def score_execution_against_gold(
    execution_result: Dict[str, Any],
    gold_item: Dict[str, Any],
    ground_truth_chunk_ids: Set[str],
) -> Dict[str, Any]:
    """
    Offline evaluation scoring against ground truth benchmark labels.
    """
    is_unanswerable = gold_item.get("is_unanswerable", False)
    is_refusal = execution_result["is_refusal"]
    retrieved_ids = execution_result["retrieved_chunk_ids"]

    # Classification
    if is_unanswerable:
        outcome = "CORRECT_REFUSAL" if is_refusal else "FALSE_ANSWER"
    else:
        has_gold = any(cid in ground_truth_chunk_ids for cid in retrieved_ids[:5])
        if is_refusal:
            outcome = "FALSE_REFUSAL"
        elif has_gold:
            outcome = "GROUNDED_ANSWER"
        else:
            outcome = "UNSUPPORTED_ANSWER"

    return {
        "query": execution_result["query"],
        "outcome": outcome,
        "is_unanswerable": is_unanswerable,
        "is_refusal": is_refusal,
    }


def main():
    print("=" * 80)
    print("AGENT ABLATION FRAMEWORK (STATUS: REAL API NOT CONFIGURED)")
    print("Strict label-isolated execution functions defined for live provider evaluation.")
    print("Previous simulated metrics have been marked SIMULATION_ONLY in scientific reports.")
    print("=" * 80)

if __name__ == "__main__":
    main()
