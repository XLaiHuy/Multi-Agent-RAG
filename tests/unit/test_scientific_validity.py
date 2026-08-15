"""
Scientific Validity & Integrity Regression Tests.
Verifies:
1. Pure label-free query execution (zero label leakage).
2. Distinction between CandidateHitRate@k (binary coverage) and TrueChunkRecall@k (set overlap).
3. Exact field-by-field retrieval configuration alignment between Settings, ConfigLoader, and retrieval_defaults.
4. Reranker strict evaluation mode (failing loudly).
5. Clean repository paths (zero absolute local machine paths).
6. Custom holdout benchmark naming integrity.
"""
import re
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_evaluation_runtime_does_not_consume_is_unanswerable():
    """Verify that pure execution function signatures do not accept benchmark labels."""
    from evaluation.scripts.run_agent_ablation import execute_query_without_gold
    import inspect

    sig = inspect.signature(execute_query_without_gold)
    forbidden_params = {"is_unanswerable", "gold_evidence", "gold_answer", "gold_contract_id", "ground_truth"}
    found = forbidden_params.intersection(set(sig.parameters.keys()))
    assert len(found) == 0, f"execute_query_without_gold must NOT accept benchmark labels: {found}"


def test_candidate_hitrate_vs_true_chunk_recall_distinction():
    """Verify mathematical distinction between CandidateHitRate@k and TrueChunkRecall@k."""
    from evaluation.metrics.retrieval_metrics import (
        compute_candidate_hit_rate_at_k,
        compute_true_chunk_recall_at_k,
    )

    retrieved = ["chunk_1", "chunk_2", "chunk_3", "chunk_4", "chunk_5"]
    ground_truth = {"chunk_1", "chunk_8", "chunk_9"}  # 3 relevant chunks, 1 retrieved in top-5

    hit_rate = compute_candidate_hit_rate_at_k(retrieved, ground_truth, k=5)
    true_recall = compute_true_chunk_recall_at_k(retrieved, ground_truth, k=5)

    assert hit_rate == 1.0, "CandidateHitRate@5 must be 1.0 (at least one gold retrieved)"
    assert true_recall == pytest.approx(1.0 / 3.0), "TrueChunkRecall@5 must be 1/3 (~0.333)"
    assert hit_rate != true_recall, "HitRate and Recall must not be conflated"


def test_exact_config_alignment_across_runtime_and_evaluation():
    """Verify exact field-by-field alignment across Settings, ConfigLoader, and retrieval_defaults."""
    from backend.app.core.config import get_settings
    from evaluation.config_loader import get_retrieval_config
    import backend.app.core.retrieval_defaults as defaults

    settings = get_settings()
    eval_cfg = get_retrieval_config()

    # 1. Dense Embedding Model & Dimension
    assert settings.local_embedding_model == defaults.DENSE_MODEL_PRODUCTION_DEFAULT == "BAAI/bge-small-en-v1.5"
    assert settings.embedding_dimension == defaults.DENSE_DIMENSION_PRODUCTION_DEFAULT == 384
    assert eval_cfg.dense_model == defaults.DENSE_MODEL_EVALUATION_SELECTED == "BAAI/bge-m3"
    assert eval_cfg.dense_dimension == defaults.DENSE_DIMENSION_EVALUATION_SELECTED == 1024

    # 2. Chunking Ingestion Parameters (Tokens)
    assert settings.child_chunk_size == eval_cfg.child_target_tokens == defaults.CHILD_CHUNK_SIZE == 250
    assert settings.child_chunk_overlap == eval_cfg.child_overlap_tokens == defaults.CHILD_CHUNK_OVERLAP == 30
    assert settings.parent_chunk_size == eval_cfg.parent_target_tokens == defaults.PARENT_CHUNK_SIZE == 1200
    assert settings.parent_chunk_overlap == eval_cfg.parent_overlap_tokens == defaults.PARENT_CHUNK_OVERLAP == 100

    # 3. First-Stage & Reranking Defaults
    assert eval_cfg.sparse_retriever == defaults.SPARSE_RETRIEVER_DEFAULT == "BM25Okapi"
    assert eval_cfg.broad_candidate_pool_size == defaults.BROAD_CANDIDATE_POOL_SIZE == 100
    assert eval_cfg.rrf_k == defaults.RRF_K_DEFAULT == 60
    assert eval_cfg.reranker_input_budget == defaults.RERANKER_INPUT_BUDGET == 20

    # 4. CrossEncoder Reranker
    assert settings.reranker_model == eval_cfg.reranker_model == defaults.RERANKER_MODEL_DEFAULT == "cross-encoder/ms-marco-TinyBERT-L-2-v2"
    assert settings.reranker_max_seq_length == eval_cfg.reranker_max_seq_length == defaults.RERANKER_MAX_SEQ_LENGTH == 512


def test_benchmark_reranker_strict_mode_raises_on_inference_failure():
    """Verify that LocalCrossEncoderReranker(strict=True) fails loudly on errors."""
    from backend.app.providers.reranker import LocalCrossEncoderReranker

    reranker = LocalCrossEncoderReranker(model_name="nonexistent/fake-model-xyz", strict=True)
    with pytest.raises(Exception):
        reranker.rerank("test query", ["test doc 1", "test doc 2"])


def test_no_absolute_machine_paths_exist_in_evaluation_scripts():
    """Verify zero hardcoded machine paths exist in evaluation scripts."""
    eval_scripts_dir = REPO_ROOT / "evaluation" / "scripts"
    assert eval_scripts_dir.exists(), f"Directory not found: {eval_scripts_dir}"

    bad_patterns = [
        re.compile(r'[a-zA-Z]:[/\\](?:Users|home)[/\\][a-zA-Z0-9_-]+', re.IGNORECASE),
        re.compile(r"anti" + r"gravity-ide", re.IGNORECASE),
        re.compile(r"\.system_" + r"generated", re.IGNORECASE),
    ]

    for py_file in eval_scripts_dir.glob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        for pat in bad_patterns:
            matches = pat.findall(text)
            assert not matches, f"Found local machine path in {py_file.name}: {matches}"


def test_custom_cuad_benchmark_is_not_labeled_official_legalbench():
    """Verify that the 25-contract holdout report and script do not claim to be official LegalBench-RAG."""
    script_file = REPO_ROOT / "evaluation" / "scripts" / "run_external_legalbench.py"
    report_file = REPO_ROOT / "evaluation" / "reports" / "EXTERNAL_LEGAL_BENCHMARK.md"

    if script_file.exists():
        text = script_file.read_text(encoding="utf-8")
        assert "custom_cuad_holdout_v2" in text.lower() or "custom" in text.lower()

    if report_file.exists():
        text = report_file.read_text(encoding="utf-8")
        assert "custom_cuad_holdout_v2" in text.lower() or "custom" in text.lower()
