"""
Central Retrieval Configuration Loader for Evaluation & Runtime Alignment.
Derives configuration values from backend.app.core.retrieval_defaults and evaluation JSON.
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional

from backend.app.core.retrieval_defaults import (
    DENSE_MODEL_PRODUCTION_DEFAULT,
    DENSE_DIMENSION_PRODUCTION_DEFAULT,
    DENSE_MODEL_EVALUATION_SELECTED,
    DENSE_DIMENSION_EVALUATION_SELECTED,
    CHILD_CHUNK_SIZE,
    CHILD_CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE,
    PARENT_CHUNK_OVERLAP,
    STRUCTURAL_METADATA_ENABLED,
    STRUCTURAL_METADATA_TEMPLATE,
    SPARSE_RETRIEVER_DEFAULT,
    BROAD_CANDIDATE_POOL_SIZE,
    RRF_K_DEFAULT,
    CANDIDATE_REDUCTION_STRATEGY,
    MAX_CHILD_CHUNKS_PER_PARENT,
    RERANKER_INPUT_BUDGET,
    SOFT_ROUTING_ENABLED_DEFAULT,
    SOFT_ROUTING_ALPHA_DEFAULT,
    SOFT_ROUTING_BETA_DEFAULT,
    RERANKER_MODEL_DEFAULT,
    RERANKER_TOP_N_DEFAULT,
    RERANKER_MAX_SEQ_LENGTH,
    ADAPTIVE_BYPASS_ENABLED_DEFAULT,
    CONSENSUS_GATE_THRESHOLD,
)

CONFIG_FILE_PATH = Path(__file__).resolve().parent / "configs" / "retrieval_final_config_v3_1.json"


class RetrievalConfig:
    """Encapsulates retrieval configuration loaded from canonical defaults and optional JSON overlay."""

    def __init__(self, overlay_dict: Optional[Dict[str, Any]] = None):
        self._overlay = overlay_dict or {}

    @classmethod
    def load_default(cls) -> "RetrievalConfig":
        if CONFIG_FILE_PATH.exists():
            with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls(data)
        return cls({})

    # Ingestion & Chunking
    @property
    def chunker_type(self) -> str:
        return self._overlay.get("ingestion", {}).get("chunker_type", "StructureAwareParentChildChunker")

    @property
    def child_target_tokens(self) -> int:
        return self._overlay.get("ingestion", {}).get("child_target_tokens", CHILD_CHUNK_SIZE)

    @property
    def child_overlap_tokens(self) -> int:
        return self._overlay.get("ingestion", {}).get("child_overlap_tokens", CHILD_CHUNK_OVERLAP)

    @property
    def parent_target_tokens(self) -> int:
        return self._overlay.get("ingestion", {}).get("parent_target_tokens", PARENT_CHUNK_SIZE)

    @property
    def parent_overlap_tokens(self) -> int:
        return self._overlay.get("ingestion", {}).get("parent_overlap_tokens", PARENT_CHUNK_OVERLAP)

    @property
    def structural_metadata_enabled(self) -> bool:
        return self._overlay.get("ingestion", {}).get("structural_metadata_enabled", STRUCTURAL_METADATA_ENABLED)

    @property
    def structural_metadata_format(self) -> str:
        return self._overlay.get("ingestion", {}).get("structural_metadata_format", STRUCTURAL_METADATA_TEMPLATE)

    # First Stage Retrieval
    @property
    def dense_model(self) -> str:
        return self._overlay.get("first_stage_retrieval", {}).get("dense_model_default", DENSE_MODEL_EVALUATION_SELECTED)

    @property
    def dense_dimension(self) -> int:
        return self._overlay.get("first_stage_retrieval", {}).get("dense_dimension", DENSE_DIMENSION_EVALUATION_SELECTED)

    @property
    def sparse_retriever(self) -> str:
        return self._overlay.get("first_stage_retrieval", {}).get("sparse_retriever", SPARSE_RETRIEVER_DEFAULT)

    @property
    def broad_candidate_pool_size(self) -> int:
        return self._overlay.get("first_stage_retrieval", {}).get("broad_candidate_pool_size", BROAD_CANDIDATE_POOL_SIZE)

    @property
    def rrf_k(self) -> int:
        return RRF_K_DEFAULT

    @property
    def soft_routing_enabled(self) -> bool:
        return self._overlay.get("first_stage_retrieval", {}).get("soft_routing_boost", {}).get("enabled", SOFT_ROUTING_ENABLED_DEFAULT)

    @property
    def soft_routing_alpha(self) -> float:
        return self._overlay.get("first_stage_retrieval", {}).get("soft_routing_boost", {}).get("document_title_alpha", SOFT_ROUTING_ALPHA_DEFAULT)

    @property
    def soft_routing_beta(self) -> float:
        return self._overlay.get("first_stage_retrieval", {}).get("soft_routing_boost", {}).get("section_heading_beta", SOFT_ROUTING_BETA_DEFAULT)

    # Reduction / Truncation
    @property
    def max_child_chunks_per_parent(self) -> int:
        return MAX_CHILD_CHUNKS_PER_PARENT

    @property
    def reranker_input_budget(self) -> int:
        return self._overlay.get("candidate_reduction", {}).get("reranker_input_budget", RERANKER_INPUT_BUDGET)

    # Second Stage Reranking
    @property
    def reranker_model(self) -> str:
        return self._overlay.get("second_stage_reranking", {}).get("reranker_model", RERANKER_MODEL_DEFAULT)

    @property
    def reranker_top_n(self) -> int:
        return self._overlay.get("second_stage_reranking", {}).get("top_n_output", RERANKER_TOP_N_DEFAULT)

    @property
    def reranker_max_seq_length(self) -> int:
        return self._overlay.get("second_stage_reranking", {}).get("max_seq_length", RERANKER_MAX_SEQ_LENGTH)

    @property
    def adaptive_bypass_enabled(self) -> bool:
        return self._overlay.get("second_stage_reranking", {}).get("adaptive_bypass_enabled", ADAPTIVE_BYPASS_ENABLED_DEFAULT)

    @property
    def consensus_gate_threshold(self) -> float:
        return self._overlay.get("second_stage_reranking", {}).get("consensus_gate_threshold", CONSENSUS_GATE_THRESHOLD)


def get_retrieval_config() -> RetrievalConfig:
    """Get the active retrieval configuration."""
    return RetrievalConfig.load_default()
