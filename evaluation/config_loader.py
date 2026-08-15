"""
Central Retrieval Configuration Loader for Evaluation & Runtime Alignment.
Serves as the single source of truth for retrieval, chunking, and reranking parameters.
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional

CONFIG_FILE_PATH = Path(__file__).resolve().parent / "configs" / "retrieval_final_config_v3_1.json"
FALLBACK_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "retrieval_final_config_v3.json"


class RetrievalConfig:
    """Encapsulates retrieval configuration loaded from the single source of truth."""

    def __init__(self, config_dict: Dict[str, Any]):
        self._cfg = config_dict

    @classmethod
    def load_default(cls) -> "RetrievalConfig":
        target = CONFIG_FILE_PATH if CONFIG_FILE_PATH.exists() else FALLBACK_CONFIG_PATH
        if not target.exists():
            raise FileNotFoundError(f"Retrieval config not found at: {target}")
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data)

    @property
    def raw(self) -> Dict[str, Any]:
        return self._cfg

    # Ingestion & Chunking
    @property
    def chunker_type(self) -> str:
        return self._cfg.get("ingestion", {}).get("chunker_type", "StructureAwareParentChildChunker")

    @property
    def child_target_tokens(self) -> int:
        return self._cfg.get("ingestion", {}).get("child_target_tokens", 250)

    @property
    def child_overlap_tokens(self) -> int:
        return self._cfg.get("ingestion", {}).get("child_overlap_tokens", 30)

    @property
    def parent_target_tokens(self) -> int:
        return self._cfg.get("ingestion", {}).get("parent_target_tokens", 1200)

    @property
    def parent_overlap_tokens(self) -> int:
        return self._cfg.get("ingestion", {}).get("parent_overlap_tokens", 100)

    @property
    def structural_metadata_enabled(self) -> bool:
        return self._cfg.get("ingestion", {}).get("structural_metadata_enabled", True)

    @property
    def structural_metadata_format(self) -> str:
        return self._cfg.get("ingestion", {}).get(
            "structural_metadata_format", "[Document: {doc_title}] [Section: {section_path}]\n{chunk_text}"
        )

    # First Stage Retrieval
    @property
    def dense_model(self) -> str:
        return self._cfg.get("first_stage_retrieval", {}).get("dense_model_default", "BAAI/bge-m3")

    @property
    def dense_dimension(self) -> int:
        return self._cfg.get("first_stage_retrieval", {}).get("dense_dimension", 1024)

    @property
    def sparse_retriever(self) -> str:
        return self._cfg.get("first_stage_retrieval", {}).get("sparse_retriever", "BM25Okapi")

    @property
    def broad_candidate_pool_size(self) -> int:
        return self._cfg.get("first_stage_retrieval", {}).get("broad_candidate_pool_size", 100)

    @property
    def rrf_k(self) -> int:
        return 60

    @property
    def soft_routing_enabled(self) -> bool:
        return self._cfg.get("first_stage_retrieval", {}).get("soft_routing_boost", {}).get("enabled", False)

    @property
    def soft_routing_alpha(self) -> float:
        return self._cfg.get("first_stage_retrieval", {}).get("soft_routing_boost", {}).get("document_title_alpha", 0.10)

    @property
    def soft_routing_beta(self) -> float:
        return self._cfg.get("first_stage_retrieval", {}).get("soft_routing_boost", {}).get("section_heading_beta", 0.10)

    # Reduction / Truncation
    @property
    def max_child_chunks_per_parent(self) -> int:
        return 2

    @property
    def reranker_input_budget(self) -> int:
        return self._cfg.get("candidate_reduction", {}).get("reranker_input_budget", 20)

    # Second Stage Reranking
    @property
    def reranker_model(self) -> str:
        return self._cfg.get("second_stage_reranking", {}).get("reranker_model", "cross-encoder/ms-marco-TinyBERT-L-2-v2")

    @property
    def reranker_top_n(self) -> int:
        return self._cfg.get("second_stage_reranking", {}).get("top_n_output", 10)

    @property
    def reranker_max_seq_length(self) -> int:
        return self._cfg.get("second_stage_reranking", {}).get("max_seq_length", 512)

    @property
    def adaptive_bypass_enabled(self) -> bool:
        return self._cfg.get("second_stage_reranking", {}).get("adaptive_bypass_enabled", True)

    @property
    def consensus_gate_threshold(self) -> float:
        return self._cfg.get("second_stage_reranking", {}).get("consensus_gate_threshold", 0.88)


def get_retrieval_config() -> RetrievalConfig:
    """Get the active retrieval configuration."""
    return RetrievalConfig.load_default()
