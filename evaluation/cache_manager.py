"""
Evaluation Cache Manager for Reusable Intermediate Retrieval Artifacts.
Caches deterministic parsed chunks, dense embeddings, BM25 indices, and candidate pools.
Does NOT cache final metrics (HitRate, MRR, nDCG) — metrics are always recomputed.
"""
import os
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np

logger = logging.getLogger("eval_cache")

CACHE_DIR = Path(__file__).resolve().parent / "cache"


def compute_cache_key(
    manifest_hash: str,
    child_target_tokens: int,
    child_overlap_tokens: int,
    parent_target_tokens: int,
    parent_overlap_tokens: int,
    dense_model: str,
    structural_metadata_version: str = "v1",
) -> str:
    """Computes a strict cryptographic cache key."""
    raw_str = (
        f"{manifest_hash}_"
        f"c{child_target_tokens}o{child_overlap_tokens}_"
        f"p{parent_target_tokens}o{parent_overlap_tokens}_"
        f"{dense_model}_"
        f"{structural_metadata_version}"
    )
    return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()[:24]


class EvaluationCache:
    """Manages serialization and deserialization of deterministic intermediate evaluation artifacts."""

    def __init__(self, cache_key: str):
        self.cache_key = cache_key
        self.dir = CACHE_DIR / cache_key
        self.dir.mkdir(parents=True, exist_ok=True)

    def is_complete(self) -> bool:
        """Returns True if all required intermediate artifacts exist in the cache."""
        required_files = [
            "metadata.json",
            "canonical_chunks.json",
            "dense_embeddings.npy",
            "query_embeddings.npy",
            "bm25_candidates_top100.json",
            "dense_candidates_top100.json",
            "rrf_candidates_top100.json",
            "gold_chunk_mapping.json",
        ]
        return all((self.dir / f).exists() for f in required_files)

    def save_corpus_chunks(self, chunks_data: List[Dict[str, Any]]):
        (self.dir / "canonical_chunks.json").write_text(
            json.dumps(chunks_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def load_corpus_chunks(self) -> List[Dict[str, Any]]:
        return json.loads((self.dir / "canonical_chunks.json").read_text(encoding="utf-8"))

    def save_dense_embeddings(self, embeddings: np.ndarray, chunk_ids: List[str]):
        np.save(self.dir / "dense_embeddings.npy", embeddings)
        (self.dir / "dense_chunk_ids.json").write_text(json.dumps(chunk_ids), encoding="utf-8")

    def load_dense_embeddings(self) -> Tuple[np.ndarray, List[str]]:
        emb = np.load(self.dir / "dense_embeddings.npy")
        c_ids = json.loads((self.dir / "dense_chunk_ids.json").read_text(encoding="utf-8"))
        return emb, c_ids

    def save_query_embeddings(self, query_embeddings: np.ndarray, query_keys: List[str]):
        np.save(self.dir / "query_embeddings.npy", query_embeddings)
        (self.dir / "query_keys.json").write_text(json.dumps(query_keys), encoding="utf-8")

    def load_query_embeddings(self) -> Tuple[np.ndarray, List[str]]:
        q_emb = np.load(self.dir / "query_embeddings.npy")
        q_keys = json.loads((self.dir / "query_keys.json").read_text(encoding="utf-8"))
        return q_emb, q_keys

    def save_retrieval_candidates(
        self,
        bm25_top100: Dict[str, List[Tuple[str, float]]],
        dense_top100: Dict[str, List[Tuple[str, float]]],
        rrf_top100: Dict[str, List[Tuple[str, float]]],
        gold_mapping: Dict[str, List[str]],
    ):
        (self.dir / "bm25_candidates_top100.json").write_text(
            json.dumps(bm25_top100, indent=2), encoding="utf-8"
        )
        (self.dir / "dense_candidates_top100.json").write_text(
            json.dumps(dense_top100, indent=2), encoding="utf-8"
        )
        (self.dir / "rrf_candidates_top100.json").write_text(
            json.dumps(rrf_top100, indent=2), encoding="utf-8"
        )
        (self.dir / "gold_chunk_mapping.json").write_text(
            json.dumps(gold_mapping, indent=2), encoding="utf-8"
        )

    def load_retrieval_candidates(self) -> Tuple[
        Dict[str, List[Tuple[str, float]]],
        Dict[str, List[Tuple[str, float]]],
        Dict[str, List[Tuple[str, float]]],
        Dict[str, List[str]],
    ]:
        b100 = json.loads((self.dir / "bm25_candidates_top100.json").read_text(encoding="utf-8"))
        d100 = json.loads((self.dir / "dense_candidates_top100.json").read_text(encoding="utf-8"))
        r100 = json.loads((self.dir / "rrf_candidates_top100.json").read_text(encoding="utf-8"))
        gmap = json.loads((self.dir / "gold_chunk_mapping.json").read_text(encoding="utf-8"))
        return b100, d100, r100, gmap

    def save_metadata(self, meta: Dict[str, Any]):
        meta["cache_key"] = self.cache_key
        meta["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        (self.dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def load_metadata(self) -> Dict[str, Any]:
        return json.loads((self.dir / "metadata.json").read_text(encoding="utf-8"))
