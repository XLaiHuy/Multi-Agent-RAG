"""
Role- & ACL-Aware Exact and Semantic Cache.
Enforces tenant isolation, ACL boundary scoping, bounded capacity, TTL expiration, and namespace invalidation.
Prevents cross-department / cross-role data leaks (e.g. Finance doc cached cannot leak to HR).
"""
import time
import json
import hashlib
import threading
from typing import Dict, Any, Optional, List, Tuple
from collections import OrderedDict
import numpy as np

from backend.app.providers.interfaces import CacheStore


def compute_acl_scope_hash(tenant_id: str, role: str, corpus_version: str = "v1", embedding_model: str = "bge-m3") -> str:
    """Computes a cryptographically distinct namespace for caching."""
    raw = f"{tenant_id}::{role}::{corpus_version}::{embedding_model}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class SemanticCacheEntry:
    def __init__(self, query: str, query_vector: np.ndarray, result: Dict[str, Any], expires_at: float):
        self.query = query
        self.query_vector = query_vector # normalized
        self.result = result
        self.expires_at = expires_at
        self.hit_count = 0
        self.last_accessed = time.time()


class BoundedSemanticCache(CacheStore):
    """
    In-memory / Redis-compatible cache implementation with:
    - Namespace scoping (tenant_id + role + corpus_version)
    - Fast L1 exact match dictionary (O(1))
    - Bounded L2 cosine similarity search (capacity bounded with LRU eviction)
    - Automatic TTL expiration
    """

    def __init__(self, max_entries_per_namespace: int = 1000, default_ttl_seconds: int = 86400):
        self.max_entries = max_entries_per_namespace
        self.default_ttl = default_ttl_seconds
        
        # exact_cache: {namespace: OrderedDict[exact_query_hash, (result, expires_at)]}
        self._exact_cache: Dict[str, OrderedDict[str, Tuple[Dict[str, Any], float]]] = {}
        
        # semantic_cache: {namespace: List[SemanticCacheEntry]}
        self._semantic_cache: Dict[str, List[SemanticCacheEntry]] = {}
        
        self._lock = threading.Lock()

    def get_exact(self, namespace: str, key: str) -> Optional[Dict[str, Any]]:
        """L1 Exact cache lookup."""
        with self._lock:
            ns_cache = self._exact_cache.get(namespace)
            if not ns_cache:
                return None

            key_hash = hashlib.md5(key.strip().lower().encode("utf-8")).hexdigest()
            if key_hash in ns_cache:
                result, expires_at = ns_cache[key_hash]
                if time.time() > expires_at:
                    del ns_cache[key_hash]
                    return None
                # Mark as recently used
                ns_cache.move_to_end(key_hash)
                return result
            return None

    def set_exact(
        self, namespace: str, key: str, value: Dict[str, Any], ttl_seconds: int = 3600
    ) -> None:
        """L1 Exact cache store."""
        with self._lock:
            if namespace not in self._exact_cache:
                self._exact_cache[namespace] = OrderedDict()

            ns_cache = self._exact_cache[namespace]
            key_hash = hashlib.md5(key.strip().lower().encode("utf-8")).hexdigest()

            # Evict LRU if capacity reached
            if len(ns_cache) >= self.max_entries:
                ns_cache.popitem(last=False)

            expires_at = time.time() + ttl_seconds
            ns_cache[key_hash] = (value, expires_at)

    def get_semantic(
        self,
        namespace: str,
        query_vector: List[float],
        similarity_threshold: float = 0.96,
    ) -> Optional[Dict[str, Any]]:
        """
        L2 Semantic cache lookup within isolated namespace.
        Cosine similarity >= threshold -> Hit.
        """
        with self._lock:
            entries = self._semantic_cache.get(namespace)
            if not entries:
                return None

            q_vec = np.array(query_vector, dtype=np.float32)
            q_norm = np.linalg.norm(q_vec)
            if q_norm == 0:
                return None
            q_vec = q_vec / q_norm

            now = time.time()
            valid_entries = []
            best_entry: Optional[SemanticCacheEntry] = None
            best_sim = -1.0

            for entry in entries:
                if now > entry.expires_at:
                    continue # Expired
                valid_entries.append(entry)

                sim = float(np.dot(q_vec, entry.query_vector))
                if sim > best_sim and sim >= similarity_threshold:
                    best_sim = sim
                    best_entry = entry

            # Update pruned list
            self._semantic_cache[namespace] = valid_entries

            if best_entry:
                best_entry.hit_count += 1
                best_entry.last_accessed = now
                cached_res = dict(best_entry.result)
                cached_res["cache_similarity"] = best_sim
                cached_res["cached"] = True
                return cached_res

            return None

    def set_semantic(
        self,
        namespace: str,
        query: str,
        query_vector: List[float],
        result: Dict[str, Any],
        ttl_seconds: int = 86400,
    ) -> None:
        """L2 Semantic cache store with normalized vector and bounded LRU eviction."""
        with self._lock:
            if namespace not in self._semantic_cache:
                self._semantic_cache[namespace] = []

            entries = self._semantic_cache[namespace]

            q_vec = np.array(query_vector, dtype=np.float32)
            q_norm = np.linalg.norm(q_vec)
            if q_norm == 0:
                return
            q_vec = q_vec / q_norm

            # Evict LRU if capacity exceeded
            if len(entries) >= self.max_entries:
                entries.sort(key=lambda e: e.last_accessed)
                entries.pop(0)

            expires_at = time.time() + ttl_seconds
            entry = SemanticCacheEntry(
                query=query,
                query_vector=q_vec,
                result=result,
                expires_at=expires_at,
            )
            entries.append(entry)

    def invalidate_namespace(self, namespace: str) -> None:
        """Invalidate all cached entries for a namespace."""
        with self._lock:
            if namespace in self._exact_cache:
                del self._exact_cache[namespace]
            if namespace in self._semantic_cache:
                del self._semantic_cache[namespace]


_global_cache: Optional[BoundedSemanticCache] = None


def get_cache_store() -> BoundedSemanticCache:
    global _global_cache
    if _global_cache is None:
        _global_cache = BoundedSemanticCache()
    return _global_cache
