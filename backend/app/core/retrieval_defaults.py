"""
Canonical Retrieval Defaults Module (Single Source of Truth).
Neutral shared definitions used by both runtime Settings and Evaluation Config Loader.
"""

# Embedding Models
DENSE_MODEL_PRODUCTION_DEFAULT = "BAAI/bge-small-en-v1.5"
DENSE_DIMENSION_PRODUCTION_DEFAULT = 384

DENSE_MODEL_EVALUATION_SELECTED = "BAAI/bge-m3"
DENSE_DIMENSION_EVALUATION_SELECTED = 1024

# Ingestion Chunking Parameters (Tokens)
CHILD_CHUNK_SIZE = 250
CHILD_CHUNK_OVERLAP = 30
PARENT_CHUNK_SIZE = 1200
PARENT_CHUNK_OVERLAP = 100

STRUCTURAL_METADATA_ENABLED = True
STRUCTURAL_METADATA_TEMPLATE = "[Document: {doc_title}] [Section: {section_path}]\n{chunk_text}"

# First-Stage Retrieval Parameters
SPARSE_RETRIEVER_DEFAULT = "BM25Okapi"
BROAD_CANDIDATE_POOL_SIZE = 100
RRF_K_DEFAULT = 60

# Candidate Reduction Strategy
CANDIDATE_REDUCTION_STRATEGY = "parent_dedup_then_rrf_order_topk_truncation"
MAX_CHILD_CHUNKS_PER_PARENT = 2
RERANKER_INPUT_BUDGET = 20

# Soft Routing Boost (Evaluated: Keep Optional / Marginal, default False)
SOFT_ROUTING_ENABLED_DEFAULT = False
SOFT_ROUTING_ALPHA_DEFAULT = 0.10
SOFT_ROUTING_BETA_DEFAULT = 0.10

# Second-Stage CrossEncoder Reranking
RERANKER_MODEL_DEFAULT = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
RERANKER_TOP_N_DEFAULT = 10
RERANKER_MAX_SEQ_LENGTH = 512
ADAPTIVE_BYPASS_ENABLED_DEFAULT = True
CONSENSUS_GATE_THRESHOLD = 0.88
