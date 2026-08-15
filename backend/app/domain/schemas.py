"""
Pydantic Schemas for API Requests, Responses, Domain DTOs, and Citations.
"""
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


class CitationItem(BaseModel):
    """Exact supporting evidence for grounded answers."""
    document_id: str = Field(description="Unique document identifier")
    document_version: int = Field(default=1, description="Version of the document")
    filename: str = Field(description="Name of the source document")
    page: int = Field(description="1-based page number")
    section_path: List[str] = Field(default_factory=list, description="Hierarchical section path (e.g. ['Article 8', '8.2'])")
    block_id: str = Field(description="Canonical block identifier")
    bbox: Optional[Dict[str, float]] = Field(default=None, description="Bounding box coordinates {x0, y0, x1, y1}")
    supporting_text: str = Field(description="Exact verbatim excerpt from source")
    score: Optional[float] = Field(default=None, description="Retrieval / relevance score")


class ExecutionStats(BaseModel):
    """Developer / Observability statistics for the query."""
    routing_ms: float = 0.0
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    generation_ms: float = 0.0
    verification_ms: float = 0.0
    total_ms: float = 0.0
    llm_calls_count: int = 0
    estimated_tokens: int = 0
    confidence_score: float = 0.0
    retrieval_path: str = "fast_hybrid"
    cache_hit: bool = False


class StructuredAnswer(BaseModel):
    """Canonical internal and external answer representation."""
    answer: str
    citations: List[CitationItem] = Field(default_factory=list)
    verification_status: Literal["grounded", "partially_grounded", "unsupported", "skipped", "unknown_error"] = "grounded"
    confidence_score: float = 1.0
    retrieval_path: str = "hybrid"
    stats: Optional[ExecutionStats] = None


# --- QA Chat Schemas ---

class ChatMessageDTO(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    citations: Optional[List[CitationItem]] = None


class ContractQARequest(BaseModel):
    query: str
    conv_id: Optional[str] = None
    document_ids: Optional[List[str]] = Field(default=None, description="Optional filter to specific contracts")
    chat_history: Optional[List[ChatMessageDTO]] = Field(default_factory=list)


class ContractQAResponse(BaseModel):
    conv_id: str
    answer: str
    citations: List[CitationItem]
    verification_status: str
    stats: ExecutionStats


# --- Comparison Schemas ---

class ContractCompareRequest(BaseModel):
    contract_a_id: str
    contract_b_id: str
    facets: Optional[List[str]] = Field(
        default=None,
        description="List of clauses/facets to compare (e.g. ['Termination', 'Governing Law', 'Liability Cap'])"
    )


class CompareFacetResult(BaseModel):
    facet_name: str
    contract_a_findings: str
    contract_a_citations: List[CitationItem] = Field(default_factory=list)
    contract_b_findings: str
    contract_b_citations: List[CitationItem] = Field(default_factory=list)
    key_differences: str
    risk_assessment: Optional[str] = None


class ContractCompareResponse(BaseModel):
    contract_a_id: str
    contract_b_id: str
    contract_a_name: str
    contract_b_name: str
    summary_comparison: str
    facet_comparisons: List[CompareFacetResult]
    stats: ExecutionStats


# --- Risk Review Schemas ---

class RiskReviewRequest(BaseModel):
    document_id: str
    custom_rules: Optional[List[str]] = None


class RiskClauseFinding(BaseModel):
    rule_id: str
    rule_name: str
    severity: Literal["low", "medium", "high", "critical"]
    clause_title: str
    clause_text: str
    risk_explanation: str
    recommendation: str
    citations: List[CitationItem] = Field(default_factory=list)


class RiskReviewResponse(BaseModel):
    document_id: str
    document_name: str
    overall_risk_level: Literal["low", "medium", "high", "critical"]
    total_risks_detected: int
    findings: List[RiskClauseFinding]
    stats: ExecutionStats


# --- Ingestion & Document Schemas ---

class IngestionJobStatus(BaseModel):
    job_id: str
    document_id: str
    status: Literal["QUEUED", "PARSING", "OCR", "NORMALIZING", "CHUNKING", "EMBEDDING", "INDEXING", "READY", "FAILED"]
    progress_pct: int
    error_message: Optional[str] = None
    created_at: str
    updated_at: str


class DocumentInfo(BaseModel):
    id: str
    filename: str
    file_type: str
    char_count: int
    page_count: int
    is_scanned: bool
    created_by: str
    created_at: str
    status: str = "READY"


class DocumentUploadResponse(BaseModel):
    status: str = "success"
    document_id: str
    job_id: str
    filename: str
    message: str


# --- Auth Schemas ---

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_info: Dict[str, Any]


class UserProfile(BaseModel):
    username: str
    full_name: str
    role: str
    tenant_id: str
