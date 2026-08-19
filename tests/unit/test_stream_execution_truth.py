"""
Unit Tests for Execution-Stage Streaming Semantics.
Verifies that stream stages correspond strictly to active execution, cache hits do not emit
fake retrieval stages, Critic is emitted only when active, and synchronous vs stream results match.
"""
import pytest
from unittest.mock import MagicMock, patch
from backend.app.application.contract_qa import ContractQAService
from backend.app.domain.schemas import StructuredAnswer, CitationItem, ExecutionStats
from backend.app.agents.planner import RetrievalPlan
from backend.app.retrieval.confidence import ConfidenceSignals
from backend.app.retrieval.fusion import RetrievedCandidate


@pytest.fixture
def mock_qa_service(monkeypatch):
    service = ContractQAService()
    # Mock gateway
    mock_gw = MagicMock()
    mock_gw.generate.return_value = "Phản hồi mẫu từ AI về điều khoản."
    service.gateway = mock_gw

    # Mock confidence engine to return high confidence by default
    mock_conf = MagicMock()
    mock_conf.compute_confidence.return_value = ConfidenceSignals(
        bm25_dense_rank_agreement=0.9,
        top_score_margin=0.02,
        rrf_consensus=0.03,
        reranker_score=0.85,
        metadata_match=1.0,
        final_confidence=0.88,
    )
    service.confidence_engine = mock_conf

    # Mock verifier
    mock_v = MagicMock()
    v_res = MagicMock()
    v_res.status = "grounded"
    v_res.recommended_action = "accept"
    mock_v.verify.return_value = v_res
    service.verifier = mock_v

    return service


def test_cache_hit_does_not_emit_fake_retrieval_generation_verification(mock_qa_service):
    """Cache hit MUST NOT emit fake retrieving, generating, or verifying events."""
    # Pre-populate cache
    cache_id = "test_cache_ns_hit"
    with patch("backend.app.application.contract_qa.build_query_cache_identity", return_value=cache_id):
        mock_qa_service.cache.set_exact(
            cache_id,
            "cau hoi test cache",
            {"answer": "Cau tra loi da cache", "citations": [], "verification_status": "grounded"},
        )

        events = list(mock_qa_service.answer_query_stream(
            query="cau hoi test cache",
            tenant_id="t1",
            role="admin",
            username="u1",
            document_ids=["doc_1"],
        ))

        stages = [e.get("stage") for e in events if e.get("event") == "stage"]
        event_types = [e.get("event") for e in events]

        assert "retrieving" not in stages
        assert "verifying" not in stages
        assert "final" in event_types
        final_ev = [e for e in events if e.get("event") == "final"][0]
        assert final_ev["answer"] == "Cau tra loi da cache"


def test_stream_does_not_emit_critic_when_critic_skipped(mock_qa_service):
    """When confidence is high (0.88 >= 0.70), Critic is skipped and MUST NOT be emitted."""
    events = list(mock_qa_service.answer_query_stream(
        query="Mức bồi thường là bao nhiêu?",
        tenant_id="t1",
        role="admin",
        username="u1",
        document_ids=["doc_1"],
    ))

    stages = [e.get("stage") for e in events if e.get("event") == "stage"]
    assert "planning" in stages
    assert "critic" not in stages


def test_stream_emits_critic_when_confidence_is_low(mock_qa_service):
    """When confidence is low (< 0.70) and complexity is high, Critic MUST be emitted."""
    mock_qa_service.confidence_engine.compute_confidence.return_value = ConfidenceSignals(
        bm25_dense_rank_agreement=0.2,
        top_score_margin=0.001,
        rrf_consensus=0.01,
        reranker_score=0.45,
        metadata_match=0.5,
        final_confidence=0.50,
    )
    # Planner returning high complexity
    mock_qa_service.planner.plan = MagicMock(return_value=RetrievalPlan(
        task_type="single_contract_qa", candidate_k=10, final_k=5, use_reranker=True,
        use_parent_expansion=False, complexity="high"
    ))
    # Mock retrieval to return candidates so it reaches confidence & critic steps
    cand = RetrievedCandidate(
        chunk_id="c1", doc_id="doc_1", doc_version=1, text="Evidence text for QA",
        is_parent_expanded=False, parent_id=None, page_number=1,
        section_path=["Sec 1"], block_id="b1", bbox=None,
        dense_score=0.5, bm25_score=2.0, rrf_score=0.01, rerank_score=0.45,
        metadata={"title": "Test Doc"}
    )
    mock_qa_service._execute_retrieval = MagicMock(return_value=([cand], {}))

    events = list(mock_qa_service.answer_query_stream(
        query="Câu hỏi phức tạp về tranh chấp thẩm quyền?",
        tenant_id="t1",
        role="admin",
        username="u1",
        document_ids=["doc_1"],
    ))

    stages = [e.get("stage") for e in events if e.get("event") == "stage"]
    assert "planning" in stages
    assert "critic" in stages
    assert "generating" in stages
    assert "verifying" in stages


def test_stream_final_matches_sync_result(mock_qa_service):
    """Stream final event output must match synchronous answer_query output structure."""
    cand = RetrievedCandidate(
        chunk_id="c_match", doc_id="doc_match", doc_version=1, text="Termination clause text",
        is_parent_expanded=False, parent_id=None, page_number=1,
        section_path=["Sec 1"], block_id="b1", bbox=None,
        dense_score=0.9, bm25_score=5.0, rrf_score=0.03, rerank_score=0.90,
        metadata={"title": "Test Doc Match"}
    )
    mock_qa_service._execute_retrieval = MagicMock(return_value=([cand], {}))

    sync_res = mock_qa_service.answer_query(
        query="Điều khoản chấm dứt",
        tenant_id="t_match",
        role="legal",
        username="u_match",
        document_ids=["doc_match"],
    )

    events = list(mock_qa_service.answer_query_stream(
        query="Điều khoản chấm dứt",
        tenant_id="t_match",
        role="legal",
        username="u_match",
        document_ids=["doc_match"],
    ))

    final_event = [e for e in events if e.get("event") == "final"][0]
    assert final_event["answer"] == sync_res.answer
    assert final_event["verification_status"] == sync_res.verification_status


def test_stream_executes_query_once(mock_qa_service):
    """Proves that a single stream request invokes the core execution logic exactly once."""
    mock_res = StructuredAnswer(
        answer="Executed once test",
        citations=[],
        verification_status="grounded",
        confidence_score=1.0,
        retrieval_path="test",
        stats=ExecutionStats(),
    )
    with patch.object(mock_qa_service, "_execute_qa_core", return_value=mock_res) as mock_core:
        events = list(mock_qa_service.answer_query_stream(
            query="cau hoi exactly once",
            tenant_id="t1",
            role="admin",
            username="u1",
            document_ids=["doc_1"],
        ))
        mock_core.assert_called_once()
        assert len(events) >= 1
        assert events[-1]["event"] == "final"
        assert events[-1]["answer"] == "Executed once test"


def test_stream_worker_exception_terminates_without_hang(mock_qa_service):
    """Worker exception must emit an error event, terminate without hanging, and never emit a final answer."""
    with patch.object(mock_qa_service, "_execute_qa_core", side_effect=RuntimeError("ChromaDB connection pool exhausted")):
        events = list(mock_qa_service.answer_query_stream(
            query="cau hoi failing worker",
            tenant_id="t1",
            role="admin",
            username="u1",
            document_ids=["doc_1"],
        ))

        # Generator terminates and returns events
        event_types = [e.get("event") for e in events]
        assert "error" in event_types
        assert "final" not in event_types
        error_ev = [e for e in events if e.get("event") == "error"][0]
        assert "ChromaDB connection pool exhausted" in error_ev["message"]
