"""
Agent Reasoning Tests with Mocked LLM Gateways.
Tests:
1. Retrieval Planner Agent (simple bypass vs complex planning)
2. Evidence Critic Agent (coverage calculation, bounded loop stop at 2)
3. Answer Verifier Agent (grounded, unsupported, and API failure fallback to 'unknown_error')
"""
import pytest
from unittest.mock import MagicMock

from backend.app.agents.planner import RetrievalPlannerAgent, RetrievalPlan
from backend.app.agents.critic import EvidenceCriticAgent, EvidenceCriticEvaluation
from backend.app.agents.verifier import AnswerVerifierAgent, AnswerVerificationResult
from backend.app.retrieval.fusion import RetrievedCandidate


def test_planner_agent_deterministic_bypass():
    mock_gateway = MagicMock()
    planner = RetrievalPlannerAgent(gateway=mock_gateway)

    # Simple greeting should bypass LLM call
    plan = planner.plan("Hello there, how are you?")
    assert plan.task_type == "conversational"
    assert plan.complexity == "low"
    assert plan.use_reranker is False
    assert mock_gateway.generate_structured.call_count == 0 # Bypassed!


def test_planner_agent_complex_plan():
    mock_gateway = MagicMock()
    mock_gateway.generate_structured.return_value = {
        "task_type": "single_contract_qa",
        "complexity": "medium",
        "requires_exact_match": True,
        "requires_semantic_search": True,
        "requires_multi_document": False,
        "use_multi_query": False,
        "use_hyde": False,
        "use_parent_expansion": True,
        "use_reranker": True,
        "candidate_k": 20,
        "final_k": 5,
        "reasoning": "Standard single contract QA.",
    }
    planner = RetrievalPlannerAgent(gateway=mock_gateway)
    plan = planner.plan("What are the indemnification terms in Section 12?")

    assert plan.task_type == "single_contract_qa"
    assert plan.use_parent_expansion is True
    assert plan.use_reranker is True
    assert plan.final_k == 5


def test_critic_agent_bounded_loop_enforcement():
    mock_gateway = MagicMock()
    critic = EvidenceCriticAgent(gateway=mock_gateway)

    cand = RetrievedCandidate(
        chunk_id="c1", doc_id="d1", doc_version=1, text="Partial text",
        is_parent_expanded=False, parent_id="p1", page_number=1,
        section_path=["Sec 1"], block_id="b1", bbox=None
    )

    # When retrieval_attempt is 2, it MUST force proceed regardless of LLM output
    critique = critic.evaluate_evidence(
        query="What is the exact maximum liability cap amount?",
        candidates=[cand],
        retrieval_attempt=2,
    )

    assert critique.recommended_action == "proceed"
    assert "Max retrieval attempts" in critique.reasoning
    assert mock_gateway.generate_structured.call_count == 0 # Bounded enforcement without extra LLM call


def test_verifier_agent_error_status_handling():
    mock_gateway = MagicMock()
    # Simulate API failure during verification
    mock_gateway.generate_structured.side_effect = RuntimeError("Gemini API 503 Service Unavailable")

    verifier = AnswerVerifierAgent(gateway=mock_gateway)
    result = verifier.verify(
        query="What is the notice period?",
        answer="The notice period is 30 days.",
        evidence_texts=["Some reference text"],
        regeneration_count=0,
    )

    # CRITICAL ACCEPTANCE CRITERIA: Verifier failure MUST NOT be silently 'grounded'
    assert result.status == "unknown_error"
    assert result.status != "grounded"
