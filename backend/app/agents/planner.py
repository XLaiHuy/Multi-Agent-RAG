"""
Agent 1: Retrieval Planner.
Classifies query complexity and determines optimal retrieval strategy.
Bypasses LLM planning with fast deterministic rules for trivially simple queries.
"""
from typing import Dict, Any, Optional, List, Literal
from pydantic import BaseModel, Field

from backend.app.core.config import get_settings
from backend.app.providers.gemini_gateway import get_gemini_gateway, GeminiAPIGateway


class RetrievalPlan(BaseModel):
    task_type: Literal["single_contract_qa", "contract_comparison", "contract_risk_review", "conversational"]
    complexity: Literal["low", "medium", "high"]
    requires_exact_match: bool = True
    requires_semantic_search: bool = True
    requires_multi_document: bool = False
    use_multi_query: bool = False
    use_hyde: bool = False
    use_parent_expansion: bool = True
    use_reranker: bool = False
    candidate_k: int = 15
    final_k: int = 5
    decomposition_facets: List[str] = Field(default_factory=list)
    reasoning: str = ""


class RetrievalPlannerAgent:
    """
    Reasoning Agent responsible for planning adaptive retrieval pipelines.
    """

    def __init__(self, gateway: Optional[GeminiAPIGateway] = None):
        self.gateway = gateway or get_gemini_gateway()

    def is_trivially_simple(self, query: str) -> bool:
        """Deterministic check for simple greetings or conversational inquiries."""
        q = query.strip().lower()
        words = q.split()
        if len(words) <= 6:
            greetings = [
                "chào", "xin chào", "hello", "hi", "hey", "cảm ơn", "thank",
                "tạm biệt", "bye", "how are you", "good morning", "good afternoon"
            ]
            if any(g in q for g in greetings):
                return True
        return False

    def plan(self, query: str, context_docs_count: int = 1) -> RetrievalPlan:
        """
        Determines the optimal retrieval plan for a given user query.
        """
        # Fast deterministic path for direct greetings
        if self.is_trivially_simple(query):
            return RetrievalPlan(
                task_type="conversational",
                complexity="low",
                requires_exact_match=False,
                requires_semantic_search=False,
                requires_multi_document=False,
                use_multi_query=False,
                use_parent_expansion=False,
                use_reranker=False,
                candidate_k=0,
                final_k=0,
                reasoning="Trivially simple conversational query, no retrieval needed.",
            )

        # Fast deterministic check for multi-document comparison
        q_lower = query.lower()
        is_comparison = any(k in q_lower for k in ["compare", "so sánh", "khác biệt", "contrast", "vs", "versus"])
        if is_comparison and context_docs_count > 1:
            return RetrievalPlan(
                task_type="contract_comparison",
                complexity="high",
                requires_exact_match=True,
                requires_semantic_search=True,
                requires_multi_document=True,
                use_multi_query=True,
                use_parent_expansion=True,
                use_reranker=True,
                candidate_k=25,
                final_k=8,
                reasoning="Multi-document contract comparison requires query decomposition and reranking.",
            )

        prompt = f"""You are the Retrieval Planner Agent for an Enterprise Contract Intelligence Platform.
Analyze the user's contract query and formulate an optimal, cost-effective retrieval plan.

User Query: "{query}"
Target Documents in Scope: {context_docs_count}

Guidelines:
- Choose 'low' complexity for straightforward single-topic lookups (e.g. "What is the termination period in Section 4?").
- Choose 'medium' complexity for semantic, multi-paragraph questions. Enable reranker and parent expansion.
- Choose 'high' complexity for complex multi-aspect legal inquiries or comparisons across multiple contracts.
- Only enable multi_query or hyde if the query is ambiguous or complex."""

        try:
            plan_dict = self.gateway.generate_structured(
                prompt=prompt,
                schema=RetrievalPlan,
                model_type="planner",
                temperature=0.0,
            )
            return RetrievalPlan(**plan_dict)
        except Exception as e:
            # Safe deterministic fallback
            return RetrievalPlan(
                task_type="single_contract_qa",
                complexity="medium",
                requires_exact_match=True,
                requires_semantic_search=True,
                requires_multi_document=context_docs_count > 1,
                use_multi_query=False,
                use_parent_expansion=True,
                use_reranker=True,
                candidate_k=20,
                final_k=5,
                reasoning=f"Fallback plan due to planner error: {e}",
            )


_planner_agent_instance: Optional[RetrievalPlannerAgent] = None


def get_retrieval_planner() -> RetrievalPlannerAgent:
    global _planner_agent_instance
    if _planner_agent_instance is None:
        _planner_agent_instance = RetrievalPlannerAgent()
    return _planner_agent_instance
