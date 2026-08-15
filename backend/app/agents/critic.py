"""
Agent 2: Evidence Critic.
Evaluates retrieved evidence blocks against the user's query aspects.
Runs only when retrieval confidence is below threshold (< 0.70) or task is complex.
Enforces finite, controlled recovery actions with a hard loop budget (max 2 attempts).
"""
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field

from backend.app.providers.gemini_gateway import get_gemini_gateway, GeminiAPIGateway
from backend.app.retrieval.fusion import RetrievedCandidate


class EvidenceCriticEvaluation(BaseModel):
    sufficient: bool = Field(description="True if the retrieved text provides enough factual evidence to answer the query")
    coverage: float = Field(description="Estimated factual coverage ratio between 0.0 and 1.0")
    covered_aspects: List[str] = Field(default_factory=list, description="Aspects adequately covered by evidence")
    missing_aspects: List[str] = Field(default_factory=list, description="Specific factual elements or clauses missing from evidence")
    recommended_action: Literal["proceed", "expand_query", "decompose"] = Field(
        description="Finite allowed recovery action"
    )
    expansion_queries: List[str] = Field(default_factory=list, description="Specific targeted search queries if action is expand_query")
    reasoning: str = ""


class EvidenceCriticAgent:
    """
    Reasoning Agent responsible for auditing retrieved evidence sufficiency.
    """

    def __init__(self, gateway: Optional[GeminiAPIGateway] = None):
        self.gateway = gateway or get_gemini_gateway()

    def evaluate_evidence(
        self,
        query: str,
        candidates: List[RetrievedCandidate],
        retrieval_attempt: int = 1,
    ) -> EvidenceCriticEvaluation:
        """
        Audits candidates against the query and returns structured critique.
        """
        # If max attempts reached, force proceed to avoid unbounded recovery loops
        if retrieval_attempt >= 2:
            return EvidenceCriticEvaluation(
                sufficient=len(candidates) > 0,
                coverage=0.5 if len(candidates) > 0 else 0.0,
                covered_aspects=["partial_evidence"],
                missing_aspects=[],
                recommended_action="proceed",
                reasoning="Max retrieval attempts (2) reached. Forcing progression to answer synthesis.",
            )

        if not candidates:
            return EvidenceCriticEvaluation(
                sufficient=False,
                coverage=0.0,
                covered_aspects=[],
                missing_aspects=["entire_query"],
                recommended_action="expand_query",
                expansion_queries=[f"contract clause regarding {query}"],
                reasoning="Zero documents retrieved.",
            )

        # Build concise evidence excerpts for the critic
        context_parts = []
        for i, c in enumerate(candidates[:6], 1):
            sec = " > ".join(c.section_path) if c.section_path else "General"
            text_snippet = c.text[:600]
            context_parts.append(f"[Evidence {i}] (Doc: {c.doc_id}, Sec: {sec}):\n{text_snippet}")

        evidence_text = "\n\n".join(context_parts)

        prompt = f"""You are the Evidence Critic Agent for an Enterprise Contract Intelligence Platform.
Evaluate whether the retrieved evidence contains sufficient, direct factual support to answer the legal contract question.

User Question: "{query}"

Retrieved Evidence Excerpts:
{evidence_text}

Instructions:
1. Determine if the evidence contains all required numbers, dates, terms, and clause conditions requested.
2. If important details are missing, list them in `missing_aspects` and specify `recommended_action: "expand_query"` along with 1-2 targeted search queries in `expansion_queries`.
3. If the evidence is sufficient, set `sufficient: true` and `recommended_action: "proceed"`."""

        try:
            critique_dict = self.gateway.generate_structured(
                prompt=prompt,
                schema=EvidenceCriticEvaluation,
                model_type="critic",
                temperature=0.0,
            )
            return EvidenceCriticEvaluation(**critique_dict)
        except Exception as e:
            # Fallback
            return EvidenceCriticEvaluation(
                sufficient=True,
                coverage=0.7,
                covered_aspects=["general"],
                missing_aspects=[],
                recommended_action="proceed",
                reasoning=f"Bypassed critic due to evaluation error: {e}",
            )


_critic_agent_instance: Optional[EvidenceCriticAgent] = None


def get_evidence_critic() -> EvidenceCriticAgent:
    global _critic_agent_instance
    if _critic_agent_instance is None:
        _critic_agent_instance = EvidenceCriticAgent()
    return _critic_agent_instance
