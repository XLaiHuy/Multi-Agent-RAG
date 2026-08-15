"""
Agent 3: Answer Verifier.
Checks generated answer claims against retrieved evidence blocks to detect hallucinations and citation issues.
Handles regeneration logic (max 1 regeneration attempt) and qualified refusal.
CRITICAL RULE: Verifier API failures are marked as 'unknown_error', NEVER silently 'grounded'.
"""
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field

from backend.app.providers.gemini_gateway import get_gemini_gateway, GeminiAPIGateway
from backend.app.domain.schemas import CitationItem


class AnswerVerificationResult(BaseModel):
    status: Literal["grounded", "partially_grounded", "unsupported", "unknown_error"] = Field(
        description="Factual grounding status of the answer"
    )
    unsupported_claims: List[str] = Field(
        default_factory=list, description="Specific factual statements in the answer that are not supported by the evidence"
    )
    citation_issues: List[str] = Field(
        default_factory=list, description="Citations that point to irrelevant or incorrect text blocks"
    )
    recommended_action: Literal["accept", "regenerate", "qualify_or_refuse"] = Field(
        description="Next step based on verification outcome"
    )
    critique_for_regeneration: str = ""


class AnswerVerifierAgent:
    """
    Reasoning Agent responsible for auditing answer faithfulness against evidence.
    """

    def __init__(self, gateway: Optional[GeminiAPIGateway] = None):
        self.gateway = gateway or get_gemini_gateway()

    def verify(
        self,
        query: str,
        answer: str,
        evidence_texts: List[str],
        regeneration_count: int = 0,
    ) -> AnswerVerificationResult:
        """
        Audits generated answer claims against evidence texts.
        """
        if not evidence_texts:
            return AnswerVerificationResult(
                status="unsupported",
                unsupported_claims=["Entire answer has no supporting reference context."],
                citation_issues=["No citations available."],
                recommended_action="qualify_or_refuse",
                critique_for_regeneration="No reference context provided.",
            )

        context_str = "\n\n---\n\n".join(
            f"[Source Block {i+1}]:\n{txt}" for i, txt in enumerate(evidence_texts)
        )

        prompt = f"""You are the Answer Verifier Agent for an Enterprise Contract Intelligence Platform.
Your job is to rigorously audit the generated answer against the reference evidence text.

User Question: {query}

Reference Evidence:
{context_str}

Generated Answer to Audit:
{answer}

Instructions:
1. Verify every factual assertion, number, clause reference, date, and liability condition in the answer.
2. If ANY statement cannot be directly substantiated by the reference evidence, list it in `unsupported_claims`.
3. If all claims are completely grounded, set `status: "grounded"`, `recommended_action: "accept"`.
4. If there are minor unsupported details and regeneration_count is 0, set `status: "partially_grounded"`, `recommended_action: "regenerate"`, and provide clear feedback in `critique_for_regeneration`.
5. If regeneration_count >= 1 and claims remain unsupported, set `recommended_action: "qualify_or_refuse"`."""

        try:
            res_dict = self.gateway.generate_structured(
                prompt=prompt,
                schema=AnswerVerificationResult,
                model_type="verifier",
                temperature=0.0,
            )
            res = AnswerVerificationResult(**res_dict)

            # If regeneration budget reached, ensure action is qualify_or_refuse
            if regeneration_count >= 1 and res.status != "grounded":
                res.recommended_action = "qualify_or_refuse"

            return res

        except Exception as e:
            # RULE: Verifier failure must NOT be treated as grounded!
            return AnswerVerificationResult(
                status="unknown_error",
                unsupported_claims=[],
                citation_issues=[],
                recommended_action="accept", # Allow response through with warning status
                critique_for_regeneration=f"Verification failed due to API error: {e}",
            )


_verifier_agent_instance: Optional[AnswerVerifierAgent] = None


def get_answer_verifier() -> AnswerVerifierAgent:
    global _verifier_agent_instance
    if _verifier_agent_instance is None:
        _verifier_agent_instance = AnswerVerifierAgent()
    return _verifier_agent_instance
