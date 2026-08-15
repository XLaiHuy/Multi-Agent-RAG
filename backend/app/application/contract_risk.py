"""
Contract Risk Review Application Service.
Combines deterministic regex/keyword business rules with context-aware LLM interpretation.
Pinpoints exact risky clauses with citations, risk ratings, and actionable mitigation recommendations.
"""
import time
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from backend.app.core.config import get_settings
from backend.app.providers.gemini_gateway import get_gemini_gateway, GeminiAPIGateway
from backend.app.domain.risk_rules import RiskRuleEngine, DEFAULT_RISK_RULES, RiskRuleDefinition
from backend.app.retrieval.bm25 import get_bm25_retriever
from backend.app.domain.schemas import (
    RiskReviewResponse, RiskClauseFinding, CitationItem, ExecutionStats
)

logger = logging.getLogger("contract_risk")


class LLMRiskInterpretation(BaseModel):
    is_true_risk: bool = Field(description="True if the clause actually constitutes an operational or legal risk")
    severity: str = Field(description="low | medium | high | critical")
    risk_explanation: str = Field(description="Clear explanation of the legal or financial liability")
    recommendation: str = Field(description="Actionable redline revision or negotiation advice")


class ContractRiskService:
    """
    Evaluates contract text against configurable risk rules and provides expert LLM legal risk analysis.
    """

    def __init__(self):
        self.settings = get_settings()
        self.gateway = get_gemini_gateway()
        self.bm25 = get_bm25_retriever()
        self.rule_engine = RiskRuleEngine()

    def review_contract_risks(
        self,
        document_id: str,
        document_name: str,
        tenant_id: str,
        role: str,
        username: str,
        custom_rule_ids: Optional[List[str]] = None,
    ) -> RiskReviewResponse:
        """
        Executes hybrid rule-based and LLM-assisted contract risk audit.
        """
        start_time = time.perf_counter()
        stats = ExecutionStats()
        stats.retrieval_path = "contract_risk_review"

        # 1. Pull all chunks belonging to this document from BM25 index
        matching_chunks = []
        for cid, doc_text, meta in zip(self.bm25.chunk_ids, self.bm25.documents, self.bm25.metadatas):
            if meta.get("doc_id") == document_id and meta.get("tenant_id") == tenant_id:
                matching_chunks.append((cid, doc_text, meta))

        if not matching_chunks:
            # Document chunks not in memory: return empty finding
            stats.total_ms = (time.perf_counter() - start_time) * 1000
            return RiskReviewResponse(
                document_id=document_id,
                document_name=document_name,
                overall_risk_level="low",
                total_risks_detected=0,
                findings=[],
                stats=stats,
            )

        # 2. Run deterministic rule scanner across all blocks
        rule_hits = [] # List of (rule_match, chunk_id, text, meta)
        for cid, text, meta in matching_chunks:
            matches = self.rule_engine.scan_block_deterministic(text)
            for m in matches:
                if custom_rule_ids and m["rule_id"] not in custom_rule_ids:
                    continue
                rule_hits.append((m, cid, text, meta))

        findings: List[RiskClauseFinding] = []
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}

        # 3. For each detected potential risk, run targeted LLM contextual interpretation
        for match_info, cid, text, meta in rule_hits[:8]: # Bounded check to control budget
            rule_name = match_info["rule_name"]
            default_sev = match_info["severity"]

            prompt = f"""You are a Senior Corporate Contract Risk Assessor.
A deterministic risk rule '{rule_name}' was triggered on the following contract clause:

Clause Text:
"{text}"

Context Metadata:
Section Path: {meta.get('section_path', [])}
Page: {meta.get('page_number', 1)}

Instructions:
1. Evaluate if this clause represents a genuine legal/commercial risk or if it contains standard market exceptions.
2. If genuine risk, provide severity (low, medium, high, critical), concise risk explanation, and redline revision recommendation."""

            try:
                interpretation = self.gateway.generate_structured(
                    prompt=prompt,
                    schema=LLMRiskInterpretation,
                    model_type="critic",
                    temperature=0.0,
                )
                stats.llm_calls_count += 1

                if interpretation.get("is_true_risk", True):
                    sev = interpretation.get("severity", default_sev).lower()
                    if sev not in severity_counts:
                        sev = "medium"
                    severity_counts[sev] += 1

                    citation = CitationItem(
                        document_id=document_id,
                        document_version=int(meta.get("doc_version", 1)),
                        filename=document_name,
                        page=int(meta.get("page_number", 1)),
                        section_path=meta.get("section_path", []),
                        block_id=meta.get("block_id", cid),
                        bbox=meta.get("bbox"),
                        supporting_text=text[:300],
                        score=1.0,
                    )

                    sec_title = " > ".join(meta.get("section_path", [])) if meta.get("section_path") else rule_name
                    finding = RiskClauseFinding(
                        rule_id=match_info["rule_id"],
                        rule_name=rule_name,
                        severity=sev,
                        clause_title=sec_title,
                        clause_text=text,
                        risk_explanation=interpretation.get("risk_explanation", "Potential risk detected by rule."),
                        recommendation=interpretation.get("recommendation", "Review clause terms with counterparty."),
                        citations=[citation],
                    )
                    findings.append(finding)

            except Exception as e:
                logger.error(f"[RiskReview] Error interpreting rule {match_info['rule_id']}: {e}")
                # Fallback to rule output
                severity_counts[default_sev] += 1
                finding = RiskClauseFinding(
                    rule_id=match_info["rule_id"],
                    rule_name=rule_name,
                    severity=default_sev,
                    clause_title=rule_name,
                    clause_text=text,
                    risk_explanation=f"Clause matched deterministic rule patterns for {rule_name}.",
                    recommendation="Review clause and ensure mutual liability caps and standard cure periods.",
                    citations=[],
                )
                findings.append(finding)

        # 4. Compute overall document risk rating
        if severity_counts["critical"] > 0:
            overall_risk = "critical"
        elif severity_counts["high"] > 0:
            overall_risk = "high"
        elif severity_counts["medium"] > 0:
            overall_risk = "medium"
        else:
            overall_risk = "low"

        stats.total_ms = (time.perf_counter() - start_time) * 1000

        return RiskReviewResponse(
            document_id=document_id,
            document_name=document_name,
            overall_risk_level=overall_risk,
            total_risks_detected=len(findings),
            findings=findings,
            stats=stats,
        )


_risk_service_instance: Optional[ContractRiskService] = None


def get_contract_risk_service() -> ContractRiskService:
    global _risk_service_instance
    if _risk_service_instance is None:
        _risk_service_instance = ContractRiskService()
    return _risk_service_instance
