"""
Contract Comparison Application Service.
Implements Query Decomposition and independent facet retrieval for parallel contracts.
Constructs structured side-by-side comparison matrices with exact citations.
"""
import time
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from backend.app.core.config import get_settings
from backend.app.providers.gemini_gateway import get_gemini_gateway, GeminiAPIGateway
from backend.app.application.contract_qa import get_contract_qa_service, ContractQAService
from backend.app.domain.schemas import (
    ContractCompareResponse, CompareFacetResult, CitationItem, ExecutionStats
)

logger = logging.getLogger("contract_compare")

DEFAULT_COMPARISON_FACETS = [
    "Term and Termination Rights",
    "Notice Period Requirements",
    "Limitation of Liability & Liability Cap",
    "Indemnification Obligations",
    "Governing Law & Dispute Resolution",
    "Payment Terms and Price Adjustments",
    "Confidentiality & Data Protection",
]


class FacetSynthesis(BaseModel):
    contract_a_summary: str
    contract_b_summary: str
    key_differences: str
    risk_assessment: str


class ComparisonSynthesis(BaseModel):
    executive_summary: str
    facet_results: List[FacetSynthesis]


class ContractCompareService:
    """
    Decomposes multi-contract comparison queries and performs independent retrieval per contract/facet.
    """

    def __init__(self):
        self.settings = get_settings()
        self.gateway = get_gemini_gateway()
        self.qa_service = get_contract_qa_service()

    def compare_contracts(
        self,
        contract_a_id: str,
        contract_b_id: str,
        contract_a_name: str,
        contract_b_name: str,
        tenant_id: str,
        role: str,
        username: str,
        custom_facets: Optional[List[str]] = None,
    ) -> ContractCompareResponse:
        """
        Executes independent facet retrieval and structured comparative synthesis.
        """
        start_time = time.perf_counter()
        stats = ExecutionStats()
        stats.retrieval_path = "contract_comparison_decomposed"

        facets = custom_facets or DEFAULT_COMPARISON_FACETS
        facet_results: List[CompareFacetResult] = []

        # Retrieve evidence independently for each facet and each document
        for facet in facets:
            facet_query = f"Clause and conditions regarding {facet}"

            # Contract A independent retrieval
            plan_a = self.qa_service.planner.plan(facet_query, context_docs_count=1)
            cands_a = self.qa_service._execute_retrieval(
                query=facet_query,
                plan=plan_a,
                tenant_id=tenant_id,
                allowed_doc_ids=[contract_a_id],
                top_k=4,
                use_rerank=True,
            )
            citations_a = [
                CitationItem(
                    document_id=contract_a_id,
                    document_version=c.doc_version,
                    filename=contract_a_name,
                    page=c.page_number,
                    section_path=c.section_path,
                    block_id=c.block_id,
                    bbox=c.bbox,
                    supporting_text=c.text[:300],
                    score=c.rerank_score or c.rrf_score,
                )
                for c in cands_a
            ]
            text_a = "\n\n".join(c.text for c in cands_a) if cands_a else "No specific clause found."

            # Contract B independent retrieval
            plan_b = self.qa_service.planner.plan(facet_query, context_docs_count=1)
            cands_b = self.qa_service._execute_retrieval(
                query=facet_query,
                plan=plan_b,
                tenant_id=tenant_id,
                allowed_doc_ids=[contract_b_id],
                top_k=4,
                use_rerank=True,
            )
            citations_b = [
                CitationItem(
                    document_id=contract_b_id,
                    document_version=c.doc_version,
                    filename=contract_b_name,
                    page=c.page_number,
                    section_path=c.section_path,
                    block_id=c.block_id,
                    bbox=c.bbox,
                    supporting_text=c.text[:300],
                    score=c.rerank_score or c.rrf_score,
                )
                for c in cands_b
            ]
            text_b = "\n\n".join(c.text for c in cands_b) if cands_b else "No specific clause found."

            # Synthesize contrast for this specific facet
            facet_prompt = f"""Compare the following two contracts on the specific topic: '{facet}'.

[Contract A: {contract_a_name}]
{text_a}

[Contract B: {contract_b_name}]
{text_b}

Provide a structured contrast:
1. Summary for Contract A
2. Summary for Contract B
3. Key differences
4. Practical risk assessment for both parties."""

            try:
                facet_synth = self.gateway.generate_structured(
                    prompt=facet_prompt,
                    schema=FacetSynthesis,
                    model_type="generation",
                    temperature=0.1,
                )
                stats.llm_calls_count += 1
                facet_results.append(
                    CompareFacetResult(
                        facet_name=facet,
                        contract_a_findings=facet_synth.get("contract_a_summary", text_a[:200]),
                        contract_a_citations=citations_a,
                        contract_b_findings=facet_synth.get("contract_b_summary", text_b[:200]),
                        contract_b_citations=citations_b,
                        key_differences=facet_synth.get("key_differences", "Differences not explicitly detailed."),
                        risk_assessment=facet_synth.get("risk_assessment", "Standard operational terms."),
                    )
                )
            except Exception as e:
                logger.error(f"[Compare] Facet '{facet}' synthesis error: {e}")
                facet_results.append(
                    CompareFacetResult(
                        facet_name=facet,
                        contract_a_findings=text_a[:250],
                        contract_a_citations=citations_a,
                        contract_b_findings=text_b[:250],
                        contract_b_citations=citations_b,
                        key_differences="Analysis fallback due to generation error.",
                        risk_assessment="Review source citations directly.",
                    )
                )

        # Generate overall executive summary
        summary_prompt = f"""You are a Principal Legal Counsel.
Provide a high-level executive summary comparing '{contract_a_name}' vs '{contract_b_name}'
based on the analyzed facets: {', '.join(facets)}.
Highlight which contract is more favorable and outline critical risk discrepancies."""

        try:
            exec_summary = self.gateway.generate(
                prompt=summary_prompt,
                model_type="generation",
                temperature=0.2,
            )
            stats.llm_calls_count += 1
        except Exception:
            exec_summary = f"Side-by-side comparison of {len(facets)} legal facets completed across both agreements."

        stats.total_ms = (time.perf_counter() - start_time) * 1000

        return ContractCompareResponse(
            contract_a_id=contract_a_id,
            contract_b_id=contract_b_id,
            contract_a_name=contract_a_name,
            contract_b_name=contract_b_name,
            summary_comparison=exec_summary,
            facet_comparisons=facet_results,
            stats=stats,
        )


_compare_service_instance: Optional[ContractCompareService] = None


def get_contract_compare_service() -> ContractCompareService:
    global _compare_service_instance
    if _compare_service_instance is None:
        _compare_service_instance = ContractCompareService()
    return _compare_service_instance
