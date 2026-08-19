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
    "Thời hạn hợp đồng & Quyền chấm dứt (Term & Termination)",
    "Quy định thời hạn báo trước (Notice Period Requirements)",
    "Giới hạn trách nhiệm & Mức trần bồi thường (Limitation of Liability)",
    "Nghĩa vụ bồi thường thiệt hại bên thứ ba (Indemnification)",
    "Luật áp dụng & Cơ quan giải quyết tranh chấp (Governing Law & Dispute Resolution)",
    "Điều khoản thanh toán & Điều chỉnh giá (Payment Terms & Pricing)",
    "Bảo mật thông tin & Bảo vệ dữ liệu (Confidentiality & Data Protection)",
    "Quyền Sở hữu Trí tuệ (Intellectual Property Rights)",
]


class FacetSynthesis(BaseModel):
    contract_a_summary: str = Field(description="Tóm tắt ngắn gọn quy định của Hợp đồng A bằng tiếng Việt")
    contract_b_summary: str = Field(description="Tóm tắt ngắn gọn quy định của Hợp đồng B bằng tiếng Việt")
    key_differences: str = Field(description="Điểm khác biệt cốt lõi giữa 2 hợp đồng bằng tiếng Việt")
    risk_assessment: str = Field(description="Nhận định rủi ro và đánh giá bên nào có lợi thế hơn bằng tiếng Việt")


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
        Executes decomposed, facet-by-facet comparison across two contracts.
        """
        start_time = time.perf_counter()
        stats = ExecutionStats()
        stats.retrieval_path = "contract_comparison_decomposed"

        facets = custom_facets or DEFAULT_COMPARISON_FACETS
        facet_results: List[CompareFacetResult] = []

        # Retrieve evidence independently for each facet and each document
        for facet in facets:
            facet_query = f"Điều khoản và quy định liên quan đến {facet}"

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
            text_a = "\n\n".join(c.text for c in cands_a) if cands_a else "Không tìm thấy điều khoản quy định cụ thể."

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
            text_b = "\n\n".join(c.text for c in cands_b) if cands_b else "Không tìm thấy điều khoản quy định cụ thể."

            # Synthesize contrast for this specific facet in Vietnamese
            facet_prompt = f"""Bạn là Cố vấn Pháp lý Doanh nghiệp Cấp cao. Hãy so sánh hai bản hợp đồng sau đây về khía cạnh: '{facet}'.

[Hợp đồng A: {contract_a_name}]
{text_a}

[Hợp đồng B: {contract_b_name}]
{text_b}

Yêu cầu cung cấp phân tích có cấu trúc bằng TIẾNG VIỆT:
1. contract_a_summary: Tóm tắt ngắn gọn nội dung của Hợp đồng A.
2. contract_b_summary: Tóm tắt ngắn gọn nội dung của Hợp đồng B.
3. key_differences: Điểm khác biệt quan trọng nhất giữa hai bên.
4. risk_assessment: Đánh giá rủi ro pháp lý/thương mại thực tế và nhận định hợp đồng nào bảo vệ quyền lợi tốt hơn."""

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
                        key_differences=facet_synth.get("key_differences", "Không có sự khác biệt đáng kể."),
                        risk_assessment=facet_synth.get("risk_assessment", "Điều khoản theo thông lệ thị trường."),
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
                        key_differences="Xem xét trích dẫn trực tiếp.",
                        risk_assessment="Kiểm tra điều khoản gốc.",
                    )
                )

        # Generate overall executive summary in Vietnamese
        summary_prompt = f"""Bạn là Giám đốc Pháp chế Doanh nghiệp (General Counsel).
Hãy viết một bản tóm tắt điều hành cấp cao (Executive Summary) bằng TIẾNG VIỆT so sánh giữa '{contract_a_name}' và '{contract_b_name}'
dựa trên các khía cạnh đã phân tích: {', '.join(facets)}.
Nêu rõ hợp đồng nào đem lại lợi thế thương mại và bảo vệ an toàn pháp lý tốt hơn cho doanh nghiệp, cùng các lưu ý đàm phán quan trọng."""

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
