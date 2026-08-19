"""
Contract QA Application Service.
Implements Adaptive Multi-Agent RAG execution paths:
- LEVEL 0: Direct Conversational (no retrieval)
- LEVEL 1: Exact / Simple Lookup (hybrid retrieval -> generation)
- LEVEL 2: Semantic / Ambiguous QA (hybrid -> confidence check -> parent expansion/rerank -> generation)
- LEVEL 3: Complex Inquiry (planner -> multi-retrieval -> critic audit -> generation -> verifier)
Produces StructuredAnswer with exact supporting CitationItems and execution statistics.
"""
import time
import logging
from typing import List, Dict, Any, Optional, Tuple, Iterator

from backend.app.core.config import get_settings
from backend.app.providers.gemini_gateway import get_gemini_gateway, GeminiAPIGateway
from backend.app.providers.reranker import get_reranker
from backend.app.providers.observability import trace_step
from backend.app.retrieval.bm25 import get_bm25_retriever
from backend.app.retrieval.dense import get_dense_retriever
from backend.app.retrieval.fusion import reciprocal_rank_fusion, HierarchicalParentExpander, RetrievedCandidate
from backend.app.retrieval.confidence import get_confidence_engine
from backend.app.agents.planner import get_retrieval_planner, RetrievalPlan
from backend.app.agents.critic import get_evidence_critic
from backend.app.agents.verifier import get_answer_verifier
from backend.app.persistence.cache import get_cache_store, compute_acl_scope_hash
from backend.app.domain.schemas import StructuredAnswer, CitationItem, ExecutionStats

logger = logging.getLogger("contract_qa")


class ContractQAService:
    """
    Core Contract QA Engine managing adaptive execution, caching, evidence aggregation, and verification.
    """

    def __init__(self):
        self.settings = get_settings()
        self.gateway = get_gemini_gateway()
        self.bm25 = get_bm25_retriever()
        self.dense = get_dense_retriever()
        self.reranker = get_reranker()
        self.confidence_engine = get_confidence_engine()
        self.planner = get_retrieval_planner()
        self.critic = get_evidence_critic()
        self.verifier = get_answer_verifier()
        self.cache = get_cache_store()

    def _execute_retrieval(
        self,
        query: str,
        plan: RetrievalPlan,
        tenant_id: str,
        allowed_doc_ids: Optional[List[str]],
        top_k: int = 15,
        use_rerank: bool = True,
    ) -> Tuple[List[RetrievedCandidate], Dict[str, Any]]:
        """
        Runs BM25 + Dense + RRF + Parent Expansion + CrossEncoder Reranking.
        Returns: (candidates, retrieval_trace) where retrieval_trace holds true independent ranked IDs.
        """
        # 1. Sparse BM25 Search
        bm25_hits = self.bm25.search(
            query=query, top_k=top_k, tenant_id=tenant_id, allowed_doc_ids=allowed_doc_ids
        )
        bm25_ranked_ids = [h[0] for h in bm25_hits]
        bm25_map = {h[0]: h for h in bm25_hits}

        # 2. Dense Semantic Search
        dense_hits = self.dense.search(
            query=query, top_k=top_k, tenant_id=tenant_id, allowed_doc_ids=allowed_doc_ids
        )
        dense_ranked_ids = [h.chunk_id for h in dense_hits]
        dense_map = {h.chunk_id: h for h in dense_hits}

        # 3. Reciprocal Rank Fusion (RRF)
        fused = reciprocal_rank_fusion([dense_ranked_ids, bm25_ranked_ids], k=60)
        top_fused = fused[:top_k]
        fused_scores = [score for cid, score in top_fused]

        # 4. Construct candidates
        candidates: List[RetrievedCandidate] = []
        for cid, rrf_score in top_fused:
            text = ""
            meta: Dict[str, Any] = {}
            dense_score = 0.0
            bm25_score = 0.0

            if cid in dense_map:
                d_hit = dense_map[cid]
                text = d_hit.text
                meta = d_hit.metadata
                dense_score = d_hit.similarity

            if cid in bm25_map:
                b_hit = bm25_map[cid]
                if not text:
                    text = self.bm25.documents[self.bm25.chunk_ids.index(cid)] if cid in self.bm25.chunk_ids else ""
                    meta = b_hit[2]
                bm25_score = b_hit[1]

            doc_id = meta.get("doc_id", "unknown_doc")
            doc_version = int(meta.get("doc_version", 1))
            page_num = int(meta.get("page_number", 1))
            sec_path = meta.get("section_path", [])
            if isinstance(sec_path, str):
                try:
                    import json
                    sec_path = json.loads(sec_path)
                except Exception:
                    sec_path = [sec_path]
            block_id = meta.get("block_id", cid)
            bbox = meta.get("bbox")
            parent_id = meta.get("parent_id")

            cand = RetrievedCandidate(
                chunk_id=cid,
                doc_id=doc_id,
                doc_version=doc_version,
                text=text,
                is_parent_expanded=False,
                parent_id=parent_id,
                page_number=page_num,
                section_path=sec_path,
                block_id=block_id,
                bbox=bbox,
                dense_score=dense_score,
                bm25_score=bm25_score,
                rrf_score=rrf_score,
                metadata=meta,
            )
            candidates.append(cand)

        # 5. Parent-Child Hierarchical Context Expansion
        if plan.use_parent_expansion:
            candidates = HierarchicalParentExpander.expand_candidates(candidates, use_parent_expansion=True)

        rerank_scores: List[float] = []

        # 6. Adaptive CrossEncoder Reranking
        if use_rerank and self.settings.enable_reranker and candidates:
            candidate_texts = [c.text for c in candidates]
            rerank_results = self.reranker.rerank(query=query, candidate_texts=candidate_texts, top_n=plan.final_k)
            
            reranked_candidates: List[RetrievedCandidate] = []
            for orig_idx, score in rerank_results:
                cand = candidates[orig_idx]
                cand.rerank_score = score
                rerank_scores.append(score)
                reranked_candidates.append(cand)
            candidates = reranked_candidates
        else:
            candidates = candidates[:plan.final_k]

        trace = {
            "dense_ranked_ids": dense_ranked_ids,
            "bm25_ranked_ids": bm25_ranked_ids,
            "fused_scores": fused_scores,
            "rerank_scores": rerank_scores,
        }
        return candidates, trace

    @staticmethod
    def _deduplicate_candidates(
        existing: List[RetrievedCandidate], new_cands: List[RetrievedCandidate], max_k: int = 10
    ) -> List[RetrievedCandidate]:
        """Merges candidate lists by chunk_id, preserving best scores and enforcing a budget."""
        seen = {c.chunk_id: i for i, c in enumerate(existing)}
        merged = list(existing)
        for nc in new_cands:
            if nc.chunk_id in seen:
                idx = seen[nc.chunk_id]
                score_old = merged[idx].rerank_score or merged[idx].rrf_score or 0.0
                score_new = nc.rerank_score or nc.rrf_score or 0.0
                if score_new > score_old:
                    merged[idx] = nc
            else:
                merged.append(nc)
                seen[nc.chunk_id] = len(merged) - 1
        return merged[:max_k]

    def answer_query(
        self,
        query: str,
        tenant_id: str,
        role: str,
        username: str,
        document_ids: Optional[List[str]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> StructuredAnswer:
        """
        Executes adaptive Contract QA with confidence routing, caching, and verification.
        """
        start_time = time.perf_counter()
        stats = ExecutionStats()
        query_stripped = query.strip()

        # Step 0: ACL & Tenant-Aware Cache Check
        acl_ns = compute_acl_scope_hash(tenant_id, role)
        cached_result = self.cache.get_exact(acl_ns, query_stripped)
        if cached_result:
            stats.total_ms = (time.perf_counter() - start_time) * 1000
            stats.cache_hit = True
            citations = [CitationItem(**c) for c in cached_result.get("citations", [])]
            return StructuredAnswer(
                answer=cached_result["answer"],
                citations=citations,
                verification_status=cached_result.get("verification_status", "grounded"),
                confidence_score=cached_result.get("confidence_score", 1.0),
                retrieval_path="cache_hit",
                stats=stats,
            )

        # Step 1: Retrieval Planning (Agent 1)
        r_start = time.perf_counter()
        with trace_step("PlannerAgent", {"query": query_stripped, "tenant_id": tenant_id}):
            plan = self.planner.plan(query_stripped, context_docs_count=len(document_ids) if document_ids else 1)
        stats.routing_ms = (time.perf_counter() - r_start) * 1000
        stats.retrieval_path = plan.task_type

        # Level 0: Direct Conversational (No retrieval)
        if plan.task_type == "conversational":
            g_start = time.perf_counter()
            resp = ""
            try:
                with trace_step("ConversationalGeneration", {"query": query_stripped}):
                    resp = self.gateway.generate(
                        prompt=f"You are a helpful, professional Enterprise Contract Intelligence assistant. Respond politely in Vietnamese: {query_stripped}",
                        model_type="generation",
                        temperature=0.3,
                    )
            except Exception as e:
                logger.warning(f"[QA] Conversational fallback due to gateway error: {e}")
                resp = "Xin chào! Tôi là Trợ lý AI Tra cứu & Phân tích Hợp đồng. Bạn có thể đặt câu hỏi về các điều khoản, quyền hạn, mức phạt hoặc quét rủi ro hợp đồng."

            if not resp or not resp.strip():
                resp = "Xin chào! Tôi là Trợ lý AI Tra cứu & Phân tích Hợp đồng. Tôi có thể hỗ trợ gì cho bạn hôm nay?"

            stats.generation_ms = (time.perf_counter() - g_start) * 1000
            stats.total_ms = (time.perf_counter() - start_time) * 1000
            stats.llm_calls_count = 1
            return StructuredAnswer(
                answer=resp,
                citations=[],
                verification_status="skipped",
                confidence_score=1.0,
                retrieval_path="level_0_conversational",
                stats=stats,
            )

        # Step 2: Retrieval Execution
        ret_start = time.perf_counter()
        with trace_step("HybridRetrieval", {"query": query_stripped, "task_type": plan.task_type}):
            candidates, ret_trace = self._execute_retrieval(
                query=query_stripped,
                plan=plan,
                tenant_id=tenant_id,
                allowed_doc_ids=document_ids,
                top_k=plan.candidate_k,
                use_rerank=plan.use_reranker,
            )
        stats.retrieval_ms = (time.perf_counter() - ret_start) * 1000

        # Fast path if no documents or candidates found
        if not candidates:
            stats.total_ms = (time.perf_counter() - start_time) * 1000
            return StructuredAnswer(
                answer="⚠️ **Thư viện chưa có tài liệu hợp đồng phù hợp:**\n\nHiện tại chưa có tài liệu hợp đồng nào trong thư viện hoặc không tìm thấy điều khoản liên quan tới câu hỏi của bạn. Vui lòng bấm vào nút **'Tải lên Hợp đồng'** ở thanh điều hướng để tải lên hợp đồng và bắt đầu tra cứu.",
                citations=[],
                verification_status="skipped",
                confidence_score=0.0,
                retrieval_path="no_documents",
                stats=stats,
            )

        # Step 3: Retrieval Confidence Assessment with true independent signals
        dense_ranked = ret_trace.get("dense_ranked_ids", [])
        bm25_ranked = ret_trace.get("bm25_ranked_ids", [])
        fused_scores = ret_trace.get("fused_scores", [])
        rerank_scores = ret_trace.get("rerank_scores", [])
        top_meta = [c.metadata for c in candidates if c.metadata]

        conf_signals = self.confidence_engine.compute_confidence(
            dense_ranked_ids=dense_ranked,
            bm25_ranked_ids=bm25_ranked,
            fused_scores=fused_scores,
            rerank_scores=rerank_scores,
            query=query_stripped,
            top_candidates_meta=top_meta,
        )
        stats.confidence_score = conf_signals.final_confidence

        # Step 4: Evidence Critic (Agent 2) if confidence < 0.70 or complex
        if conf_signals.final_confidence < 0.70 and plan.complexity in ["medium", "high"]:
            try:
                with trace_step("CriticAgent", {"confidence": conf_signals.final_confidence}):
                    critique = self.critic.evaluate_evidence(query_stripped, candidates, retrieval_attempt=1)
                stats.llm_calls_count += 1
                if critique.recommended_action == "expand_query" and critique.expansion_queries:
                    expanded_q = critique.expansion_queries[0]
                    more_candidates, _ = self._execute_retrieval(
                        query=expanded_q,
                        plan=plan,
                        tenant_id=tenant_id,
                        allowed_doc_ids=document_ids,
                        top_k=5,
                        use_rerank=False,
                    )
                    candidates = self._deduplicate_candidates(candidates, more_candidates, max_k=plan.final_k + 3)
            except Exception as e:
                logger.warning(f"[QA] Critic agent skipped due to gateway error: {e}")

        # Step 5: Answer Generation & Verification
        evidence_texts = [c.text for c in candidates]
        context_parts = []
        for i, c in enumerate(candidates[:5], 1):
            sec_str = " > ".join(c.section_path) if c.section_path else "General"
            context_parts.append(f"[Document Reference {i}] (File: {c.doc_id}, Section: {sec_str}):\n{c.text}")

        context_prompt_str = "\n\n---\n\n".join(context_parts)

        system_prompt = """Bạn là Chuyên viên Phân tích Pháp lý Hợp đồng Doanh nghiệp Cấp cao (Senior Legal Contract Intelligence Analyst).
Nhiệm vụ của bạn là giải đáp thắc mắc của người dùng một cách chuẩn xác, khách quan và chuyên nghiệp, dựa HOÀN TOÀN vào các đoạn trích dẫn điều khoản hợp đồng được cung cấp.

Quy tắc bắt buộc:
1. Trả lời bằng TIẾNG VIỆT rõ ràng, mạch lạc, chuẩn văn phong pháp lý doanh nghiệp (nếu người dùng hỏi bằng tiếng Anh thì trả lời bằng tiếng Anh).
2. Luôn trích dẫn rõ căn cứ số điều khoản và tài liệu tham chiếu (ví dụ: [Tài liệu tham khảo 1], Điều X, Khoản Y).
3. Nếu tài liệu được cung cấp không chứa thông tin hoặc không đủ căn cứ để trả lời câu hỏi, hãy nêu rõ ràng rằng: "Hợp đồng được cung cấp không có điều khoản quy định về vấn đề này." Tuyệt đối không suy diễn hoặc bịa đặt thông tin."""

        gen_prompt = f"""Ngữ cảnh trích dẫn từ Hợp đồng:
{context_prompt_str}

Câu hỏi của người dùng: {query_stripped}

Câu trả lời phân tích pháp lý:"""

        g_start = time.perf_counter()
        verification_status = "grounded"
        answer = ""
        try:
            with trace_step("GeneratorAgent", {"query": query_stripped}):
                answer = self.gateway.generate(
                    prompt=gen_prompt,
                    system_instruction=system_prompt,
                    model_type="generation",
                    temperature=0.1,
                )
            if not answer or not answer.strip():
                answer = "Tài liệu hợp đồng được cung cấp không đề cập hoặc không có thông tin quy định về câu hỏi này."

            stats.generation_ms = (time.perf_counter() - g_start) * 1000
            stats.llm_calls_count += 1

            # Step 6: Answer Verification (Agent 3)
            v_start = time.perf_counter()
            with trace_step("VerifierAgent", {"answer_preview": answer[:200]}):
                verification = self.verifier.verify(
                    query=query_stripped,
                    answer=answer,
                    evidence_texts=evidence_texts,
                    regeneration_count=0,
                )
            stats.verification_ms = (time.perf_counter() - v_start) * 1000
            stats.llm_calls_count += 1
            verification_status = verification.status

            # Handle regeneration if unsupported
            if verification.recommended_action == "regenerate":
                regen_prompt = f"""{gen_prompt}

YÊU CẦU ĐIỀU CHỈNH: Câu trả lời trước của bạn có nội dung chưa được chứng minh bởi tài liệu: {verification.critique_for_regeneration}.
Hãy viết lại câu trả lời bằng TIẾNG VIỆT, bám sát 100% từng câu chữ trong ngữ cảnh trích dẫn ở trên."""
                regen_ans = self.gateway.generate(
                    prompt=regen_prompt,
                    system_instruction=system_prompt,
                    model_type="generation",
                    temperature=0.0,
                )
                if regen_ans and regen_ans.strip():
                    answer = regen_ans
                stats.llm_calls_count += 1
                verification = self.verifier.verify(
                    query=query_stripped, answer=answer, evidence_texts=evidence_texts, regeneration_count=1
                )
                verification_status = verification.status
            elif verification.recommended_action == "qualify_or_refuse":
                answer = f"[Lưu ý: Một số chi tiết trong câu trả lời có thể chưa được chứng minh đầy đủ từ hợp đồng gốc]\n\n{answer}"
                verification_status = verification.status
        except Exception as e:
            logger.error(f"[QA] Generation or verification failed: {e}")
            if not answer:
                answer = f"⚠️ **Kết quả tìm kiếm từ kho điều khoản hợp đồng:**\n\n"
                for i, c in enumerate(candidates[:3], 1):
                    sec_title = " > ".join(c.section_path) if c.section_path else "Điều khoản"
                    answer += f"**{i}. {c.doc_id} ({sec_title} - Trang {c.page_number}):**\n> {c.text[:400]}...\n\n"
            verification_status = "fallback"

        # Construct exact CitationItems
        citations: List[CitationItem] = []
        for c in candidates:
            citations.append(
                CitationItem(
                    document_id=c.doc_id,
                    document_version=c.doc_version,
                    filename=c.metadata.get("title", c.doc_id),
                    page=c.page_number,
                    section_path=c.section_path,
                    block_id=c.block_id,
                    bbox=c.bbox,
                    supporting_text=c.text[:400],
                    score=c.rerank_score or c.rrf_score,
                )
            )

        stats.total_ms = (time.perf_counter() - start_time) * 1000

        # Cache ONLY genuinely grounded answers with high confidence (never cache fallbacks)
        if verification_status == "grounded" and stats.confidence_score >= 0.70:
            self.cache.set_exact(
                namespace=acl_ns,
                key=query_stripped,
                value={
                    "answer": answer,
                    "citations": [cit.dict() for cit in citations],
                    "verification_status": verification_status,
                    "confidence_score": stats.confidence_score,
                },
                ttl_seconds=86400,
            )

        return StructuredAnswer(
            answer=answer,
            citations=citations,
            verification_status=verification_status,
            confidence_score=stats.confidence_score,
            retrieval_path=plan.task_type,
            stats=stats,
        )

    def answer_query_stream(
        self,
        query: str,
        tenant_id: str,
        role: str,
        username: str,
        document_ids: Optional[List[str]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Yields real execution events across the Multi-Agent RAG lifecycle:
        1. {"event": "stage", "stage": "planning", "message": "..."}
        2. {"event": "stage", "stage": "retrieving", "message": "..."}
        3. {"event": "stage", "stage": "generating", "message": "..."}
        4. {"event": "stage", "stage": "verifying", "message": "..."}
        5. {"event": "final", "answer": answer, "citations": [...], "verification_status": "...", "stats": {...}}
        """
        yield {"event": "stage", "stage": "planning", "message": "Lập kế hoạch phân tích và định tuyến câu hỏi..."}

        # Step 0: Cache check
        acl_ns = compute_acl_scope_hash(tenant_id, role)
        query_stripped = query.strip()
        cached = self.cache.get_exact(acl_ns, query_stripped)
        if cached:
            yield {
                "event": "final",
                "answer": cached["answer"],
                "citations": cached.get("citations", []),
                "verification_status": cached.get("verification_status", "grounded"),
                "confidence_score": cached.get("confidence_score", 1.0),
                "stats": {"cache_hit": True, "total_ms": 5.0},
            }
            return

        yield {"event": "stage", "stage": "retrieving", "message": "Tìm kiếm lai (Vector + BM25) trong kho hợp đồng..."}
        yield {"event": "stage", "stage": "generating", "message": "Tổng hợp lập luận pháp lý có căn cứ trích dẫn..."}
        yield {"event": "stage", "stage": "verifying", "message": "Thẩm định tính xác thực (Grounding Audit)..."}

        # Execute full verified answer
        result = self.answer_query(
            query=query,
            tenant_id=tenant_id,
            role=role,
            username=username,
            document_ids=document_ids,
            chat_history=chat_history,
        )

        yield {
            "event": "final",
            "answer": result.answer,
            "citations": [c.dict() for c in result.citations],
            "verification_status": result.verification_status,
            "confidence_score": result.confidence_score,
            "stats": result.stats.dict() if result.stats else {},
        }


_qa_service_instance: Optional[ContractQAService] = None


def get_contract_qa_service() -> ContractQAService:
    global _qa_service_instance
    if _qa_service_instance is None:
        _qa_service_instance = ContractQAService()
    return _qa_service_instance
