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
from typing import List, Dict, Any, Optional, Iterator

from backend.app.core.config import get_settings
from backend.app.providers.gemini_gateway import get_gemini_gateway, GeminiAPIGateway
from backend.app.providers.reranker import get_reranker
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
    ) -> List[RetrievedCandidate]:
        """Runs BM25 + Dense + RRF + Parent Expansion + CrossEncoder Reranking."""
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

        # 4. Construct candidates
        candidates: List[RetrievedCandidate] = []
        for cid, rrf_score in top_fused:
            meta = {}
            text = ""
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

        # 6. Adaptive CrossEncoder Reranking
        if use_rerank and self.settings.enable_reranker and candidates:
            candidate_texts = [c.text for c in candidates]
            rerank_results = self.reranker.rerank(query=query, candidate_texts=candidate_texts, top_n=plan.final_k)
            
            reranked_candidates: List[RetrievedCandidate] = []
            for orig_idx, score in rerank_results:
                cand = candidates[orig_idx]
                cand.rerank_score = score
                reranked_candidates.append(cand)
            return reranked_candidates

        return candidates[:plan.final_k]

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
        plan = self.planner.plan(query_stripped, context_docs_count=len(document_ids) if document_ids else 1)
        stats.routing_ms = (time.perf_counter() - r_start) * 1000
        stats.retrieval_path = plan.task_type

        # Level 0: Direct Conversational (No retrieval)
        if plan.task_type == "conversational":
            g_start = time.perf_counter()
            resp = ""
            try:
                resp = self.gateway.generate(
                    prompt=f"You are a helpful, professional Enterprise Contract Intelligence assistant. Respond politely to: {query_stripped}",
                    model_type="generation",
                    temperature=0.3,
                )
            except Exception as e:
                logger.warning(f"[QA] Conversational fallback due to gateway rate limit/error: {e}")
                resp = "Hello! I am your Enterprise Contract Intelligence assistant. You can ask detailed questions about your uploaded agreements, compare clauses, or review contract risks."

            if not resp or not resp.strip():
                resp = "Hello! I am your Enterprise Contract Intelligence assistant. How can I assist you with your agreements today?"

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
        candidates = self._execute_retrieval(
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
                answer="⚠️ **Thư viện chưa có tài liệu hợp đồng phù hợp:**\n\nHiện tại chưa có tài liệu hợp đồng nào trong thư viện hoặc không tìm thấy điều khoản liên quan tới câu hỏi của bạn. Vui lòng bấm vào nút **'Upload'** ở thanh điều hướng trên cùng để tải lên hợp đồng (PDF, DOCX, Markdown, JSON) và bắt đầu tra cứu.",
                citations=[],
                verification_status="skipped",
                confidence_score=0.0,
                retrieval_path="no_documents",
                stats=stats,
            )

        # Step 3: Retrieval Confidence Assessment
        dense_ids = [c.chunk_id for c in candidates]
        fused_scores = [c.rrf_score for c in candidates]
        rerank_scores = [c.rerank_score for c in candidates if c.rerank_score is not None]
        top_meta = [c.metadata for c in candidates if c.metadata]

        conf_signals = self.confidence_engine.compute_confidence(
            dense_ranked_ids=dense_ids,
            bm25_ranked_ids=dense_ids,
            fused_scores=fused_scores,
            rerank_scores=rerank_scores,
            query=query_stripped,
            top_candidates_meta=top_meta,
        )
        stats.confidence_score = conf_signals.final_confidence

        # Step 4: Evidence Critic (Agent 2) if confidence < 0.70 or complex
        if conf_signals.final_confidence < 0.70 and plan.complexity in ["medium", "high"]:
            try:
                critique = self.critic.evaluate_evidence(query_stripped, candidates, retrieval_attempt=1)
                stats.llm_calls_count += 1
                if critique.recommended_action == "expand_query" and critique.expansion_queries:
                    expanded_q = critique.expansion_queries[0]
                    more_candidates = self._execute_retrieval(
                        query=expanded_q,
                        plan=plan,
                        tenant_id=tenant_id,
                        allowed_doc_ids=document_ids,
                        top_k=5,
                        use_rerank=False,
                    )
                    candidates.extend(more_candidates)
            except Exception as e:
                logger.warning(f"[QA] Critic agent skipped due to gateway error: {e}")

        # Step 5: Answer Generation & Verification
        evidence_texts = [c.text for c in candidates]
        context_parts = []
        for i, c in enumerate(candidates[:5], 1):
            sec_str = " > ".join(c.section_path) if c.section_path else "General"
            context_parts.append(f"[Document Reference {i}] (File: {c.doc_id}, Section: {sec_str}):\n{c.text}")

        context_prompt_str = "\n\n---\n\n".join(context_parts)

        system_prompt = """You are a Senior Legal Contract Intelligence Analyst.
Answer the user's question with precise factual accuracy based solely on the provided reference excerpts.
Always cite the exact supporting document and section (e.g. [Document Reference X])."""

        gen_prompt = f"""Reference Contract Context:
{context_prompt_str}

User Question: {query_stripped}

Answer:"""

        g_start = time.perf_counter()
        v_status = "grounded"
        answer = ""
        try:
            answer = self.gateway.generate(
                prompt=gen_prompt,
                system_instruction=system_prompt,
                model_type="generation",
                temperature=0.1,
            )
            if not answer or not answer.strip():
                answer = "The provided contract documents do not specify or mention information regarding this query."

            stats.generation_ms = (time.perf_counter() - g_start) * 1000
            stats.llm_calls_count += 1

            # Step 6: Answer Verification (Agent 3)
            v_start = time.perf_counter()
            verification = self.verifier.verify(
                query=query_stripped,
                answer=answer,
                evidence_texts=evidence_texts,
                regeneration_count=0,
            )
            stats.verification_ms = (time.perf_counter() - v_start) * 1000
            stats.llm_calls_count += 1
            v_status = verification.status

            # Handle regeneration if unsupported
            if verification.recommended_action == "regenerate":
                regen_prompt = f"""{gen_prompt}

IMPORTANT CORRECTION: Your previous answer contained unsupported statements: {verification.critique_for_regeneration}.
Strictly adhere only to the verbatim factual context provided above."""
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
                v_status = verification.status
            elif verification.recommended_action == "qualify_or_refuse":
                answer = f"[Notice: Certain details in the contract could not be fully substantiated.]\n\n{answer}"
                v_status = verification.status
        except Exception as e:
            logger.error(f"[QA] Generation failed: {e}")
            if not answer:
                answer = f"⚠️ **Kết quả tìm kiếm từ kho điều khoản hợp đồng:**\n\n"
                for i, c in enumerate(candidates[:3], 1):
                    sec_title = " > ".join(c.section_path) if c.section_path else "Điều khoản"
                    answer += f"**{i}. {c.doc_id} ({sec_title} - Trang {c.page_number}):**\n> {c.text[:400]}...\n\n"
            v_status = "grounded"

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

        # Cache high-confidence grounded answers
        if verification.status == "grounded" and stats.confidence_score >= 0.70:
            self.cache.set_exact(
                namespace=acl_ns,
                key=query_stripped,
                value={
                    "answer": answer,
                    "citations": [cit.dict() for cit in citations],
                    "verification_status": verification.status,
                    "confidence_score": stats.confidence_score,
                },
                ttl_seconds=86400,
            )

        return StructuredAnswer(
            answer=answer,
            citations=citations,
            verification_status=verification.status,
            confidence_score=stats.confidence_score,
            retrieval_path=plan.task_type,
            stats=stats,
        )


_qa_service_instance: Optional[ContractQAService] = None


def get_contract_qa_service() -> ContractQAService:
    global _qa_service_instance
    if _qa_service_instance is None:
        _qa_service_instance = ContractQAService()
    return _qa_service_instance
