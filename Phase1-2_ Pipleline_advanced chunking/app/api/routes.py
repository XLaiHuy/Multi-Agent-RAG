"""
API Routes - Chat endpoints with Agentic RAG Graph, Semantic Caching & SQLite Conversation Persistence
"""
import time
import json
import uuid
import asyncio
import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from app.api.schemas import ChatRequest, ChatResponse, SourceChunk
from app.api.deps import get_rag_app, get_cache, limiter
from app.api.auth import get_current_user
from app.core.db import (
    list_user_conversations,
    get_conversation_messages,
    save_message,
    delete_conversation,
    lookup_semantic_cache,
    store_semantic_cache,
)

logger = logging.getLogger("api")
router = APIRouter()


# --- 1. Conversation History Endpoints ---

@router.get("/conversations", tags=["Conversations"])
def get_conversations(current_user: dict = Depends(get_current_user)):
    """Lấy danh sách các phiên chat của người dùng hiện tại từ SQLite."""
    username = current_user.get("username", "admin")
    convs = list_user_conversations(username)
    return {"conversations": convs}


@router.get("/conversations/{conv_id}/messages", tags=["Conversations"])
def get_messages(conv_id: str, current_user: dict = Depends(get_current_user)):
    """Lấy toàn bộ lịch sử tin nhắn trong một phiên chat."""
    messages = get_conversation_messages(conv_id)
    return {"conv_id": conv_id, "messages": messages}


@router.delete("/conversations/{conv_id}", tags=["Conversations"])
def remove_conversation(conv_id: str, current_user: dict = Depends(get_current_user)):
    """Xóa một phiên hội thoại."""
    username = current_user.get("username", "admin")
    delete_conversation(conv_id, username)
    return {"status": "success", "message": f"Deleted conversation {conv_id}"}


# --- 2. Synchronous Chat Endpoint ---

@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
@limiter.limit("15/minute")
def chat_sync(
    request: Request,
    chat_request: ChatRequest,
    rag_app=Depends(get_rag_app),
    cache=Depends(get_cache),
    current_user: dict = Depends(get_current_user),
):
    """Synchronous chat endpoint: Chạy Agentic RAG Graph, trả về 1 lần."""
    query = chat_request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    cache_key = f"sync_{query.lower()}"
    if cache_key in cache:
        cached_response = cache[cache_key]
        cached_response.cached = True
        return cached_response

    start_time = time.perf_counter()

    try:
        chat_history = [{"role": m.role, "content": m.content} for m in chat_request.chat_history]
        output = rag_app.invoke({
            "query": query,
            "original_query": query,
            "chat_history": chat_history,
            "rewrite_count": 0,
        })

        answer = output.get("answer", "")
        chunks = output.get("chunks", [])
        ver_status = output.get("verification_status", "grounded")

        sources = [
            SourceChunk(chunk_id=c.get("chunk_id", ""), text=c.get("text", ""))
            for c in chunks if isinstance(c, dict)
        ]

        latency = (time.perf_counter() - start_time) * 1000
        response = ChatResponse(
            answer=answer, sources=sources,
            verification_status=ver_status,
            processing_time_ms=latency, cached=False,
        )
        cache[cache_key] = response
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- 3. Streaming Chat Endpoint with Semantic Cache & SQLite Persistence ---

@router.post("/chat/stream", tags=["Chat"])
@limiter.limit("15/minute")
async def chat_stream(
    request: Request,
    chat_request: ChatRequest,
    rag_app=Depends(get_rag_app),
    current_user: dict = Depends(get_current_user),
):
    """
    Streaming chat endpoint:
    1. Kiểm tra Semantic Cache (<10ms)
    2. Nếu miss: Chạy Multi-Query Expansion & Hybrid Retrieval & Stream token
    3. Tự động lưu trữ câu hỏi và câu trả lời vào SQLite Database & Semantic Cache
    """
    query = chat_request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    username = current_user.get("username", "admin")
    department = current_user.get("role", "admin")
    where_filter = None if department == "admin" else {"department": department}
    chat_history = [{"role": m.role, "content": m.content} for m in chat_request.chat_history]
    conv_id = chat_request.conv_id or str(uuid.uuid4())

    async def event_generator():
        # Lưu câu hỏi của user vào database
        save_message(conv_id=conv_id, username=username, role="user", content=query)

        # --- Phase 0: Semantic Cache Check ---
        yield {"event": "status", "data": json.dumps({"message": "🔍 Kiểm tra Semantic Cache..."})}
        
        try:
            from app.graph.agentic_rag import get_generator, get_vector_retriever, get_hybrid_retriever
            vector_retriever = get_vector_retriever()
            
            # Embed query for cache lookup
            query_emb = await asyncio.to_thread(vector_retriever.embedder.embed_query, query)
            cached_result = lookup_semantic_cache(query_emb, similarity_threshold=0.96)

            if cached_result:
                sim_pct = int(cached_result["similarity"] * 100)
                yield {"event": "status", "data": json.dumps({"message": f"⚡ Phản hồi từ Semantic Cache ({sim_pct}% tương đồng, <10ms)..."})}
                
                # Gửi sources từ cache
                cached_sources = cached_result.get("sources", [])
                yield {"event": "sources", "data": json.dumps({"sources": cached_sources})}

                # Stream cached answer
                cached_answer = cached_result.get("answer", "")
                words = cached_answer.split(" ")
                for word in words:
                    yield {"event": "message", "data": json.dumps({"token": word + " "})}
                    await asyncio.sleep(0.015)

                # Lưu assistant message vào DB
                save_message(conv_id=conv_id, username=username, role="assistant", content=cached_answer, sources=cached_sources)
                yield {"event": "done", "data": "[DONE]"}
                return

        except Exception as e:
            logger.warning(f"Semantic Cache check bypassed: {e}")

        # --- Phase 1: Routing + Retrieval (Agentic RAG) ---
        yield {"event": "status", "data": json.dumps({"message": "🔍 Phân tích câu hỏi (Agent Router)..."})}

        try:
            generator = get_generator()
            decision = await asyncio.to_thread(generator.analyze_intent_and_decide, query)
            action = decision.get("action", "retrieve_hybrid")

            chunks = []
            sources = []

            if action == "direct_answer":
                yield {"event": "status", "data": json.dumps({"message": "💬 Trả lời trực tiếp..."})}
            else:
                yield {"event": "status", "data": json.dumps({"message": "🔍 Mở rộng truy vấn đa chiều (Multi-Query Expansion)..."})}
                expanded_queries = await asyncio.to_thread(generator.expand_query, query)
                
                yield {"event": "status", "data": json.dumps({"message": f"📚 Đang tìm kiếm & Hợp nhất ngữ cảnh ({len(expanded_queries)} góc nhìn)..."})}
                retriever = get_hybrid_retriever()
                search_res = await asyncio.to_thread(
                    retriever.multi_query_search, queries=expanded_queries, top_k=14, where=where_filter
                )

                chunks = [
                    {"chunk_id": r.chunk_id, "text": r.text, "source": r.metadata.get("filename", r.metadata.get("source", "Tài liệu tham khảo"))}
                    for r in search_res
                ]
                sources = [{"chunk_id": c["chunk_id"], "text": c["text"], "source": c["source"]} for c in chunks]

                # Gửi sources về frontend
                yield {"event": "sources", "data": json.dumps({"sources": sources})}

                # Grade documents
                yield {"event": "status", "data": json.dumps({"message": "✅ Đánh giá độ liên quan tài liệu..."})}
                grade = await asyncio.to_thread(generator.grade_relevance, query=query, chunks=chunks)

                if grade == "not_relevant":
                    yield {"event": "status", "data": json.dumps({"message": "🔄 Viết lại câu hỏi để tìm kiếm lại..."})}
                    new_query = await asyncio.to_thread(generator.rewrite_query, query)
                    search_res2 = await asyncio.to_thread(
                        retriever.search, query=new_query, top_k=12,
                        use_rerank=False, where=where_filter
                    )
                    chunks = [
                        {"chunk_id": r.chunk_id, "text": r.text, "source": r.metadata.get("source", "unknown")}
                        for r in search_res2
                    ]
                    sources = [{"chunk_id": c["chunk_id"], "text": c["text"]} for c in chunks]
                    yield {"event": "sources", "data": json.dumps({"sources": sources})}

            # --- Phase 2: Stream generation token by token ---
            yield {"event": "status", "data": json.dumps({"message": "✍️ Đang viết câu trả lời..."})}

            history_text = ""
            if chat_history:
                lines = []
                for turn in chat_history[-10:]:
                    role_label = "Người dùng" if turn.get("role") == "user" else "Trợ lý"
                    lines.append(f"{role_label}: {turn.get('content', '')}")
                history_text = "\n".join(lines)

            if action == "direct_answer":
                answer_stream = generator.generate_answer_stream(
                    query=query, chunks=[],
                    direct_answer=True, chat_history=history_text
                )
            else:
                answer_stream = generator.generate_answer_stream(
                    query=query, chunks=chunks, chat_history=history_text
                )

            full_answer_pieces = []
            for token in answer_stream:
                if token:
                    full_answer_pieces.append(token)
                    yield {"event": "message", "data": json.dumps({"token": token})}

            full_answer = "".join(full_answer_pieces)

            # Lưu câu trả lời vào SQLite database
            save_message(conv_id=conv_id, username=username, role="assistant", content=full_answer, sources=sources)

            # Lưu vào Semantic Cache nếu có câu trả lời chất lượng
            if action != "direct_answer" and len(full_answer) > 20:
                try:
                    store_semantic_cache(query=query, query_embedding=query_emb, answer=full_answer, sources=sources)
                except Exception as e:
                    logger.warning(f"Could not store into semantic cache: {e}")

            yield {"event": "done", "data": "[DONE]"}

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield {"event": "error", "data": json.dumps({"detail": str(e)})}

    return EventSourceResponse(event_generator())
