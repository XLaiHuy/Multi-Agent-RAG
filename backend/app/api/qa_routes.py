"""
Contract QA & Conversation API Endpoints.
Provides synchronous and SSE streaming QA, verified citations, and anti-IDOR conversation persistence.
"""
import json
import uuid
import asyncio
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sse_starlette.sse import EventSourceResponse
from sqlalchemy.orm import Session

from backend.app.core.security import get_current_user, UserTokenData
from backend.app.persistence.database import get_db, ConversationRepository
from backend.app.application.contract_qa import get_contract_qa_service
from backend.app.domain.schemas import (
    ContractQARequest, ContractQAResponse, CitationItem, ExecutionStats
)

qa_router = APIRouter(prefix="/chat", tags=["Contract QA"])
conversation_router = APIRouter(prefix="/conversations", tags=["Conversations"])


# --- Synchronous QA Endpoint ---

@qa_router.post("", response_model=ContractQAResponse)
def chat_sync(
    request_data: ContractQARequest,
    current_user: UserTokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Synchronous Contract QA:
    Executes Adaptive Multi-Agent RAG pipeline and persists interaction to database.
    """
    query = request_data.query.strip()
    if not query:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query cannot be empty.")

    conv_id = request_data.conv_id or f"conv_{uuid.uuid4().hex[:12]}"
    qa_service = get_contract_qa_service()

    # Save user message
    ConversationRepository.save_message(
        db=db,
        conv_id=conv_id,
        username=current_user.username,
        tenant_id=current_user.tenant_id,
        role="user",
        content=query,
    )

    structured_answer = qa_service.answer_query(
        query=query,
        tenant_id=current_user.tenant_id,
        role=current_user.role,
        username=current_user.username,
        document_ids=request_data.document_ids,
        chat_history=[m.dict() for m in request_data.chat_history] if request_data.chat_history else None,
    )

    # Save assistant response with citations and stats
    citations_data = [c.dict() for c in structured_answer.citations]
    ConversationRepository.save_message(
        db=db,
        conv_id=conv_id,
        username=current_user.username,
        tenant_id=current_user.tenant_id,
        role="assistant",
        content=structured_answer.answer,
        citations=citations_data,
        retrieval_path=structured_answer.retrieval_path,
        verification_status=structured_answer.verification_status,
        latency_ms=structured_answer.stats.total_ms if structured_answer.stats else 0.0,
    )

    return ContractQAResponse(
        conv_id=conv_id,
        answer=structured_answer.answer,
        citations=structured_answer.citations,
        verification_status=structured_answer.verification_status,
        stats=structured_answer.stats or ExecutionStats(),
    )


# --- Streaming QA Endpoint ---

@qa_router.post("/stream")
async def chat_stream(
    request_data: ContractQARequest,
    current_user: UserTokenData = Depends(get_current_user),
):
    """
    SSE Streaming Contract QA:
    Yields progressive status updates, sources/citations, and token-by-token answer stream.
    """
    query = request_data.query.strip()
    if not query:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query cannot be empty.")

    conv_id = request_data.conv_id or f"conv_{uuid.uuid4().hex[:12]}"
    qa_service = get_contract_qa_service()

    async def event_generator():
        yield {"event": "status", "data": json.dumps({"message": "🔍 Evaluating complexity & retrieval plan..."})}
        await asyncio.sleep(0.05)

        # Run synchronous QA in threadpool
        res = await asyncio.to_thread(
            qa_service.answer_query,
            query=query,
            tenant_id=current_user.tenant_id,
            role=current_user.role,
            username=current_user.username,
            document_ids=request_data.document_ids,
            chat_history=[m.dict() for m in request_data.chat_history] if request_data.chat_history else None,
        )

        # Yield citations
        citations_list = [c.dict() for c in res.citations]
        yield {"event": "citations", "data": json.dumps({"citations": citations_list, "verification_status": res.verification_status})}

        # Stream answer tokens
        words = res.answer.split(" ")
        for w in words:
            yield {"event": "token", "data": json.dumps({"token": w + " "})}
            await asyncio.sleep(0.015)

        yield {"event": "stats", "data": json.dumps(res.stats.dict() if res.stats else {})}
        yield {"event": "done", "data": "[DONE]"}

    return EventSourceResponse(event_generator())


# --- Anti-IDOR Conversation History Endpoints ---

@conversation_router.get("")
def list_conversations(
    current_user: UserTokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List conversations belonging strictly to authenticated user and tenant."""
    convs = ConversationRepository.list_user_conversations(db, current_user.username, current_user.tenant_id)
    return {
        "conversations": [
            {
                "id": c.id,
                "title": c.title,
                "task_type": c.task_type,
                "created_at": c.created_at.isoformat(),
                "updated_at": c.updated_at.isoformat(),
            }
            for c in convs
        ]
    }


@conversation_router.get("/{conv_id}/messages")
def get_conversation_messages(
    conv_id: str,
    current_user: UserTokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    ANTI-IDOR PROTECTED:
    Returns messages only if authenticated user is the legitimate owner.
    """
    messages = ConversationRepository.get_conversation_messages_safe(
        db, conv_id, current_user.username, current_user.tenant_id
    )
    if messages is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Conversation not found or you lack permission to view this conversation.",
        )

    return {
        "conv_id": conv_id,
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "citations": m.citations_json,
                "retrieval_path": m.retrieval_path,
                "verification_status": m.verification_status,
                "created_at": m.created_at.isoformat(),
            }
            for m in messages
        ],
    }


@conversation_router.delete("/{conv_id}")
def delete_conversation(
    conv_id: str,
    current_user: UserTokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Deletes conversation if owned by current user."""
    success = ConversationRepository.delete_conversation_safe(
        db, conv_id, current_user.username, current_user.tenant_id
    )
    if not success:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Conversation not found or you lack permission to delete this conversation.",
        )
    return {"status": "success", "message": f"Deleted conversation {conv_id}"}
