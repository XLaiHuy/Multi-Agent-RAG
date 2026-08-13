import pytest
from app.core.db import (
    get_user_by_username,
    verify_password,
    save_message,
    get_conversation_messages,
    list_user_conversations,
    delete_conversation,
    store_semantic_cache,
    lookup_semantic_cache,
)


def test_sqlite_user_auth():
    admin = get_user_by_username("admin")
    assert admin is not None
    assert admin["role"] == "admin"
    assert verify_password("admin", admin["hashed_password"]) is True
    assert verify_password("wrong_pass", admin["hashed_password"]) is False


import uuid

def test_conversation_persistence():
    conv_id = f"test_conv_{uuid.uuid4().hex[:8]}"
    username = "admin"
    
    # Save user message
    save_message(conv_id=conv_id, username=username, role="user", content="Test question 1")
    # Save assistant message
    save_message(conv_id=conv_id, username=username, role="assistant", content="Test answer 1", sources=[{"chunk_id": "c1"}])

    # Check conversation list
    convs = list_user_conversations(username)
    assert any(c["id"] == conv_id for c in convs)

    # Check messages
    messages = get_conversation_messages(conv_id)
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["sources"] == [{"chunk_id": "c1"}]

    # Clean up
    delete_conversation(conv_id, username)
    convs_after = list_user_conversations(username)
    assert not any(c["id"] == conv_id for c in convs_after)


def test_semantic_cache():
    query = "Học phí đại học là bao nhiêu?"
    # Dummy 768-d vector
    dummy_emb = [0.1] * 768
    answer = "Học phí là 20 triệu/năm."

    store_semantic_cache(query=query, query_embedding=dummy_emb, answer=answer, sources=[{"chunk_id": "s1"}])

    # Lookup with exact match
    res = lookup_semantic_cache(dummy_emb, similarity_threshold=0.95)
    assert res is not None
    assert res["answer"] == answer
    assert res["similarity"] >= 0.99
