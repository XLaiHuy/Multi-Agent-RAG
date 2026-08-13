import pytest
from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "Agentic RAG API is running" in data["message"]

def test_unauthorized_chat_access():
    response = client.post("/api/chat", json={"query": "test"})
    # Expect 401 Unauthorized without Bearer Token
    assert response.status_code in [401, 403]
