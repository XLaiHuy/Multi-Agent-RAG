#!/usr/bin/env python3
"""
Unit tests verifying that the global exception handler masks sensitive details in production.
"""
import asyncio
import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi import Request
from backend.app.main import global_exception_handler


def test_global_exception_handler_masks_details_in_production():
    mock_request = MagicMock(spec=Request)
    mock_request.url.path = "/api/v1/qa/chat"
    exc = ValueError("Database password leaked in internal error: postgres://user:secret@internal:5432/db")

    with patch("backend.app.main.settings") as mock_settings:
        mock_settings.environment = "production"
        response = asyncio.run(global_exception_handler(mock_request, exc))
        assert response.status_code == 500
        body = json.loads(response.body.decode("utf-8"))
        assert body == {"error": "Internal Server Error"}
        assert "detail" not in body
        assert "password" not in str(body)


def test_global_exception_handler_includes_detail_in_development():
    mock_request = MagicMock(spec=Request)
    mock_request.url.path = "/api/v1/qa/chat"
    exc = ValueError("Sample development debug detail")

    with patch("backend.app.main.settings") as mock_settings:
        mock_settings.environment = "development"
        response = asyncio.run(global_exception_handler(mock_request, exc))
        assert response.status_code == 500
        body = json.loads(response.body.decode("utf-8"))
        assert body["error"] == "Internal Server Error"
        assert body["detail"] == "Sample development debug detail"
