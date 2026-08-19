#!/usr/bin/env python3
"""
Unit tests for the /ready database connectivity probe.
"""
import pytest
from unittest.mock import patch, MagicMock
from backend.app.main import readiness_check


def test_readiness_probe_returns_200_when_database_connected():
    with patch("backend.app.persistence.database.engine.connect") as mock_connect:
        mock_conn = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn

        resp = readiness_check()
        assert resp == {"status": "ready", "database": "connected"}


def test_readiness_probe_returns_503_when_database_fails():
    with patch("backend.app.persistence.database.engine.connect") as mock_connect:
        mock_connect.side_effect = Exception("Connection refused to postgresql server")

        resp = readiness_check()
        assert resp.status_code == 503
        import json
        body = json.loads(resp.body.decode("utf-8"))
        assert body == {"status": "unavailable", "database": "disconnected"}
