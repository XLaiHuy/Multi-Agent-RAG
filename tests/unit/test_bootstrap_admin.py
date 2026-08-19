#!/usr/bin/env python3
"""
Unit tests for production admin bootstrap script.
"""
import os
import pytest
from unittest.mock import patch, MagicMock
from scripts.bootstrap_admin import bootstrap_admin


def test_bootstrap_admin_fails_on_missing_password(monkeypatch):
    monkeypatch.delenv("BOOTSTRAP_ADMIN_PASSWORD", raising=False)
    monkeypatch.setenv("BOOTSTRAP_ADMIN_USERNAME", "admin")
    result = bootstrap_admin()
    assert result == 1


def test_bootstrap_admin_fails_on_short_password(monkeypatch):
    monkeypatch.setenv("BOOTSTRAP_ADMIN_PASSWORD", "short")
    monkeypatch.setenv("BOOTSTRAP_ADMIN_USERNAME", "admin")
    result = bootstrap_admin()
    assert result == 1


def test_bootstrap_admin_creates_tenant_and_user_when_not_exist(monkeypatch):
    monkeypatch.setenv("BOOTSTRAP_ADMIN_USERNAME", "prod_admin")
    monkeypatch.setenv("BOOTSTRAP_ADMIN_PASSWORD", "SuperSecurePassword123")
    monkeypatch.setenv("BOOTSTRAP_ADMIN_FULL_NAME", "Production Administrator")
    monkeypatch.setenv("BOOTSTRAP_ADMIN_TENANT_ID", "acme_tenant")

    mock_db = MagicMock()
    # Tenant query returns None -> create; User query returns None -> create
    mock_db.query.return_value.filter.return_value.first.return_value = None

    with patch("scripts.bootstrap_admin.SessionLocal") as mock_session_local, \
         patch("scripts.bootstrap_admin.Base.metadata.create_all"), \
         patch("scripts.bootstrap_admin.UserRepository.get_by_username", return_value=None), \
         patch("scripts.bootstrap_admin.UserRepository.create_user") as mock_create_user:

        mock_session_local.return_value.__enter__.return_value = mock_db
        mock_user = MagicMock()
        mock_user.username = "prod_admin"
        mock_user.id = "user_123"
        mock_create_user.return_value = mock_user

        result = bootstrap_admin()
        assert result == 0
        mock_db.add.assert_called_once()
        mock_create_user.assert_called_once()


def test_bootstrap_admin_is_idempotent_if_user_already_exists(monkeypatch):
    monkeypatch.setenv("BOOTSTRAP_ADMIN_USERNAME", "existing_admin")
    monkeypatch.setenv("BOOTSTRAP_ADMIN_PASSWORD", "SuperSecurePassword123")
    monkeypatch.setenv("BOOTSTRAP_ADMIN_TENANT_ID", "existing_tenant")

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = MagicMock()
    existing_user = MagicMock()
    existing_user.username = "existing_admin"
    existing_user.tenant_id = "existing_tenant"

    with patch("scripts.bootstrap_admin.SessionLocal") as mock_session_local, \
         patch("scripts.bootstrap_admin.Base.metadata.create_all"), \
         patch("scripts.bootstrap_admin.UserRepository.get_by_username", return_value=existing_user), \
         patch("scripts.bootstrap_admin.UserRepository.create_user") as mock_create_user:

        mock_session_local.return_value.__enter__.return_value = mock_db
        result = bootstrap_admin()
        assert result == 0
        mock_create_user.assert_not_called()
