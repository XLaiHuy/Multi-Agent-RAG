#!/usr/bin/env python3
"""
Unit tests for Database URL normalization and PostgreSQL / SQLite driver selection.
"""
import pytest
from backend.app.persistence.database import normalize_database_url


def test_sqlite_url_remains_unchanged():
    sqlite_url = "sqlite:///./data/contracts.db"
    assert normalize_database_url(sqlite_url) == sqlite_url


def test_postgres_protocol_normalized_to_psycopg3():
    raw_url = "postgres://user:secret@host.railway.internal:5432/railway"
    expected = "postgresql+psycopg://user:secret@host.railway.internal:5432/railway"
    assert normalize_database_url(raw_url) == expected


def test_postgresql_protocol_normalized_to_psycopg3():
    raw_url = "postgresql://user:secret@host.railway.internal:5432/railway"
    expected = "postgresql+psycopg://user:secret@host.railway.internal:5432/railway"
    assert normalize_database_url(raw_url) == expected


def test_explicit_psycopg_url_preserved():
    explicit_url = "postgresql+psycopg://user:secret@host.railway.internal:5432/railway"
    assert normalize_database_url(explicit_url) == explicit_url


def test_empty_or_none_url_handled_safely():
    assert normalize_database_url("") == ""
    assert normalize_database_url(None) is None
