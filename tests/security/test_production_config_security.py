#!/usr/bin/env python3
"""
Unit tests for production configuration validation and security gates.
"""
import pytest
from backend.app.core.config import Settings


def test_development_mode_allows_default_secret():
    s = Settings(
        _env_file=None,
        environment="development",
        jwt_secret_key="dev_insecure_jwt_secret_key_change_in_production_1234567890",
        allowed_origins="http://localhost:5173",
    )
    # Should not raise
    s.validate_security()


def test_production_rejects_missing_or_short_jwt_secret():
    s_empty = Settings(
        _env_file=None,
        environment="production",
        jwt_secret_key="",
        gemini_api_key="valid_gemini_api_key",
        allowed_origins="https://frontend.up.railway.app",
    )
    with pytest.raises(ValueError, match="JWT_SECRET_KEY must be >= 32"):
        s_empty.validate_security()

    s_short = Settings(
        _env_file=None,
        environment="production",
        jwt_secret_key="too_short_secret",
        gemini_api_key="valid_gemini_api_key",
        allowed_origins="https://frontend.up.railway.app",
    )
    with pytest.raises(ValueError, match="JWT_SECRET_KEY must be >= 32"):
        s_short.validate_security()


def test_production_rejects_default_dev_jwt_secret():
    s_dev_secret = Settings(
        _env_file=None,
        environment="production",
        jwt_secret_key="dev_insecure_jwt_secret_key_change_in_production_1234567890",
        gemini_api_key="valid_gemini_api_key",
        allowed_origins="https://frontend.up.railway.app",
    )
    with pytest.raises(ValueError, match="must not use the built-in development default secret"):
        s_dev_secret.validate_security()


def test_production_rejects_missing_gemini_api_key_when_gemini_provider():
    s_no_gemini = Settings(
        _env_file=None,
        environment="production",
        jwt_secret_key="a_very_secure_production_secret_key_1234567890!",
        llm_provider="gemini",
        gemini_api_key="",
        allowed_origins="https://frontend.up.railway.app",
    )
    with pytest.raises(ValueError, match="GEMINI_API_KEY must be set"):
        s_no_gemini.validate_security()


def test_production_rejects_wildcard_cors_origin():
    s_wildcard = Settings(
        _env_file=None,
        environment="production",
        jwt_secret_key="a_very_secure_production_secret_key_1234567890!",
        gemini_api_key="valid_gemini_api_key",
        allowed_origins="*",
    )
    with pytest.raises(ValueError, match="must not contain wildcard"):
        s_wildcard.validate_security()


def test_allowed_origins_parsing():
    s = Settings(
        _env_file=None,
        allowed_origins="https://app.example.com, https://admin.example.com "
    )
    origins = s.get_allowed_origins()
    assert origins == ["https://app.example.com", "https://admin.example.com"]
