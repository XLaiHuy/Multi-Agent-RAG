"""
Unit Tests for Contract Risk Rule Engine.
"""
import pytest
from backend.app.domain.risk_rules import RiskRuleEngine


def test_detect_unlimited_liability():
    engine = RiskRuleEngine()
    sample_text = (
        "In no event shall either party's liability be limited, and each party shall bear unlimited liability "
        "for any indirect or consequential damages."
    )
    matches = engine.scan_block_deterministic(sample_text)
    rule_ids = [m["rule_id"] for m in matches]
    assert "RULE_UNLIMITED_LIABILITY" in rule_ids


def test_detect_auto_renewal():
    engine = RiskRuleEngine()
    sample_text = (
        "This Agreement shall automatically renew for successive one year periods unless either party gives notice."
    )
    matches = engine.scan_block_deterministic(sample_text)
    rule_ids = [m["rule_id"] for m in matches]
    assert "RULE_AUTO_RENEWAL" in rule_ids


def test_detect_excessive_notice_period():
    engine = RiskRuleEngine()
    sample_text = (
        "Either party may terminate this agreement by providing at least 90 days prior written notice."
    )
    matches = engine.scan_block_deterministic(sample_text)
    rule_ids = [m["rule_id"] for m in matches]
    assert "RULE_LONG_NOTICE_PERIOD" in rule_ids


def test_clean_standard_clause_no_false_positives():
    engine = RiskRuleEngine()
    sample_text = (
        "The Supplier shall provide standard maintenance services during normal business hours Monday to Friday."
    )
    matches = engine.scan_block_deterministic(sample_text)
    assert len(matches) == 0
