"""
Unit Tests for Deterministic Numeric Risk Predicates (Thresholds, Boundaries, and Negations).
"""
import pytest
from backend.app.domain.risk_rules import RiskRuleEngine


def test_penalty_strictly_above_8_percent_triggers():
    engine = RiskRuleEngine()
    
    # 10% penalty -> Should trigger
    text_10 = "Bên vi phạm phải chịu mức phạt vi phạm là 10% tổng giá trị hợp đồng."
    matches_10 = engine.scan_block_deterministic(text_10)
    rule_ids_10 = [m["rule_id"] for m in matches_10]
    assert "RULE_EXCESSIVE_PENALTY_VN" in rule_ids_10
    
    # 20% penalty -> Should trigger
    text_20 = "In case of default, a penalty of 20% shall be imposed."
    matches_20 = engine.scan_block_deterministic(text_20)
    rule_ids_20 = [m["rule_id"] for m in matches_20]
    assert "RULE_EXCESSIVE_PENALTY_VN" in rule_ids_20


def test_penalty_at_or_below_8_percent_does_not_trigger():
    engine = RiskRuleEngine()
    
    # 8% statutory limit -> Compliant, should NOT trigger
    text_8 = "Mức phạt vi phạm là 8% giá trị phần nghĩa vụ bị vi phạm theo Luật Thương mại."
    matches_8 = engine.scan_block_deterministic(text_8)
    rule_ids_8 = [m["rule_id"] for m in matches_8]
    assert "RULE_EXCESSIVE_PENALTY_VN" not in rule_ids_8

    # 5% penalty -> Compliant, should NOT trigger
    text_5 = "Hai bên thống nhất mức phạt vi phạm là 5% tổng giá trị đợt giao hàng."
    matches_5 = engine.scan_block_deterministic(text_5)
    rule_ids_5 = [m["rule_id"] for m in matches_5]
    assert "RULE_EXCESSIVE_PENALTY_VN" not in rule_ids_5


def test_penalty_negation_exclusion_does_not_trigger():
    engine = RiskRuleEngine()
    
    # Explicit 'không vượt quá 8%' -> Compliant, should NOT trigger
    text_neg = "Mức phạt vi phạm hợp đồng do các bên thỏa thuận nhưng không vượt quá 8% giá trị phần nghĩa vụ vi phạm."
    matches_neg = engine.scan_block_deterministic(text_neg)
    rule_ids_neg = [m["rule_id"] for m in matches_neg]
    assert "RULE_EXCESSIVE_PENALTY_VN" not in rule_ids_neg


def test_notice_period_threshold_boundary():
    engine = RiskRuleEngine()
    
    # 90 days notice (> 60) -> Triggers
    text_90 = "Hợp đồng có thể được chấm dứt khi có thông báo trước ít nhất 90 ngày bằng văn bản."
    matches_90 = engine.scan_block_deterministic(text_90)
    rule_ids_90 = [m["rule_id"] for m in matches_90]
    assert ("RULE_LONG_NOTICE_PERIOD" in rule_ids_90 or "RULE_EXCESSIVE_NOTICE_PERIOD" in rule_ids_90)

    # 30 days notice (<= 60) -> Does NOT trigger
    text_30 = "Bên A có quyền đơn phương chấm dứt hợp đồng sau khi gửi thông báo trước 30 ngày cho Bên B."
    matches_30 = engine.scan_block_deterministic(text_30)
    rule_ids_30 = [m["rule_id"] for m in matches_30]
    assert "RULE_LONG_NOTICE_PERIOD" not in rule_ids_30
    assert "RULE_EXCESSIVE_NOTICE_PERIOD" not in rule_ids_30
