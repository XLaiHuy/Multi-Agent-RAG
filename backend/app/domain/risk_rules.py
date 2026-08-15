"""
Configurable Contract Risk Rules.
Combines deterministic regex/keyword heuristics with context-aware LLM reasoning rules.
"""
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class RiskRuleDefinition:
    rule_id: str
    name: str
    category: str
    severity: str # "low" | "medium" | "high" | "critical"
    description: str
    regex_patterns: List[str]
    keywords: List[str]
    default_enabled: bool = True
    threshold_days: Optional[int] = None


DEFAULT_RISK_RULES: List[RiskRuleDefinition] = [
    RiskRuleDefinition(
        rule_id="RULE_UNLIMITED_LIABILITY",
        name="Unlimited Liability / Missing Cap",
        category="Liability",
        severity="critical",
        description="The contract contains unlimited liability, uncapped damages, or lacks a clear monetary cap on damages.",
        regex_patterns=[
            r"(?i)\bunlimited\s+liability\b",
            r"(?i)\bshall\s+not\s+be\s+subject\s+to\s+any\s+(cap|limitation)\b",
            r"(?i)\bwithout\s+(limitation|cap)\s+as\s+to\s+amount\b",
            r"(?i)\bno\s+limitation\s+of\s+liability\b",
        ],
        keywords=["unlimited liability", "uncapped damages", "no cap", "without limitation of liability"],
    ),
    RiskRuleDefinition(
        rule_id="RULE_AUTO_RENEWAL",
        name="Automatic Evergreen Renewal",
        category="Term & Renewal",
        severity="medium",
        description="The contract automatically renews unless notice of non-renewal is provided well in advance.",
        regex_patterns=[
            r"(?i)\bautomatically\s+renew(s|ed|ing)?\b",
            r"(?i)\bshall\s+be\s+automatically\s+extended\b",
            r"(?i)\bsuccessive\s+(one|1|two|2|three|3|year|month)\s+(year|month|period)s?\b",
        ],
        keywords=["automatically renew", "auto-renewal", "evergreen", "successive periods"],
    ),
    RiskRuleDefinition(
        rule_id="RULE_LONG_NOTICE_PERIOD",
        name="Excessive Termination Notice Period",
        category="Termination",
        severity="high",
        description="Termination notice period exceeds 60 days, limiting operational flexibility.",
        regex_patterns=[
            r"(?i)\b(60|90|120|180)\s+days?'?\s+(prior\s+)?(written\s+)?notice\b",
            r"(?i)\bnotice\s+of\s+not\s+less\s+than\s+(60|90|120|180)\s+days\b",
        ],
        keywords=["60 days notice", "90 days notice", "120 days prior notice"],
        threshold_days=60,
    ),
    RiskRuleDefinition(
        rule_id="RULE_ONE_SIDED_INDEMNITY",
        name="One-Sided / Broad Indemnification",
        category="Indemnity",
        severity="high",
        description="Imposes unilateral, broad indemnification obligations against all third-party claims.",
        regex_patterns=[
            r"(?i)\bindemnify,\s+defend\s+and\s+hold\s+harmless\b",
            r"(?i)\bagainst\s+any\s+and\s+all\s+claims,\s+losses,\s+damages\b",
            r"(?i)\bsolely\s+indemnify\b",
        ],
        keywords=["hold harmless", "any and all claims", "indemnify and defend", "unilateral indemnity"],
    ),
    RiskRuleDefinition(
        rule_id="RULE_MISSING_TERMINATION_CONVENIENCE",
        name="No Termination for Convenience",
        category="Termination",
        severity="medium",
        description="Contract can only be terminated for cause/material breach, locking the party in.",
        regex_patterns=[
            r"(?i)\bmay\s+only\s+terminate\s+for\s+(material\s+)?breach\b",
            r"(?i)\bno\s+right\s+to\s+terminate\s+for\s+convenience\b",
        ],
        keywords=["terminate for cause only", "no termination for convenience"],
    ),
    RiskRuleDefinition(
        rule_id="RULE_UNFAVORABLE_GOVERNING_LAW",
        name="Foreign / Non-Standard Dispute Forum",
        category="Jurisdiction",
        severity="medium",
        description="Disputes must be settled under foreign law or in an inconvenient arbitration forum.",
        regex_patterns=[
            r"(?i)\bgoverned\s+by\s+the\s+laws\s+of\s+(the\s+State\s+of\s+Delaware|England\s+and\s+Wales|Singapore|Hong\s+Kong)\b",
            r"(?i)\bexclusive\s+jurisdiction\s+of\s+the\s+courts\s+of\b",
        ],
        keywords=["governing law", "exclusive jurisdiction", "arbitration rules"],
    ),
]


class RiskRuleEngine:
    """Evaluates text blocks against configured risk rules."""

    def __init__(self, rules: Optional[List[RiskRuleDefinition]] = None):
        self.rules = rules or DEFAULT_RISK_RULES

    def scan_block_deterministic(self, text: str) -> List[Dict[str, Any]]:
        """Run regex and keyword checks on a text block."""
        matches = []
        for rule in self.rules:
            if not rule.default_enabled:
                continue

            matched_patterns = []
            for pattern in rule.regex_patterns:
                if re.search(pattern, text):
                    matched_patterns.append(pattern)

            matched_keywords = []
            text_lower = text.lower()
            for kw in rule.keywords:
                if kw.lower() in text_lower:
                    matched_keywords.append(kw)

            if matched_patterns or len(matched_keywords) >= 1:
                matches.append({
                    "rule_id": rule.rule_id,
                    "rule_name": rule.name,
                    "severity": rule.severity,
                    "category": rule.category,
                    "matched_patterns": matched_patterns,
                    "matched_keywords": matched_keywords,
                })
        return matches
