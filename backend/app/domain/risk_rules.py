"""
Configurable Contract Risk Rules.
Combines deterministic regex/keyword heuristics with context-aware LLM reasoning rules.
Fully covers Vietnamese Commercial/Civil Codes and International Commercial Standards (Common Law / UCC / GDPR).
"""
import re
from typing import List, Dict, Any, Optional, Tuple
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
    # -------------------------------------------------------------
    # 1. TÀI CHÍNH, BỒI THƯỜNG & PHẠT VI PHẠM (LIABILITY & PENALTIES)
    # -------------------------------------------------------------
    RiskRuleDefinition(
        rule_id="RULE_UNLIMITED_LIABILITY",
        name="Trách nhiệm bồi thường vô hạn / Không có trần thiệt hại (Unlimited Liability)",
        category="Liability",
        severity="critical",
        description="Hợp đồng không quy định mức trần bồi thường tối đa (Liability Cap) hoặc bắt bồi thường vô hạn với mọi thiệt hại phát sinh.",
        regex_patterns=[
            r"(?i)\bunlimited\s+liability\b",
            r"(?i)\bshall\s+not\s+be\s+subject\s+to\s+any\s+(cap|limitation)\b",
            r"(?i)\bwithout\s+(limitation|cap)\s+as\s+to\s+amount\b",
            r"(?i)\bno\s+limitation\s+of\s+liability\b",
            r"(?i)\bkhông\s+giới\s+hạn\s+trách\s+nhiệm(\s+bồi\s+thường)?\b",
            r"(?i)\bchịu\s+trách\s+nhiệm\s+toàn\s+bộ\s+mọi\s+thiệt\s+hại\b",
            r"(?i)\bkhông\s+bị\s+giới\s+hạn\s+bởi\s+bất\s+kỳ\s+mức\s+trần\b",
        ],
        keywords=["unlimited liability", "uncapped damages", "no cap", "without limitation of liability", "không giới hạn trách nhiệm", "bồi thường toàn bộ thiệt hại"],
    ),

    RiskRuleDefinition(
        rule_id="RULE_EXCESSIVE_PENALTY_VN",
        name="Phạt vi phạm vượt khung luật định Việt Nam - Trần 8% (Excessive Penalties)",
        category="Penalties",
        severity="critical",
        description="Theo Điều 301 Luật Thương mại VN, mức phạt vi phạm tối đa là 8% giá trị phần nghĩa vụ bị vi phạm. Quy định mức phạt >8% (ví dụ 10%, 20%, 50%) có nguy cơ bị tuyên vô hiệu.",
        regex_patterns=[
            r"(?i)\bphạt\s+vi\s+phạm\s+(\d{1,2}|[1-9]\d{2,})%\b",
            r"(?i)\bmức\s+phạt\s+(\d{1,2}|[1-9]\d{2,})%\b",
            r"(?i)\bpenalty\s+of\s+(\d{1,2}|[1-9]\d{2,})%\b",
            r"(?i)\bliquidated\s+damages\s+of\s+(\d{1,2})%\b",
        ],
        keywords=["phạt vi phạm", "mức phạt vi phạm", "penalty of", "liquidated damages"],
    ),

    RiskRuleDefinition(
        rule_id="RULE_FOREIGN_CURRENCY_VN",
        name="Vi phạm Pháp lệnh Ngoại hối Việt Nam (Foreign Currency Restriction)",
        category="Payment & Compliance",
        severity="critical",
        description="Giao dịch nội địa giữa các pháp nhân/cá nhân cư trú tại Việt Nam không được niêm yết hoặc thanh toán bằng USD/ngoại tệ, trừ trường hợp được Ngân hàng Nhà nước cấp phép.",
        regex_patterns=[
            r"(?i)\bthanh\s+toán\s+bằng\s+(USD|đô\s+la\s+Mỹ|ngoại\s+tệ)\b",
            r"(?i)\bgiá\s+trị\s+hợp\s+đồng\s+được\s+quy\s+định\s+bằng\s+USD\b",
            r"(?i)\bpayable\s+in\s+USD\b",
            r"(?i)\bdenominated\s+in\s+US\s+Dollars\b",
        ],
        keywords=["thanh toán bằng USD", "đô la Mỹ", "payable in USD", "denominated in USD", "ngoại tệ"],
    ),

    # -------------------------------------------------------------
    # 2. BẢO VỆ DỮ LIỆU & SỞ HỮU TRÍ TUỆ (DATA PRIVACY & IP)
    # -------------------------------------------------------------
    RiskRuleDefinition(
        rule_id="RULE_DATA_PRIVACY_PDP",
        name="Tuân thủ Bảo vệ Dữ liệu Cá nhân / Rò rỉ Dữ liệu (NĐ 13/2023 & GDPR)",
        category="Data Privacy",
        severity="high",
        description="Hợp đồng liên quan đến xử lý dữ liệu khách hàng/nhân sự nhưng thiếu cam kết tuân thủ Nghị định 13/2023/NĐ-CP hoặc GDPR, hoặc đẩy toàn bộ trách nhiệm khi có sự cố rò rỉ dữ liệu.",
        regex_patterns=[
            r"(?i)\bxử\s+lý\s+dữ\s+liệu\s+cá\s+nhân\b",
            r"(?i)\bpersonal\s+data\s+processing\b",
            r"(?i)\bdata\s+breach\s+notification\b",
            r"(?i)\bnghị\s+định\s+13/2023\b",
            r"(?i)\bGDPR\s+compliance\b",
        ],
        keywords=["dữ liệu cá nhân", "xử lý dữ liệu", "personal data", "data breach", "nghị định 13", "GDPR"],
    ),

    RiskRuleDefinition(
        rule_id="RULE_BROAD_IP_TRANSFER",
        name="Chuyển nhượng/Tước đoạt quyền Sở hữu Trí tuệ (Broad IP Assignment)",
        category="Intellectual Property",
        severity="critical",
        description="Điều khoản buộc bên bạn phải chuyển giao toàn bộ quyền tác giả, mã nguồn, sáng chế phát sinh cho đối tác thay vì chỉ cấp quyền sử dụng có giới hạn (License).",
        regex_patterns=[
            r"(?i)\bassigns\s+all\s+(right,\s+title\s+and\s+interest|intellectual\s+property)\b",
            r"(?i)\bwork\s+made\s+for\s+hire\b",
            r"(?i)\bchuyển\s+giao\s+toàn\s+bộ\s+quyền\s+sở\s+hữu\s+trí\s+tuệ\b",
            r"(?i)\bthuộc\s+quyền\s+sở\s+hữu\s+độc\s+quyền\s+của\s+(bên\s+[AB]|khách\s+hàng)\b",
            r"(?i)\bsole\s+and\s+exclusive\s+property\s+of\b",
        ],
        keywords=["assigns all right", "work made for hire", "chuyển giao toàn bộ quyền sở hữu trí tuệ", "quyền sở hữu độc quyền", "exclusive property"],
    ),

    RiskRuleDefinition(
        rule_id="RULE_ONE_SIDED_INDEMNITY",
        name="Bồi thường đơn phương cho bên thứ ba (One-Sided Broad Indemnification)",
        category="Indemnity",
        severity="high",
        description="Buộc một bên phải đứng ra bồi thường, bào chữa và gánh chịu toàn bộ thiệt hại/kiện tụng từ bên thứ ba một cách bất bình đẳng.",
        regex_patterns=[
            r"(?i)\bindemnify,\s+defend\s+and\s+hold\s+harmless\b",
            r"(?i)\bagainst\s+any\s+and\s+all\s+claims,\s+losses,\s+damages\b",
            r"(?i)\bsolely\s+indemnify\b",
            r"(?i)\bbồi\s+thường\s+và\s+giữ\s+cho\s+không\s+bị\s+thiệt\s+hại\b",
            r"(?i)\bgánh\s+chịu\s+toàn\s+bộ\s+khiếu\s+nại\s+từ\s+bên\s+thứ\s+ba\b",
        ],
        keywords=["hold harmless", "indemnify and defend", "bồi thường và giữ cho", "khiếu nại bên thứ ba", "indemnity"],
    ),

    # -------------------------------------------------------------
    # 3. THỜI HẠN, CHẤM DỨT & GIA HẠN (TERM, RENEWAL & TERMINATION)
    # -------------------------------------------------------------
    RiskRuleDefinition(
        rule_id="RULE_AUTO_RENEWAL",
        name="Tự động gia hạn hợp đồng vô thời hạn (Evergreen Auto-Renewal)",
        category="Term & Renewal",
        severity="medium",
        description="Hợp đồng tự động gia hạn thêm chu kỳ mới nếu không gửi văn bản từ chối trước thời hạn, dễ làm doanh nghiệp bị kẹt nghĩa vụ ngoài ý muốn.",
        regex_patterns=[
            r"(?i)\bautomatically\s+renew(s|ed|ing)?\b",
            r"(?i)\bshall\s+be\s+automatically\s+extended\b",
            r"(?i)\bsuccessive\s+(one|1|two|2|three|3|year|month)\s+(year|month|period)s?\b",
            r"(?i)\btự\s+động\s+gia\s+hạn(\s+thêm)?\b",
            r"(?i)\btự\s+động\s+kéo\s+dài\s+thời\s+hạn\b",
        ],
        keywords=["automatically renew", "auto-renewal", "evergreen", "tự động gia hạn", "tự động kéo dài"],
    ),

    RiskRuleDefinition(
        rule_id="RULE_LONG_NOTICE_PERIOD",
        name="Thời hạn báo trước chấm dứt quá dài (Excessive Termination Notice)",
        category="Termination",
        severity="high",
        description="Yêu cầu thời hạn thông báo trước khi chấm dứt hợp đồng quá dài (> 60 ngày), làm giảm sự linh hoạt vận hành của doanh nghiệp.",
        regex_patterns=[
            r"(?i)\b(60|90|120|180)\s+days?'?\s+(prior\s+)?(written\s+)?notice\b",
            r"(?i)\bnotice\s+of\s+not\s+less\s+than\s+(60|90|120|180)\s+days\b",
            r"(?i)\bthông\s+báo\s+trước\s+(ít\s+nhất\s+)?(60|90|120|180)\s+ngày\b",
        ],
        keywords=["60 days notice", "90 days notice", "120 days prior notice", "thông báo trước 60 ngày", "thông báo trước 90 ngày"],
        threshold_days=60,
    ),

    RiskRuleDefinition(
        rule_id="RULE_MISSING_TERMINATION_CONVENIENCE",
        name="Không có quyền đơn phương chấm dứt tự nguyện (No Termination for Convenience)",
        category="Termination",
        severity="medium",
        description="Hợp đồng chỉ cho phép chấm dứt khi có vi phạm nghiêm trọng (for cause), khóa chặt các bên không thể chủ động dừng hợp đồng khi nhu cầu thay đổi.",
        regex_patterns=[
            r"(?i)\bmay\s+only\s+terminate\s+for\s+(material\s+)?breach\b",
            r"(?i)\bno\s+right\s+to\s+terminate\s+for\s+convenience\b",
            r"(?i)\bchỉ\s+được\s+chấm\s+dứt\s+khi\s+(có\s+)?vi\s+phạm\s+nghiêm\s+trọng\b",
            r"(?i)\bkhông\s+được\s+đơn\s+phương\s+chấm\s+dứt\s+hợp\s+đồng\b",
        ],
        keywords=["terminate for cause only", "no termination for convenience", "chỉ được chấm dứt khi vi phạm", "không được đơn phương"],
    ),

    # -------------------------------------------------------------
    # 4. HẠN CHẾ CẠNH TRANH, ĐỘC QUYỀN & THẨM QUYỀN TÒA ÁN
    # -------------------------------------------------------------
    RiskRuleDefinition(
        rule_id="RULE_NON_COMPETE_RESTRICTION",
        name="Hạn chế Cạnh tranh & Tuyển dụng nhân sự (Non-Compete & Non-Solicitation)",
        category="Restrictive Covenants",
        severity="high",
        description="Cấm bên bạn kinh doanh cùng ngành hoặc cấm tuyển dụng nhân viên sau khi kết thúc hợp đồng với mức phạt lớn hoặc thời hạn cấm quá dài (> 1-2 năm).",
        regex_patterns=[
            r"(?i)\bshall\s+not\s+(engage\s+in|compete\s+with)\b",
            r"(?i)\bnon-compete\s+period\s+of\b",
            r"(?i)\bshall\s+not\s+solicit\s+or\s+hire\s+any\s+employee\b",
            r"(?i)\bkhông\s+được\s+kinh\s+doanh\s+trong\s+cùng\s+lĩnh\s+vực\b",
            r"(?i)\bcấm\s+cạnh\s+tranh\b",
            r"(?i)\bkhông\s+được\s+lôi\s+kéo(\s+hoặc\s+tuyển\s+dụng)?\s+nhân\s+viên\b",
        ],
        keywords=["non-compete", "non-solicitation", "cấm cạnh tranh", "không được tuyển dụng nhân viên", "lôi kéo nhân sự"],
    ),

    RiskRuleDefinition(
        rule_id="RULE_UNFAVORABLE_GOVERNING_LAW",
        name="Luật áp dụng & Cơ quan giải quyết tranh chấp bất lợi (Jurisdiction & Forum)",
        category="Jurisdiction",
        severity="high",
        description="Hợp đồng bắt buộc giải quyết tranh chấp tại Tòa án/Trọng tài nước ngoài (Singapore, London, Delaware, Hong Kong...) làm phát sinh chi phí tố tụng và đi lại cực lớn.",
        regex_patterns=[
            r"(?i)\bgoverned\s+by\s+the\s+laws\s+of\s+(the\s+State\s+of\s+Delaware|England\s+and\s+Wales|Singapore|Hong\s+Kong|New\s+York)\b",
            r"(?i)\bexclusive\s+jurisdiction\s+of\s+the\s+courts\s+of\b",
            r"(?i)\bSingapore\s+International\s+Arbitration\s+Centre\b",
            r"(?i)\btoà\s+án\s+(nước\s+ngoài|Singapore|Anh\s+quốc|Hoa\s+Kỳ)\b",
            r"(?i)\btrọng\s+tài\s+quốc\s+tế\s+tại\s+nước\s+ngoài\b",
        ],
        keywords=["governing law", "exclusive jurisdiction", "SIAC", "Delaware law", "English law", "tòa án nước ngoài", "luật áp dụng"],
    ),

    RiskRuleDefinition(
        rule_id="RULE_UNILATERAL_MODIFICATION",
        name="Đơn phương thay đổi điều khoản hoặc giá (Unilateral Price/Term Change)",
        category="Commercial Terms",
        severity="high",
        description="Cho phép đối tác có quyền tự ý sửa đổi điều khoản hoặc tăng giá dịch vụ mà không cần sự đồng ý bằng văn bản của bên bạn.",
        regex_patterns=[
            r"(?i)\bmay\s+modify\s+(these\s+terms|the\s+agreement)\s+at\s+any\s+time\b",
            r"(?i)\bsole\s+discretion\s+to\s+(increase|modify|change)\s+fees\b",
            r"(?i)\bcó\s+quyền\s+thay\s+đổi\s+điều\s+khoản\s+bất\s+kỳ\s+lúc\s+nào\b",
            r"(?i)\btự\s+ý\s+điều\s+chỉnh\s+(giá|phí|biểu\s+phí)\b",
            r"(?i)\bwithout\s+prior\s+written\s+consent\b",
        ],
        keywords=["modify at any time", "sole discretion to increase", "thay đổi điều khoản bất kỳ lúc nào", "tự ý điều chỉnh giá"],
    ),
]


class RiskRuleEngine:
    """
    Evaluates text blocks against configured risk rules with deterministic structured predicates:
    1. Regex / Keyword candidate detection
    2. Structured numeric extraction & threshold predicates (e.g. Penalty > 8%, Notice > 60 days)
    3. Safe negation & compliance exclusion handling (e.g. 'không vượt quá 8%')
    """

    def __init__(self, rules: Optional[List[RiskRuleDefinition]] = None):
        self.rules = rules or DEFAULT_RISK_RULES

    @staticmethod
    def _check_excessive_penalty_vn(text: str) -> Tuple[bool, Optional[float]]:
        """
        Extracts penalty percentage and verifies if it strictly exceeds the 8% statutory cap
        under Article 301, Vietnam Commercial Law.
        Returns: (is_excessive, extracted_pct)
        """
        # If text explicitly states 'không vượt quá 8%' or 'tối đa 8%', it is compliant
        if re.search(r"(?i)(không\s+vượt\s+quá|tối\s+đa|not\s+exceed(ing)?|maximum\s+of)\s+8\s*%", text):
            return False, 8.0

        # Look for penalty patterns with numbers
        patterns = [
            r"(?i)(?:phạt\s+vi\s+phạm|mức\s+phạt|penalty\s+of|liquidated\s+damages\s+of|fine\s+of)\s*(?:là|is)?\s*(\d+(?:\.\d+)?)\s*%",
            r"(?i)(\d+(?:\.\d+)?)\s*%\s*(?:trên\s+tổng\s+giá\s+trị|tiền\s+phạt|phạt\s+vi\s+phạm|as\s+a\s+penalty)",
        ]
        for p in patterns:
            for match in re.finditer(p, text):
                try:
                    pct = float(match.group(1))
                    if pct > 8.0:
                        return True, pct
                except (ValueError, IndexError):
                    continue

        return False, None

    @staticmethod
    def _check_excessive_notice(text: str, threshold_days: int = 60) -> Tuple[bool, Optional[int]]:
        """
        Extracts notice period days and checks if it exceeds the threshold (e.g. > 60 days).
        Returns: (is_excessive, extracted_days)
        """
        patterns = [
            r"(?i)(?:thông\s+báo\s+trước|notice\s+of|prior\s+notice\s+of|providing|giving)\s*(?:ít\s+nhất|at\s+least)?\s*(\d+)\s*(?:ngày|days)",
            r"(?i)(\d+)\s*(?:ngày|days)\s*(?:trước\s+khi\s+chấm\s+dứt|prior\s+to\s+termination|prior\s+written\s+notice|prior\s+notice|notice)",
        ]
        for p in patterns:
            for match in re.finditer(p, text):
                try:
                    days = int(match.group(1))
                    if days > threshold_days:
                        return True, days
                except (ValueError, IndexError):
                    continue
        return False, None

    def scan_block_deterministic(self, text: str) -> List[Dict[str, Any]]:
        """Run regex, keyword, and numeric predicate checks on a text block."""
        matches = []
        text_lower = text.lower()

        for rule in self.rules:
            if not rule.default_enabled:
                continue

            # Specialized predicate for excessive penalties (> 8% cap)
            if rule.rule_id == "RULE_EXCESSIVE_PENALTY_VN":
                is_excessive, extracted_pct = self._check_excessive_penalty_vn(text)
                if is_excessive:
                    matches.append({
                        "rule_id": rule.rule_id,
                        "rule_name": rule.name,
                        "severity": rule.severity,
                        "category": rule.category,
                        "matched_patterns": [f"Extracted penalty: {extracted_pct}% > 8% cap"],
                        "matched_keywords": ["phạt vi phạm"],
                        "extracted_value": extracted_pct,
                    })
                continue

            # Specialized predicate for excessive notice periods (> 60 days)
            if rule.rule_id in ["RULE_LONG_NOTICE_PERIOD", "RULE_EXCESSIVE_NOTICE_PERIOD"]:
                is_excessive, extracted_days = self._check_excessive_notice(text, threshold_days=rule.threshold_days or 60)
                if is_excessive:
                    matches.append({
                        "rule_id": rule.rule_id,
                        "rule_name": rule.name,
                        "severity": rule.severity,
                        "category": rule.category,
                        "matched_patterns": [f"Extracted notice: {extracted_days} days > 60 days threshold"],
                        "matched_keywords": ["thông báo trước"],
                        "extracted_value": extracted_days,
                    })
                continue

            matched_patterns = []
            for pattern in rule.regex_patterns:
                if re.search(pattern, text):
                    matched_patterns.append(pattern)

            matched_keywords = []
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

