"""
Legal HyDE (Hypothetical Document Embeddings) Transformer:
Transforms a layperson user query into a hypothetical legal statutory passage or legal terminology expansion.
Enables dense vector models to bridge the semantic gap between layperson phrasing and formal legal legalese.
"""
from typing import Optional
from app.generation.generator import LLMGenerator

HYDE_LEGAL_PROMPT = """Bạn là một chuyên gia Pháp lý và Luật sư Doanh nghiệp.
Hãy viết một đoạn dự thảo điều khoản hợp đồng hoặc trích đoạn văn bản luật ngắn (tầm 2-3 câu) liên quan trực tiếp đến thắc mắc của người dùng bên dưới.
Sử dụng văn phong pháp lý chính thức, thuật ngữ luật học và cấu trúc quy định hành chính.

Câu hỏi của người dùng: "{query}"

Dự thảo đoạn văn bản pháp lý giả định (chỉ viết đoạn văn bản luật, không giải thích):"""

LEGAL_TERM_MAPPING = {
    "đuổi việc": "đơn phương chấm dứt hợp đồng lao động sa thải",
    "nghỉ việc": "chấm dứt hợp đồng lao động trợ cấp nghỉ việc",
    "bồi thường": "trách nhiệm bồi thường thiệt hại vi phạm hợp đồng",
    "bảo mật": "nghĩa vụ bảo vệ bí mật kinh doanh dữ liệu cá nhân",
    "lương": "tiền lương phụ cấp quy chế trả lương",
    "phát phạt": "phạt vi phạm hợp đồng chế tài tài chính",
    "hóa đơn": "hóa đơn điện tử giá trị gia tăng VAT chứng từ hợp lệ",
    "chuyển khoản": "thanh toán qua tài khoản ngân hàng không dùng tiền mặt",
}


class LegalHyDETransformer:
    def __init__(self, generator: Optional[LLMGenerator] = None):
        self.generator = generator

    def transform(self, query: str, use_llm: bool = False) -> str:
        """
        Transforms query into hypothetical legal document text.
        If use_llm=True and LLM is configured, calls Gemini to draft hypothetical legal clause.
        Otherwise, uses fast term-expansion rule set.
        """
        if use_llm and self.generator:
            try:
                prompt = HYDE_LEGAL_PROMPT.format(query=query)
                hypo_doc = self.generator.generate_direct(prompt)
                if hypo_doc and len(hypo_doc.strip()) > 10:
                    return f"{query} {hypo_doc.strip()}"
            except Exception as e:
                print(f"[LegalHyDE] Warning LLM generation error: {e}")

        # Offline / Rule-based Legal Term Expansion fallback (<1ms)
        expanded_terms = []
        q_lower = query.lower()
        for k, v in LEGAL_TERM_MAPPING.items():
            if k in q_lower:
                expanded_terms.append(v)

        if expanded_terms:
            return f"{query} {' '.join(expanded_terms)}"

        return query
