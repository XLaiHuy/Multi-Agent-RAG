import json
import openai
from typing import Literal
from pydantic import BaseModel, Field

from google import genai
from google.genai import types
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import Settings, get_settings


class AgentDecision(BaseModel):
    action: Literal["direct_answer", "retrieve_hybrid", "retrieve_graph"] = Field(
        description="Hành động được chọn dựa trên loại câu hỏi"
    )
    reasoning: str = Field(
        description="Giải thích ngắn gọn lý do chọn hành động này"
    )


class VerificationResult(BaseModel):
    status: Literal["grounded", "hallucinated"] = Field(
        description="Trạng thái kiểm định câu trả lời"
    )
    comment: str = Field(
        description="Nhận xét ngắn gọn lý do đánh giá grounded hoặc hallucinated"
    )


class LLMGenerator:

    """
    A class to handle LLM generation, supporting both Google GenAI (Gemini) 
    and OpenAI-compatible endpoints (like local Ollama).
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.provider = self.settings.llm_provider
        self.model = self.settings.llm_model
        
        if self.provider == "gemini":
            self.gemini_client = genai.Client(api_key=self.settings.gemini_api_key)
            self.openai_client = None
        elif self.provider == "ollama":
            self.gemini_client = None
            self.openai_client = openai.OpenAI(
                base_url=self.settings.ollama_base_url,
                api_key="ollama" # Dummy key for local Ollama
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def generate_answer(self, query: str, chunks: list[dict], verification_comment: str = "", chat_history: str = "") -> str:
        if not chunks:
            return "Không có tài liệu tham khảo nào được tìm thấy để trả lời câu hỏi."

        # Format context chunks
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("source", "unknown")
            text = chunk.get("text", "")
            context_parts.append(f"Tài liệu [{i}] (Nguồn: {source}):\n{text}")
        
        context_str = "\n\n".join(context_parts)
        
        correction_instruction = ""
        if verification_comment:
            correction_instruction = f"\nLƯU Ý QUAN TRỌNG: Câu trả lời trước đó của bạn đã bị từ chối với lý do: '{verification_comment}'. Hãy cực kỳ cẩn thận, TUYỆT ĐỐI không bịa đặt (hallucinate) và chỉ dựa vào tài liệu tham khảo.\n"

        history_section = ""
        if chat_history:
            history_section = f"\nLịch sử hội thoại gần nhất:\n{chat_history}\n"

        prompt = f"""Bạn là một chuyên gia AI và trợ lý tài liệu thông minh, tin cậy của Đại học Mở.
Nhiệm vụ của bạn là trả lời câu hỏi của người dùng một cách rõ ràng, mạch lạc và sâu sắc, kết hợp kiến thức chuyên môn với các tài liệu tham khảo được cung cấp bên dưới.

Nguyên tắc trả lời:
1. Luôn ưu tiên trích dẫn và liên hệ trực tiếp tới các thông tin có trong tài liệu tham khảo (sử dụng định dạng [Tài liệu X]).
2. Đối với các câu hỏi khái niệm, định nghĩa hoặc nguyên lý: Hãy giải thích một cách dễ hiểu, đầy đủ, sau đó liên hệ trực tiếp với các chiến lược, thông số và phương pháp kỹ thuật được đề cập trong tài liệu tham khảo.
3. Nếu tài liệu hoàn toàn không liên quan đến chủ đề câu hỏi, hãy thông báo lịch sự cho người dùng.
4. Trình bày khoa học với các gạch đầu dòng, định dạng Markdown rõ ràng.{history_section}{correction_instruction}
Tài liệu tham khảo:
{context_str}

Câu hỏi: {query}

Câu trả lời:"""

        @retry(
            reraise=True,
            stop=stop_after_attempt(2),
            wait=wait_exponential(multiplier=1, min=1, max=3),
            retry=retry_if_exception_type(Exception),
        )
        def _call_api():
            if self.provider == "gemini":
                res = self.gemini_client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                )
                return res.text if res else ""
            else:
                res = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2
                )
                return res.choices[0].message.content if res.choices else ""

        try:
            response_text = _call_api()
            if not response_text:
                return "Không thể tạo câu trả lời từ LLM."
            return response_text
        except Exception as e:
            print(f"[LLMGenerator] Generation error: {e}")
            return "Đã xảy ra lỗi trong quá trình tạo câu trả lời."

    def generate_answer_stream(self, query: str, chunks: list[dict], verification_comment: str = "", chat_history: str = "", direct_answer: bool = False):
        history_section = ""
        if chat_history:
            history_section = f"\nLịch sử hội thoại:\n{chat_history}\n"

        if direct_answer:
            prompt = (
                f"Bạn là một trợ lý AI thông minh của Đại học Mở.{history_section}\n"
                f"Câu hỏi: {query}\n\n"
                "Hãy trả lời ngắn gọn, lịch sự, có tham chiếu ngữ cảnh nếu phù hợp.\n\nCâu trả lời:"
            )
        else:
            if not chunks:
                yield "Không có tài liệu tham khảo nào được tìm thấy để trả lời câu hỏi."
                return

            context_parts = []
            for i, chunk in enumerate(chunks, 1):
                source = chunk.get("source", "unknown")
                text = chunk.get("text", "")
                context_parts.append(f"Tài liệu [{i}] (Nguồn: {source}):\n{text}")
            context_str = "\n\n".join(context_parts)

            correction_instruction = ""
            if verification_comment:
                correction_instruction = f"\nLƯU Ý: Câu trả lời trước đã bị từ chối: '{verification_comment}'. Chỉ dựa vào tài liệu.\n"

            prompt = f"""Bạn là một chuyên gia AI và trợ lý tài liệu thông minh, tin cậy của Đại học Mở.
Nhiệm vụ của bạn là trả lời câu hỏi của người dùng một cách rõ ràng, mạch lạc và sâu sắc, kết hợp kiến thức chuyên môn với các tài liệu tham khảo được cung cấp bên dưới.

Nguyên tắc trả lời:
1. Luôn ưu tiên trích dẫn và liên hệ trực tiếp tới các thông tin có trong tài liệu tham khảo (sử dụng định dạng [Tài liệu X]).
2. Đối với các câu hỏi khái niệm, định nghĩa hoặc nguyên lý: Hãy giải thích một cách dễ hiểu, đầy đủ, sau đó liên hệ trực tiếp với các chiến lược, thông số và phương pháp kỹ thuật được đề cập trong tài liệu tham khảo.
3. Nếu tài liệu hoàn toàn không liên quan đến chủ đề câu hỏi, hãy thông báo lịch sự cho người dùng.
4. Trình bày khoa học với các gạch đầu dòng, định dạng Markdown rõ ràng.{history_section}{correction_instruction}
Tài liệu tham khảo:
{context_str}

Câu hỏi: {query}

Câu trả lời:"""

        try:
            if self.provider == "gemini":
                response = self.gemini_client.models.generate_content_stream(
                    model=self.model,
                    contents=prompt
                )
                for chunk in response:
                    if chunk.text:
                        yield chunk.text
            else:
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    stream=True
                )
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
        except Exception as e:
            print(f"[LLMGenerator] Stream generation error: {e}")
            yield " Đã xảy ra lỗi khi tạo câu trả lời (Stream)."

    def grade_relevance(self, query: str, chunks: list[dict]) -> str:
        if not chunks:
            return "not_relevant"

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get("text", "")
            context_parts.append(f"Tài liệu [{i}]:\n{text}")
        context_str = "\n\n".join(context_parts)

        prompt = f"""Bạn là một chuyên gia đánh giá thông tin. Hãy kiểm tra xem các tài liệu tham khảo dưới đây có chứa thông tin hữu ích và liên quan trực tiếp để trả lời câu hỏi hay không.

Tài liệu tham khảo:
{context_str}

Câu hỏi: {query}

Hãy trả lời bằng cách viết chính xác chữ "YES" (nếu có tài liệu liên quan) hoặc "NO" (nếu không có bất kỳ tài liệu nào liên quan). Không viết thêm bất kỳ từ nào khác ngoài YES hoặc NO.

Trả lời:"""

        @retry(
            reraise=True,
            stop=stop_after_attempt(2),
            wait=wait_exponential(multiplier=1, min=1, max=3),
            retry=retry_if_exception_type(Exception),
        )
        def _call_api():
            if self.provider == "gemini":
                res = self.gemini_client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                )
                return res.text if res else "NO"
            else:
                res = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                return res.choices[0].message.content if res.choices else "NO"

        try:
            result_text = _call_api().strip().upper()
            if "YES" in result_text:
                return "relevant"
            return "not_relevant"
        except Exception as e:
            print(f"[LLMGenerator] Grading error: {e}")
            return "not_relevant"

    def rewrite_query(self, query: str) -> str:
        prompt = f"""Bạn là một trợ lý tối ưu hóa tìm kiếm. Hãy viết lại câu hỏi dưới đây của người dùng thành một câu hỏi hoặc từ khóa tìm kiếm mới rõ ràng hơn, nhiều ngữ nghĩa hơn và tối ưu hơn cho công cụ tìm kiếm vector (vector database search).
Hãy cố gắng giữ nguyên ý nghĩa cốt lõi của câu hỏi ban đầu, chỉ diễn đạt lại từ ngữ rõ ràng và chi tiết hơn.
Chỉ trả về duy nhất câu hỏi/từ khóa tìm kiếm mới sau khi viết lại, không viết thêm giải thích hay dẫn giải nào khác.

Câu hỏi ban đầu: {query}

Câu hỏi mới viết lại:"""

        @retry(
            reraise=True,
            stop=stop_after_attempt(2),
            wait=wait_exponential(multiplier=1, min=1, max=3),
            retry=retry_if_exception_type(Exception),
        )
        def _call_api():
            if self.provider == "gemini":
                res = self.gemini_client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                )
                return res.text if res else query
            else:
                res = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                return res.choices[0].message.content if res.choices else query

        try:
            response_text = _call_api()
            new_query = response_text.strip().strip('"').strip("'")
            return new_query
        except Exception as e:
            print(f"[LLMGenerator] Rewrite error: {e}")
            return query

    def expand_query(self, query: str) -> list[str]:
        """
        Multi-Query Expansion: Generates 2-3 query variations including English technical terms,
        synonyms, and specific facets to achieve high-recall multi-perspective retrieval.
        """
        prompt = f"""Bạn là một chuyên gia tìm kiếm thông tin tài liệu khoa học và kỹ thuật.
Từ câu hỏi của người dùng dưới đây, hãy tạo ra 2 đến 3 câu hỏi phụ hoặc từ khóa tìm kiếm bổ trợ (bao gồm cả thuật ngữ tiếng Anh chuyên ngành, tên bảng biểu hoặc khía cạnh kỹ thuật liên quan) nhằm giúp tìm kiếm bao quát 100% thông tin trong tài liệu.

Yêu cầu định dạng:
Mỗi câu hỏi/từ khóa phụ nằm trên MỘT dòng riêng biệt. Không đánh số, không thêm gạch đầu dòng, không thêm lời dẫn giải.

Câu hỏi của người dùng: {query}

Các câu truy vấn mở rộng:"""

        @retry(
            reraise=True,
            stop=stop_after_attempt(2),
            wait=wait_exponential(multiplier=1, min=1, max=3),
            retry=retry_if_exception_type(Exception),
        )
        def _call_api():
            if self.provider == "gemini":
                res = self.gemini_client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                )
                return res.text if res else ""
            else:
                res = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                return res.choices[0].message.content if res.choices else ""

        try:
            raw = _call_api().strip()
            lines = [line.strip().strip("-").strip("•").strip('"').strip("'").strip() for line in raw.split("\n") if line.strip()]
            queries = [query] + [l for l in lines if l and l.lower() != query.lower()]
            return queries[:4]
        except Exception as e:
            print(f"[LLMGenerator] Expand query error: {e}")
            return [query]

    def analyze_intent_and_decide(self, query: str) -> dict:
        prompt = f"""Bạn là một Agent phân tích ý định câu hỏi của người dùng. Hãy phân loại câu hỏi dưới đây vào một trong 3 nhóm xử lý:

1. "direct_answer": Dành cho các câu chào hỏi xã giao (như "chào bạn", "cảm ơn"), các phép tính toán đơn giản (như "1+1"), hoặc câu hỏi giao tiếp cơ bản không liên quan đến tài liệu.
2. "retrieve_hybrid": Dành cho các câu hỏi tra cứu thông tin thông thường từ tài liệu, khái niệm, tóm tắt, tham số, tìm kiếm từ khóa.
3. "retrieve_graph": Dành cho các câu hỏi mang tính chất kết nối mạng lưới (Graph), đa chiều, hỏi về mối quan hệ giữa nhiều thực thể, hoặc yêu cầu tổng hợp thông tin qua nhiều bước (multi-hop reasoning).

Câu hỏi của người dùng: {query}"""

        @retry(
            reraise=True,
            stop=stop_after_attempt(2),
            wait=wait_exponential(multiplier=1, min=1, max=3),
            retry=retry_if_exception_type(Exception),
        )
        def _call_api():
            if self.provider == "gemini":
                res = self.gemini_client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=AgentDecision,
                    ),
                )
                if res and res.text:
                    return json.loads(res.text)
                return {}
            else:
                # Ollama standard JSON output parsing
                res = self.openai_client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=AgentDecision,
                    temperature=0.0
                )
                if res and res.choices:
                    return res.choices[0].message.parsed.model_dump()
                return {}

        try:
            data = _call_api()
            if data:
                return {
                    "action": data.get("action", "retrieve_hybrid"),
                    "reasoning": data.get("reasoning", "")
                }
        except Exception as e:
            print(f"[LLMGenerator] Intent analysis parsing error: {e}. Fallback to 'retrieve_hybrid'.")
        
        return {"action": "retrieve_hybrid", "reasoning": "Fallback due to parsing error."}

    def verify_answer_groundedness(self, query: str, chunks: list[dict], answer: str) -> dict:
        if not chunks:
            return {
                "status": "skipped",
                "comment": "Không có tài liệu truy xuất để đối soát."
            }

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get("text", "")
            context_parts.append(f"Tài liệu [{i}]:\n{text}")
        context_str = "\n\n".join(context_parts)

        prompt = f"""Bạn là một Subagent giám định tính trung thực của thông tin. Nhiệm vụ của bạn là kiểm tra xem câu trả lời bên dưới có thực sự căn cứ (grounded) trên các tài liệu tham khảo hay bị tự bịa đặt (hallucinated).

Tài liệu tham khảo:
{context_str}

Câu hỏi: {query}

Câu trả lời cần kiểm định:
{answer}"""

        @retry(
            reraise=True,
            stop=stop_after_attempt(2),
            wait=wait_exponential(multiplier=1, min=1, max=3),
            retry=retry_if_exception_type(Exception),
        )
        def _call_api():
            if self.provider == "gemini":
                res = self.gemini_client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=VerificationResult,
                    ),
                )
                if res and res.text:
                    return json.loads(res.text)
                return {}
            else:
                # Ollama structured output parsing
                res = self.openai_client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=VerificationResult,
                    temperature=0.0
                )
                if res and res.choices:
                    return res.choices[0].message.parsed.model_dump()
                return {}

        try:
            data = _call_api()
            if data:
                return {
                    "status": data.get("status", "grounded"),
                    "comment": data.get("comment", "")
                }
        except Exception as e:
            print(f"[LLMGenerator] Verification parsing error: {e}.")
        
        return {"status": "grounded", "comment": "Bỏ qua kiểm định do lỗi API."}
