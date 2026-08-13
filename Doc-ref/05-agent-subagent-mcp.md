# Module 5 — Agent, Subagent, MCP

## 1. Phân biệt các khái niệm hay bị nhầm lẫn

| Khái niệm | Định nghĩa | Đặc điểm ra quyết định |
|---|---|---|
| Chain | Chuỗi bước cố định, luôn chạy theo 1 thứ tự | Không có quyết định động |
| Workflow | Chuỗi bước có thể rẽ nhánh nhưng **luật rẽ nhánh do người viết định nghĩa trước** (if/else rõ ràng) | Rẽ nhánh nhưng deterministic |
| Router | Một bước chọn 1 trong N nhánh dựa trên phân loại (có thể dùng LLM để phân loại) | Quyết định 1 lần, không lặp |
| Agent | LLM **tự quyết định** hành động tiếp theo (gọi tool nào, dừng khi nào) dựa trên trạng thái hiện tại — không có luồng cố định trước | Quyết định động, có thể lặp |
| Tool-using agent | Agent có quyền gọi tool bên ngoài (search, calculator, API...) | — |
| Multi-agent | Nhiều agent độc lập, mỗi agent có vai trò/context riêng, phối hợp qua giao tiếp hoặc điều phối viên | — |
| Subagent | Một agent con được gọi bởi agent chính để xử lý 1 nhiệm vụ con chuyên biệt, kết quả trả về cho agent cha | Phạm vi hẹp hơn agent chính |
| Supervisor agent | Agent điều phối, quyết định giao việc cho subagent nào | Không tự làm việc chi tiết, chỉ điều phối |
| Specialized agent | Agent được thiết kế/prompt riêng cho 1 loại việc cụ thể (vd: chỉ retrieval, chỉ verification) | — |

**Nguyên tắc chọn:** nếu luồng có thể vẽ ra một sơ đồ if/else cố định trước khi chạy → đó là **workflow/graph deterministic** (Module 04), không cần "agent". Chỉ gọi là agent khi **LLM thực sự cần tự quyết định bước tiếp theo dựa trên ngữ cảnh chưa biết trước**.

## 2. Kiến trúc đề xuất cho project

**Phương án đơn giản (khuyến nghị P0 cho 2 ngày):** không multi-agent — dùng LangGraph deterministic (Graph 2 ở Module 04) làm xương sống, chỉ thêm **1 node dạng agent** ở bước quyết định "cần retrieval hay không / cần rewrite hay không". Đây là "agentic RAG nhẹ", đủ để nói trong CV là "có agent ra quyết định động" mà không phải quản lý nhiều agent giao tiếp nhau.

**Phương án nâng cao (P1, chỉ làm khi Graph 2 đã ổn định):**
```
Supervisor / Main Agent
├── Retrieval Agent   — chọn retriever phù hợp (vector-only / hybrid / cần rewrite)
├── Answer Agent        — sinh câu trả lời có citation
└── Verification Agent    — kiểm tra câu trả lời có bám sát context không, có hallucinate không
```
State chia sẻ: `query`, `chunks`, `answer`, `verification_result` — supervisor đọc state này để quyết định bước kế tiếp (trả lời user hay yêu cầu retrieval agent tìm lại).

**Cảnh báo over-engineering:** multi-agent với 3 agent giao tiếp nhau tăng latency (mỗi agent = ít nhất 1 lượt gọi LLM), tăng độ phức tạp debug, và **không tự động tăng chất lượng** nếu vấn đề gốc là retrieval yếu chứ không phải thiếu điều phối. Chỉ thêm subagent khi bạn đã đo được (bằng Module 06) rằng vấn đề nằm ở chỗ cần một bước quyết định động, không phải ở retrieval hay prompt.

## 3. Cách tránh vòng lặp agent & giới hạn resource

- **Giới hạn số bước tối đa** (`max_iterations`) trong state — agent buộc dừng sau N bước dù chưa "hài lòng".
- **Giới hạn token budget** cho mỗi phiên — cắt sớm nếu vượt ngưỡng, trả lời với cảnh báo thay vì tiếp tục gọi LLM vô hạn.
- **Timeout tổng cho toàn bộ graph**, không chỉ từng node.
- **Log quyết định của agent**: mỗi lần agent chọn hành động, ghi lại `{step, action_chosen, reason}` — vừa để debug, vừa là bằng chứng "agent thực sự ra quyết định động" khi giải thích trong phỏng vấn.

```python
# app/agents/decision_log.py
import structlog
logger = structlog.get_logger()

def log_agent_decision(step: int, action: str, reason: str, state_snapshot: dict) -> None:
    logger.info("agent_decision", step=step, action=action, reason=reason,
                query=state_snapshot.get("query"))
```

## 4. Code skeleton — node dạng agent quyết định retrieval

```python
# app/agents/retrieval_agent.py
from typing import Literal

def decide_retrieval_strategy(query: str) -> Literal["skip", "vector_only", "hybrid"]:
    """LLM phân loại: câu hỏi có cần retrieval không, và cần chiến lược nào.
    Đây là điểm 'ra quyết định động' — không hardcode if/else theo từ khóa."""
    prompt = f"""Phân loại câu hỏi sau vào đúng 1 trong 3 nhãn: skip, vector_only, hybrid.
- skip: câu chào hỏi/xã giao, không cần tra cứu tài liệu.
- vector_only: câu hỏi diễn giải ý nghĩa, không có thuật ngữ/mã số cụ thể.
- hybrid: câu hỏi có thuật ngữ, mã số, tên riêng cần khớp chính xác.
Câu hỏi: "{query}"
Chỉ trả về đúng 1 từ trong 3 nhãn trên."""
    label = call_llm(prompt).strip().lower()
    if label not in {"skip", "vector_only", "hybrid"}:
        return "hybrid"  # fallback an toàn
    return label  # type: ignore
```

## 5. MCP (Model Context Protocol) — có cần cho project không?

**MCP là gì:** một giao thức chuẩn hóa (do Anthropic phát triển, nay được nhiều bên áp dụng) để LLM/agent **kết nối với data source và tool bên ngoài** theo cách thống nhất, thay vì mỗi tool phải viết tích hợp riêng. Một MCP server expose các "tool" và "resource"; bất kỳ MCP client (agent) nào cũng gọi được theo cùng 1 chuẩn.

**Khác gì với LangChain tool calling thông thường:** LangChain tool calling là cách bạn tự định nghĩa hàm Python và bind vào LLM trong chính codebase của bạn — gắn chặt với app của bạn. MCP hướng tới **khả năng tái sử dụng liên hệ thống**: bạn expose retriever của mình như một MCP server, thì bất kỳ agent nào (Claude Desktop, IDE, agent khác) cũng dùng lại được mà không cần viết lại tích hợp.

**Có cần cho MVP 2 ngày không? Không (P2/optional).** Lý do:
- Project của bạn là 1 hệ thống RAG độc lập có API riêng (FastAPI) — không cần chuẩn hóa để nhiều client khác nhau cùng gọi.
- Setup MCP server + client tốn thêm thời gian không phục vụ trực tiếp Definition of Done.

**Khi nào đáng làm (mở rộng sau MVP):** nếu bạn muốn retriever của mình được gọi từ Claude Desktop hoặc từ 1 agent hệ thống khác, expose nó như 1 MCP server (`retrieve_documents` tool) là cách chuẩn — đây là điểm cộng đẹp cho CV nếu còn thời gian ở P2, nhưng **không đánh đổi với các mục P0**.

## 6. Bài tập 7 — Agent tool calling (P0/P1)

**Mục tiêu:** có 1 điểm quyết định động thực sự trong hệ thống.
**Yêu cầu:** implement `decide_retrieval_strategy` ở trên, tích hợp làm node đầu tiên trong Graph 2 (thay thế bước cố định `hybrid retrieval` bằng bước động: skip/vector_only/hybrid).
**Kiến thức cần dùng:** Module 04 (conditional edge) + LLM classification prompt.
**Tiêu chí hoàn thành:** test với 3 loại câu hỏi (xã giao / diễn giải / có mã số cụ thể) và agent chọn đúng nhãn cho cả 3, có log quyết định.
**Lỗi thường gặp:** agent trả về nhãn ngoài 3 lựa chọn (do prompt không đủ chặt) → luôn có fallback an toàn như code mẫu.

## 7. Validation Module 5

- [ ] Có ít nhất 1 node LLM ra quyết định động (không phải if/else hardcode theo keyword)
- [ ] Quyết định của agent được log lại có lý do
- [ ] Có giới hạn cứng (max_iterations hoặc tương đương) chống vòng lặp/tốn resource vô hạn
- [ ] Giải thích được ranh giới rõ ràng: phần nào của hệ thống là deterministic workflow, phần nào là agentic

## 8. Quiz kiểm tra hiểu biết

1. Khác nhau cốt lõi giữa "workflow" và "agent" là gì?
   *Đáp: Workflow rẽ nhánh theo luật cố định người viết định nghĩa trước; agent để LLM tự quyết định hành động tiếp theo dựa trên ngữ cảnh runtime.*
2. Vì sao multi-agent không phải lúc nào cũng tốt hơn hệ thống đơn giản?
   *Đáp: Mỗi agent thêm vào là thêm ít nhất 1 lượt gọi LLM (tăng latency/cost) và tăng độ phức tạp debug, mà không tự động cải thiện chất lượng nếu vấn đề gốc nằm ở nơi khác (vd: retrieval yếu).*
3. MCP giải quyết vấn đề gì mà LangChain tool calling thông thường không giải quyết?
   *Đáp: MCP chuẩn hóa cách expose/tiêu thụ tool giữa nhiều hệ thống khác nhau, cho phép tái sử dụng tool/data source liên hệ thống thay vì gắn chặt vào 1 codebase.*
4. Vì sao MCP được xếp P2 cho project này?
   *Đáp: Project chỉ cần 1 API riêng phục vụ chính nó, không cần nhiều client khác nhau cùng gọi chuẩn hóa — chi phí setup không phục vụ trực tiếp Definition of Done trong 2 ngày.*
5. Làm sao tránh agent bị vòng lặp vô hạn hoặc tốn resource không kiểm soát?
   *Đáp: Giới hạn số bước tối đa, giới hạn token budget, timeout tổng cho graph, và log lại từng quyết định để phát hiện bất thường.*

Đi tiếp: mở file `06-evaluation.md`.
