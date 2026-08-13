# Module 4 — Graph Fundamentals, LangGraph, GraphRAG

## 1. Kiến thức graph tối thiểu cần biết

| Khái niệm | Giải thích ngắn |
|---|---|
| Node | Một đơn vị xử lý (một hàm/bước) |
| Edge | Đường nối chỉ định node nào chạy sau node nào |
| Directed graph | Edge có hướng — A→B khác B→A |
| DAG (Directed Acyclic Graph) | Graph có hướng, không có vòng lặp |
| Cycle | Có đường đi quay lại node cũ — cần điều kiện dừng, nếu không sẽ infinite loop |
| State | Dữ liệu được truyền và cập nhật xuyên suốt graph (query, chunks, answer, số lần retry...) |
| State transition | Node nhận state, xử lý, trả về state đã cập nhật |
| Conditional edge | Edge có điều kiện: "nếu X thì đi node A, nếu Y thì đi node B" |
| Routing | Cơ chế chọn node tiếp theo dựa trên state hiện tại |
| Finite-state machine | Hệ thống có tập trạng thái hữu hạn và luật chuyển trạng thái rõ ràng — LangGraph về bản chất là FSM cho LLM workflow |
| Workflow graph | Graph mô tả logic nghiệp vụ ở mức thiết kế |
| Execution graph | Graph thực sự chạy tại runtime, có thể có loop, retry, parallel |

## 2. Ví dụ minh họa — RAG graph đơn giản

```
START → Analyze Query → Retrieve Documents → Grade Documents → Generate Answer → Verify Answer → END
```
`Grade Documents` là một **conditional edge**: nếu tài liệu không liên quan → quay lại rewrite query rồi retrieve lại (cycle có giới hạn số lần lặp).

## 3. Khi nào graph tốt hơn chuỗi function tuần tự?

Chuỗi function (`retrieve() → generate()`) đủ dùng khi luồng **luôn cố định**. Graph cần thiết khi:
- Có **rẽ nhánh động** dựa trên kết quả bước trước (tài liệu tốt/xấu → đi hướng khác nhau)
- Có **vòng lặp có kiểm soát** (rewrite rồi thử lại, giới hạn N lần)
- Cần **quan sát/log từng bước** riêng biệt để debug hoặc để agent quyết định
- Cần **checkpoint/resume** (dừng giữa chừng, khôi phục lại sau)

Nếu pipeline của bạn không có rẽ nhánh, đừng dùng LangGraph cho có — một hàm tuần tự đơn giản hơn, dễ debug hơn.

## 4. LangGraph — các khái niệm cốt lõi

| Khái niệm | Vai trò |
|---|---|
| `StateGraph` | Object định nghĩa graph, gắn state schema + node + edge |
| State schema | `TypedDict`/Pydantic model định nghĩa cấu trúc dữ liệu chạy xuyên graph |
| Node function | Hàm `(state) -> partial_state_update` |
| Edge / Conditional edge | Nối node cố định hoặc theo điều kiện (`add_conditional_edges`) |
| START / END | Node đặc biệt đánh dấu điểm vào/ra |
| Reducer | Hàm định nghĩa cách merge state cũ + state mới (vd: list append thay vì overwrite) |
| Message state | Pattern state dạng list message (giống chat history), phổ biến cho agent |
| Persistence / Checkpointing | Lưu lại state giữa các bước — cho phép resume, time-travel debug |
| Human-in-the-loop / Interrupt | Dừng graph để chờ con người xác nhận trước khi tiếp tục |
| Retry | Cơ chế thử lại node khi lỗi (LangGraph hỗ trợ policy retry) |
| Loop | Cycle có kiểm soát bằng counter trong state |
| Parallel execution | Nhiều node chạy song song rồi hợp nhất kết quả |
| Subgraph | Một graph con được nhúng vào graph lớn — dùng cho subagent |
| Streaming | Trả kết quả từng phần khi node đang chạy, thay vì đợi xong toàn bộ |

## 5. Xây dựng 3 graph tăng dần

### Graph 1 — Basic RAG (P0)
```
retrieve → generate
```
```python
# app/graph/basic_rag.py
from typing import TypedDict

class RAGState(TypedDict):
    query: str
    chunks: list[dict]
    answer: str

def retrieve_node(state: RAGState) -> dict:
    chunks = hybrid_retrieve(state["query"])   # từ Module 03
    return {"chunks": chunks}

def generate_node(state: RAGState) -> dict:
    answer = generate_answer(state["query"], state["chunks"])
    return {"answer": answer}

# graph.add_node("retrieve", retrieve_node)
# graph.add_node("generate", generate_node)
# graph.add_edge(START, "retrieve")
# graph.add_edge("retrieve", "generate")
# graph.add_edge("generate", END)
```

### Graph 2 — Corrective RAG (P0)
```
analyze_query → retrieve → grade_documents → generate → verify
                    ↑                │
                    └── rewrite_query ┘  (nếu grade = "không liên quan", tối đa 1 lần)
```
```python
# app/graph/corrective_rag.py
from typing import TypedDict, Literal

class CRAGState(TypedDict):
    query: str
    original_query: str
    chunks: list[dict]
    grade: Literal["relevant", "not_relevant"]
    rewrite_count: int
    answer: str

MAX_REWRITE = 1  # giới hạn bắt buộc — tránh infinite loop

def grade_documents_node(state: CRAGState) -> dict:
    grade = grade_relevance(state["query"], state["chunks"])  # gọi LLM chấm điểm
    return {"grade": grade}

def route_after_grading(state: CRAGState) -> str:
    if state["grade"] == "not_relevant" and state["rewrite_count"] < MAX_REWRITE:
        return "rewrite_query"
    return "generate"

def rewrite_query_node(state: CRAGState) -> dict:
    new_query = rewrite_query(state["original_query"])
    return {"query": new_query, "rewrite_count": state["rewrite_count"] + 1}

# graph.add_conditional_edges("grade_documents", route_after_grading,
#     {"rewrite_query": "rewrite_query", "generate": "generate"})
# graph.add_edge("rewrite_query", "retrieve")   # đây là cycle — có counter chặn loop vô hạn
```

**Điểm gây infinite loop:** quên tăng `rewrite_count`, hoặc điều kiện dừng không kiểm tra đúng field. Luôn có counter cứng (`MAX_REWRITE`) chứ không chỉ dựa vào "LLM tự biết dừng".

### Graph 3 — Agentic RAG (P1, làm sau khi Graph 2 ổn định)
Agent (xem Module 05) tự quyết định: có cần retrieval không, dùng retriever nào (vector-only/hybrid), có cần rewrite không, có cần gọi subagent verification không, đã đủ bằng chứng để trả lời chưa. Đây là graph có node dạng "agent reasoning" thay vì luồng cố định.

## 6. GraphRAG là gì? (khác LangGraph!)

Dễ nhầm lẫn: **LangGraph** là framework orchestration (cách tổ chức các bước xử lý), còn **GraphRAG** là một **kỹ thuật retrieval** dựa trên knowledge graph — dữ liệu được biểu diễn thành entity + relationship (đồ thị tri thức) thay vì chỉ chunk văn bản phẳng, retrieval sau đó truy vấn qua các mối quan hệ giữa entity (vd: "công ty A liên quan gì tới sự kiện B qua chuỗi quan hệ nào").

- **Khi nào cần GraphRAG:** câu hỏi đòi hỏi tổng hợp qua nhiều entity/quan hệ mà chunk-based retrieval không nắm được (vd: "ai từng làm việc chung với X trong dự án nào"), hoặc dữ liệu vốn có cấu trúc quan hệ mạnh (tổ chức, quy trình, citation network).
- **Vì sao KHÔNG làm ngay trong 2 ngày:** cần xây dựng knowledge graph (extract entity/relationship bằng LLM, lưu vào graph DB như Neo4j), tốn thời gian setup + tune vượt xa scope MVP.
- **Đánh dấu P2.** Nếu muốn thể hiện đã hiểu khái niệm này trong CV/phỏng vấn, chỉ cần giải thích đúng bản chất — không cần implement.

## 7. Bài tập 5 — LangGraph cơ bản (P0)

**Mục tiêu:** dựng Graph 1, chạy thành công end-to-end.
**Yêu cầu:** `retrieve → generate`, test với 3 câu hỏi khác nhau.
**Tiêu chí hoàn thành:** state truyền đúng giữa các node (không mất dữ liệu), answer sinh ra có dùng chunks đã retrieve (không phải trả lời "trống").

## 8. Bài tập 6 — Query rewrite có giới hạn (P0/P1)

**Mục tiêu:** dựng Graph 2, thêm conditional edge.
**Yêu cầu:** khi `grade_documents` trả "not_relevant", rewrite query và retrieve lại — **tối đa 1 lần**, sau đó bắt buộc đi tiếp `generate` (kèm cảnh báo "độ tin cậy thấp" nếu vẫn not_relevant).
**Lỗi thường gặp:** infinite loop khi quên counter; hoặc rewrite query nhưng không thực sự thay đổi gì so với query gốc (do prompt rewrite quá yếu).

## 9. Validation Module 4

- [ ] Graph 1 chạy hết state không lỗi, log rõ input/output từng node
- [ ] Graph 2 có conditional edge chạy đúng cả 2 nhánh (relevant và not_relevant)
- [ ] Có counter chặn vòng lặp vô hạn, đã test case buộc phải rewrite
- [ ] Giải thích được bằng lời (không nhìn code) luồng đi của Graph 2 — đây là câu hỏi chắc chắn gặp khi phỏng vấn

## 10. Quiz kiểm tra hiểu biết

1. Khác biệt cốt lõi giữa LangGraph và GraphRAG là gì?
   *Đáp: LangGraph là framework điều phối luồng xử lý (orchestration); GraphRAG là kỹ thuật retrieval dựa trên đồ thị tri thức (entity-relationship), không phải công cụ điều phối.*
2. Vì sao cycle trong LangGraph luôn cần một cơ chế giới hạn?
   *Đáp: Không có giới hạn, điều kiện "chưa đạt yêu cầu → lặp lại" có thể chạy vô hạn nếu LLM không bao giờ tự đánh giá "đạt".*
3. Reducer trong LangGraph dùng để làm gì?
   *Đáp: Định nghĩa cách hợp nhất state cũ và state mới khi 1 node trả về update — ví dụ append vào list thay vì ghi đè.*
4. Khi nào KHÔNG nên dùng LangGraph mà chỉ cần gọi hàm tuần tự?
   *Đáp: Khi pipeline không có rẽ nhánh động, không có vòng lặp, không cần checkpoint — một chuỗi hàm đơn giản dễ debug hơn.*
5. Vì sao GraphRAG bị xếp vào P2 cho project 2 ngày?
   *Đáp: Đòi hỏi xây dựng knowledge graph (extract entity/relationship, setup graph DB) — khối lượng công việc vượt quá scope MVP 2 ngày.*

Đi tiếp: mở file `05-agent-subagent-mcp.md`.
