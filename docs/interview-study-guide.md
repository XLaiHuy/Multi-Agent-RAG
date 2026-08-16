# Multi-Agent Safe-RAG: Master Interview Study Guide & Deep-Dive Curriculum

> **Mục tiêu**: Chuyển đổi từ hiểu biết bề mặt (vibe-coding / thuộc lòng README) sang **nắm vững bản chất kỹ thuật (system understanding)**: trace code thật, giải thích tường tận quyết định thiết kế (design decisions), bảo vệ các con số metric, phát hiện các điểm chênh lệch giữa docs và runtime, và tự tin vượt qua các vòng phỏng vấn kỹ thuật Senior/Staff AI Engineer.

---

## 🎯 Phương pháp học tập cốt lõi

Thay vì đọc README rồi cố học thuộc, phương pháp học hiệu quả nhất là:
$$\text{Concept} \longrightarrow \text{Trace Code Thật} \longrightarrow \text{Tại sao thiết kế như vậy?} \longrightarrow \text{Metric Độc Lập} \longrightarrow \text{Interviewer Hỏi Ngược}$$

### Nguyên tắc vàng khi phỏng vấn
1. **Không giả vờ tự viết từng dòng code 100%**: Mục tiêu không phải nói dối, mà là chứng minh: *"Tôi hiểu sâu sắc kiến trúc hệ thống, tôi biết tại sao từng component tồn tại, tôi biết rõ trade-off, giới hạn (limitations) và có đầy đủ năng lực để maintain, debug và scale hệ thống."*
2. **Runtime Code > Tests/Evaluation Artifacts > Docs > README**: Code thực tế chạy như thế nào là chân lý duy nhất. Nếu Docs/README có điểm nói quá hoặc chưa wired hoàn toàn (ví dụ: OCR provider injection, 4 agents vs 3 agents + generation step), hãy thẳng thắn phân tích sự khác biệt đó như một Senior Engineer.

---

## 📑 Mục lục chương trình học

- [Prompt 0 — Master Tutor (System Prompt dùng đầu tiên)](#-prompt-0--master-tutor-dùng-đầu-tiên)
- [8 Bài học chuyên sâu (Prompts 1 – 8)](#-8-bài-học-chuyên-sâu)
  - [Lesson 1: Toàn cảnh RAG & Request Lifecycle](#prompt-1--toàn-cảnh-rag-và-request-lifecycle)
  - [Lesson 2: Ingestion, OCR, CanonicalDocument, Parent–Child Chunking](#prompt-2--ingestion-ocr-canonicaldocument-parentchild-chunking)
  - [Lesson 3: Embedding, BM25, Hybrid Retrieval, RRF, CrossEncoder Reranker](#prompt-3--embedding-bm25-hybrid-retrieval-rrf-reranker)
  - [Lesson 4: Document-Scoped Retrieval, Multi-Tenant ACL & Collision Drop](#prompt-4--document-scoped-retrieval-acl-và-tại-sao-nó-là-contribution-quan-trọng)
  - [Lesson 5: Multi-Agent Layer: Planner, Critic, Generator Step, Verifier](#prompt-5--multi-agent-planner-critic-generator-step-verifier)
  - [Lesson 6: Evaluation & Defending CV Metrics](#prompt-6--evaluation-phải-hiểu-mọi-con-số-trên-cv)
  - [Lesson 7: Backend Systems Engineering, FastAPI, Database & Security](#prompt-7--backend-api-db-security-frontend)
  - [Lesson 8: Limitations, Technical Debt & "Nếu làm lại"](#prompt-8--limitations-và-nếu-làm-lại)
- [2 Bài luyện phỏng vấn thực chiến (Prompts 9 – 10)](#-2-bài-luyện-phỏng-vấn-thực-chiến)
  - [Prompt 9: Mock Interview AI Engineer](#prompt-9--mock-interview)
  - [Prompt 10: "Bắt chết Vibe-Coder" (Anti-Vibe-Coding Torture Test)](#prompt-10--bắt-chết-vibe-coder)
- [Lộ trình học tập đề xuất (Study Roadmap)](#-lộ-trình-học-tập-đề-xuất)

---

## 👑 Prompt 0 — Master Tutor (Dùng đầu tiên)

*Copy nguyên prompt này vào một chat mới có GitHub connector kết nối repo `XLaiHuy/Multi-Agent-RAG` (commit `e5ffc0919d65d5ac0bce344f0d783b3752960c5f`):*

```text
Bạn là Senior AI/ML Engineer và technical interviewer đang giúp tôi thực sự hiểu project của chính tôi để chuẩn bị phỏng vấn.

Repo bắt buộc dùng làm source of truth:
GitHub: XLaiHuy/Multi-Agent-RAG
Commit cần ưu tiên:
e5ffc0919d65d5ac0bce344f0d783b3752960c5f

Bối cảnh:
- Tôi là sinh viên AI Engineer.
- Project này phần lớn được xây bằng AI-assisted/vibe coding.
- Tôi hiểu RAG ở mức cơ bản nhưng KHÔNG chắc mình hiểu đầy đủ implementation.
- Mục tiêu không phải học thuộc README, mà phải có khả năng giải thích, defend design decisions, đọc code và trả lời follow-up trong interview.
- Tôi cần phát hiện cả những chỗ README/docs nói khác runtime code.

QUY TẮC TUYỆT ĐỐI:
1. Trước khi giải thích bất kỳ subsystem nào, hãy inspect source code thật trong repo.
2. Runtime code > tests/evaluation artifacts > docs > README.
3. Nếu docs và implementation không khớp, phải ghi rõ:
   - Docs claim gì
   - Runtime thực sự làm gì
   - Tôi nên nói gì trong interview
4. Không tự suy diễn feature chưa implement.
5. Không gọi một component là production-ready nếu code không chứng minh được.
6. Mỗi thuật ngữ mới phải giải thích bằng trực giác trước, sau đó mới tới kỹ thuật.
7. Luôn chỉ rõ file/class/function liên quan.
8. Không dump hàng trăm dòng code. Chỉ lấy đoạn logic quan trọng và diễn giải bằng pseudocode.
9. Với metric, giải thích:
   - metric đo cái gì
   - numerator/denominator
   - tại sao project cần metric đó
   - metric cao/thấp nói lên điều gì
10. Sau mỗi lesson, kiểm tra tôi bằng câu hỏi interview. Đừng đưa đáp án ngay.

PHƯƠNG PHÁP DẠY:

Mỗi topic phải theo format:

A. Intuition
Giải thích như tôi mới biết RAG.

B. Role in my system
Component này giải quyết vấn đề gì trong Multi-Agent Safe-RAG?

C. Actual runtime flow
Trace từ input → function/class → output.

D. Concrete example
Dùng một câu hỏi hợp đồng thực tế, ví dụ:
"What is the liability cap and does it exclude gross negligence?"

E. Why this design?
Tại sao dùng giải pháp hiện tại thay vì giải pháp đơn giản hơn?

F. Alternatives & trade-offs
Ví dụ:
- dense only vs hybrid
- RRF vs weighted score
- bi-encoder vs cross-encoder
- flat chunking vs parent-child
- one-shot RAG vs agentic RAG

G. Failure modes
Nó có thể fail ở đâu?

H. Interview answer
Cho tôi:
- bản 20 giây
- bản 60 giây
- bản deep-dive 2 phút

I. Checkpoint
5 câu hỏi từ easy → hard.
Dừng và chờ tôi trả lời.

MỤC TIÊU CUỐI:
Sau khóa học, tôi phải có thể tự vẽ và giải thích từ đầu đến cuối:

Upload document
→ parsing/OCR
→ canonical representation
→ parent-child chunking
→ embedding/indexing
→ document scoping
→ BGE-M3 dense retrieval
→ BM25
→ RRF
→ CrossEncoder
→ parent expansion
→ Planner
→ Critic
→ generation
→ Verifier
→ citation/refusal
→ ACL/security
→ API/frontend
→ evaluation
→ metrics
→ limitations

Đầu tiên KHÔNG dạy chi tiết ngay.

Hãy:
1. Audit repo.
2. Tạo system map từ code thật.
3. Chia thành curriculum 8 lesson.
4. Chỉ rõ prerequisite knowledge của từng lesson.
5. Chỉ ra những claim nào trong README/docs tôi cần đặc biệt cẩn thận khi phỏng vấn.
6. Sau đó bắt đầu Lesson 1.
```

---

## 📚 8 Bài học chuyên sâu

### Prompt 1 — Toàn cảnh RAG và Request Lifecycle

> **Trọng tâm**: Hiểu rõ hai luồng dữ liệu độc lập (Ingestion Lifecycle vs Query Lifecycle), vai trò của từng tầng, chuyển đổi biểu diễn dữ liệu và trả lời project walk-through.

```text
Hãy dạy tôi Lesson 1: End-to-End Architecture của repo XLaiHuy/Multi-Agent-RAG.

Không đi sâu code từng thuật toán trước.

Tôi cần hiểu chính xác một user request đi qua hệ thống như thế nào.

Trace hai lifecycle riêng:

1. INGESTION:
upload contract
→ API
→ parsing
→ canonical document
→ chunking
→ embedding
→ dense/BM25 indexing
→ persistence
→ ready

2. QUERY:
user chooses document + asks question
→ authentication/ACL
→ planner
→ retrieval
→ dense + BM25
→ RRF
→ reranking
→ parent expansion
→ critic
→ generation
→ verifier
→ citations/refusal
→ response

Đối với từng bước:
- input là gì?
- output là gì?
- class/function/file nào chịu trách nhiệm?
- dữ liệu thay đổi representation như thế nào?
- bước đó tồn tại để giải quyết vấn đề gì?

Vẽ một ASCII architecture diagram.

Sau đó dùng một contract giả và câu:
"What is the liability cap and does it exclude gross negligence?"
để trace toàn bộ query.

Cuối cùng:
- cho tôi bản giải thích project 30 giây
- 60 giây
- 2 phút
- hỏi tôi 7 câu interview và chờ tôi trả lời.
```

---

### Prompt 2 — Ingestion, OCR, CanonicalDocument, Parent–Child Chunking

> **Trọng tâm**: Xử lý tài liệu phi cấu trúc, cấu trúc phân cấp Canonical representation, Parent-Child chunking (~250 tok / ~1200 tok), và phân tích trung thực hiện trạng OCR.

```text
Dạy tôi subsystem ingestion của XLaiHuy/Multi-Agent-RAG từ code thật.

Tập trung vào:
- supported document formats
- NativePDFParser
- MasterDocumentParser
- OCRGatingAnalyzer
- OCRProvider
- CanonicalDocument / CanonicalPage / CanonicalBlock
- headings, tables, bbox, section_path
- StructureAwareParentChildChunker
- child chunks
- parent chunks
- overlap
- metadata
- indexing boundaries

Đặc biệt kiểm tra kỹ OCR.

Tôi biết parser có OCR-related implementation nhưng hãy xác định:
1. OCR code tồn tại ở đâu?
2. OCR gating quyết định như thế nào?
3. OCRProvider được truyền ở đâu?
4. Runtime ingestion hiện tại có thực sự truyền OCRProvider hay không?
5. Do đó feature OCR hiện ở trạng thái:
   implemented / partially wired / runtime active?
6. Tôi được phép nói gì và KHÔNG được nói gì khi interview?

Giải thích tại sao legal RAG không nên dùng naive fixed-character splitting.

Dùng một clause dài có:
Section 8 - Limitation of Liability
8.1 ...
8.2 ...
8.3 ...

và minh họa nó biến thành parent và child chunks như thế nào.

Cuối cùng hỏi tôi 7 câu interview.
```

---

### Prompt 3 — Embedding, BM25, Hybrid Retrieval, RRF, Reranker

> **Trọng tâm**: First principles của Dense vs Sparse, cơ chế toán học của BM25 và RRF ($k=60$), tại sao dùng Bi-Encoder + Cross-Encoder 2-stage reranking, và bài toán số cụ thể.

```text
Dạy tôi retrieval pipeline của XLaiHuy/Multi-Agent-RAG từ first principles đến implementation.

Tôi cần hiểu sâu nhưng trực quan:

1. Embedding là gì?
2. BGE-M3 biến query/document thành gì?
3. Dense similarity hoạt động thế nào?
4. Dense retrieval giỏi và yếu ở đâu?
5. BM25 hoạt động trực giác thế nào?
6. TF, IDF, term saturation và document length normalization là gì?
7. Tại sao legal documents cần BM25 song song với dense retrieval?
8. RRF là gì?
9. Giải thích công thức:
   score(d) = Σ 1/(k + rank_i(d))
10. Tại sao k=60?
11. Vì sao RRF dùng rank thay vì raw cosine/BM25 scores?
12. CrossEncoder khác bi-encoder như thế nào?
13. Tại sao retrieve nhiều rồi rerank ít?
14. Parent expansion xảy ra khi nào?

Sau đó inspect source thật và trace:

query
→ dense top-k
→ BM25 top-k
→ RRF
→ dedup
→ CrossEncoder
→ top-k
→ parent expansion

Cho một ví dụ với 5 candidate chunks và tự tính RRF bằng số cụ thể.

Sau đó giải thích các lựa chọn thay thế:
- dense only
- BM25 only
- weighted score fusion
- RRF
- rerank all chunks
- no reranker

Cuối cùng hỏi tôi 10 câu interview từ intern đến strong junior.
```

---

### Prompt 4 — Document-Scoped Retrieval, ACL và tại sao nó là contribution quan trọng

> **Trọng tâm**: Bản chất của Corpus-Wide vs Document-Scoped retrieval, hiện tượng Cross-contract collision (81.97% vs 28.67%), Multi-tenant security isolation, Anti-IDOR.

```text
Dạy tôi thật sâu về "document-scoped retrieval" trong XLaiHuy/Multi-Agent-RAG.

Tôi không muốn chỉ thuộc câu:
"scope retrieval to active document".

Hãy giải thích:

1. Corpus-wide retrieval là gì?
2. Cross-contract collision xảy ra như thế nào?
3. Tại sao legal contracts đặc biệt dễ bị collision?
4. doc_id filtering xảy ra trước hay sau retrieval?
5. Vì sao filtering trước retrieval tốt hơn retrieve-global-then-filter?
6. Dense index được scoped thế nào?
7. BM25 được scoped thế nào?
8. tenant_id / role / document ownership liên quan thế nào?
9. security boundary khác retrieval relevance boundary ở đâu?
10. anti-IDOR là gì?

Dùng ví dụ 3 contracts đều có "Limitation of Liability" nhưng số tiền khác nhau.

Trace code thật.

Sau đó giải thích benchmark:
- document-scoped Hit@10 = 81.97%
- corpus-wide collision baseline = 28.67%

Không chỉ nói số.
Giải thích experiment đang chứng minh hypothesis gì và không chứng minh gì.

Cuối bài hỏi tôi các câu interviewer có thể dùng để bắt lỗi architecture/security.
```

---

### Prompt 5 — Multi-Agent: Planner, Critic, Generator step, Verifier

> **Trọng tâm**: Kiến trúc agentic thực tế trong runtime code, phân biệt 3 agent classes (Planner/Critic/Verifier) + generation step, cơ chế verification & refusal, và tính toán chi phí token/latency.

```text
Dạy tôi agentic layer của XLaiHuy/Multi-Agent-RAG từ CODE, không dựa riêng vào tên "Multi-Agent".

Tôi cần biết chính xác:

- Retrieval Planner làm gì?
- input/output schema?
- direct QA vs comparison vs risk review?
- Evidence Critic làm gì?
- evidence sufficiency nghĩa là gì?
- khi nào retrieval lại?
- retry/retrieval loop bounded như thế nào?
- generation xảy ra ở đâu?
- Generator có phải một Agent class runtime riêng không?
- Answer Verifier làm gì?
- verifier kiểm citation/grounding như thế nào?
- insufficient evidence/refusal được quyết định thế nào?
- LLM calls/query được bounded ra sao?

Hãy đặc biệt audit mismatch giữa:
- README
- docs
- runtime code

Nếu README gọi Planner/Critic/Generator/Verifier nhưng runtime chỉ có 3 agent classes và một generation step, giải thích chính xác cách tôi nên diễn đạt.

Sau đó so sánh:

Traditional RAG:
retrieve → prompt → LLM

vs

My bounded agentic RAG:
plan → retrieve → critique → optional retrieve → generate → verify

Giải thích:
- lợi ích
- latency/cost penalty
- khi nào overengineering
- tại sao không cần LangGraph
- LangGraph sẽ thay đổi orchestration như thế nào nếu dùng

Cuối cùng hỏi tôi 10 câu interview cực khó.
```

---

### Prompt 6 — Evaluation: Phải hiểu mọi con số trên CV

> **Trọng tâm**: Bảo vệ toàn bộ metric trên CV, hiểu rõ công thức, numerator, denominator, strict child mapping vs parent expansion, và protocol đánh giá độc lập (Layer A / Layer B).

```text
Tôi cần defend mọi metric của Multi-Agent-RAG đang ghi trên CV.

Dùng frozen evaluation artifacts của commit:
e5ffc0919d65d5ac0bce344f0d783b3752960c5f

Dạy tôi từng metric:

Retrieval:
- HitRate@1
- HitRate@5
- HitRate@10
- CandidateHitRate vs final HitRate
- MRR
- nDCG
- TrueChunkRecall
- Parent HitRate

Generation:
- strict balanced answerability accuracy (72.50%)
- inclusive balanced answerability accuracy (74.50%)
- answerable acceptance rate (67.00%)
- strict unanswerable refusal rate (78.00%)
- inclusive unanswerable refusal rate (82.00%)
- valid explicit citation compliance (98.51%)
- child citation hit (85.07%)
- child citation coverage (62.00%)
- parent citation hit (92.54%) / coverage (68.00%)
- citation precision macro (80.97%)
- citation precision micro (73.53%)
- citation recall (63.00%)
- wrong-document citation (0 / 140 observed)

Cho mỗi metric:
1. Definition
2. Formula
3. Tiny numerical example
4. What it measures
5. What it DOES NOT measure
6. Why it matters in legal RAG
7. Exact result in my project
8. How an interviewer may challenge it

Sau đó giải thích dataset split:
- CUAD
- dev vs frozen holdout
- 25 unseen contracts
- N=294 retrieval
- N=200 end-to-end
- 100 answerable + 100 unanswerable
- why held-out contracts matter

Cuối cùng bắt tôi tự giải thích:
"81.97% HitRate@10 and 0.5214 MRR"
và
"72.5% balanced accuracy and 80.97% citation precision"

Chấm câu trả lời của tôi thật nghiêm.
```

---

### Prompt 7 — Backend, API, DB, Security, Frontend

> **Trọng tâm**: Kỹ thuật phần mềm (SWE) trong hệ thống AI: FastAPI, Clean Architecture (Domain/Service/Router/Provider), In-Memory vs Persistent storage, Multi-Tenant RBAC, SSE streaming.

```text
Dạy tôi phần AI Engineering/software engineering của XLaiHuy/Multi-Agent-RAG.

Trace actual source cho:

React frontend
→ FastAPI
→ auth
→ route
→ service
→ retrieval/agent pipeline
→ persistence
→ response/SSE

Tôi cần hiểu:
- FastAPI dùng để làm gì?
- router/service/provider/domain separation là gì?
- dependency injection đang làm ra sao?
- SQLAlchemy dùng ở đâu?
- SQLite lưu gì?
- ChromaDB lưu gì?
- BM25 index nằm ở đâu?
- JWT authentication
- tenant_id
- roles
- ACL
- anti-IDOR
- cache isolation
- SSE
- retry/rate limiting/circuit breaker nếu runtime có

Đừng mô tả PostgreSQL/Redis như production dependency nếu repo runtime hiện tại không thực sự deploy chúng.

Với mỗi component, giải thích:
"Why not just put everything in one Python script?"

Sau đó hỏi tôi system design interview questions:
- scale to 100k contracts?
- multiple tenants?
- concurrent users?
- vector DB growth?
- async ingestion?
- failed embedding jobs?
- LLM outage?
- duplicate uploads?
```

---

### Prompt 8 — Limitations và “Nếu làm lại”

> **Trọng tâm**: Nhìn nhận khách quan các điểm yếu kỹ thuật, technical debt, hạn chế của evaluation, trade-off chi phí/độ trễ, và trả lời câu hỏi kinh điển: *"Nếu có thêm 1 tháng, bạn sẽ cải tiến điều gì?"*.

```text
Audit XLaiHuy/Multi-Agent-RAG như một Senior AI Engineer khó tính.

Tôi muốn biết project hiện tại YẾU ở đâu.

Phân loại:

1. Correctness issues
2. Architecture debt
3. Docs/runtime mismatches
4. Evaluation limitations
5. Scalability limitations
6. Latency/cost limitations
7. OCR limitations
8. Security limitations
9. Retrieval limitations
10. Agentic-design limitations

Không được cố khen project.

Với mỗi vấn đề:
- current implementation
- why it is a limitation
- severity
- what I should say if interviewer finds it
- how I would improve it

Sau đó tạo câu trả lời cho:
"If you had another month, what would you improve?"

Tôi muốn câu trả lời có priority:
P0 correctness/security
P1 retrieval/evaluation
P2 UX/scalability
P3 advanced features

Cuối cùng hỏi tôi 10 adversarial interview questions.
```

---

## 🥊 2 Bài luyện phỏng vấn thực chiến

### Prompt 9 — Mock Interview

> **Cách dùng**: Chạy sau khi đã học xong 8 bài học trên để mô phỏng một buổi phỏng vấn kỹ thuật thực tế với Senior AI Engineer.

```text
Act as a Senior AI Engineer interviewer.

You have access to:
XLaiHuy/Multi-Agent-RAG
commit e5ffc0919d65d5ac0bce344f0d783b3752960c5f

Interview me for an AI Engineer Intern / Junior position.

RULES:
- Ask ONE question at a time.
- Do not give the answer before I respond.
- Start easy, then drill down based on my exact answer.
- If I use a buzzword, challenge me:
  "What does that actually mean?"
- If I mention an algorithm, ask for intuition or formula.
- If I mention a metric, ask how it was calculated.
- If I mention architecture, ask why not a simpler alternative.
- If I claim something inconsistent with repo code, call it out.
- Include coding/system-design questions when relevant.

Focus:
RAG
embeddings
BM25
RRF
CrossEncoder
chunking
document-scoped retrieval
multi-agent reasoning
citation grounding
evaluation
FastAPI
security
OCR
latency/cost
limitations

After each answer:
1. Score 0-10
2. What was correct
3. What was vague/wrong
4. Give a stronger answer
5. Ask the next deeper question.

Do not go easy on me.

Start with:
"Walk me through your Multi-Agent Safe-RAG project."
```

---

### Prompt 10 — “Bắt chết Vibe-Coder” (Anti-Vibe-Coding Torture Test)

> **Cách dùng**: Đặt giả định interviewer nghi ngờ ứng viên chỉ vibe-code và học thuộc lòng README; đào sâu vào chi tiết implementation, data structure, edge cases và discrepancy.

```text
Assume you suspect I vibe-coded this repository and may not understand my own code.

Repo:
XLaiHuy/Multi-Agent-RAG
commit e5ffc0919d65d5ac0bce344f0d783b3752960c5f

Your job is to expose shallow understanding.

Inspect the repo and create 30 questions that distinguish:

A. Someone who merely read README
from
B. Someone who actually understands and could maintain the system.

Questions must target:
- data structures
- exact execution order
- design rationale
- formulas
- failure modes
- code ownership boundaries
- config values
- input/output of components
- benchmark methodology
- discrepancies between docs and runtime
- features that look implemented but are not fully wired
- agent orchestration
- security boundaries
- latency/cost

Do NOT show answers yet.

Ask me one question at a time.
After I answer, assess whether my understanding is:
- Memorized
- Conceptual
- Implementation-level
- Interview-ready

Keep drilling until I can explain it without buzzwords.
```

---

## 🗓️ Lộ trình học tập đề xuất

```text
Day 1: Nền tảng kiến trúc & Tiền xử lý
├── Prompt 1 ──► End-to-End Architecture & Request Lifecycles
└── Prompt 2 ──► Ingestion, CanonicalDocument, OCR Status & Parent-Child Chunking

Day 2: Tìm kiếm kết hợp & Phạm vi an toàn
├── Prompt 3 ──► BGE-M3 Dense + BM25Okapi + RRF (k=60) + TinyBERT CrossEncoder
└── Prompt 4 ──► Document-Scoped Retrieval vs Collision Drop (81.97% vs 28.67%) + Multi-Tenant ACL

Day 3: Tác tử suy luận & Đo lường khoa học
├── Prompt 5 ──► Multi-Agent Execution Stack (Planner, Critic, Generator, Verifier)
└── Prompt 6 ──► Evaluation Metrics Defense (Strict Hit@10, MRR, Balanced Accuracy, Precision)

Day 4: Hệ thống kỹ thuật & Giới hạn
├── Prompt 7 ──► Backend System Design, FastAPI, Dependency Injection, Persistence
└── Prompt 8 ──► Limitations, Technical Debt & "Nếu làm lại trong 1 tháng"

Day 5+: Luyện phỏng vấn thực chiến
├── Prompt 9  ──► Full Mock Technical Interview
└── Prompt 10 ──► Anti-Vibe-Coding Deep Drill Test
```

### Tiêu chí vượt qua từng bài học
Bạn **chỉ nên chuyển sang bài tiếp theo** khi có thể tự trình bày trôi chảy bằng lời mà không cần nhìn tài liệu.

*Ví dụ trước khi qua Lesson Retrieval, bạn phải tự nói được câu sau:*
> *"Khi người dùng truy vấn một hợp đồng, hệ thống giới hạn không gian ứng viên vào duy nhất tài liệu đó trước để tránh nhiễu chéo giữa các hợp đồng. Câu hỏi được encode cho dense retrieval bằng BGE-M3, đồng thời chạy BM25 lexical search. Hai danh sách xếp hạng được kết hợp bằng Reciprocal Rank Fusion ($k=60$) vì điểm cosine và điểm BM25 có phân phối khác nhau. Top ứng viên sau fusion được Cross-Encoder rerank chính xác, rồi đoạn bằng chứng con (~250 tokens) được mở rộng về đoạn cha (~1200 tokens) để cung cấp đầy đủ ngữ cảnh điều khoản cho bước sinh câu trả lời."*

---

## 🔗 Liên kết tài liệu tham khảo trong Repo

- [Architecture & Ingestion Pipeline](architecture.md)
- [CV Project Entry Source](cv-project-entry.md)
- [Evaluation Methodology & Benchmark Splits](evaluation.md)
- [Portfolio Summary & Engineering Decisions](portfolio-summary.md)
- [Multi-Tenant Security & ACL Proof](security.md)
- [Step-by-Step Reproducibility Guide](reproducibility.md)
- [Phase 6.1 Final Scientific Sign-Off](../evaluation/reports/PHASE6_1_FINAL_SCIENTIFIC_SIGNOFF.md)
- [Phase 4.2 Master Metric Integrity Report](../evaluation/reports/PHASE4_2_FINAL_METRIC_INTEGRITY.md)
