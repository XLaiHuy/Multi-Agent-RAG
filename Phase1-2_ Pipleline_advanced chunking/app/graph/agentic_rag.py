from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

from app.retrieval.vector_retriever import VectorRetriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.generation.generator import LLMGenerator


# 1. Define the State schema for Agentic RAG
class AgenticState(TypedDict):
    query: str                                                      # Current query being searched/processed
    original_query: str                                             # User's original input query
    chat_history: list[dict]                                        # Last N turns of conversation [{role, content}]
    routing_action: Literal["direct_answer", "retrieve_vector", "retrieve_hybrid"] # Decision by Agent Router
    routing_reasoning: str                                          # Reason for the decision
    chunks: list[dict]                                              # Retrieved text chunks
    grade: Literal["relevant", "not_relevant"]                      # Document relevance grade
    rewrite_count: int                                              # Counter for rewrite loops
    answer: str                                                     # Generated final answer
    verification_status: Literal["grounded", "hallucinated", "skipped"] # Subagent verification status
    verification_comment: str                                       # Subagent verification notes


# 2. Lazy initialization of dependencies
_vector_retriever = None
_hybrid_retriever = None
_generator = None

def get_vector_retriever() -> VectorRetriever:
    global _vector_retriever
    if _vector_retriever is None:
        _vector_retriever = VectorRetriever()
    return _vector_retriever

def get_hybrid_retriever() -> HybridRetriever:
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever()
    return _hybrid_retriever

def get_generator() -> LLMGenerator:
    global _generator
    if _generator is None:
        _generator = LLMGenerator()
    return _generator


# 3. Node Definitions

def agent_reasoning_node(state: AgenticState) -> dict:
    """
    Agent Router Node: Analyzes intent of the current query and decides action:
    'direct_answer', 'retrieve_vector', or 'retrieve_hybrid'.
    """
    query = state["query"]
    decision = get_generator().analyze_intent_and_decide(query)
    
    return {
        "routing_action": decision["action"],
        "routing_reasoning": decision["reasoning"]
    }


def retrieve_vector_node(state: AgenticState) -> dict:
    """
    Retrieves documents using Dense Vector Search (Vector-only).
    """
    query = state["query"]
    search_results = get_vector_retriever().search(query=query, top_k=5)
    
    chunks = [
        {
            "chunk_id": res.chunk_id,
            "text": res.text,
            "source": res.source,
            "similarity": res.similarity,
        }
        for res in search_results
    ]
    return {"chunks": chunks}


def retrieve_hybrid_node(state: AgenticState) -> dict:
    """
    Retrieves documents using Hybrid Search (Vector + BM25 + Cross-Encoder Rerank).
    """
    query = state["query"]
    search_results = get_hybrid_retriever().search(query=query, top_k=5, use_rerank=True)
    
    chunks = [
        {
            "chunk_id": res.chunk_id,
            "text": res.text,
            "source": res.source,
            "similarity": res.similarity,
        }
        for res in search_results
    ]
    return {"chunks": chunks}


def grade_documents_node(state: AgenticState) -> dict:
    """
    Grades document relevancy unless direct_answer was selected.
    """
    routing_action = state.get("routing_action", "retrieve_hybrid")
    if routing_action == "direct_answer":
        return {"grade": "relevant"}
        
    query = state["query"]
    chunks = state.get("chunks", [])
    
    grade = get_generator().grade_relevance(query=query, chunks=chunks)
    return {"grade": grade}


def rewrite_query_node(state: AgenticState) -> dict:
    """
    Rewrites the query when retrieved documents are irrelevant.
    """
    original_query = state["original_query"]
    rewrite_count = state.get("rewrite_count", 0)
    
    new_query = get_generator().rewrite_query(original_query)
    
    return {
        "query": new_query,
        "rewrite_count": rewrite_count + 1
    }


def generate_node(state: AgenticState) -> dict:
    """
    Generates the answer using retrieved chunks and chat_history for context.
    If direct_answer, generates conversational response (with history).
    """
    query = state["query"]
    routing_action = state.get("routing_action", "retrieve_hybrid")
    chunks = state.get("chunks", [])
    grade = state.get("grade", "relevant")
    chat_history = state.get("chat_history", [])

    # Build history prefix
    history_text = ""
    if chat_history:
        lines = []
        for turn in chat_history[-10:]:  # Giới hạn 10 tin nhắn gần nhất
            role = "Người dùng" if turn.get("role") == "user" else "Trợ lý"
            lines.append(f"{role}: {turn.get('content', '')}")
        history_text = "\n".join(lines)

    if routing_action == "direct_answer":
        if history_text:
            prompt = (
                f"Lịch sử hội thoại:\n{history_text}\n\n"
                f"Câu hỏi hiện tại: {query}\n\n"
                "Hãy trả lời ngắn gọn, lịch sự, có tham chiếu ngữ cảnh trước nếu phù hợp."
            )
        else:
            prompt = f"Bạn là một trợ lý AI thông minh. Hãy trả lời câu hỏi sau ngắn gọn, lịch sự:\n\n{query}"

        answer_res = get_generator().client.models.generate_content(
            model=get_generator().model,
            contents=prompt,
        )
        answer = answer_res.text if answer_res and answer_res.text else "Xin chào! Tôi có thể giúp gì cho bạn?"
    else:
        answer = get_generator().generate_answer(
            query=query, chunks=chunks, chat_history=history_text
        )
        if grade == "not_relevant":
            warning_prefix = (
                "[Cảnh báo: Tài liệu truy xuất không chứa đủ thông tin liên quan. "
                "Câu trả lời mang tính chất tham khảo chung.]\n\n"
            )
            answer = warning_prefix + answer

    return {"answer": answer}


def verify_answer_node(state: AgenticState) -> dict:
    """
    Verification Subagent Node: Checks if the generated answer is grounded in retrieved chunks.
    """
    routing_action = state.get("routing_action", "retrieve_hybrid")
    if routing_action == "direct_answer":
        return {
            "verification_status": "skipped",
            "verification_comment": "Bỏ qua kiểm định cho câu hỏi giao tiếp trực tiếp."
        }
        
    query = state["query"]
    chunks = state.get("chunks", [])
    answer = state.get("answer", "")
    
    verification = get_generator().verify_answer_groundedness(
        query=query, chunks=chunks, answer=answer
    )
    
    return {
        "verification_status": verification["status"],
        "verification_comment": verification["comment"]
    }


# 4. Routing Functions

def route_initial_action(state: AgenticState) -> str:
    """
    Routes from agent reasoning node based on decision: direct_answer, retrieve_vector, or retrieve_hybrid.
    """
    return state.get("routing_action", "retrieve_hybrid")


def route_after_grading(state: AgenticState) -> str:
    """
    Decides whether to rewrite query or proceed to answer generation.
    """
    grade = state.get("grade", "relevant")
    rewrite_count = state.get("rewrite_count", 0)
    
    if grade == "not_relevant" and rewrite_count < 1:
        return "rewrite_query"
    return "generate"


# 5. Build the Agentic RAG Graph

workflow = StateGraph(AgenticState)

# Add Nodes
workflow.add_node("agent_reasoning", agent_reasoning_node)
workflow.add_node("retrieve_vector", retrieve_vector_node)
workflow.add_node("retrieve_hybrid", retrieve_hybrid_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("rewrite_query", rewrite_query_node)
workflow.add_node("generate", generate_node)
workflow.add_node("verify_answer", verify_answer_node)

# Connect Edges
workflow.add_edge(START, "agent_reasoning")

# Conditional Edge from Router
workflow.add_conditional_edges(
    "agent_reasoning",
    route_initial_action,
    {
        "direct_answer": "generate",
        "retrieve_vector": "retrieve_vector",
        "retrieve_hybrid": "retrieve_hybrid",
    }
)

# Connect Retrievers to Grader
workflow.add_edge("retrieve_vector", "grade_documents")
workflow.add_edge("retrieve_hybrid", "grade_documents")

# Conditional Edge after Grading
workflow.add_conditional_edges(
    "grade_documents",
    route_after_grading,
    {
        "rewrite_query": "rewrite_query",
        "generate": "generate"
    }
)

# Rewrite loop back to Agent Router
workflow.add_edge("rewrite_query", "agent_reasoning")

# Verification & End
workflow.add_edge("generate", "verify_answer")
workflow.add_edge("verify_answer", END)

# Compile runnable app
app = workflow.compile()
