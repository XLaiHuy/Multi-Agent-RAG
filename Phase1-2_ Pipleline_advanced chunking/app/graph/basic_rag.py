from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from app.retrieval.hybrid_retriever import HybridRetriever
from app.generation.generator import LLMGenerator

class BasicState(TypedDict):
    query: str
    chunks: list[dict]
    answer: str

def get_retriever() -> HybridRetriever:
    if not hasattr(get_retriever, "_instance"):
        get_retriever._instance = HybridRetriever()
    return get_retriever._instance

def get_generator() -> LLMGenerator:
    if not hasattr(get_generator, "_instance"):
        get_generator._instance = LLMGenerator()
    return get_generator._instance

def retrieve_node(state: BasicState) -> dict:
    retriever = get_retriever()
    chunks = retriever.search(state["query"], top_k=3)
    return {"chunks": chunks}

def generate_node(state: BasicState) -> dict:
    generator = get_generator()
    answer = generator.generate_answer(state["query"], state["chunks"])
    return {"answer": answer}

builder = StateGraph(BasicState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("generate", generate_node)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

app = builder.compile()
