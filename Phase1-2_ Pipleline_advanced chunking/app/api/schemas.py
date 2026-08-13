from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class SourceChunk(BaseModel):
    chunk_id: str
    text: str
    score: Optional[float] = None

class ChatHistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    query: str = Field(..., description="The question asked by the user")
    conv_id: Optional[str] = Field(default=None, description="Conversation session ID")
    chat_history: List[ChatHistoryItem] = Field(
        default_factory=list,
        description="Last N messages for conversation memory"
    )

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceChunk] = Field(default_factory=list)
    verification_status: Optional[str] = None
    processing_time_ms: Optional[float] = None
    cached: bool = False
