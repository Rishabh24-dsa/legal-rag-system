"""
models.py — Pydantic request/response schemas for the Legal KG Search API.
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    # Kept for backward compatibility with older frontend builds; the backend
    # is English-only and intentionally ignores this field.
    language: Optional[str] = None
    clear_history: Optional[bool] = False


class SourceNode(BaseModel):
    title: str
    path: str
    node_id: str


class ChatResponse(BaseModel):
    answer: str
    detected_language: str
    sources: List[SourceNode] = []
    confidence: float
    session_id: Optional[str] = None
    conversation_turn: Optional[int] = None


class ConversationMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[str] = None


class ConversationHistoryResponse(BaseModel):
    session_id: str
    history: List[ConversationMessage]
    total_turns: int


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    total_nodes: int
    llm_model: str


class IndexStatsResponse(BaseModel):
    total_sections: int
    total_nodes: int
    top_level_sections: List[str]
    last_updated: Optional[str]


class SuggestedQuestionsResponse(BaseModel):
    questions: List[str]
