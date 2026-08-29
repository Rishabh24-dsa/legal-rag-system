"""
main.py — FastAPI backend for Legal Knowledge Graph Search.
HTTP layer only: routes, request/response shaping. No business logic here —
that lives in rag_engine.py / legal_kg_engine.py / legal_search_engine.py.
English only – no multilingual support.
"""

import logging
import os
import time
import traceback
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from models import (
    ChatRequest, ChatResponse, HealthResponse, IndexStatsResponse,
    SuggestedQuestionsResponse, SourceNode, ConversationHistoryResponse,
    ConversationMessage,
)
from rag_engine import get_rag_engine

load_dotenv()

log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# /debug/chat returns full internal state, including tracebacks. It's a
# genuinely useful dev aid, but must not ship reachable in production —
# gate it behind an explicit flag rather than the endpoint's mere existence.
DEBUG_ENDPOINTS_ENABLED = os.getenv("DEBUG", "false").lower() == "true"

limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Legal KG Search API...")
    try:
        get_rag_engine()
        logger.info("Legal KG engine ready")
    except Exception as e:
        logger.error(f"Engine init error: {e}")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Legal Knowledge Graph Search API",
    description="Master Index + Lazy Loading + LRU Cache + BM25 Search with Conversation Memory",
    version="2.2.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

cors_origins = os.getenv(
    "CORS_ORIGINS", "http://localhost:5173,http://localhost:3000,http://localhost:8000"
).split(",")
app.add_middleware(
    CORSMiddleware,
    # Previously hardcoded to ["*"] with allow_credentials=True — that
    # combination is invalid per the CORS spec (browsers must reject a
    # wildcard origin when credentials are allowed), so credentialed
    # requests could silently fail. Use the parsed, configurable origin
    # list instead.
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({time.time() - start:.2f}s)")
    return response


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    try:
        engine = get_rag_engine()
        stats = engine.engine.get_stats()
        return HealthResponse(
            status="ok",
            index_loaded=True,
            total_nodes=stats["master_index_stats"]["total_files"],
            llm_model=os.getenv("LLM_MODEL", "llama3.2:3b"),
        )
    except Exception as e:
        return HealthResponse(status=f"degraded: {e}", index_loaded=False, total_nodes=0, llm_model="unknown")


@app.get("/index/stats", response_model=IndexStatsResponse, tags=["Knowledge Base"])
async def index_stats():
    engine = get_rag_engine()
    stats = engine.engine.get_stats()
    master = stats["master_index_stats"]
    return IndexStatsResponse(
        total_sections=master["total_files"],
        total_nodes=master["total_triples"],
        top_level_sections=list(engine.engine.master_index.file_metadata.keys()),
        last_updated=None,
    )


@app.get("/suggestions", response_model=SuggestedQuestionsResponse, tags=["Chat"])
async def suggestions():
    return SuggestedQuestionsResponse(questions=[
        "What are the NDPS circulars on sampling?",
        "Tell me about the Bureau of Police Research and Development",
        "Explain the BNS Act",
        "What is BNSS?",
        "Give me information about forensic science laboratory procedures",
        "What is the procedure for sealing exhibits?",
        "Who publishes the FSL circulars?",
        "Tell me more about the NDPS Act",
        "What are the penalties under NDPS?",
    ])


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
@limiter.limit(f"{os.getenv('RATE_LIMIT_PER_MINUTE', '30')}/minute")
async def chat(request: Request, body: ChatRequest):
    session_id = body.session_id or str(uuid.uuid4())
    logger.info(f"[{session_id}] '{body.message[:60]}' (clear_history={body.clear_history})")

    try:
        engine = get_rag_engine()

        if body.clear_history:
            engine.clear_history(session_id)
            logger.info(f"[{session_id}] History cleared")

        result = await engine.answer(body.message, session_id=session_id, include_history=True)

        history = engine.get_history(session_id)
        conversation_turn = len(history) // 2  # each turn is one user + one assistant message

    except Exception as e:
        logger.error(f"[{session_id}] RAG engine exception:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"AI engine error: {type(e).__name__}: {e}")

    return ChatResponse(
        answer=result.answer,
        detected_language="en",
        sources=[SourceNode(title=s[:100], path="", node_id=str(i)) for i, s in enumerate(result.sources)],
        confidence=result.confidence,
        session_id=session_id,
        conversation_turn=conversation_turn,
    )


@app.get("/conversation/{session_id}", response_model=ConversationHistoryResponse, tags=["Chat"])
async def get_conversation_history(session_id: str):
    """Get conversation history for a session."""
    engine = get_rag_engine()
    history = engine.get_history(session_id)
    messages = [
        ConversationMessage(role=msg["role"], content=msg["content"], timestamp=None)
        for msg in history
    ]
    return ConversationHistoryResponse(session_id=session_id, history=messages, total_turns=len(messages) // 2)


@app.delete("/conversation/{session_id}", tags=["Chat"])
async def clear_conversation(session_id: str):
    """Clear conversation history for a session."""
    engine = get_rag_engine()
    engine.clear_history(session_id)
    return {"status": "success", "message": f"Conversation history cleared for session {session_id}", "session_id": session_id}


@app.post("/conversation/new", tags=["Chat"])
async def new_conversation():
    """Generate a new session ID for a fresh conversation."""
    return {"session_id": str(uuid.uuid4()), "message": "New conversation started"}


if DEBUG_ENDPOINTS_ENABLED:
    @app.post("/debug/chat", tags=["Debug"])
    async def debug_chat(request: Request, body: ChatRequest):
        """Dev-only: same engine call as /chat, but returns full internal
        state including raw tracebacks. Only registered when DEBUG=true —
        never gated by client-side checks alone, since a stack trace leaks
        file paths and library versions to anyone who can reach it."""
        session_id = body.session_id or "debug_" + str(uuid.uuid4())[:8]
        debug_info = {
            "message": body.message,
            "session_id": session_id,
            "engine_ready": True,
            "error": None,
            "traceback": None,
            "answer": None,
            "has_history": False,
        }
        try:
            engine = get_rag_engine()
            result = await engine.answer(body.message, session_id=session_id, include_history=True)
            debug_info["answer"] = result.answer
            debug_info["confidence"] = result.confidence
            debug_info["sources"] = result.sources
            debug_info["has_history"] = engine.session_exists(session_id)
        except Exception as e:
            debug_info["error"] = f"{type(e).__name__}: {e}"
            debug_info["traceback"] = traceback.format_exc()
            logger.error(f"Debug chat error:\n{traceback.format_exc()}")
        return JSONResponse(content=debug_info)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("BACKEND_HOST", "0.0.0.0"),
        port=int(os.getenv("BACKEND_PORT", "8000")),
        reload=True,
    )
