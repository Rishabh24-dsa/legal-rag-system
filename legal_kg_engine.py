"""
legal_kg_engine.py
Orchestration layer: conversation memory, prompt assembly, and calling the
LLM client. Wraps LegalKGSearchEngine (the retrieval engine) for use by the
FastAPI backend. English only — no language detection.
"""

import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from legal_search_engine import LegalKGSearchEngine, SearchResult
from llm_client import get_llm_client, build_grounded_prompt, build_history_only_prompt

logger = logging.getLogger(__name__)

# Confidence bands. These intentionally stay well short of 1.0: even a
# "strong" retrieval + successful synthesis is a heuristic estimate, not a
# calibrated probability, so it's capped at 0.95.
_MAX_SYNTHESIS_CONFIDENCE = 0.95
_MIN_SYNTHESIS_CONFIDENCE = 0.5
_HISTORY_ONLY_CONFIDENCE = 0.55
_RAW_FACTS_FALLBACK_CAP = 0.6
_NO_ANSWER_CONFIDENCE = 0.0


@dataclass
class KGResult:
    answer: str
    sources: List[str] = field(default_factory=list)
    confidence: float = 0.0
    detected_language: str = "en"


class ConversationMemory:
    """In-memory, per-session conversation history. Not persisted — history
    is lost on restart and doesn't share state across server instances.
    Fine for a local single-user tool; a multi-instance deployment would
    need this backed by Redis or a database, keyed by session_id."""

    def __init__(self, max_history: int = 10):
        self.sessions: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        self.max_history = max_history

    def add_message(self, session_id: str, role: str, content: str):
        self.sessions[session_id].append({"role": role, "content": content, "timestamp": None})
        # Keep only the last max_history user/assistant turns.
        if len(self.sessions[session_id]) > self.max_history * 2:
            self.sessions[session_id] = self.sessions[session_id][-self.max_history * 2:]

    def get_history(self, session_id: str, last_n_turns: int = 5) -> List[Dict[str, str]]:
        if session_id not in self.sessions:
            return []
        return self.sessions[session_id][-last_n_turns * 2:]

    def get_history_text(self, session_id: str, last_n_turns: int = 5) -> str:
        history = self.get_history(session_id, last_n_turns)
        if not history:
            return ""
        lines = ["### Previous Conversation:"]
        for msg in history:
            speaker = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{speaker}: {msg['content']}")
        return "\n".join(lines) + "\n"

    def clear_session(self, session_id: str):
        self.sessions.pop(session_id, None)

    def session_exists(self, session_id: str) -> bool:
        return session_id in self.sessions


def _compute_synthesis_confidence(result: SearchResult, num_facts_used: int) -> float:
    """Heuristic confidence for a successful retrieval + LLM synthesis,
    derived from Master Index file-match strength, within-file BM25
    strength, and how many facts were actually available to synthesize
    from. Not a calibrated probability — it's meant to rank "how well did
    retrieval do" rather than certify correctness."""
    file_component = min(result.file_match_score / 6.0, 1.0)
    bm25_component = min(result.bm25_top_score / 4.0, 1.0)
    facts_component = min(num_facts_used / 10.0, 1.0)
    weighted = 0.4 * file_component + 0.35 * bm25_component + 0.25 * facts_component
    span = _MAX_SYNTHESIS_CONFIDENCE - _MIN_SYNTHESIS_CONFIDENCE
    return round(_MIN_SYNTHESIS_CONFIDENCE + weighted * span, 2)


class LegalKGWrapper:
    def __init__(self):
        self.engine = LegalKGSearchEngine()
        self.memory = ConversationMemory(max_history=15)  # retain up to 15 turns server-side

    async def answer(self, query: str, session_id: Optional[str] = None,
                      include_history: bool = True) -> KGResult:
        if session_id is None:
            session_id = str(uuid.uuid4())

        self.memory.add_message(session_id, "user", query)

        result = self.engine.search(query, max_kgs=3)

        if not result.facts:
            return await self._answer_from_history_only(query, session_id, include_history)

        return await self._answer_from_facts(query, session_id, result, include_history)

    async def _answer_from_history_only(self, query: str, session_id: str, include_history: bool) -> KGResult:
        """No facts matched. Try answering from conversation history alone
        (handles pure follow-ups like "what about the penalty for that")
        before giving up."""
        if include_history and self.memory.session_exists(session_id):
            history_context = self.memory.get_history_text(session_id, last_n_turns=3)
            if history_context:
                try:
                    client = get_llm_client()
                    prompt = build_history_only_prompt(history_context, query)
                    answer_text = await client.generate(prompt)
                    self.memory.add_message(session_id, "assistant", answer_text)
                    return KGResult(answer=answer_text, sources=[], confidence=_HISTORY_ONLY_CONFIDENCE)
                except Exception as e:
                    logger.warning(f"[{session_id}] History-only answer failed: {e}")

        answer_text = "I do not know based on the available legal knowledge graph."
        return KGResult(answer=answer_text, confidence=_NO_ANSWER_CONFIDENCE)

    async def _answer_from_facts(self, query: str, session_id: str, result: SearchResult,
                                  include_history: bool) -> KGResult:
        facts_used = result.facts[:20]
        facts_context = "\n".join(facts_used)
        history_context = self.memory.get_history_text(session_id, last_n_turns=5) if include_history else ""

        try:
            client = get_llm_client()
            prompt = build_grounded_prompt(facts_context, query, history_context)
            answer_text = await client.generate(prompt)
            confidence = _compute_synthesis_confidence(result, len(facts_used))
        except Exception as e:
            logger.warning(f"[{session_id}] LLM synthesis failed, falling back to raw facts: {e}")
            answer_text = "Relevant facts:\n" + "\n".join(result.facts[:10])
            confidence = min(_compute_synthesis_confidence(result, len(facts_used)), _RAW_FACTS_FALLBACK_CAP)

        self.memory.add_message(session_id, "assistant", answer_text)
        return KGResult(answer=answer_text, sources=result.facts[:5], confidence=confidence)

    def clear_history(self, session_id: str):
        self.memory.clear_session(session_id)

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        return self.memory.get_history(session_id)

    def session_exists(self, session_id: str) -> bool:
        return self.memory.session_exists(session_id)
