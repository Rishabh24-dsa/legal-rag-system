"""
llm_client.py — Local LLM client for the legal knowledge-graph assistant.

Talks to a local Ollama server (not a cloud API). This module used to be
named gemini_client.py with a GeminiClient class, a leftover from an early
project pivot away from the Gemini API — it has never called Gemini. It was
renamed here to match what the code actually does.
"""

import os
import json
import logging
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# Shared system instruction. Rule 1 used to read "You can use your own
# knowledge", which directly contradicted the later "do not hallucinate"
# instruction. Reworded so the model is only licensed to use general
# language ability to phrase/connect facts, not to introduce new claims.
_SYSTEM_INSTRUCTION = (
    "You are a legal assistant for Indian law. Answer strictly from the "
    "knowledge graph facts (and, if provided, conversation history) given "
    "to you below."
)

_GROUNDED_RULES = """### Rules:
1. Use only the facts provided below. You may use general reasoning to
   connect or phrase them, but do not introduce legal claims that are not
   present in the facts.
2. If multiple facts are relevant, synthesize them into a concise answer.
3. For relationship questions, clearly state the subject, relationship, and
   object.
4. For yes/no questions, answer with "Yes" or "No" followed by a brief
   justification from the facts.
5. Read the question carefully and address every part of it.
6. Format the answer in plain English prose; use bullet points only when
   listing multiple distinct items.
7. Do not print raw triples (e.g. "X --[relation]--> Y") in the output —
   translate them into natural language.
8. If a requested full form or definition is not present in the facts, say
   you don't know rather than guessing."""


def build_grounded_prompt(facts_context: str, question: str, history_context: str = "") -> str:
    """Prompt used for the main retrieval-grounded answer path."""
    context_block = ""
    if history_context:
        context_block += history_context + "\n"
    context_block += "### Knowledge Graph Facts (triples in format: subject --[relation]--> object):\n"
    context_block += facts_context

    return f"""{_SYSTEM_INSTRUCTION}

{_GROUNDED_RULES}

{context_block}

### Question:
{question}

### Answer:"""


def build_history_only_prompt(history_context: str, question: str) -> str:
    """Prompt used when retrieval finds no facts but prior conversation
    history exists (e.g. a pure follow-up like "what about the penalty for that")."""
    return f"""{_SYSTEM_INSTRUCTION}

The knowledge graph search returned no directly matching facts for this
question, so answer using the conversation history below if it resolves
the question (for example, a follow-up that refers to something already
established earlier). If the history doesn't answer it either, say so
politely rather than guessing.

{history_context}

### Current question:
{question}

### Answer:"""


class OllamaClient:
    """Thin async client for a local Ollama server's /api/chat endpoint."""

    def __init__(self):
        self.base_url = OLLAMA_BASE_URL.rstrip("/")
        self.model = LLM_MODEL
        self._verify_connection()

    def _verify_connection(self):
        """Best-effort startup check that Ollama is reachable and the model
        is pulled. Never raises — a slow/unavailable Ollama shouldn't block
        API startup, only degrade the /chat endpoint at request time."""
        import urllib.request

        try:
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=3) as r:
                data = json.loads(r.read())
                available = [m["name"] for m in data.get("models", [])]
                if not any(self.model in m for m in available):
                    logger.warning(f"Model '{self.model}' not found locally. Run: ollama pull {self.model}")
                else:
                    logger.info(f"Ollama model '{self.model}' available")
        except Exception as e:
            logger.warning(f"Cannot reach Ollama at {self.base_url}: {e}")

    async def generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": LLM_TEMPERATURE,
                "num_predict": 1024,
                "num_ctx": 4096,
            },
        }

        async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
            response = await client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            text = data.get("message", {}).get("content", "").strip()
            if not text:
                raise RuntimeError("Ollama returned an empty response")
            return text


_client: Optional[OllamaClient] = None


def get_llm_client() -> OllamaClient:
    global _client
    if _client is None:
        _client = OllamaClient()
    return _client
