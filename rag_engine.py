"""
rag_engine.py — Provides the LegalKGWrapper singleton.

Thin indirection layer so main.py doesn't need to know how the engine is
constructed.
"""

import logging
from typing import Optional

from legal_kg_engine import LegalKGWrapper, KGResult

logger = logging.getLogger(__name__)

_engine: Optional[LegalKGWrapper] = None

def get_rag_engine() -> LegalKGWrapper:
    global _engine
    if _engine is None:
        _engine = LegalKGWrapper()
    return _engine

# Re-export for compatibility
RAGResult = KGResult