#!/usr/bin/env python3
"""
legal_search_engine.py — Legal Knowledge Graph Search Engine
Master Index + Lazy Loading + LRU Cache + BM25 Search.

No FastAPI or LLM awareness lives here on purpose: this module can be
reused as a standalone CLI or plugged into a different chat layer unchanged.
"""

import json
import logging
import os
import re
import time
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()
logger = logging.getLogger(__name__)

# ========== Configuration ==========

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

KG_ROOT_ENV = os.getenv("KG_DATA_PATH", "")
if KG_ROOT_ENV:
    KG_ROOT = KG_ROOT_ENV if os.path.isabs(KG_ROOT_ENV) else os.path.join(_BASE_DIR, KG_ROOT_ENV)
else:
    KG_ROOT = os.path.join(_BASE_DIR, "knowledge_graphs")

MASTER_INDEX_FILE = os.path.join(KG_ROOT, "metadata.json")
CACHE_SIZE = 3

# Previously hardcoded to True, which forced a full corpus rescan (and a
# metadata.json rewrite) on every startup. uvicorn's --reload file watcher
# monitors this same project directory, so that rewrite was observed to
# trigger an unnecessary "change detected" restart right after boot. Set
# FORCE_REBUILD=true to opt back into an unconditional full rescan (e.g.
# after manually editing KG files without touching their mtimes).
FORCE_REBUILD = os.getenv("FORCE_REBUILD", "false").lower() == "true"

STOPWORDS = {
    "the", "and", "for", "with", "from", "this", "that", "were", "was",
    "what", "who", "where", "when", "why", "how", "is", "are", "be",
}

os.makedirs(KG_ROOT, exist_ok=True)
logger.info(f"Knowledge Graph root: {KG_ROOT}")


class MasterIndex:
    """Cheap, file-level index: maps keywords and entity names to the KG
    files that mention them, without holding any file's full triple set in
    memory. Rebuilt incrementally — only files that are new or changed
    since the last build are rescanned."""

    def __init__(self, root_folder: str = KG_ROOT, index_file: str = MASTER_INDEX_FILE):
        self.root_folder = root_folder
        self.index_file = index_file
        self.keyword_to_files: Dict[str, list] = defaultdict(list)
        self.node_to_files: Dict[str, list] = defaultdict(list)
        self.file_metadata: Dict[str, dict] = {}
        self._build_or_refresh_index()

    def _discover_kg_files(self) -> Dict[str, float]:
        """Return {relative_path: mtime} for every KG JSON file on disk."""
        found = {}
        for root, _dirs, files in os.walk(self.root_folder):
            for file in files:
                if not file.endswith(".json") or file == "metadata.json":
                    continue
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, self.root_folder)
                found[relative_path] = os.path.getmtime(file_path)
        return found

    def _build_or_refresh_index(self):
        on_disk = self._discover_kg_files()

        previous_metadata: Dict[str, dict] = {}
        previous_keyword_to_files: Dict[str, list] = defaultdict(list)
        previous_node_to_files: Dict[str, list] = defaultdict(list)
        if not FORCE_REBUILD and os.path.exists(self.index_file):
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                previous_metadata = saved.get("file_metadata", {})
                previous_keyword_to_files = defaultdict(list, saved.get("keyword_to_files", {}))
                previous_node_to_files = defaultdict(list, saved.get("node_to_files", {}))
            except Exception as e:
                logger.warning(f"Could not read existing master index, doing a full rebuild: {e}")

        changed_or_new = [
            path for path, mtime in on_disk.items()
            if FORCE_REBUILD
            or path not in previous_metadata
            or previous_metadata[path].get("mtime") != mtime
        ]
        removed = [path for path in previous_metadata if path not in on_disk]

        if not changed_or_new and not removed and previous_metadata:
            logger.info(f"Master index up to date: {len(previous_metadata)} KG files, no changes detected")
            self.file_metadata = previous_metadata
            self.keyword_to_files = previous_keyword_to_files
            self.node_to_files = previous_node_to_files
            return

        logger.info(
            f"Refreshing master index: {len(changed_or_new)} new/changed file(s), "
            f"{len(removed)} removed file(s)"
        )

        # Start from the previous index and patch in the delta, so unchanged
        # files never need to be re-read from disk.
        self.file_metadata = {k: v for k, v in previous_metadata.items() if k in on_disk}
        self.keyword_to_files = defaultdict(list, {
            kw: [p for p in paths if p in on_disk and p not in changed_or_new]
            for kw, paths in previous_keyword_to_files.items()
        })
        self.node_to_files = defaultdict(list, {
            node: [p for p in paths if p in on_disk and p not in changed_or_new]
            for node, paths in previous_node_to_files.items()
        })

        for relative_path in changed_or_new:
            file_path = os.path.join(self.root_folder, relative_path)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                triples = data.get("triples", data.get("edges", []))
                if not triples:
                    continue
                keywords, nodes = self._extract_keywords_and_nodes(triples)
                self.file_metadata[relative_path] = {
                    "path": file_path,
                    "triple_count": len(triples),
                    "keywords": list(keywords),
                    "nodes": list(nodes),
                    "folder": os.path.dirname(file_path),
                    "mtime": on_disk[relative_path],
                }
                for kw in keywords:
                    self.keyword_to_files[kw].append(relative_path)
                for node in nodes:
                    self.node_to_files[node].append(relative_path)
                logger.debug(f"{relative_path}: {len(triples)} triples, {len(keywords)} keywords")
            except Exception as e:
                logger.warning(f"Error reading {file_path}: {e}")

        self._save_index()
        logger.info(f"Master index ready: {len(self.file_metadata)} KG files")

    @staticmethod
    def _extract_keywords_and_nodes(triples: list) -> Tuple[Set[str], Set[str]]:
        # Only the first 200 triples per file are scanned for keyword/node
        # extraction (a deliberate cap on Master Index build cost). All of a
        # file's triples are still loaded and searchable once that file is
        # selected as a candidate — this cutoff only affects whether a very
        # large file's later, distinctive entities help it get *selected* in
        # the first place.
        keywords: Set[str] = set()
        nodes: Set[str] = set()
        for src, _rel, tgt in triples[:200]:
            src_lower, tgt_lower = src.lower(), tgt.lower()
            nodes.add(src_lower)
            nodes.add(tgt_lower)
            for word in src_lower.split() + tgt_lower.split():
                if len(word) >= 3 and word not in STOPWORDS:
                    keywords.add(word)
        return keywords, nodes

    def _save_index(self):
        index_data = {
            "keyword_to_files": dict(self.keyword_to_files),
            "node_to_files": dict(self.node_to_files),
            "file_metadata": self.file_metadata,
            "build_time": time.time(),
        }
        os.makedirs(self.root_folder, exist_ok=True)
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)

    def find_relevant_kgs(self, question: str, max_kgs: int = 3) -> List[Tuple[str, int, dict]]:
        """Score each KG file by keyword/node overlap with the question.
        A full entity-name (node) match counts double a loose keyword match
        — a heuristic weighting, not one tuned against a labeled eval set."""
        question_lower = question.lower()
        keywords = re.findall(r"\b[a-z]{3,}\b", question_lower)
        keywords = [k for k in keywords if k not in STOPWORDS]

        scores: Dict[str, int] = defaultdict(int)
        for kw in keywords:
            for file_path in self.keyword_to_files.get(kw, []):
                scores[file_path] += 1
        for kw in keywords:
            for file_path in self.node_to_files.get(kw, []):
                scores[file_path] += 2

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for file_path, score in ranked[:max_kgs]:
            if file_path in self.file_metadata:
                results.append((file_path, score, self.file_metadata[file_path]))
        return results

    def get_stats(self) -> dict:
        total_triples = sum(m.get("triple_count", 0) for m in self.file_metadata.values())
        return {
            "total_files": len(self.file_metadata),
            "total_triples": total_triples,
            "unique_keywords": len(self.keyword_to_files),
            "unique_nodes": len(self.node_to_files),
        }


class LRUCache(OrderedDict):
    def __init__(self, capacity: int):
        super().__init__()
        self.capacity = capacity

    def get(self, key):
        if key not in self:
            return None
        self.move_to_end(key)
        return self[key]

    def put(self, key, value):
        if key in self:
            self.move_to_end(key)
        self[key] = value
        if len(self) > self.capacity:
            oldest = next(iter(self))
            logger.debug(f"Cache evicted: {oldest}")
            del self[oldest]


class KGCache:
    """Lazily loads and BM25-indexes individual KG files, keeping only the
    CACHE_SIZE most recently used files' full triple sets in memory."""

    def __init__(self, cache_size: int = CACHE_SIZE):
        self.cache = LRUCache(cache_size)

    def load_kg(self, file_path: str) -> Dict:
        cached = self.cache.get(file_path)
        if cached:
            logger.debug(f"Cache hit: {os.path.basename(file_path)}")
            return cached

        logger.debug(f"Loading: {os.path.basename(file_path)}")
        start_time = time.time()
        full_path = os.path.join(KG_ROOT, file_path)
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            triples = data.get("triples", data.get("edges", []))
            kg_index = self._build_kg_index(triples)
            kg_index["triples"] = triples
            kg_index["file_path"] = file_path
            self.cache.put(file_path, kg_index)
            logger.debug(f"Loaded {len(triples)} triples in {time.time() - start_time:.2f}s")
            return kg_index
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return {"triples": [], "node_out": {}, "node_in": {}, "bm25": None, "node_names": []}

    @staticmethod
    def _build_kg_index(triples: List) -> Dict:
        node_out = defaultdict(list)
        node_in = defaultdict(list)
        node_names: Set[str] = set()
        for src, rel, tgt in triples:
            src_lower, tgt_lower = src.lower(), tgt.lower()
            node_names.add(src_lower)
            node_names.add(tgt_lower)
            node_out[src_lower].append((tgt_lower, rel.lower()))
            node_in[tgt_lower].append((src_lower, rel.lower()))
        node_names = list(node_names)
        tokenized_nodes = [name.split() for name in node_names]
        bm25 = BM25Okapi(tokenized_nodes) if tokenized_nodes else None
        return {
            "node_names": node_names,
            "bm25": bm25,
            "node_out": dict(node_out),
            "node_in": dict(node_in),
        }

    @staticmethod
    def search_in_kg(kg_index: Dict, question: str, max_facts: int = 30, top_k_nodes: int = 5) -> Tuple[List[str], float]:
        """Returns (facts, top_bm25_score). top_bm25_score is 0.0 when the
        substring-fallback path was used instead of a real BM25 match."""
        node_names = kg_index.get("node_names", [])
        bm25 = kg_index.get("bm25")
        node_out = kg_index.get("node_out", {})
        node_in = kg_index.get("node_in", {})
        if not bm25 or not node_names:
            return [], 0.0

        query_tokens = question.lower().split()
        if not query_tokens:
            return [], 0.0

        scores = bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k_nodes]
        matched_nodes = [node_names[i] for i in top_indices if scores[i] > 0]
        top_score = float(max((scores[i] for i in top_indices), default=0.0))

        if not matched_nodes:
            # BM25 found nothing sharing tokens with any node; fall back to a
            # plain substring check so a query still surfaces *something*.
            for node in node_names:
                if any(kw in node for kw in query_tokens):
                    matched_nodes.append(node)
                    if len(matched_nodes) >= top_k_nodes:
                        break

        facts = []
        for node in matched_nodes:
            for neighbor, rel in node_out.get(node, []):
                facts.append(f"{node} --[{rel}]--> {neighbor}")
            for pred, rel in node_in.get(node, []):
                facts.append(f"{pred} --[{rel}]--> {node}")

        seen = set()
        unique_facts = []
        for fact in facts:
            if fact not in seen:
                seen.add(fact)
                unique_facts.append(fact)

        return unique_facts[:max_facts], top_score

    def get_stats(self) -> dict:
        return {
            "cache_size": len(self.cache),
            "cache_capacity": self.cache.capacity,
            "cached_files": list(self.cache.keys()),
        }


@dataclass
class SearchResult:
    """Facts plus enough retrieval-quality signal to derive an honest
    confidence estimate downstream, instead of a hardcoded constant."""
    facts: List[str] = field(default_factory=list)
    file_match_score: int = 0       # best Master Index file score
    bm25_top_score: float = 0.0     # best within-file BM25 score
    files_searched: List[str] = field(default_factory=list)


class LegalKGSearchEngine:
    def __init__(self):
        logger.info("Legal Knowledge Graph Search Engine — Master Index + LRU Cache + BM25")
        self.master_index = MasterIndex()
        stats = self.master_index.get_stats()
        logger.info(
            f"Master Index: {stats['total_files']} KG files, {stats['total_triples']:,} triples, "
            f"{stats['unique_keywords']:,} keywords, {stats['unique_nodes']:,} nodes"
        )
        self.cache = KGCache()
        self.query_count = 0
        self.total_search_time = 0.0

    def search(self, question: str, max_kgs: int = 2) -> SearchResult:
        self.query_count += 1
        start_time = time.time()

        relevant_kgs = self.master_index.find_relevant_kgs(question, max_kgs)
        if not relevant_kgs:
            logger.debug(f"Query #{self.query_count}: no relevant KG files found")
            return SearchResult()

        result = SearchResult(file_match_score=relevant_kgs[0][1])
        for kg_path, _score, metadata in relevant_kgs:
            result.files_searched.append(kg_path)
            kg_index = self.cache.load_kg(kg_path)
            facts, bm25_top_score = self.cache.search_in_kg(kg_index, question, max_facts=20)
            result.facts.extend(facts)
            result.bm25_top_score = max(result.bm25_top_score, bm25_top_score)
            if len(result.facts) >= 30:
                break

        search_time = time.time() - start_time
        self.total_search_time += search_time
        logger.info(
            f"Query #{self.query_count}: {len(result.facts)} facts from "
            f"{len(result.files_searched)} file(s) in {search_time:.3f}s"
        )
        result.facts = result.facts[:30]
        return result

    def get_stats(self) -> dict:
        avg_time = self.total_search_time / self.query_count if self.query_count > 0 else 0
        return {
            "total_queries": self.query_count,
            "avg_search_time": avg_time,
            "cache_stats": self.cache.get_stats(),
            "master_index_stats": self.master_index.get_stats(),
        }
