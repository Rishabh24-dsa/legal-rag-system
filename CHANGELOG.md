# Changelog — code quality pass

Everything below is a real behavior/quality fix, cross-referenced to the
section of `Legal_KG_Interview_Prep.docx` that originally described it as
a "bug I found." The doc itself is unchanged — use this file to update
your talking points from "here's the bug I found" to "here's the bug I
found and fixed."

## Fixed

- **Naming mismatch (doc §9, "Gemini/Ollama naming story")** —
  `gemini_client.py` → `llm_client.py`, `GeminiClient` → `OllamaClient`,
  `get_gemini_client()` → `get_llm_client()`, and the `gemini_model` field
  on `HealthResponse`/`/health` → `llm_model`. The class never called
  Gemini; it only ever talked to a local Ollama server.

- **CORS wildcard + credentials bug (doc §9)** — `main.py` parsed
  `CORS_ORIGINS` into a list but never used it; `CORSMiddleware` was
  hardcoded to `allow_origins=["*"]` with `allow_credentials=True`, a
  combination browsers are spec-required to reject. Now the parsed
  `cors_origins` list is actually passed to the middleware.

- **Unauthenticated `/debug/chat` (doc §9)** — the route (raw tracebacks,
  no auth) is now only registered at all when `DEBUG=true` is set. With
  `DEBUG` unset/false it 404s instead of existing reachable-but-undocumented
  in a production deployment.

- **`BUILD_INDEX` always `True` → self-triggered dev-server reload (doc
  §9)** — `MasterIndex` now compares each KG file's mtime against what's
  stored in `metadata.json` and only re-scans files that are new or
  changed, instead of unconditionally rescanning everything and rewriting
  `metadata.json` on every startup. Verified: a second startup with no KG
  file changes logs "no changes detected" and does not touch
  `metadata.json`, so uvicorn's `--reload` watcher no longer sees a
  self-inflicted change right after boot. Set `FORCE_REBUILD=true` to opt
  back into an unconditional full rescan.

- **Hardcoded confidence constants (doc §9, §12)** — the four fixed
  floats (0.85 / 0.7 / 0.6 / 0.0) are replaced with a confidence computed
  from actual retrieval signal: the Master Index file-match score, the
  within-file BM25 top score, and how many facts were available to
  synthesize from (see `_compute_synthesis_confidence` in
  `legal_kg_engine.py`). It's explicitly documented in-code as a heuristic
  ranking signal, not a calibrated probability — that honesty point from
  the doc still stands, it's just no longer *also* a hardcoded-constant
  problem. History-only and "I don't know" paths keep fixed, clearly
  lower/zero values since there's no retrieval signal to derive from
  there.

- **"You can use your own knowledge" prompt ambiguity (doc §6)** — rule 1
  is reworded to the exact fix the doc proposed: "Use only the facts
  provided... you may use general reasoning to connect them, but do not
  introduce legal claims not present in the facts."

- **No `.gitignore` (doc §9)** — added, covering `.venv/`, `__pycache__/`,
  `*.pyc`, `tempCodeRunnerFile.py`, `.env`, and the generated
  `knowledge_graphs/metadata.json`.
  `tempCodeRunnerFile.py` (an empty VS Code leftover) is removed. The
  108MB `.venv/` and compiled `__pycache__/*.pyc` are excluded from this
  delivered copy, matching what `.gitignore` says shouldn't be
  version-controlled in the first place.

- **`env` file not actually loaded (new find, not in the doc)** — the
  config template was named `env`, not `.env`; `load_dotenv()` looks for
  `.env` by default, so unless the app was launched from a shell that had
  separately sourced it, none of those variables were actually reaching
  the process. Renamed to `.env.example` — copy it to `.env` to use it.

## Deliberately left alone

These are real, documented design trade-offs, not bugs — changing them
would make the doc's answers about them inaccurate, so they're untouched:
`CACHE_SIZE=3`, the 2:1 node-vs-keyword weighting in `find_relevant_kgs`,
the first-200-triples cutoff in Master Index keyword extraction,
`max_history=15` / `last_n_turns=5`, the `language` field being accepted
but ignored, the in-memory (non-persisted) `ConversationMemory`, and the
overall `main.py` → `rag_engine.py` → `legal_kg_engine.py` →
`legal_search_engine.py` layering.

## Also cleaned up (not previously flagged)

- Replaced ad-hoc `print()` debug output in `legal_search_engine.py` and
  `legal_kg_engine.py` with the `logging` module, respecting `LOG_LEVEL`
  — the old `sys.stdout` redirect hack in `LegalKGWrapper.__init__` used
  to suppress those prints is gone since there's nothing left to suppress.
- Removed a bare `except:` in the history-only fallback path (now
  `except Exception as e:` with a logged warning).
- Removed `GeminiClient.formulate_answer()` — dead code; nothing called
  it, and the two prompts that *are* used were duplicated inline in
  `legal_kg_engine.py`. Both now come from shared builders
  (`build_grounded_prompt` / `build_history_only_prompt`) in
  `llm_client.py`.
- Added type hints and docstrings throughout.
