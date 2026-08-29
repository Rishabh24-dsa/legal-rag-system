# Legal Knowledge Graph Search

A retrieval-and-synthesis assistant for Indian legal/police procedure
(NDPS Act, BNSS, SOPs, circulars, etc.). It retrieves facts from a set of
knowledge-graph files (subject–relation–object triples) using a two-tier
BM25-based search, then has a local Ollama LLM synthesize an answer
strictly grounded in those facts, with per-session conversation memory for
follow-up questions.

## Architecture

```
main.py               FastAPI routes (HTTP layer only)
  -> rag_engine.py     Singleton accessor for the wrapper below
  -> legal_kg_engine.py   Conversation memory, prompt assembly, confidence scoring
       -> legal_search_engine.py   Master Index + LRU cache + BM25 retrieval
       -> llm_client.py            Ollama client + prompt templates
models.py              Pydantic request/response schemas
```

Retrieval is two-tier:
1. **Master Index** — a lightweight keyword/entity → file mapping, built
   once and refreshed incrementally (only files that are new or changed
   since the last build get rescanned).
2. **KG cache** — lazily loads the 2–3 most relevant files' full triple
   sets into an LRU cache and runs BM25 within them to find matching
   entities, then pulls their connected facts.

See `CHANGELOG.md` for the full list of fixes made during the code-quality
pass, and `Legal_KG_Interview_Prep.docx` for a deeper design/talking-points
walkthrough.

## Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy the env template and adjust if needed
cp .env.example .env

# 4. Make sure Ollama is running locally with the model pulled
ollama pull llama3.2:3b
ollama serve                    # if not already running

# 5. Run the API
python main.py
# -> http://localhost:8000  (docs at /docs)
```

## Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama server |
| `LLM_MODEL` | `llama3.2:3b` | Model name (must be pulled) |
| `LLM_TIMEOUT` | `60` | Seconds before an LLM call times out |
| `LLM_TEMPERATURE` | `0.0` | Sampling temperature |
| `BACKEND_HOST` / `BACKEND_PORT` | `0.0.0.0` / `8000` | Server bind address |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `RATE_LIMIT_PER_MINUTE` | `30` | Per-client rate limit on `/chat` |
| `CORS_ORIGINS` | `http://localhost:5173,http://localhost:3000` | Comma-separated allowed origins |
| `KG_DATA_PATH` | `knowledge_graphs` | Folder containing the KG JSON files |
| `DEBUG` | `false` | Set `true` to enable `/debug/chat` (dev only — returns raw tracebacks) |
| `FORCE_REBUILD` | `false` | Set `true` to force a full Master Index rescan on startup |

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Health check, model name, index status |
| `/chat` | POST | Ask a question (`message`, optional `session_id`, `clear_history`) |
| `/conversation/{session_id}` | GET | Get history for a session |
| `/conversation/{session_id}` | DELETE | Clear history for a session |
| `/conversation/new` | POST | Get a fresh `session_id` |
| `/suggestions` | GET | Sample questions |
| `/index/stats` | GET | Master Index stats |
| `/debug/chat` | POST | Only registered when `DEBUG=true` |

Interactive docs at `/docs` once the server is running.

## Knowledge graph data

Place KG JSON files (each with a `triples` or `edges` array of
`[subject, relation, object]`) anywhere under `knowledge_graphs/`
(subfolders are fine). `metadata.json` in that folder is a generated
index cache — don't hand-edit it, and don't commit it (it's gitignored).

`convert_pkl_to_triples.py` converts NetworkX pickle graphs into this
triples format, if that's your source data.

## Testing

`LegalKGWrapper` accepts an injectable `llm_client`, so you can test the
retrieval/prompt/memory logic without a live Ollama connection:

```python
from legal_kg_engine import LegalKGWrapper

class FakeClient:
    async def generate(self, prompt, system_instruction=None):
        return "stub answer"

wrapper = LegalKGWrapper(llm_client=FakeClient())
result = await wrapper.answer("What is the procedure for Zero FIR?", session_id="test")
