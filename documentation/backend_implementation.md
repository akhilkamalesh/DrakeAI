# Implementation Plan: DrakeAI Backend System Architecture

Implement the end-to-end DrakeAI backend system as specified in [data_flow.md](../product_specs/data_flow.md). This system will power hybrid semantic & metadata retrieval over Drake's catalog using PostgreSQL with `pgvector`, a 2-LLM LangGraph state machine, and a FastAPI REST service connected to the Streamlit UI.

---

## User Review Required

> [!IMPORTANT]
> **LLM Model & Provider Selection:**
> The plan configures the system to use local **Ollama** by default (`http://localhost:11434`), supporting models already present on your machine (`gpt-oss:latest` or `gemma3:270m`), with automatic fallback to OpenAI / Gemini if API keys are provided in `.env`.
> 
> We will configure `OLLAMA_MODEL` in `.env` (defaulting to `gpt-oss:latest` for high quality reasoning, or `gemma3:270m` for ultra-fast local turnaround).

> [!NOTE]
> **Dependencies to Install in `./venv`:**
> - `fastapi`
> - `pydantic`
> - `langgraph`
> - `langchain-core`
> - `langchain-ollama`

---

## Proposed Changes

```
DrakeAI/
├── backend/
│   ├── __init__.py
│   ├── config.py             # Environment configuration & settings
│   ├── db.py                 # PostgreSQL connection pool & context graph query builder
│   ├── embeddings.py         # 1024-d zero-padded SentenceTransformer embedder
│   ├── models.py             # Pydantic data contracts (Requests, Responses, Filters)
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py          # AgentState TypedDict definition
│   │   ├── nodes.py          # Node 1 (Guardrail/Intent), Node 2 (Retrieval), Node 3 (Response Formatter)
│   │   └── workflow.py       # LangGraph DAG definition and compiler
│   ├── main.py               # FastAPI application entrypoint (POST /api/chat, GET /health)
│   └── test_backend.py       # Integration verification script
└── main.py                   # Updated Streamlit UI to render rich sources & album art
```

---

### Dependencies & Environment

#### [MODIFY] [.env](../.env)
- Add backend server configurations:
  - `BACKEND_HOST=0.0.0.0`
  - `BACKEND_PORT=8000`
  - `OLLAMA_BASE_URL=http://localhost:11434`
  - `OLLAMA_MODEL=gpt-oss:latest`

---

### Backend Core & Database Layer

#### [NEW] [config.py](../backend/config.py)
- Defines application settings using `pydantic` and `dotenv`.
- Manages `DATABASE_URL`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `EMBEDDING_MODEL_NAME` (`all-MiniLM-L6-v2`), and `TARGET_EMBEDDING_DIM` (`1024`).

#### [NEW] [db.py](../backend/db.py)
- Manages a thread-safe connection pool using `psycopg2.pool.ThreadedConnectionPool`.
- Registers `pgvector` on connection checkout.
- Implements `execute_context_graph_query(query_vector, filters)` executing the exact context graph SQL from section 3 of `data_flow.md`:
  - Joins `stanza`, `track`, and `album`.
  - Filters on `personal_feel`, `release_year_before`, `release_year_after`, `album_name`, `album_type`.
  - Orders by `s.embedding <=> :query_vector::vector(1024) ASC LIMIT :limit`.

#### [NEW] [embeddings.py](../backend/embeddings.py)
- Implements `SentenceTransformerEmbedder` singleton.
- Loads `all-MiniLM-L6-v2` (384-d) and normalizes & pads with zeros to 1024 dimensions, maintaining exact parity with the seeded `stanza` table embeddings.

---

### Schemas & State Contracts

#### [NEW] [models.py](../backend/models.py)
- Defines Pydantic models matching section 3 and 4 of `data_flow.md`:
  - `MetadataFilters`:
    - `personal_feel`: Optional enum string matching the 10 categories
    - `release_year_before`: Optional int
    - `release_year_after`: Optional int
    - `album_name`: Optional str
    - `album_type`: Optional str
    - `limit`: int = 3
  - `IntentOutput`: `is_relevant: bool`, `rejection_message: Optional[str]`, `semantic_query: Optional[str]`, `metadata_filters: MetadataFilters`.
  - `SourceItem`: `track_name`, `album_name`, `release_date`, `album_art_url`, `spotify_url`, `quoted_stanzas`.
  - `ChatRequest`: `prompt: str`, `history: List[Dict[str, str]]`.
  - `ChatResponse`: `response: str`, `sources: List[SourceItem]`.

#### [NEW] [state.py](../backend/graph/state.py)
- Implements `AgentState` TypedDict:
  - `prompt`: str
  - `history`: List[Dict[str, str]]
  - `is_relevant`: bool
  - `rejection_message`: Optional[str]
  - `semantic_query`: Optional[str]
  - `metadata_filters`: Dict[str, Any]
  - `query_vector`: Optional[List[float]]
  - `retrieved_context`: List[Dict[str, Any]]
  - `final_response`: str
  - `sources`: List[Dict[str, Any]]

---

### LangGraph Workflow & Nodes

#### [NEW] [nodes.py](../backend/graph/nodes.py)
- **`guardrail_intent_node` (LLM Call #1):**
  - Evaluates user prompt and history against Drake discography/music scope.
  - If out-of-scope, sets `is_relevant=False` and standard rejection message.
  - If in-scope, extracts semantic query themes and metadata filters (`personal_feel`, date limits, album filters).
  - Robust JSON parser with Pydantic validation and heuristic regex fallback for local LLMs.
- **`hybrid_retrieval_node` (Deterministic Python Node):**
  - Computes 1024-d padded dense vector embedding using `SentenceTransformerEmbedder`.
  - Executes parameterized context graph SQL via `db.py`.
  - Stores retrieved rows in `retrieved_context`.
- **`response_formatter_node` (LLM Call #2):**
  - Synthesizes final conversational response answering the user prompt using retrieved stanzas.
  - Strictly cites Track Name, Album Name, and Release Year for quoted lyrics.
  - Assembles structured `sources` list with album artwork, Spotify URLs, and quoted stanzas.
- **`out_of_scope_node`:**
  - Early exit node populating `final_response` with the rejection message and empty sources.

#### [NEW] [workflow.py](../backend/graph/workflow.py)
- Builds and compiles the LangGraph `StateGraph`:
  - Entry node: `guardrail_intent`
  - Conditional edge:
    - If `is_relevant == False` -> `out_of_scope` -> `END`
    - If `is_relevant == True` -> `hybrid_retrieval` -> `response_formatter` -> `END`

---

### FastAPI Service & Integration

#### [NEW] [main.py (Backend)](../backend/main.py)
- Creates FastAPI instance with CORS middleware.
- `GET /health`: Health check verifying DB connection and embedder status.
- `POST /api/chat`: Takes `ChatRequest`, invokes compiled LangGraph workflow with `prompt` and `history`, and returns `ChatResponse`.

#### [MODIFY] [main.py (Frontend)](../main.py)
- Enhances existing Streamlit frontend to display rich sources returned from `/api/chat`:
  - Renders album artwork thumbnails.
  - Renders clickable Spotify track links.
  - Preserves full conversational history in `st.session_state`.

---

## Verification Plan

### Automated & Integration Tests
1. **Backend Unit & Integration Suite (`backend/test_backend.py`):**
   - Test 1: Health check endpoint `/health`.
   - Test 2: Out-of-scope query guardrail check (e.g. `"What is the capital of France?"` -> early exit rejection message).
   - Test 3: In-scope semantic query (e.g. `"Show me vulnerable lyrics about heartbreak"` -> validates context graph SQL retrieval and response citations).
   - Test 4: Filtered query (e.g. `"Show me stanzas with personal_feel 'Late-Night Confessional' before 2018"` -> verifies SQL filter constraints and pgvector similarity).

### Execution Command
```bash
./venv/bin/python -m backend.test_backend
```

### Manual Verification
- Launch FastAPI:
  ```bash
  ./venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
  ```
- Test query through running Streamlit app at `http://localhost:8501`.
- Verify rich responses, citations, Spotify links, and album cover images display in UI.
