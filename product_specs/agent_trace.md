# Implementation Plan: Visualize Agent Breakdown in Chatbot

Visualize the complete agent reasoning and execution pipeline in the DrakeAI chatbot:
1. **Query Analysis & Acoustic Profiling:** Key word extraction, audio feature creation (acoustic targets, hard filters), and metadata limits.
2. **Hybrid Context Retrieval (Pulled Songs):** Candidate tracks and stanzas initially pulled from the database before vetting.
3. **Parent-Child Vetting Agent (Judge):** Verification decisions (`APPROVED` / `REJECTED`), confidence scores, and rationales evaluating candidate parent lyrics and acoustics.
4. **Interactive Lifecycle:** Show the agent thinking live as it progresses, and collapse the thinking breakdown into an accordion once the final result is ready.

---

## User Review Required

> [!IMPORTANT]
> **Collapsible Experience & Visual Format:**
> - While the agent is running, Streamlit's `st.status(..., expanded=True)` displays the live thinking steps in real time as the backend processes the query.
> - Once the response is synthesized, the status widget automatically updates to `state="complete", expanded=False`, collapsing into a sleek summary bar directly above the response.
> - Clicking this bar expands the full visual breakdown at any time.
> - Every prior assistant response in chat history will also feature a collapsible `st.expander` showing its complete agent trace.

---

## Proposed Changes

```
DrakeAI/
├── backend/
│   ├── graph/
│   │   ├── state.py         # Add pulled_tracks, vetting_decisions, agent_trace to AgentState
│   │   ├── nodes.py         # Capture extracted keywords, audio profile, pulled songs, vetting log, compile agent_trace
│   │   └── workflow.py      # Maintain graph execution
│   ├── models.py            # Add agent_trace to ChatResponse model
│   ├── main.py              # Add POST /api/chat/stream SSE/ndjson endpoint and include agent_trace in /api/chat
│   └── test_backend.py      # Unit and integration tests for agent trace & streaming endpoint
└── main.py                  # Streamlit UI: CSS styling, live thinking streaming, collapsible breakdown renderer
```

---

### Backend Schema & Graph Tracing

#### [MODIFY] [backend/graph/state.py](file:///Users/akhilkamalesh/Documents/DrakeAI/backend/graph/state.py)
- Extend `AgentState` TypedDict to include:
  - `extracted_keywords: List[str]`
  - `pulled_tracks: List[Dict[str, Any]]`
  - `vetting_decisions: List[Dict[str, Any]]`
  - `agent_trace: Dict[str, Any]`

#### [MODIFY] [backend/graph/nodes.py](file:///Users/akhilkamalesh/Documents/DrakeAI/backend/graph/nodes.py)
- **`guardrail_intent_node`**:
  - Extract salient query keywords (e.g. emotion keywords, era references, music style terms).
  - Package query analysis into state: extracted keywords, semantic query expansion, audio feature creation (targets, filters, sort direction), and metadata limits (limit, personal feel, era, album).
- **`hybrid_retrieval_node`**:
  - Record `pulled_tracks`: candidate tracks retrieved from the context graph (track title, album, year, similarity, hybrid score, valence, energy, tempo, danceability, lyric snippet).
- **`vet_track_node`**:
  - Record `vetting_decisions`: for each candidate evaluated by the vetting agent, record `track_name`, `album_name`, `status` (`APPROVED` or `REJECTED`), `confidence`, `reason`, and acoustic features.
  - Track retries if vetting rejection causes retrieval re-execution.
- **`response_formatter_node`**:
  - Compile the structured `agent_trace` containing the complete 3-stage breakdown + reasoning analysis, and attach it to state.

#### [MODIFY] [backend/models.py](file:///Users/akhilkamalesh/Documents/DrakeAI/backend/models.py)
- Add `agent_trace: Optional[Dict[str, Any]] = None` to `ChatResponse`.

#### [MODIFY] [backend/main.py](file:///Users/akhilkamalesh/Documents/DrakeAI/backend/main.py)
- Update `/api/chat` to populate `agent_trace` in `ChatResponse`.
- Add `POST /api/chat/stream`:
  - Uses `StreamingResponse` with `application/x-ndjson`.
  - Streams events progressively as `chat_graph.stream(initial_state)` executes:
    - `intent`: query keywords, audio features, metadata limits
    - `retrieval`: pulled candidate songs
    - `vetting`: vetting judge decisions per track
    - `complete`: final response, sources, and full `agent_trace`

---

### Frontend Chatbot (Streamlit)

#### [MODIFY] [main.py](file:///Users/akhilkamalesh/Documents/DrakeAI/main.py)
- Add custom CSS for:
  - Agent thinking container (`.agent-breakdown-card`, `.stage-header`, `.pill-tag`, `.metric-chip`, `.vetting-card`).
  - Distinct badge colors for `APPROVED` (emerald green) and `REJECTED` (rose red).
  - Clean card styling for pulled songs with acoustic stats and lyric preview.
- Implement helper `render_agent_breakdown(trace: Dict[str, Any])`:
  - **Stage 1: Intent & Acoustic Profiling**: Displays extracted keywords as pills, audio feature targets & filters as acoustic chips, and metadata limits.
  - **Stage 2: Hybrid Retrieval (Pulled Songs)**: Displays all tracks pulled from the database before vetting with similarity %, hybrid score %, audio features, and lyric snippets.
  - **Stage 3: Parent-Child Vetting Agent**: Displays candidate cards with approval status, confidence score, and vetting rationale.
  - **Stage 4: Reasoning & Synthesis**: Displays the musicological reasoning analysis.
- Update Chat Execution Loop:
  - Calls `/api/chat/stream` (with graceful fallback to `/api/chat`).
  - Uses `with st.status("🧠 DrakeAI Agent Pipeline Thinking...", expanded=True) as status_box:`
  - Progressively renders each stage in the status container as it arrives.
  - Upon completion, calls `status_box.update(label="🧠 Agent Thinking Breakdown (Query Analysis → Pulled Songs → Vetting)", state="complete", expanded=False)` so the entire breakdown is neatly collapsed right above the final response.
  - Stores `agent_trace` in `st.session_state.messages` for each assistant turn so all past messages maintain collapsible thinking dropdowns.

---

## Verification Plan

### Automated Tests
- Run `unittest` in `backend/test_backend.py` to verify:
  - Extraction of keywords, audio targets, and metadata limits.
  - State tracking of pulled songs and vetting decisions.
  - `ChatResponse` serialization containing `agent_trace`.
  - Streaming endpoint `/api/chat/stream` yielding valid NDJSON events.
- Command:
  ```bash
  HF_HUB_OFFLINE=1 ./venv/bin/python -m unittest backend/test_backend.py
  ```

### Manual Verification
- Test user queries in Streamlit UI:
  1. Sad query: `"What are the top 3 saddest Drake songs?"`
     - Verify Stage 1 shows keywords (`sad`, `saddest`), audio target (`valence: 0.20`, `energy: 0.35`, `sort: valence_asc`), limit 3.
     - Verify Stage 2 shows pulled candidate songs (e.g. *Marvins Room*, *Doing It Wrong*, etc.) with acoustics.
     - Verify Stage 3 shows vetting judge approval/rejection with rationale and confidence.
     - Verify thinking box is open while generating and collapses upon completion.
     - Verify expanding the collapsed box displays the clean breakdown.
  2. Hype query: `"Show me high-energy club bangers"`
     - Verify high energy acoustic filters and targets.
  3. Filter query: `"Introspective songs before 2016"`
     - Verify era limit `release_year_before: 2016`.
