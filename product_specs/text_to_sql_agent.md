# Text-to-SQL Agent Specification

The **Text-to-SQL Agent** is an intelligent query synthesis component in the DrakeAI LangGraph pipeline. It operates between the Router (`guardrail_intent_node`) and the Knowledge Retrieval Layer (`hybrid_retrieval_node`), dynamically translating natural language queries, semantic expansions, audio profiles, and metadata constraints into flexible, parameterized PostgreSQL queries.

---

## 1. Problem Context

Previously, the retrieval layer relied on a static parameterized SQL query (`CONTEXT_GRAPH_SQL`) that strictly ranked results by vector cosine distance against `s.embedding`. This static approach suffered from significant limitations across common music discovery use cases:
- **Specific Track Queries:** Queries like *"Tell me about God's Plan"* or *"Play Headlines"* were forced through vector semantic comparison against lyric stanzas instead of filtering by track title (`t.name`).
- **Exact Lyric Matching:** Quoted lyric lookups (e.g. *"running through the 6 with my woes"*) often returned semantically similar vibes rather than pinning the exact track or stanza containing the lyric.
- **Acoustic Superlatives:** Inquiries for *"the saddest Drake song"* or *"fastest bangers"* were constrained by static ordering rather than dynamically sorting by acoustic feature extremes (e.g., `valence ASC`, `energy DESC`, `tempo ASC`).

---

## 2. Architecture & Pipeline Placement

The Text-to-SQL Agent is positioned directly after the Intent Router and before context retrieval:

```mermaid
flowchart TD
    User([User Query]) --> Router[Node 1: Guardrail & Intent Router]
    Router -->|is_relevant == false| OutOfScope[Early Exit: Guidance Message]
    Router -->|is_relevant == true| TextToSQL[Node 1.5: Text-to-SQL Agent]
    
    subgraph KnowledgeLayer["Knowledge Retrieval Layer"]
        TextToSQL --> Embedder[SentenceTransformer 1024-d]
        Embedder --> ExecSQL[Execute Dynamic SQL\nPostgreSQL + pgvector]
        ExecSQL --> HybridRank[Hybrid Audio/Semantic Scorer]
    end
    
    HybridRank --> Vetting[Node 2: Parent-Child Vetting Judge]
    Vetting -->|Rejected (retry < 2)| ExecSQL
    Vetting -->|Approved| Reasoning[Node 3: Drake's Thematic Reflection]
    Reasoning --> Formatter[Node 4: Response Formatter & Citations]
    Formatter --> Deliver([Streamlit UI & Stream])
```

---

## 3. Query Generation Strategies

Based on user intent, the Text-to-SQL Agent produces queries classified into 4 distinct strategies:

| Query Type | Trigger Pattern | SQL Strategy | Example Query |
|---|---|---|---|
| **`song_request`** | Mention of a Drake song title or *"song/track <title>"* | Filters `t.name ILIKE %(target_track_name)s` ordered sequentially by `s.chunk_index ASC`. | *"Tell me about God's Plan"* |
| **`lyric_match`** | Quoted strings (`"..."`) or lyrical phrases (*"lyrics where..."*, *"says..."*) | Filters `(s.lyric_chunk ILIKE %(target_lyric)s OR t.lyrics ILIKE %(target_lyric)s)` ordered by semantic similarity. | *"Find the song where Drake says 'running through the 6 with my woes'"* |
| **`acoustic_filter`** | Acoustic superlatives (*"saddest"*, *"highest energy"*, *"slowest"*) or explicit boundaries | Orders by acoustic columns (e.g. `COALESCE(taf.valence, af.valence) ASC`) alongside vector cosine similarity. | *"What are the top 3 saddest songs on Take Care?"* |
| **`hybrid_semantic`** | Thematic, mood, or open-ended inquiries | Orders by pgvector cosine distance (`s.embedding <=> %(query_vector)s::vector(1024) ASC`) with optional album/year constraints. | *"Introspective relationship lyrics from albums released before 2018"* |

---

## 4. Context Graph Projection Contract

To ensure downstream agents (`vet_track_node`, `reasoning_agent_node`, `response_formatter_node`) and UI source cards receive all necessary relational attributes, every generated query MUST project the standard context graph schema:

```sql
SELECT 
    s.id AS stanza_id,
    s.lyric_chunk,
    s.chunk_index,
    s.personal_feel,
    1 - (s.embedding <=> %(query_vector)s::vector(1024)) AS similarity,
    t.id AS track_id,
    t.name AS track_name,
    t.lyrics AS track_lyrics,
    t.disc_number,
    t.external_urls AS track_urls,
    a.id AS album_id,
    a.name AS album_name,
    a.album_type,
    a.release_date,
    a.images_url AS album_art_url,
    COALESCE(taf.valence, af.valence) AS valence,
    COALESCE(taf.energy, af.energy) AS energy,
    COALESCE(taf.danceability, af.danceability) AS danceability,
    COALESCE(taf.tempo, af.tempo) AS tempo,
    COALESCE(taf.acousticness, af.acousticness) AS acousticness,
    COALESCE(taf.loudness, af.loudness) AS loudness,
    COALESCE(taf.speechiness, af.speechiness) AS speechiness,
    COALESCE(taf.mode, af.mode) AS mode,
    COALESCE(taf.key, af.key) AS key,
    COALESCE(taf.instrumentalness, af.instrumentalness) AS instrumentalness,
    COALESCE(taf.liveness, af.liveness) AS liveness
FROM stanza s
JOIN track t ON s.song_id = t.id
JOIN album a ON t.album_id = a.id
LEFT JOIN track_audio_feature taf ON t.id = taf.track_id
LEFT JOIN audio_feature af ON t.id = af.song_id
WHERE (%(excluded_track_ids)s IS NULL OR NOT (t.id = ANY(%(excluded_track_ids)s)))
...
LIMIT %(fetch_limit)s;
```

---

## 5. Security & Safety Sandbox

All queries generated by the Text-to-SQL Agent are validated by `validate_sql_safety(sql: str)` in `backend/db.py` prior to execution:
1. **Read-Only Enforcement:** Queries must strictly start with `SELECT` or `WITH`.
2. **DDL/DML Rejection:** Any presence of mutating statements (`INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `TRUNCATE`, `CREATE`, `GRANT`, `EXECUTE`, etc.) is immediately blocked.
3. **Multi-Statement Prevention:** Disallows stacked statements separated by semicolons.
4. **Catalog Table Whitelisting:** Enforces that `FROM` and `JOIN` clauses access only authorized catalog tables: `stanza`, `track`, `album`, `track_audio_feature`, `audio_feature`, or declared CTEs.
5. **Parameter Binding:** Values are passed via named psycopg2 placeholders (`%(param)s`) to prevent SQL injection vulnerabilities.

---

## 6. Deterministic Fallback Engine

If the LLM call times out, is offline, or produces invalid SQL syntax, `_heuristic_text_to_sql` provides deterministic rule-based query construction:
- Matches track names against `KNOWN_TRACKS` while preventing false-positive matches on common musical words (e.g. *"energy"*, *"over"*).
- Extracts quoted lyric substrings and maps them to `s.lyric_chunk ILIKE %(target_lyric)s`.
- Applies acoustic superlatives and sort directions (`valence_asc`, `energy_desc`, `tempo_asc`).
- Automatically falls back to the default vector similarity query if no specific pattern is identified.

If dynamic query execution returns 0 rows due to overly strict conditions, `search_context_graph` automatically falls back to standard context graph retrieval to ensure the user is never stranded with an empty result.

---

## 7. Data Contracts & State Representation

### 7.1 Pydantic Model (`backend/models.py`)
```python
class SQLAgentOutput(BaseModel):
    sql: str = Field(description="Generated PostgreSQL query string.")
    params: Dict[str, Any] = Field(default_factory=dict, description="Named parameters dictionary.")
    explanation: str = Field(default="", description="Natural language description of query strategy.")
    query_type: str = Field(default="hybrid_semantic", description="Categorization of retrieval intent.")
```

### 7.2 LangGraph State (`AgentState`)
```python
class AgentState(TypedDict):
    ...
    generated_sql: Optional[str]
    sql_params: Optional[Dict[str, Any]]
    sql_explanation: Optional[str]
    query_type: Optional[str]
```

### 7.3 Trace & Streaming Event
```json
{
  "step": "sql_generation",
  "query": "SELECT ... FROM stanza s JOIN track t ON s.song_id = t.id ... WHERE t.name ILIKE %(target_track_name)s ...",
  "strategy": "Exact track name lookup for 'God's Plan' ordered by lyrical progression.",
  "query_type": "song_request",
  "params": {"target_track_name": "%God's Plan%"}
}
```

---

## 8. Frontend Visualization (Streamlit)

In the Streamlit chat interface (`main.py`):
1. **Live Thinking Container:** When streaming events from `/api/chat/stream`, displays:
   `🛠️ Stage 2 Complete: Dynamic Text-to-SQL query synthesized`
2. **Accordion Breakdown Card:** Renders a dedicated **Stage 2: Dynamic SQL Generation** card containing:
   - Retrieval strategy explanation.
   - Query type badge (`song_request`, `lyric_match`, `acoustic_filter`, `hybrid_semantic`).
   - Formatted SQL code block with syntax highlighting (`st.code(..., language="sql")`).
   - Bound parameters caption.