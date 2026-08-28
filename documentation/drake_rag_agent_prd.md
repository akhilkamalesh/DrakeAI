# Product Requirements Document (PRD): Drake Lyric & Audio Intelligence Agent

## 1. Executive Summary
**Product Vision:** A highly specialized Retrieval-Augmented Generation (RAG) agent designed to explore the discography of recording artist Drake (approx. 190 singles and 11 studio albums). The application will allow users to query lyrics based on semantic meaning and filter results using technical musical attributes (e.g., tempo, energy, valence).
**Objective:** Deliver an end-to-end, locally hosted hybrid search system that seamlessly combines mathematical vector similarity (lyrical intent) with traditional relational data filtering (Spotify audio features) in a single workflow. 

---

## 2. Target Use Cases
* **Semantic Vibe Search:** A user queries for abstract concepts like, *"What are some profound Drake songs about loyalty?"* The system retrieves stanzas mapped to the conceptual vector space rather than relying on exact keyword matches.
* **Hybrid Feature Filtering:** A user queries, *"Show me introspective Drake lyrics, but only from upbeat, high-energy tracks released before 2018."* The system executes a simultaneous search filtering for semantic meaning alongside strict numeric constraints (`energy > 0.8`, `release_date < '2018-01-01'`).
* **Attribution & Citation:** The LLM generates natural chat responses that explicitly cite the track title, album, and release year for every stanza quoted.

---

## 3. System Architecture & Tech Stack

### Frontend: Streamlit
* **Purpose:** Rapid UI prototyping for a chat-based interface.
* **Functionality:** Captures user prompts, manages conversational history via `st.session_state`, and renders the LLM's response. Decoupled from the backend, communicating exclusively via REST API.

### Backend: FastAPI & Pydantic
* **Purpose:** Asynchronous API orchestration.
* **Functionality:** Receives frontend queries, validates LLM parameter outputs using strict Pydantic schemas, and manages I/O operations without blocking concurrent requests. 

### Database: PostgreSQL with `pgvector`
* **Purpose:** Unified storage for both relational metadata and dense vectors, eliminating the need for a separate vector database.
* **Functionality:** Executes single SQL queries that combine vector similarity math (`<->`) with relational `JOIN` operations across tables (`albums`, `songs`, `stanzas`).

### Agent Router: Kimi (Open Source)
* **Purpose:** Intent extraction and tool execution.
* **Functionality:** Analyzes the user's prompt, determines the required filters (e.g., date ranges, audio feature thresholds), and generates the hybrid SQL query to pass to the PostgreSQL database tool.

---

## 4. Data Strategy & Ingestion Pipeline

To ensure low latency and rate-limit resilience, data will not be fetched live from the Spotify API during user queries. Instead, a one-time ETL batch process will seed the PostgreSQL database.

### Phase 1: Batch API Extraction
* Query `GET /v1/artists/{id}/albums` to retrieve the catalog.
* Query `GET /v1/albums/{id}/tracks` to map all track IDs.
* Utilize `GET /v1/audio-features?ids=...` to batch-download metrics (tempo, energy, valence) for up to 100 tracks per request.
* Cache all API responses locally in a staging directory (`/data/raw_audio_features.json`).

### Phase 2: Structural Chunking
* Parse raw lyric files using a Python script. 
* Split text strictly by double line breaks (`\n\n`) to ensure chunks represent cohesive stanzas/verses.
* Assign sequential `chunk_index` values to enable small-to-big context window retrieval if required.

### Phase 3: Embedding & Database Seeding
* Pass each stanza through the embedding model to generate the dense vector representation.
* Upsert the structured records into the PostgreSQL schema, establishing foreign key relations between the `stanzas`, `songs`, and `albums` tables.

---

## 5. Future Roadmap & Extensibility

### V1.5: Machine Learning Classifications
* **Feature:** Integrate a custom ML classification model to append subjective categorization (e.g., `"Late Night Drive"`, `"Hype"`, `"Introspective"`) to individual stanzas.
* **Implementation:** Since the architecture utilizes PostgreSQL, this requires no database migration. A batch job will simply append a new `personal_feel` VARCHAR column to the `stanzas` table, instantly exposing it to the Kimi agent for future hybrid SQL queries.

### V2.0: Deep Audio Structure Tooling
* **Feature:** Enable the agent to answer deep music theory questions (e.g., beat switches, time signature changes).
* **Implementation:** Introduce a secondary tool for the Kimi agent to query highly structured JSON data from Spotify's `/v1/audio-analysis/{id}` endpoint, ensuring this heavy payload is kept separate from the lightweight vector search index.
