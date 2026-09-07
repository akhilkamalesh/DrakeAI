# Knowledge Layer

4. **Knowledge Retrieval Layer (Hybrid Query Builder):**
   - The backend encodes the semantic query string into a 1024-dimensional dense vector using the workspace embedder (`SentenceTransformer('all-MiniLM-L6-v2')` zero-padded to 1024 dimensions).
   - Executes a parameterized SQL query against PostgreSQL with `pgvector` (`<->` / `<=>`).
   - **4.1 Context Graph Retrieval:** The query performs a multi-table JOIN across `stanza`, `track`, and `album` to retrieve the complete context graph (stanza lyrics, sequential chunk index, personal feel, track metadata, album artwork, and Spotify URLs).

## Functionality:
    - Search context graph based on semantic similarity, audio feature similarity, and metadata constraints
    - This should take the approach of a hard filter as well as a weighted hybrid score to rank the remaining candidates based on the query