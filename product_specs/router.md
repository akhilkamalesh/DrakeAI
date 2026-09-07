# Router for context awareness

## Data Flow Step
**Context Awareness & Guardrail (Router):** The backend evaluates the query against the domain scope (Drake's discography, albums, and lyrics):
   - **Out-of-Scope:** If the query is unrelated, the system halts early and returns a domain-specific guidance message:  
     *"I specialize in Drake's discography, albums, and lyrics. Please ask a question related to Drake's music!"*
   - **In-Scope:** The system extracts structured query intent, semantic themes, and metadata constraints into a typed schema.

## Functionality
* **Extraction:** The system should extract structured query intent, create generic audio_features based on query, and have metadata constraints
    - The enhancement here would be to extract the keyword that is also extracted for semantic search and to generate audio features based on it
    - example: top 3 saddest drake songs:
        1. Saddest would be enhanced and the enhancement would be semantic query (keep as is)
        2. Saddest would also have a generated audio feature as to what would be classified as sad
        3. Metadata constraints (order by desc limit 3 since we are looking for the top 3)