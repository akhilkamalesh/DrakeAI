# Stanza Creation

This document will hold the specification for splitting up lyrics into smaller chunks and generating embeddings for them. The creation implements **Phase 2: Structural Chunking** as described in the [PRD](../../documentation/drake_rag_agent_prd.md).

## 1. Objectives:
* **Chunking:** Split up lyrics into smaller chunks of 5-7 lines each.
* **Embedding:** Apply vector embedding to the stanzas
* **Table Creation:** Create the stanza table based on schema.json

## 2. Embedding Model:
* **Embedding Model:** Sentence Transformers 'all-MiniLM-L6-v2' is the current embedding model used to generate embeddings for the lyric chunks. The model is a bi-encoder that encodes text into dense vector embeddings of size 1024. If this doesn't work, use any open source embedding model that is available locally

## 3. Personal Feel Model:
* **Personal Feel Model:** Local Llama 3.1 with QLoRA for classification for personal_feel. Categories are listed below
```json
[
  {
    "category": "Late-Night Confessional",
    "feel": "Moody, atmospheric, and emotionally vulnerable. Usually features stripped-down or underwater-sounding R&B production. The lyrics focus on longing, past relationships, and late-night regrets.",
    "example_tracks": [
      "Jungle",
      "Marvins Room",
      "Fire & Desire"
    ]
  },
  {
    "category": "Triumphant Flex",
    "feel": "High-energy, boastful, and aggressive. This is the sound of asserting dominance in the rap game, counting wins, and brushing off detractors.",
    "example_tracks": [
      "Energy",
      "Trophies",
      "Started From the Bottom"
    ]
  },
  {
    "category": "Paranoid & Guarded",
    "feel": "Anxious, defensive, and weary of fame. The focus is on fake friends, industry snakes, and the isolation that comes with being at the top.",
    "example_tracks": [
      "Fake Love",
      "Trust Issues",
      "No Friends In The Industry"
    ]
  },
  {
    "category": "Time-Stamp Introspection",
    "feel": "Stream-of-consciousness, highly lyrical, and deeply personal. These tracks (often named with a time and location) act as status updates where he directly addresses peers, airs grievances, and reflects on his career trajectory over stripped-back beats.",
    "example_tracks": [
      "5 AM in Toronto",
      "4 PM in Calabasas",
      "7am on Bridle Path"
    ]
  },
  {
    "category": "Toxic & Petty",
    "feel": "Bitter, vindictive, and unapologetically self-centered regarding romantic fallouts. Instead of longing, the tone is accusatory, pointing out an ex's flaws or flexing on how much better he is without them.",
    "example_tracks": [
      "Childs Play",
      "Jaded",
      "Hotline Bling"
    ]
  },
  {
    "category": "Global Groove / Island Infusion",
    "feel": "Tropical, rhythmic, and danceable. These tracks heavily borrow from Dancehall, Afrobeats, or UK Funky, focusing on warm weather, dancing, and fleeting romances.",
    "example_tracks": [
      "Controlla",
      "Blem",
      "One Dance"
    ]
  },
  {
    "category": "Pop Crossover / Radio R&B",
    "feel": "Melodic, upbeat, and accessible. Designed for mass appeal, these tracks lean heavily into pop structures, catchy hooks, and universally relatable themes of love or infatuation.",
    "example_tracks": [
      "Hold On, We're Going Home",
      "Find Your Love",
      "In My Feelings"
    ]
  },
  {
    "category": "Hard-Hitting / Mob Tie",
    "feel": "Menacing, aggressive, and street-oriented. Often featuring heavy trap beats and collaborations with street rappers, the themes center on loyalty, mob ties, and veiled threats.",
    "example_tracks": [
      "Knife Talk",
      "Nonstop",
      "Sneakin'"
    ]
  },
  {
    "category": "Crew Loyalty & Brotherhood",
    "feel": "Celebratory and unified. The focus is entirely on his inner circle (OVO), honoring the people who stayed loyal before the fame and sharing the wealth with his team.",
    "example_tracks": [
      "God's Plan",
      "Crew Love",
      "Mob Ties"
    ]
  },
  {
    "category": "The Club Anthem",
    "feel": "Pure hype. Made for VIP sections and festivals, these tracks prioritize heavy bassdrops, memorable catchphrases, and an infectious, party-ready tempo over lyrical density.",
    "example_tracks": [
      "Jimmy Cooks",
      "Way 2 Sexy",
      "Rich Flex"
    ]
  }
]

```
## 4. Data Transformations & DB Schema Alignment
From tracks.json, for each track.lyrics, extract that and apply embedding. Save that into the output schema below

The output schema must match the PostgreSQL database tables:
```json
"id": {
    "type": "SERIAL",
    "nullable": false,
    "description": "Unique auto-incrementing primary key for the stanza"
    },
    "song_id": {
        "type": "VARCHAR(255)",
        "nullable": false,
        "description": "Foreign key referencing the track table (track.id)"
    },
    "lyric_chunk": {
        "type": "TEXT",
        "nullable": false,
        "description": "Text content of the lyric stanza/chunk"
    },
    "chunk_index": {
        "type": "INTEGER",
        "nullable": false,
        "description": "Sequential index of the stanza within the track's lyrics"
    },
    "embedding": {
        "type": "vector(1024)",
        "nullable": false,
        "description": "Dense vector embedding representing the lyric chunk for semantic search"
    },
    "personal_feel": {
        "type": "VARCHAR(255)",
        "nullable": true,
        "description": "Subjective categorization classification for the stanza (e.g., 'Late Night Drive', 'Hype', 'Introspective')"
    }
```

## 5. Operational Requirements

### 5.1 Logging & Deduplication
* Print real-time execution logs (e.g. `Processing spotify_id {spotify-id} 12/80: ...`).

### 5.2 Output Files
* Raw response payload written to `data/stanza.json`.




