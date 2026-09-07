# Product Requirements Document (PRD): Audio Feature Integration & Hybrid Multi-Modal Retrieval

## 1. Executive Summary

### 1.1 Product Vision
The Drake Lyric & Audio Intelligence Agent combines dense semantic vector retrieval (`pgvector`) with Spotify-standard acoustic audio features (`valence`, `energy`, `danceability`, `tempo`, `acousticness`, `speechiness`, `loudness`, `mode`). By integrating acoustic features into the retrieval pipeline, the system transforms from a pure lyrical quote-finder into a multi-modal music intelligence engine capable of distinguishing musical vibe, mood, and instrumentation alongside lyrical themes.

### 1.2 Core Objectives
1. **Intelligent Audio Feature Extraction (Router):** Extract both semantic themes and technical audio feature profiles (hard filter bounds and soft target vectors) from natural language user queries during intent routing.
2. **Hybrid Context Graph Retrieval (Knowledge Layer):** Update the SQL context graph builder to JOIN `stanza`, `track`, `album`, and `track_audio_feature` (with fallback to `audio_feature`), applying parameterized hard SQL filters and multi-modal weighted hybrid scoring ($S_{\text{hybrid}} = \alpha \cdot S_{\text{semantic}} + (1 - \alpha) \cdot S_{\text{audio}}$).
3. **Automatic Filter Relaxation:** If strict hard filter boundaries yield fewer candidates than requested, automatically relax the filter bounds (widening thresholds by 15-20%) and rely on soft weighted hybrid scoring to rank the closest musical candidates.
4. **Multi-Modal Vetting & Attribution:** Provide the Vetting Agent and Reasoning Agent with acoustic metrics, enabling musical validation (e.g. verifying that a candidate for "saddest track" actually possesses low valence) and rich attribution in final responses and UI cards.

---

## 2. End-to-End Data Flow Architecture

```mermaid
flowchart TD
    User([User Prompt: 'Top 3 saddest Drake songs']) --> Router[Node 1: Guardrail & Intent Router]

    subgraph Extraction["Node 1: Intent Extraction"]
        Router --> SemExp[Semantic Query Expansion\n'sadness, heartbreak, weeping, despair']
        Router --> AudioProf[Audio Feature Generator\nTarget: valence=0.20, energy=0.35, tempo=80\nHard Filter: valence <= 0.40, energy <= 0.55]
        Router --> MetaFilt[Metadata Constraints\nlimit=3, order=hybrid_desc]
    end

    Extraction --> Retrieval[Node 2: Hybrid Query Builder & Retriever]

    subgraph KnowledgeLayer["Knowledge Retrieval Layer"]
        Retrieval --> VectorEnc[SentenceTransformer 1024-d Vector]
        Retrieval --> ContextGraphSQL[Parameterized Context Graph SQL Query]
        ContextGraphSQL --> DB[(PostgreSQL + pgvector)]
        DB -->|JOIN stanza + track + album + track_audio_feature| CandidatePool[Candidate Stanzas with Audio Metrics]
        CandidatePool --> HybridScorer[Weighted Hybrid Ranking Engine\nScore = α · Sim_semantic + (1-α) · Sim_audio]
        CandidatePool -->|If candidates < limit| Relaxation[Automatic Filter Relaxation Engine]
    end

    HybridScorer --> Vetting[Node 3: Multi-Modal Track Vetting Agent\nChecks Lyric Match + Audio Mood Alignment]
    Vetting --> Reasoning[Node 4: Reasoning & Analysis Agent\nExplains Lyrical & Acoustic Fit]
    Reasoning --> Formatter[Node 5: Response Formatter & UI\nGenerates Citations + Audio Feature Badges]
```

---

## 3. Audio Feature Domain Model & Taxonomy

The system leverages 11 core Spotify-standard audio metrics persisted in the `track_audio_feature` table:

| Metric | Type / Range | Musical Definition | Example Drake Regimes |
|---|---|---|---|
| **valence** | Float [0.0, 1.0] | Musical positiveness, cheerfulness, and euphoria vs. sadness and depression | Low (0.15 - 0.35): *Marvins Room*, *Jaded*, *Doing It Wrong*<br>High (0.60 - 0.85): *In My Feelings*, *One Dance* |
| **energy** | Float [0.0, 1.0] | Perceptual measure of intensity, volume, and noise | Low (0.20 - 0.45): *Jungle*, *Passionfruit*<br>High (0.70 - 0.90): *Mob Ties*, *Nonstop*, *Energy* |
| **danceability** | Float [0.0, 1.0] | Suitability for dancing (tempo, rhythm stability, beat strength) | High (0.75 - 0.92): *Hotline Bling*, *Controlla*, *One Dance* |
| **tempo** | Float (BPM, 40 - 220) | Overall pace / beats per minute | Slow (65 - 90 BPM): *Marvins Room*<br>Upbeat (110 - 130 BPM): *Nice For What* |
| **acousticness** | Float [0.0, 1.0] | Confidence measure of acoustic instruments vs. synthesizers | High (0.35 - 0.70): Melodic piano/guitar intros, *Look What You've Done* |
| **speechiness** | Float [0.0, 1.0] | Presence of spoken/rapped words vs. sung vocals | High (0.15 - 0.40): Dense rap verses (*Knife Talk*, *Tuscan Leather*) |
| **loudness** | Float (dB, -60.0 to 0.0) | Overall track loudness | Punchy (-6.5 to -4.0 dB); Ambient (-12.0 to -8.0 dB) |
| **mode** | Integer (0 or 1) | Modality of track (0 = Minor, 1 = Major) | Minor (0): Melancholic / Aggressive; Major (1): Bright / Uplifting |
| **key** | Integer (0 to 11) | Pitch class (0 = C, 1 = C#, ..., 11 = B) | Harmonic key classification |

---

## 4. Component Functional Specifications

### 4.1 Node 1: Guardrail & Intent Extractor (Router)

#### Functional Requirements:
1. **Keyword-to-Audio Profiling:** When a user prompt includes mood, sonic, or tempo descriptors (e.g. "saddest", "hype banger", "acoustic", "danceable", "slow bpm", "aggressive trap"), the router extracts:
   - **Target Audio Feature Vector (`AudioFeatureTargets`):** Ideal metric values (e.g. `{"valence": 0.20, "energy": 0.35}`) used for soft distance ranking.
   - **Audio Feature Filter Bounds (`AudioFeatureFilters`):** Upper and lower numerical bounds used as hard SQL `WHERE` constraints (e.g. `valence <= 0.40`, `energy <= 0.55`).
2. **Deterministic & Heuristic Fallback:** When the LLM is offline or in test environments, a regex and keyword-driven audio profiling engine maps queries deterministically to standard regimes:
   - `sad` / `saddest` / `heartbreak` / `gutwrenching` $\rightarrow$ `valence <= 0.40`, `energy <= 0.55`, `target_valence = 0.20`, `target_energy = 0.35`, `target_tempo = 80`
   - `hype` / `club` / `banger` / `party` $\rightarrow$ `danceability >= 0.70`, `energy >= 0.70`, `target_danceability = 0.82`, `target_energy = 0.80`, `target_valence = 0.65`
   - `introspective` / `late night` / `confessional` $\rightarrow$ `energy <= 0.60`, `target_energy = 0.45`, `target_valence = 0.35`, `target_tempo = 90`
   - `aggressive` / `mob tie` / `drill` $\rightarrow$ `energy >= 0.65`, `speechiness >= 0.15`, `mode = 0`, `target_energy = 0.80`
   - `island` / `afrobeats` / `dancehall` $\rightarrow$ `danceability >= 0.75`, `valence >= 0.55`, `target_danceability = 0.85`
3. **Limit & Ranking Orientation:** Queries like "top 3 saddest" automatically configure `limit = 3` and audio feature alignment sorting.

#### Pydantic Schema Additions:
```python
class AudioFeatureFilters(BaseModel):
    min_valence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_valence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    min_energy: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_energy: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    min_danceability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_danceability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    min_tempo: Optional[float] = Field(default=None, ge=40.0, le=250.0)
    max_tempo: Optional[float] = Field(default=None, ge=40.0, le=250.0)
    min_acousticness: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_acousticness: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    mode: Optional[int] = Field(default=None, description="0 for Minor, 1 for Major")

class AudioFeatureTargets(BaseModel):
    target_valence: Optional[float] = None
    target_energy: Optional[float] = None
    target_danceability: Optional[float] = None
    target_tempo: Optional[float] = None
    target_acousticness: Optional[float] = None
    target_speechiness: Optional[float] = None

class MetadataFilters(BaseModel):
    personal_feel: Optional[str] = None
    release_year_before: Optional[int] = None
    release_year_after: Optional[int] = None
    album_name: Optional[str] = None
    album_type: Optional[str] = None
    limit: int = 3
    audio_filters: Optional[AudioFeatureFilters] = None
    audio_targets: Optional[AudioFeatureTargets] = None
```

---

### 4.2 Node 2: Hybrid Query Builder & Retrieval Layer

#### Functional Requirements:
1. **Multi-Table SQL Context Graph:** Query joins `stanza s`, `track t`, `album a`, `LEFT JOIN track_audio_feature taf ON t.id = taf.track_id`, and `LEFT JOIN audio_feature af ON t.id = af.song_id`.
2. **Hard Filter Enforcement:** Applied strictly in the SQL `WHERE` clause:
   ```sql
   AND (%(min_valence)s IS NULL OR COALESCE(taf.valence, af.valence) >= %(min_valence)s)
   AND (%(max_valence)s IS NULL OR COALESCE(taf.valence, af.valence) <= %(max_valence)s)
   AND (%(min_energy)s IS NULL OR COALESCE(taf.energy, af.energy) >= %(min_energy)s)
   AND (%(max_energy)s IS NULL OR COALESCE(taf.energy, af.energy) <= %(max_energy)s)
   AND (%(min_danceability)s IS NULL OR COALESCE(taf.danceability, af.danceability) >= %(min_danceability)s)
   AND (%(max_danceability)s IS NULL OR COALESCE(taf.danceability, af.danceability) <= %(max_danceability)s)
   AND (%(min_tempo)s IS NULL OR COALESCE(taf.tempo, af.tempo) >= %(min_tempo)s)
   AND (%(max_tempo)s IS NULL OR COALESCE(taf.tempo, af.tempo) <= %(max_tempo)s)
   AND (%(mode)s IS NULL OR COALESCE(taf.mode, af.mode) = %(mode)s)
   ```
3. **Weighted Hybrid Scoring Formulation:**
   - **Semantic Vector Similarity ($S_{\text{semantic}}$):**
     $$S_{\text{semantic}} = 1 - (s.\text{embedding} \Leftrightarrow \text{query\_vector})$$
   - **Audio Feature Proximity ($S_{\text{audio}}$):**
     Let $k$ be the active target features (e.g. valence $v_t$, energy $e_t$). Audio distance $D_{\text{audio}}$ is computed as normalized weighted Euclidean distance:
     $$D_{\text{audio}} = \sqrt{\frac{\sum_{i=1}^{k} w_i \cdot (\text{feature}_i - \text{target}_i)^2}{\sum_{i=1}^{k} w_i}}$$
     $$S_{\text{audio}} = 1.0 - \min(1.0, D_{\text{audio}})$$
   - **Composite Hybrid Score ($S_{\text{hybrid}}$):**
     $$S_{\text{hybrid}} = \alpha \cdot S_{\text{semantic}} + (1 - \alpha) \cdot S_{\text{audio}}$$
     Default parameter balance: $\alpha = 0.60$ for semantic queries with audio constraints, or $\alpha = 0.35$ for explicit acoustic queries (e.g. "top 3 saddest songs with slowest tempo").
4. **Automatic Filter Relaxation Strategy:** If hard audio filters return fewer than `limit` candidates, the retriever automatically widens hard filter bounds (by $\pm 0.15$ on 0-1 metrics and $\pm 15$ BPM on tempo) and utilizes soft hybrid scoring to guarantee the top $k$ closest musical matches are returned.

---

### 4.3 Node 3 & 4: Vetting & Reasoning Agents

#### Multi-Modal Vetting:
The Vetting Agent verifies whether both the lyrics and audio features correspond with the prompt intent. If a candidate track has lyrics matching "sadness" but an energetic dancehall tempo and high valence, the vetting agent checks whether the user asked for "lyrical sadness" vs. "overall sad song" and adjusts approval accordingly.

#### Reasoning & Attribution:
The Reasoning Agent cites specific audio values to explain candidate selection:
> *"Marvins Room (Take Care, 2011) was selected as a top match due to its low valence score of 0.31 and subdued 86 BPM tempo, which musically reinforce the lyric's late-night vulnerability..."*

---

### 4.4 Node 5 & Frontend: Response Formatter & Streamlit UI

1. **Source Contract (`SourceItem`):**
   ```python
   class SourceItem(BaseModel):
       track_name: str
       album_name: str
       release_date: Optional[str] = None
       album_art_url: Optional[str] = None
       spotify_url: Optional[str] = None
       quoted_stanzas: List[str] = Field(default_factory=list)
       personal_feel: Optional[str] = None
       track_lyrics: Optional[str] = None
       match_rationale: Optional[str] = None
       # Audio Feature Attributes
       valence: Optional[float] = None
       energy: Optional[float] = None
       danceability: Optional[float] = None
       tempo: Optional[float] = None
       acousticness: Optional[float] = None
       loudness: Optional[float] = None
       mode: Optional[int] = None
   ```
2. **Frontend UI Rendering (`main.py`):**
   Each source card renders aesthetic visual badges displaying musical metrics:
   - 📉 **Valence:** `0.21 (Melancholic)`
   - ⚡ **Energy:** `0.38 (Subdued)`
   - ⏱️ **Tempo:** `74.5 BPM`
   - 💃 **Danceability:** `0.52`

---

## 5. Verification Plan

1. **Unit Tests (`backend/test_backend.py`):**
   - Verify intent extraction parses `AudioFeatureFilters` and `AudioFeatureTargets`.
   - Verify SQL query builder successfully joins `track_audio_feature` / `audio_feature` and returns audio columns.
   - Verify hard filter exclusion and automatic relaxation.
   - Verify hybrid scoring ranks tracks by combined semantic + audio distance.
2. **Integration Verification:**
   - Execute query `"top 3 saddest drake songs"` and verify retrieved tracks have `valence <= 0.40` and are ranked properly.
   - Verify Streamlit UI renders audio feature badges and Spotify links.
