"""Pydantic schemas and interface contracts for DrakeAI."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator

# Supported 10 subjective categorization categories from data_flow.md
SUPPORTED_PERSONAL_FEELS = [
    "Late-Night Confessional",
    "Triumphant Flex",
    "Paranoid & Guarded",
    "Time-Stamp Introspection",
    "Toxic & Petty",
    "Global Groove / Island Infusion",
    "Pop Crossover / Radio R&B",
    "Hard-Hitting / Mob Tie",
    "Crew Loyalty & Brotherhood",
    "The Club Anthem"
]

PersonalFeelCategory = Literal[
    "Late-Night Confessional",
    "Triumphant Flex",
    "Paranoid & Guarded",
    "Time-Stamp Introspection",
    "Toxic & Petty",
    "Global Groove / Island Infusion",
    "Pop Crossover / Radio R&B",
    "Hard-Hitting / Mob Tie",
    "Crew Loyalty & Brotherhood",
    "The Club Anthem"
]


class AudioFeatureFilters(BaseModel):
    min_valence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Minimum valence (0.0=sad, 1.0=happy)")
    max_valence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Maximum valence (0.0=sad, 1.0=happy)")
    min_energy: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Minimum energy (0.0=calm, 1.0=intense)")
    max_energy: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Maximum energy (0.0=calm, 1.0=intense)")
    min_danceability: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Minimum danceability (0.0=not danceable, 1.0=very danceable)")
    max_danceability: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Maximum danceability (0.0=not danceable, 1.0=very danceable)")
    min_tempo: Optional[float] = Field(default=None, ge=40.0, le=250.0, description="Minimum tempo in BPM")
    max_tempo: Optional[float] = Field(default=None, ge=40.0, le=250.0, description="Maximum tempo in BPM")
    min_acousticness: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Minimum acousticness")
    max_acousticness: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Maximum acousticness")
    mode: Optional[int] = Field(default=None, description="Modality (0=minor, 1=major)")


class AudioFeatureTargets(BaseModel):
    target_valence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    target_energy: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    target_danceability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    target_tempo: Optional[float] = Field(default=None, ge=40.0, le=250.0)
    target_acousticness: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    target_speechiness: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class MetadataFilters(BaseModel):
    personal_feel: Optional[str] = Field(
        default=None,
        description="One of the 10 supported personal feel categories if applicable to the user prompt."
    )
    release_year_before: Optional[int] = Field(
        default=None,
        description="Upper bound release year constraint (e.g. 2018 for tracks released before 2018)."
    )
    release_year_after: Optional[int] = Field(
        default=None,
        description="Lower bound release year constraint (e.g. 2015 for tracks released after 2015)."
    )
    album_name: Optional[str] = Field(
        default=None,
        description="Specific Drake album title filter (e.g. 'Take Care', 'Scorpion', 'Views')."
    )
    album_type: Optional[str] = Field(
        default=None,
        description="Album release type, e.g. 'album', 'single', or 'compilation'."
    )
    limit: int = Field(
        default=3,
        description="Maximum number of stanzas to retrieve (default 3, min 1, max 10)."
    )
    audio_filters: Optional[AudioFeatureFilters] = Field(
        default=None,
        description="Hard acoustic boundary constraints (min/max valence, energy, tempo, danceability, mode)."
    )
    audio_targets: Optional[AudioFeatureTargets] = Field(
        default=None,
        description="Soft target audio feature vector for weighted hybrid similarity distance ranking."
    )
    sort_by: Optional[str] = Field(
        default=None,
        description="Sorting preference, e.g. 'hybrid', 'valence_asc', 'valence_desc', 'energy_desc', 'tempo_asc'."
    )


class IntentOutput(BaseModel):
    is_relevant: bool = Field(
        description="True if the user prompt relates to Drake, his songs, albums, lyrics, or vibes. False otherwise."
    )
    rejection_message: Optional[str] = Field(
        default=None,
        description="Domain guidance message if is_relevant is False."
    )
    semantic_query: Optional[str] = Field(
        default=None,
        description="Richly expanded semantic query incorporating synonyms, definitions, emotional nuances, and lyrical examples to maximize vector cosine similarity matching."
    )
    metadata_filters: MetadataFilters = Field(
        default_factory=MetadataFilters,
        description="Structured metadata constraints extracted from the user query."
    )

    @field_validator("metadata_filters", mode="before")
    @classmethod
    def set_default_filters(cls, v):
        if v is None:
            return MetadataFilters()
        return v


class TrackVettingResult(BaseModel):
    is_match: bool = Field(
        description="True if the candidate parent track genuinely matches the user's thematic and emotional intent, False otherwise."
    )
    confidence: float = Field(
        default=1.0,
        description="Confidence score between 0.0 and 1.0."
    )
    reason: str = Field(
        default="",
        description="Analytical justification explaining why the candidate track matches or fails the query."
    )


class SourceItem(BaseModel):
    track_name: str
    album_name: str
    release_date: Optional[str] = None
    album_art_url: Optional[str] = None
    spotify_url: Optional[str] = None
    quoted_stanzas: List[str] = Field(default_factory=list)
    personal_feel: Optional[str] = None
    track_lyrics: Optional[str] = Field(
        default=None,
        description="Full lyrics of the parent track pulled into context."
    )
    match_rationale: Optional[str] = Field(
        default=None,
        description="Analytical reasoning explaining why this specific track and stanza match the user's prompt."
    )
    # Audio feature attributes
    valence: Optional[float] = Field(default=None, description="Musical positiveness (0.0 to 1.0)")
    energy: Optional[float] = Field(default=None, description="Intensity and activity (0.0 to 1.0)")
    danceability: Optional[float] = Field(default=None, description="Danceability (0.0 to 1.0)")
    tempo: Optional[float] = Field(default=None, description="Tempo in BPM")
    acousticness: Optional[float] = Field(default=None, description="Acousticness (0.0 to 1.0)")
    speechiness: Optional[float] = Field(default=None, description="Speechiness (0.0 to 1.0)")
    loudness: Optional[float] = Field(default=None, description="Loudness in dB")
    mode: Optional[int] = Field(default=None, description="Modality (0=minor, 1=major)")
    similarity: Optional[float] = Field(default=None, description="Semantic vector cosine similarity")
    hybrid_score: Optional[float] = Field(default=None, description="Multi-modal weighted hybrid score")


class TrackMatchAnalysis(BaseModel):
    track_name: str
    match_rationale: str = Field(
        description="Analysis of why this track's lyrics and mood directly match the user's query."
    )


class ReasoningAgentOutput(BaseModel):
    thematic_analysis: str = Field(
        description="Overarching thematic and lyrical analysis addressing the user's prompt across the retrieved tracks."
    )
    track_rationales: List[TrackMatchAnalysis] = Field(
        default_factory=list,
        description="Per-track match explanations linking exact lyrics/subtext to the user's query."
    )


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    prompt: str
    history: List[Dict[str, str]] = Field(default_factory=list)


class ChatResponse(BaseModel):
    response: str
    sources: List[SourceItem] = Field(default_factory=list)
    reasoning: Optional[str] = Field(
        default=None,
        description="In-depth thematic analysis and reasoning generated by the reasoning agent."
    )
    agent_trace: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured breakdown of agent thinking, intent extraction, pulled songs, and vetting."
    )

