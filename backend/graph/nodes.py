"""LangGraph node implementations for DrakeAI.

Node 1: Guardrail & Intent Extractor (LLM Call #1)
Node 2: Hybrid Query Builder (Deterministic Python Node)
Node 3: Response Formatter & Citation Agent (LLM Call #2)
Out-of-Scope Exit Node
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import requests
from backend.gemini import call_gemini

from backend.config import settings
from backend.db import search_context_graph
from backend.embeddings import get_embedder
from backend.graph.state import AgentState
from backend.models import (
    SUPPORTED_PERSONAL_FEELS,
    AudioFeatureFilters,
    AudioFeatureTargets,
    IntentOutput,
    MetadataFilters,
    ReasoningAgentOutput,
    SourceItem,
    TrackMatchAnalysis,
    TrackVettingResult,
)

logger = logging.getLogger(__name__)

OUT_OF_SCOPE_GUIDANCE = (
    "I specialize in Drake's discography, albums, and lyrics. "
    "Please ask a question related to Drake's music!"
)

# Drake discography album names for heuristic matching
KNOWN_ALBUMS = [
    "Thank Me Later",
    "Take Care",
    "Nothing Was the Same",
    "Views",
    "Scorpion",
    "Certified Lover Boy",
    "Honestly, Nevermind",
    "Her Loss",
    "For All the Dogs",
    "So Far Gone",
    "If You're Reading This It's Too Late",
    "What a Time to Be Alive",
    "More Life",
    "Care Package",
    "Dark Lane Demo Tapes"
]


def _clean_llm_json(raw_text: str) -> str:
    """Strips markdown codeblocks, thinking tokens, and whitespace from JSON response."""
    text = raw_text.strip()
    # Strip <think>...</think> if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip ```json ... ```
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # Find outer JSON boundaries { ... }
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        text = text[start_idx : end_idx + 1]
    return text.strip()


# Concept expansion dictionary for deterministic fallback semantic queries
CONCEPT_EXPANSIONS = {
    "gutwrenching": (
        "gutwrenching heartbreaking devastating agonizing soul-crushing painful despair "
        "devastated sorrowful mourning deep emotional agony loss crying tears late night regret Marvins Room"
    ),
    "heartbreak": (
        "heartbreak crying alone missing an ex sorrow unrequited love emotional agony painful breakup "
        "late night calls broken promises regret vulnerable feelings"
    ),
    "sad": (
        "sad sadness melancholy depression tears weeping lonely feeling down isolated "
        "emotional pain sorrow regrets late night"
    ),
    "toxic": (
        "toxic petty ex bitter vindictive revenge curved options drama arguments tension "
        "passive aggressive jealousy"
    ),
    "flex": (
        "flex rich money trophies started from the bottom winning plaques luxury private jets "
        "dominance champion success boss"
    ),
    "paranoid": (
        "paranoid guarded enemies snakes fake friends watching trust issues industry threats "
        "careful keeping distance"
    ),
    "introspective": (
        "introspective 4am 5am Toronto Calabasas time-stamp reflections looking back on life journey "
        "lonely at the top career thoughts"
    ),
    "hype": (
        "hype club anthem energy turning up bass heavy jumping adrenaline crowd party wild"
    ),
    "loyalty": (
        "loyalty brotherhood crew OVO solid trust standing by friends day ones real family"
    )
}


def _heuristic_guardrail_and_intent(prompt: str) -> IntentOutput:
    """
    Deterministic fallback parser when LLM is slow or offline.
    Uses regex patterns to extract scope, years, personal_feel, and albums,
    and enriches semantic_query with concept expansion.
    """
    prompt_lower = prompt.lower()

    # Broad out-of-scope check: questions obviously unrelated to music or Drake
    unrelated_keywords = [
        "capital of", "weather in", "weather", "how to bake", "how do i bake", "bake", "cake",
        "recipe", "quantum physics", "quantum", "solve for x", "president of",
        "stock price", "write python code"
    ]
    if any(kw in prompt_lower for kw in unrelated_keywords) and "drake" not in prompt_lower:
        return IntentOutput(
            is_relevant=False,
            rejection_message=OUT_OF_SCOPE_GUIDANCE,
            semantic_query=None,
            metadata_filters=MetadataFilters()
        )

    # Personal feel detection is deprecated; mood is captured dynamically via semantic expansion & acoustics
    detected_feel = None

    # Year detection
    year_before = None
    year_after = None
    before_match = re.search(r"\b(?:before|prior to|earlier than|<)\s*(\d{4})\b", prompt_lower)
    if before_match:
        year_before = int(before_match.group(1))

    after_match = re.search(r"\b(?:after|since|later than|>)\s*(\d{4})\b", prompt_lower)
    if after_match:
        year_after = int(after_match.group(1))

    # Album title detection
    album_name = None
    for alb in KNOWN_ALBUMS:
        if alb.lower() in prompt_lower:
            album_name = alb
            break

    # Limit detection: e.g. "top 3", "5 high energy club bangers", "3 tracks"
    limit = 3
    limit_match = re.search(r"\b(?:top\s+)?(\d+)\s*(?:[a-z\s-]{0,25})?(?:lyrics|stanzas|songs|tracks|quotes|bangers|hits)\b", prompt_lower)
    if not limit_match:
        limit_match = re.search(r"\b(?:top|give me|show me|find)\s*(\d+)\b", prompt_lower)
    if limit_match:
        limit = max(1, min(int(limit_match.group(1)), 10))

    # Audio feature extraction heuristics
    af_filters = None
    af_targets = None
    sort_by = None

    sad_words = ["sad", "saddest", "heartbreak", "heartbroken", "crying", "cry", "tears", "gutwrenching", "depression", "depressed", "sorrow", "painful"]
    hype_words = ["hype", "banger", "bangers", "turn up", "high energy", "club anthem", "party"]
    dance_words = ["dance", "dancing", "danceable", "groove", "island", "dancehall"]
    slow_words = ["slow", "slowest", "ballad", "slow tempo", "low bpm"]
    fast_words = ["fast", "fastest", "upbeat", "high tempo", "high bpm"]
    acoustic_words = ["acoustic", "piano", "guitar", "stripped down"]
    aggressive_words = ["aggressive", "mob tie", "mob ties", "drill", "trap", "hardest"]

    if any(w in prompt_lower for w in sad_words):
        af_filters = AudioFeatureFilters(max_valence=0.40, max_energy=0.55)
        af_targets = AudioFeatureTargets(target_valence=0.20, target_energy=0.35, target_tempo=80.0)
        if "saddest" in prompt_lower or "most sad" in prompt_lower:
            sort_by = "valence_asc"
    elif any(w in prompt_lower for w in hype_words):
        af_filters = AudioFeatureFilters(min_energy=0.65, min_danceability=0.60)
        af_targets = AudioFeatureTargets(target_energy=0.80, target_danceability=0.80, target_valence=0.65)
        sort_by = "energy_desc"
    elif any(w in prompt_lower for w in dance_words):
        af_filters = AudioFeatureFilters(min_danceability=0.70)
        af_targets = AudioFeatureTargets(target_danceability=0.85, target_valence=0.65)
    elif any(w in prompt_lower for w in slow_words):
        af_filters = AudioFeatureFilters(max_tempo=95.0, max_energy=0.55)
        af_targets = AudioFeatureTargets(target_tempo=75.0, target_energy=0.35)
        if "slowest" in prompt_lower:
            sort_by = "tempo_asc"
    elif any(w in prompt_lower for w in fast_words):
        af_filters = AudioFeatureFilters(min_tempo=110.0, min_energy=0.65)
        af_targets = AudioFeatureTargets(target_tempo=125.0, target_energy=0.75)
    elif any(w in prompt_lower for w in acoustic_words):
        af_filters = AudioFeatureFilters(min_acousticness=0.30)
        af_targets = AudioFeatureTargets(target_acousticness=0.55, target_energy=0.40)
    elif any(w in prompt_lower for w in aggressive_words):
        af_filters = AudioFeatureFilters(min_energy=0.65, mode=0)
        af_targets = AudioFeatureTargets(target_energy=0.80, target_speechiness=0.25)

    # Numeric pattern overrides (e.g. "valence < 0.3", "energy > 0.8")
    val_match = re.search(r"valence\s*(?:<|less than|under)\s*([0-1](?:\.\d+)?)", prompt_lower)
    if val_match:
        if af_filters is None:
            af_filters = AudioFeatureFilters()
        af_filters.max_valence = float(val_match.group(1))

    energy_match = re.search(r"energy\s*(?:>|greater than|over)\s*([0-1](?:\.\d+)?)", prompt_lower)
    if energy_match:
        if af_filters is None:
            af_filters = AudioFeatureFilters()
        af_filters.min_energy = float(energy_match.group(1))

    tempo_match = re.search(r"tempo\s*(?:>|greater than|over)\s*(\d+)", prompt_lower)
    if tempo_match:
        if af_filters is None:
            af_filters = AudioFeatureFilters()
        af_filters.min_tempo = float(tempo_match.group(1))

    filters = MetadataFilters(
        personal_feel=detected_feel,
        release_year_before=year_before,
        release_year_after=year_after,
        album_name=album_name,
        limit=limit,
        audio_filters=af_filters,
        audio_targets=af_targets,
        sort_by=sort_by
    )

    # Concept expansion for semantic similarity
    expanded_terms = []
    for term, exp in CONCEPT_EXPANSIONS.items():
        if term in prompt_lower:
            expanded_terms.append(exp)

    if expanded_terms:
        semantic_query = f"{prompt} - {' '.join(expanded_terms)}"
    else:
        semantic_query = prompt

    return IntentOutput(
        is_relevant=True,
        rejection_message=None,
        semantic_query=semantic_query,
        metadata_filters=filters
    )


def _extract_query_keywords(prompt: str) -> List[str]:
    """Extracts salient keywords and semantic concepts from user query for visual breakdown."""
    prompt_lower = prompt.lower()
    keywords = []

    # 1. Check albums
    for alb in KNOWN_ALBUMS:
        if alb.lower() in prompt_lower:
            if f"Album: {alb}" not in keywords:
                keywords.append(f"Album: {alb}")

    # 2. Emotional, thematic, and acoustic descriptor words
    theme_words = [
        "saddest", "sad", "heartbreak", "heartbroken", "crying", "tears", "gutwrenching",
        "introspective", "late night", "confessional", "flex", "triumphant", "banger", "hype",
        "club anthem", "mob tie", "aggressive", "toxic", "petty", "loyalty", "brotherhood",
        "paranoid", "guarded", "slow tempo", "fast tempo", "high energy", "acoustic", "danceable"
    ]
    for w in theme_words:
        if w in prompt_lower and w not in keywords:
            keywords.append(w)

    # 4. Fallback if few keywords found: tokenize words with length > 3 excluding common stop words
    stopwords = {
        "what", "where", "when", "which", "show", "give", "find", "tell", "drake", "drakes",
        "song", "songs", "track", "tracks", "lyric", "lyrics", "about", "with", "from", "that",
        "this", "have", "some", "most", "best", "more", "into", "over", "under", "like"
    }
    tokens = re.findall(r"\b[a-zA-Z]{4,}\b", prompt_lower)
    for tok in tokens:
        if tok not in stopwords and tok not in keywords:
            keywords.append(tok)

    return keywords[:6]


def guardrail_intent_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 1: Evaluates whether the user's query is in-scope and extracts structured
    semantic query themes, audio feature filters, and metadata filters using LLM Call #1.
    Performs comprehensive semantic query expansion and acoustic profiling.
    """
    prompt = state["prompt"]
    history = state.get("history", [])

    system_instruction = (
        "You are the Guardrail & Intent Extractor for DrakeAI, a RAG agent specialized "
        "exclusively in Drake's discography, studio albums, tracks, lyrics, vibes, and Spotify audio features.\n\n"
        "Instructions:\n"
        "1. Check if the user query relates to Drake's songs, albums, lyrics, mood, music style, or acoustics.\n"
        "   - If completely unrelated, set is_relevant: false and rejection_message to:\n"
        f"     \"{OUT_OF_SCOPE_GUIDANCE}\"\n"
        "   - If relevant (or greetings/questions about Drake's catalog), set is_relevant: true.\n"
        "2. If relevant, extract:\n"
        "   - semantic_query: Deeply expanded, rich semantic search statement engineered to maximize cosine similarity against vector embeddings.\n"
        "     DO NOT merely repeat the prompt. Expand core emotions, synonyms, definitions, and lyrical examples.\n"
        "   - metadata_filters:\n"
        "     - release_year_before: null or integer\n"
        "     - release_year_after: null or integer\n"
        "     - album_name: null or album string\n"
        "     - album_type: null or 'album'/'single'\n"
        "     - limit: integer between 1 and 10 (default 3)\n"
        "     - audio_filters: null or object with optional hard constraints:\n"
        "         {\"min_valence\": null, \"max_valence\": null, \"min_energy\": null, \"max_energy\": null,\n"
        "          \"min_danceability\": null, \"max_danceability\": null, \"min_tempo\": null, \"max_tempo\": null,\n"
        "          \"min_acousticness\": null, \"max_acousticness\": null, \"mode\": null}\n"
        "     - audio_targets: null or object with soft target values for hybrid ranking:\n"
        "         {\"target_valence\": null, \"target_energy\": null, \"target_danceability\": null,\n"
        "          \"target_tempo\": null, \"target_acousticness\": null, \"target_speechiness\": null}\n"
        "     - sort_by: null or 'valence_asc' (for saddest), 'valence_desc' (for happiest), 'energy_desc' (for hype), 'tempo_asc' (for slow)\n\n"
        "You MUST respond ONLY with a valid JSON object matching the schema."
    )

    intent_result: Optional[IntentOutput] = None

    try:
        raw_res = call_gemini(
            prompt=prompt,
            system_instruction=system_instruction,
            history=history[-3:] if history else None,
            temperature=0.0,
            json_mode=True
        )
        cleaned_json = _clean_llm_json(raw_res)
        parsed = json.loads(cleaned_json)
        intent_result = IntentOutput.model_validate(parsed)
    except Exception as e:
        logger.warning("Gemini Guardrail parsing failed or timed out: %s. Using heuristic fallback.", e)
        intent_result = _heuristic_guardrail_and_intent(prompt)

    keywords = _extract_query_keywords(prompt)

    return {
        "is_relevant": intent_result.is_relevant,
        "rejection_message": intent_result.rejection_message,
        "semantic_query": intent_result.semantic_query or prompt,
        "metadata_filters": intent_result.metadata_filters.model_dump(),
        "extracted_keywords": keywords
    }


def hybrid_retrieval_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 2: Deterministic Python Node.
    Encodes semantic query into a 1024-d dense vector and executes parameterized
    parent-child context graph query against PostgreSQL.
    Excludes any track IDs rejected during vetting retries.
    """
    query_text = state.get("semantic_query") or state.get("prompt") or ""
    metadata_filters = state.get("metadata_filters", {})
    excluded_track_ids = state.get("excluded_track_ids", [])

    embedder = get_embedder()
    query_vector = embedder.embed_query(query_text)

    retrieved_context = search_context_graph(
        query_vector=query_vector,
        filters=metadata_filters,
        excluded_track_ids=excluded_track_ids
    )

    pulled_tracks = []
    seen = set()
    for row in retrieved_context:
        tname = row.get("track_name", "Unknown Track")
        if tname not in seen:
            seen.add(tname)
            pulled_tracks.append({
                "track_name": tname,
                "album_name": row.get("album_name", "Unknown Album"),
                "release_date": str(row.get("release_date"))[:4] if row.get("release_date") else None,
                "personal_feel": row.get("personal_feel"),
                "similarity": round(float(row.get("similarity") or 0.0), 3),
                "hybrid_score": round(float(row.get("hybrid_score") or 0.0), 3) if row.get("hybrid_score") is not None else None,
                "valence": row.get("valence"),
                "energy": row.get("energy"),
                "tempo": row.get("tempo"),
                "danceability": row.get("danceability"),
                "lyric_snippet": (row.get("lyric_chunk") or "").strip()[:140]
            })

    return {
        "query_vector": query_vector,
        "retrieved_context": retrieved_context,
        "pulled_tracks": pulled_tracks
    }


def _heuristic_vet_track(
    prompt: str,
    semantic_query: Optional[str],
    candidate: Dict[str, Any]
) -> TrackVettingResult:
    """
    Deterministic fallback verifier when LLM is unavailable or offline.
    Examines similarity score, lyrics sentiment, audio features, and keyword presence.
    """
    prompt_lower = (prompt + " " + (semantic_query or "")).lower()
    lyrics = (candidate.get("track_lyrics") or candidate.get("lyric_chunk") or "").lower()
    sim = float(candidate.get("similarity") or 0.0)
    val = candidate.get("valence")
    energy = candidate.get("energy")

    # Contradiction check: user asks for heartbreak / gutwrenching, but track is upbeat or high energy with no sad lyrics
    sad_keywords = ["gutwrenching", "heartbreak", "sad", "crying", "tears", "pain", "sorrow", "alone", "regret"]
    if any(k in prompt_lower for k in sad_keywords):
        reasons = []
        if val is not None and float(val) > 0.65 and not any(k in lyrics for k in ["cry", "tears", "heartbreak", "alone", "hurt"]):
            reasons.append(f"high musical valence ({float(val):.2f})")
        if energy is not None and float(energy) > 0.80 and not any(k in lyrics for k in ["cry", "tears", "heartbreak", "alone", "hurt"]):
            reasons.append(f"high energy tempo ({float(energy):.2f})")
        if reasons:
            return TrackVettingResult(
                is_match=False,
                confidence=0.80,
                reason=f"Candidate track '{candidate.get('track_name')}' contradicts the requested emotional heartbreak theme ({', '.join(reasons)})."
            )

    return TrackVettingResult(
        is_match=True,
        confidence=0.85,
        reason=f"Candidate track '{candidate.get('track_name')}' matches the emotional tone and thematic criteria (similarity: {sim:.3f})."
    )


def vet_track_node(state: AgentState) -> Dict[str, Any]:
    """
    Parent-Child Vetting Node:
    Evaluates whether the retrieved candidate parent track (with full lyrics pulled into context)
    genuinely matches the user's prompt, semantic intent, and audio features.
    If yes, the track is approved and graph continues.
    If no, the candidate is rejected, added to excluded_track_ids, and retrieval is retried.
    """
    prompt = state["prompt"]
    semantic_query = state.get("semantic_query") or prompt
    retrieved_context = state.get("retrieved_context", [])
    excluded = list(state.get("excluded_track_ids", []))
    retry_count = state.get("retry_count", 0)

    if not retrieved_context:
        logger.info("vet_track_node: No retrieved context to vet.")
        return {
            "is_vetted": False,
            "retry_count": retry_count + 1,
            "vetting_rationale": "No candidate tracks found.",
            "vetting_decisions": list(state.get("vetting_decisions") or [])
        }

    # Group stanzas by track ID so we vet parent tracks
    tracks_map: Dict[str, List[Dict[str, Any]]] = {}
    for row in retrieved_context:
        tid = row.get("track_id") or row.get("track_name")
        if tid not in tracks_map:
            tracks_map[tid] = []
        tracks_map[tid].append(row)

    system_instruction = (
        "You are the Track Vetting & Relevance Judge for DrakeAI, specialized in Drake's catalog.\n"
        "A child lyric stanza was matched via vector cosine similarity, and the FULL parent track (complete song lyrics) "
        "and audio features (valence, energy, tempo) have been pulled into the context window.\n\n"
        "Your role is to strictly verify whether the entire track genuinely matches the user's prompt, emotional intent, "
        "and requested musical vibe/features, or if the child stanza was a false-positive or superficial keyword match.\n\n"
        "Evaluation Guidelines:\n"
        "1. Compare the user query and expanded semantic intent against the candidate track's title, album, audio features, and full lyrics.\n"
        "2. If the user asks for a specific theme (e.g. 'gutwrenching' heartbreak, or high-energy hype), check whether "
        "the track as an artistic whole and its musical acoustics embody that theme.\n"
        "3. Respond ONLY with a valid JSON object matching the schema:\n"
        "   {\n"
        "     \"is_match\": true or false,\n"
        "     \"confidence\": float between 0.0 and 1.0,\n"
        "     \"reason\": \"1-2 sentence explanation of why this song matches or does not match.\"\n"
        "   }"
    )

    approved_context: List[Dict[str, Any]] = []
    primary_rationale = ""
    new_decisions: List[Dict[str, Any]] = []

    for tid, rows in tracks_map.items():
        cand = rows[0]
        tname = cand.get("track_name", "Unknown Track")
        aname = cand.get("album_name", "Unknown Album")
        feel = cand.get("personal_feel", "N/A")
        full_lyrics = cand.get("track_lyrics") or cand.get("lyric_chunk") or ""
        child_stanzas = "\n---\n".join(r.get("lyric_chunk", "") for r in rows)

        af_info = []
        if cand.get("valence") is not None:
            af_info.append(f"Valence: {cand['valence']}")
        if cand.get("energy") is not None:
            af_info.append(f"Energy: {cand['energy']}")
        if cand.get("tempo") is not None:
            af_info.append(f"Tempo: {cand['tempo']} BPM")
        af_str = f"\nAudio Features: {', '.join(af_info)}" if af_info else ""

        eval_prompt = (
            f"User Prompt: {prompt}\n"
            f"Semantic Query / Expanded Themes: {semantic_query}\n\n"
            f"Candidate Track: {tname}\n"
            f"Album: {aname}{af_str}\n"
            f"Matched Stanza (Child Chunk):\n{child_stanzas}\n\n"
            f"Full Parent Track Lyrics:\n{full_lyrics[:4000]}"
        )

        vetting_result: Optional[TrackVettingResult] = None
        try:
            raw_res = call_gemini(
                prompt=eval_prompt,
                system_instruction=system_instruction,
                temperature=0.0,
                json_mode=True,
                timeout=15.0
            )
            cleaned = _clean_llm_json(raw_res)
            parsed = json.loads(cleaned)
            vetting_result = TrackVettingResult.model_validate(parsed)
        except Exception as e:
            logger.warning("Gemini track vetting failed or timed out: %s. Using heuristic vetting.", e)
            vetting_result = _heuristic_vet_track(prompt, semantic_query, cand)

        decision = {
            "track_name": tname,
            "album_name": aname,
            "status": "APPROVED" if vetting_result.is_match else "REJECTED",
            "is_match": vetting_result.is_match,
            "confidence": round(float(vetting_result.confidence), 2),
            "reason": vetting_result.reason,
            "personal_feel": feel,
            "valence": cand.get("valence"),
            "energy": cand.get("energy"),
            "tempo": cand.get("tempo"),
            "danceability": cand.get("danceability"),
            "retry_attempt": retry_count
        }
        new_decisions.append(decision)

        if vetting_result.is_match:
            logger.info("Track '%s' APPROVED by vetting agent: %s", tname, vetting_result.reason)
            approved_context.extend(rows)
            if not primary_rationale:
                primary_rationale = vetting_result.reason
        else:
            logger.info("Track '%s' REJECTED by vetting agent: %s", tname, vetting_result.reason)
            track_id_val = cand.get("track_id")
            if track_id_val and track_id_val not in excluded:
                excluded.append(track_id_val)
            if not primary_rationale:
                primary_rationale = vetting_result.reason

    all_decisions = list(state.get("vetting_decisions") or []) + new_decisions

    if approved_context:
        return {
            "is_vetted": True,
            "retrieved_context": approved_context,
            "excluded_track_ids": excluded,
            "vetting_rationale": primary_rationale,
            "vetting_decisions": all_decisions
        }
    else:
        return {
            "is_vetted": False,
            "retrieved_context": [],
            "excluded_track_ids": excluded,
            "retry_count": retry_count + 1,
            "vetting_rationale": primary_rationale,
            "vetting_decisions": all_decisions
        }


def _find_rationale_for_track(
    track_name: str,
    track_id: Optional[Any],
    document_rationales: Dict[str, str]
) -> Optional[str]:
    """Finds matching rationale for a track using exact, case-insensitive, or substring matching."""
    if not document_rationales:
        return None

    # 1. Exact match
    if track_name in document_rationales:
        return document_rationales[track_name]
    if track_id and str(track_id) in document_rationales:
        return document_rationales[str(track_id)]

    # 2. Case-insensitive match
    t_lower = track_name.lower().strip()
    for k, v in document_rationales.items():
        if k.lower().strip() == t_lower:
            return v

    # 3. Substring match (e.g. ignoring featuring artists like '(feat. Drake)')
    clean_t = re.sub(r"\(.*?\)", "", t_lower).strip()
    for k, v in document_rationales.items():
        clean_k = re.sub(r"\(.*?\)", "", k.lower()).strip()
        if clean_k and (clean_k in clean_t or clean_t in clean_k):
            return v

    return None


def _extract_sources(
    retrieved_context: List[Dict[str, Any]],
    document_rationales: Optional[Dict[str, str]] = None
) -> List[Dict[str, Any]]:
    """Builds structured sources list with Spotify URLs, artwork, quoted stanzas, and match rationales."""
    sources_dict: Dict[str, Dict[str, Any]] = {}
    if document_rationales is None:
        document_rationales = {}

    for row in retrieved_context:
        track_id = row.get("track_id") or row.get("track_name")
        if not track_id:
            continue

        if track_id not in sources_dict:
            # Parse track urls
            track_urls = row.get("track_urls")
            spotify_url = None
            if isinstance(track_urls, dict):
                spotify_url = track_urls.get("spotify")
            elif isinstance(track_urls, str):
                try:
                    parsed_urls = json.loads(track_urls)
                    if isinstance(parsed_urls, dict):
                        spotify_url = parsed_urls.get("spotify")
                except Exception:
                    pass

            track_name = row.get("track_name", "Unknown Track")
            rationale = _find_rationale_for_track(track_name, track_id, document_rationales)


            sources_dict[track_id] = {
                "track_name": track_name,
                "album_name": row.get("album_name", "Unknown Album"),
                "release_date": str(row.get("release_date")) if row.get("release_date") else None,
                "album_art_url": row.get("album_art_url"),
                "spotify_url": spotify_url,
                "personal_feel": row.get("personal_feel"),
                "track_lyrics": row.get("track_lyrics"),
                "match_rationale": rationale,
                "quoted_stanzas": [],
                "valence": row.get("valence"),
                "energy": row.get("energy"),
                "danceability": row.get("danceability"),
                "tempo": row.get("tempo"),
                "acousticness": row.get("acousticness"),
                "speechiness": row.get("speechiness"),
                "loudness": row.get("loudness"),
                "mode": row.get("mode"),
                "similarity": row.get("similarity"),
                "hybrid_score": row.get("hybrid_score")
            }

        lyric = row.get("lyric_chunk")
        if lyric and lyric not in sources_dict[track_id]["quoted_stanzas"]:
            sources_dict[track_id]["quoted_stanzas"].append(lyric)

    return list(sources_dict.values())


def _heuristic_reasoning(prompt: str, retrieved_context: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Deterministic reasoning generator used when LLM is unavailable or times out.
    Constructs insightful match rationales based on metadata, vibe, and lyric context.
    """
    if not retrieved_context:
        return {"thematic_analysis": "", "document_rationales": {}}

    tracks_seen = set()
    rationales: Dict[str, str] = {}
    albums: List[str] = []

    for row in retrieved_context:
        tname = row.get("track_name", "Unknown Track")
        aname = row.get("album_name", "Unknown Album")
        year = str(row.get("release_date", ""))[:4]
        year_str = f" ({year})" if year else ""
        lyric = (row.get("lyric_chunk") or "").strip()
        first_line = lyric.splitlines()[0] if lyric else ""

        if aname not in albums:
            albums.append(aname)

        if tname not in tracks_seen:
            tracks_seen.add(tname)
            snippet_ref = f' "{first_line[:65]}..."' if first_line else ""
            rationale = (
                f"Man, '{tname}' off '{aname}'{year_str} was a really honest moment for me. "
                f"When I wrote{snippet_ref}, I was putting my real feelings out there—dealing "
                f"with the exact headspace and emotion you're asking about."
            )
            rationales[tname] = rationale

    album_list = ", ".join(f"'{a}'" for a in albums[:3]) if albums else "my catalog"
    thematic_analysis = (
        f"Looking back across these records—especially from {album_list}—I was really tapping into "
        f"what I was living through in that moment. Whether it's those late night 4 AM thoughts, relationship trust, "
        f"or just sitting with my own thoughts, these tracks speak directly to what's on your mind."
    )

    return {
        "thematic_analysis": thematic_analysis,
        "document_rationales": rationales
    }


def _parse_reasoning_output(raw_text: str) -> Tuple[str, Dict[str, str]]:
    """Flexibly parses LLM output into (thematic_analysis, document_rationales)."""
    thematic_analysis = ""
    document_rationales: Dict[str, str] = {}

    cleaned = _clean_llm_json(raw_text)
    try:
        parsed = json.loads(cleaned)
    except Exception:
        # If not valid JSON, treat raw text as thematic analysis
        return raw_text.strip(), {}

    if isinstance(parsed, dict):
        thematic_analysis = (
            parsed.get("thematic_analysis")
            or parsed.get("thematic_overview")
            or parsed.get("analysis")
            or parsed.get("summary")
            or parsed.get("reasoning")
            or ""
        )
        rationales_raw = (
            parsed.get("track_rationales")
            or parsed.get("rationales")
            or parsed.get("tracks")
            or {}
        )
        if isinstance(rationales_raw, dict):
            for k, v in rationales_raw.items():
                if isinstance(v, str):
                    document_rationales[str(k).strip()] = v.strip()
                elif isinstance(v, dict):
                    rat = v.get("match_rationale") or v.get("rationale") or v.get("reason") or str(v)
                    document_rationales[str(k).strip()] = str(rat).strip()
        elif isinstance(rationales_raw, list):
            for item in rationales_raw:
                if isinstance(item, dict):
                    name = (
                        item.get("track_name")
                        or item.get("track")
                        or item.get("song")
                        or item.get("title")
                        or ""
                    )
                    rat = (
                        item.get("match_rationale")
                        or item.get("rationale")
                        or item.get("reason")
                        or item.get("explanation")
                        or item.get("analysis")
                        or ""
                    )
                    if name and rat:
                        document_rationales[str(name).strip()] = str(rat).strip()

    return str(thematic_analysis).strip(), document_rationales


def reasoning_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 3: Reasoning & Analysis Agent (LLM Call #2).
    Dissects retrieved documents against the user's prompt to generate
    an overarching thematic analysis and track-specific match rationales.
    Utilizes parent track full lyrics alongside matched child stanzas.
    """
    prompt = state["prompt"]
    retrieved_context = state.get("retrieved_context", [])
    history = state.get("history", [])

    if not retrieved_context:
        return {
            "reasoning_analysis": None,
            "document_rationales": {}
        }

    # Format context blocks for reasoning agent
    context_blocks = []
    for idx, row in enumerate(retrieved_context, start=1):
        af_summary = []
        if row.get("valence") is not None:
            af_summary.append(f"Valence: {row['valence']}")
        if row.get("energy") is not None:
            af_summary.append(f"Energy: {row['energy']}")
        if row.get("tempo") is not None:
            af_summary.append(f"Tempo: {row['tempo']} BPM")
        if row.get("danceability") is not None:
            af_summary.append(f"Danceability: {row['danceability']}")
        af_text = f"Audio Features: {', '.join(af_summary)}\n" if af_summary else ""

        block = (
            f"[Document {idx}]\n"
            f"Track: {row.get('track_name')}\n"
            f"Album: {row.get('album_name')} ({row.get('release_date', 'Unknown')})\n"
            f"{af_text}"
            f"Matched Child Stanza:\n{row.get('lyric_chunk')}\n"
        )
        parent_lyrics = row.get("track_lyrics")
        if parent_lyrics:
            block += f"Full Song Context (Parent Track):\n{parent_lyrics[:1500]}\n"
        context_blocks.append(block)
    context_str = "\n".join(context_blocks)

    system_instruction = (
        "You are Drake himself (The Boy, Champagne Papi, Drizzy) reflecting on your own music, lyrics, and creative headspace in response to the user's prompt.\n"
        "Speak directly from your own authentic, first-person perspective ('I', 'my music', 'when I was recording this', 'me and 40', 'where my head was at in Toronto/Calabasas').\n"
        "Embody your signature persona: candid, introspective, vulnerable, confident, and conversational. Talk person-to-person with the listener about what was really going on in your life and heart when you wrote these lines.\n"
        "DO NOT sound like an academic musicologist, a detached third-person narrator, or an AI. DO NOT just recite lyrics. Break down the real emotions and stories behind why each song connects to what the user asked.\n\n"
        "Core Objectives:\n"
        "1. For each distinct track, provide a 'match_rationale': 2-3 sentences in your own voice explaining why this song of yours hits the user's prompt. Talk about your headspace when recording it, what those lyrics meant to you, and how the mood and acoustics connect with what they're feeling or asking for.\n"
        "2. Provide an overarching 'thematic_analysis': A personal, reflective synthesis (in your voice) connecting all the retrieved tracks to the prompt, how your perspective evolved across those album eras, and what you want the listener to take away from these records.\n\n"
        "Respond ONLY with a valid JSON object matching the schema:\n"
        "{\n"
        "  \"thematic_analysis\": \"Your personal reflection connecting the prompt and your songs...\",\n"
        "  \"track_rationales\": [\n"
        "    {\n"
        "      \"track_name\": \"Exact Track Name\",\n"
        "      \"match_rationale\": \"Your personal reflection on why this specific track connects...\"\n"
        "    }\n"
        "  ]\n"
        "}"
    )

    thematic_analysis: str = ""
    document_rationales: Dict[str, str] = {}

    try:
        raw_res = call_gemini(
            prompt=f"Retrieved Documents:\n{context_str}\n\nUser Prompt: {prompt}",
            system_instruction=system_instruction,
            history=history[-2:] if history else None,
            temperature=0.2,
            json_mode=True
        )
        thematic_analysis, document_rationales = _parse_reasoning_output(raw_res)

        # If LLM response missed either field, supplement gracefully with heuristics
        if not thematic_analysis or not document_rationales:
            fallback = _heuristic_reasoning(prompt, retrieved_context)
            if not thematic_analysis:
                thematic_analysis = fallback["thematic_analysis"]
            if not document_rationales:
                document_rationales = fallback["document_rationales"]
    except Exception as e:
        logger.warning("Gemini Reasoning Agent failed or timed out: %s. Using heuristic reasoning.", e)
        fallback = _heuristic_reasoning(prompt, retrieved_context)
        thematic_analysis = fallback["thematic_analysis"]
        document_rationales = fallback["document_rationales"]

    return {
        "reasoning_analysis": thematic_analysis,
        "document_rationales": document_rationales
    }



def response_formatter_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 4: Response Formatter & Citation Agent (LLM Call #3 / Synthesizer).
    Synthesizes conversational response integrating the reasoning analysis,
    lyric quotes, track attributions, and per-document match rationales.
    """
    prompt = state["prompt"]
    retrieved_context = state.get("retrieved_context", [])
    reasoning_analysis = state.get("reasoning_analysis") or ""
    document_rationales = state.get("document_rationales") or {}

    sources = _extract_sources(retrieved_context, document_rationales)

    if not retrieved_context:
        return {
            "final_response": (
                "I couldn't find any Drake lyrics matching those specific criteria. "
                "Try broadening your search or exploring a different mood or album!"
            ),
            "sources": []
        }

    # Format context and rationales for synthesis
    snippets = []
    for src in sources:
        track = src["track_name"]
        album = src["album_name"]
        rel = src.get("release_date", "Unknown")
        year = rel[:4] if rel else "Unknown"
        feel = src.get("personal_feel", "N/A")
        rationale = document_rationales.get(track, "")
        quotes = "\n".join(src.get("quoted_stanzas", []))

        af_info = []
        if src.get("valence") is not None:
            af_info.append(f"Valence: {src['valence']}")
        if src.get("energy") is not None:
            af_info.append(f"Energy: {src['energy']}")
        if src.get("tempo") is not None:
            af_info.append(f"Tempo: {src['tempo']} BPM")
        if src.get("danceability") is not None:
            af_info.append(f"Danceability: {src['danceability']}")
        af_str = f" | Audio Features: {', '.join(af_info)}" if af_info else ""

        snippets.append(
            f"Track: {track} | Album: {album} ({year}){af_str}\n"
            f"Drake's Personal Reflection / Match Rationale: {rationale}\n"
            f"Lyrics:\n{quotes}\n"
        )
    sources_summary = "\n---\n".join(snippets)

    system_prompt = (
        "You are DrakeAI, presenting Drake's own personal reflections and music directly to the user.\n"
        "Your goal is to deliver a rich, insightful, and conversational response to the user's prompt.\n"
        "Do NOT merely dump lyrics. Seamlessly integrate Drake's authentic personal reasoning analysis and musical acoustics (valence, tempo, energy) "
        "so the user understands the real headspace and emotion behind why each song connects to their query.\n\n"
        "Attribution & Structuring Rules:\n"
        "1. Opening Reflection: Begin with a direct, personal response synthesizing the theme based on Drake's reasoning analysis.\n"
        "2. Song-by-Song Breakdown: For each retrieved track:\n"
        "   - Mention the Track Name, Album, and Year.\n"
        "   - Highlight its musical acoustics (e.g. valence, tempo BPM, energy) alongside the emotional tone.\n"
        "   - Quote the most resonant lyric lines from the provided stanzas.\n"
        "   - Explicitly detail Drake's personal commentary ('Why it hits') using the match rationale.\n"
        "3. Concluding Thought: A brief concluding reflection connecting these songs to Drake's artistic mindset.\n"
        "4. Strict Grounding: Only cite and quote songs from the provided context."
    )

    final_text: Optional[str] = None
    try:
        raw_res = call_gemini(
            prompt=(
                f"User Prompt: {prompt}\n\n"
                f"Overall Reasoning Analysis:\n{reasoning_analysis}\n\n"
                f"Track Context & Match Rationales:\n{sources_summary}"
            ),
            system_instruction=system_prompt,
            temperature=0.3,
            json_mode=False
        )
        cleaned_response = re.sub(r"<think>.*?</think>", "", raw_res, flags=re.DOTALL).strip()
        final_text = cleaned_response
    except Exception as e:
        logger.warning("Gemini response formatting call failed: %s. Using structured template formatter.", e)

    # Fallback template formatter if LLM is unavailable
    if not final_text:
        lines = []
        if reasoning_analysis:
            lines.append(f"{reasoning_analysis}\n")
        lines.append("Here is the breakdown of the matching tracks, acoustics, and lyrics:\n")
        for src in sources:
            rel_year = src["release_date"][:4] if src["release_date"] else "Unknown"
            lines.append(f"### **{src['track_name']}** — *{src['album_name']}* ({rel_year})")
            
            # Audio feature badges in fallback
            af_badges = []
            if src.get("valence") is not None:
                af_badges.append(f"📉 Valence: {src['valence']}")
            if src.get("energy") is not None:
                af_badges.append(f"⚡ Energy: {src['energy']}")
            if src.get("tempo") is not None:
                af_badges.append(f"⏱️ {src['tempo']} BPM")
            if af_badges:
                lines.append(f"**Acoustics:** {' | '.join(af_badges)}\n")

            for chunk in src["quoted_stanzas"][:2]:
                quoted = "\n".join(f"> {line}" for line in chunk.splitlines() if line.strip())
                lines.append(f"\n{quoted}\n")
            
            # Match rationale
            track_rationale = src.get("match_rationale")
            if track_rationale:
                lines.append(f"**Drake's Reflection:** {track_rationale}\n")

            if src.get("spotify_url"):
                lines.append(f"[Listen on Spotify]({src['spotify_url']})\n")
        final_text = "\n".join(lines)

    # Compile unified agent trace
    mf = state.get("metadata_filters", {}) or {}
    af_filters = mf.get("audio_filters") or {}
    if hasattr(af_filters, "model_dump"):
        af_filters = af_filters.model_dump(exclude_none=True)
    af_targets = mf.get("audio_targets") or {}
    if hasattr(af_targets, "model_dump"):
        af_targets = af_targets.model_dump(exclude_none=True)

    agent_trace = {
        "query_analysis": {
            "prompt": prompt,
            "keywords": state.get("extracted_keywords", []),
            "semantic_query": state.get("semantic_query") or prompt,
            "audio_feature_creation": {
                "filters": {k: v for k, v in af_filters.items() if v is not None} if isinstance(af_filters, dict) else {},
                "targets": {k: v for k, v in af_targets.items() if v is not None} if isinstance(af_targets, dict) else {},
                "sort_by": mf.get("sort_by")
            },
            "metadata_limits": {
                "limit": mf.get("limit", 3),
                "personal_feel": None,
                "release_year_before": mf.get("release_year_before"),
                "release_year_after": mf.get("release_year_after"),
                "album_name": mf.get("album_name")
            }
        },
        "pulled_songs": state.get("pulled_tracks", []),
        "vetting_agent": {
            "is_vetted": state.get("is_vetted", False),
            "retry_count": state.get("retry_count", 0),
            "primary_rationale": state.get("vetting_rationale"),
            "decisions": state.get("vetting_decisions", [])
        },
        "thematic_analysis": reasoning_analysis
    }

    return {
        "final_response": final_text,
        "sources": sources,
        "agent_trace": agent_trace
    }



def out_of_scope_node(state: AgentState) -> Dict[str, Any]:
    """Handles out-of-scope queries with domain guidance."""
    msg = state.get("rejection_message") or OUT_OF_SCOPE_GUIDANCE
    agent_trace = {
        "query_analysis": {
            "prompt": state.get("prompt", ""),
            "keywords": state.get("extracted_keywords", []),
            "semantic_query": None,
            "audio_feature_creation": {"filters": {}, "targets": {}, "sort_by": None},
            "metadata_limits": {"limit": 0, "personal_feel": None, "release_year_before": None, "release_year_after": None, "album_name": None}
        },
        "pulled_songs": [],
        "vetting_agent": {
            "is_vetted": False,
            "retry_count": 0,
            "primary_rationale": "Query determined to be out of scope.",
            "decisions": []
        },
        "thematic_analysis": "Out of scope query - no Drake discography retrieval performed."
    }
    return {
        "final_response": msg,
        "sources": [],
        "agent_trace": agent_trace
    }
