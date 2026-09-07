"""Database connection pool and hybrid context graph retrieval."""

import logging
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool
from pgvector.psycopg2 import register_vector

from backend.config import settings

logger = logging.getLogger(__name__)

_pool: Optional[ThreadedConnectionPool] = None


def get_db_pool() -> ThreadedConnectionPool:
    global _pool
    if _pool is None:
        logger.info("Initializing PostgreSQL connection pool...")
        _pool = ThreadedConnectionPool(
            minconn=settings.DB_POOL_MIN,
            maxconn=settings.DB_POOL_MAX,
            dsn=settings.DATABASE_URL
        )
    return _pool


def close_db_pool() -> None:
    global _pool
    if _pool is not None:
        logger.info("Closing PostgreSQL connection pool...")
        _pool.closeall()
        _pool = None


@contextmanager
def get_db_connection() -> Generator[psycopg2.extensions.connection, None, None]:
    """Context manager for acquiring and releasing pool connections."""
    pool = get_db_pool()
    conn = pool.getconn()
    try:
        register_vector(conn)
        yield conn
    finally:
        pool.putconn(conn)


def check_db_health() -> bool:
    """Checks whether the database is responsive."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                res = cur.fetchone()
                return res is not None and res[0] == 1
    except Exception as e:
        logger.error("Database health check failed: %s", e)
        return False


CONTEXT_GRAPH_SQL = """
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
WHERE 1=1
  AND (%(excluded_track_ids)s IS NULL OR NOT (t.id = ANY(%(excluded_track_ids)s)))
  AND (%(personal_feel)s IS NULL OR s.personal_feel = %(personal_feel)s)
  AND (%(release_year_before)s IS NULL OR EXTRACT(YEAR FROM a.release_date) < %(release_year_before)s)
  AND (%(release_year_after)s IS NULL OR EXTRACT(YEAR FROM a.release_date) > %(release_year_after)s)
  AND (%(album_name)s IS NULL OR a.name ILIKE '%%' || %(album_name)s || '%%')
  AND (%(album_type)s IS NULL OR a.album_type ILIKE %(album_type)s)
  AND (%(min_valence)s IS NULL OR COALESCE(taf.valence, af.valence) >= %(min_valence)s)
  AND (%(max_valence)s IS NULL OR COALESCE(taf.valence, af.valence) <= %(max_valence)s)
  AND (%(min_energy)s IS NULL OR COALESCE(taf.energy, af.energy) >= %(min_energy)s)
  AND (%(max_energy)s IS NULL OR COALESCE(taf.energy, af.energy) <= %(max_energy)s)
  AND (%(min_danceability)s IS NULL OR COALESCE(taf.danceability, af.danceability) >= %(min_danceability)s)
  AND (%(max_danceability)s IS NULL OR COALESCE(taf.danceability, af.danceability) <= %(max_danceability)s)
  AND (%(min_tempo)s IS NULL OR COALESCE(taf.tempo, af.tempo) >= %(min_tempo)s)
  AND (%(max_tempo)s IS NULL OR COALESCE(taf.tempo, af.tempo) <= %(max_tempo)s)
  AND (%(min_acousticness)s IS NULL OR COALESCE(taf.acousticness, af.acousticness) >= %(min_acousticness)s)
  AND (%(max_acousticness)s IS NULL OR COALESCE(taf.acousticness, af.acousticness) <= %(max_acousticness)s)
  AND (%(mode)s IS NULL OR COALESCE(taf.mode, af.mode) = %(mode)s)
ORDER BY s.embedding <=> %(query_vector)s::vector(1024) ASC
LIMIT %(fetch_limit)s;
"""


def _compute_audio_similarity(
    row: Dict[str, Any],
    audio_targets: Optional[Dict[str, Any]]
) -> float:
    """
    Computes normalized audio feature proximity (0.0 to 1.0) against target vector.
    Uses weighted Euclidean distance across active target features.
    """
    if not audio_targets:
        return 1.0

    target_diffs = []
    weights = []

    # Valence target (weight 2.0)
    if audio_targets.get("target_valence") is not None and row.get("valence") is not None:
        diff = float(row["valence"]) - float(audio_targets["target_valence"])
        target_diffs.append((diff ** 2) * 2.0)
        weights.append(2.0)

    # Energy target (weight 1.5)
    if audio_targets.get("target_energy") is not None and row.get("energy") is not None:
        diff = float(row["energy"]) - float(audio_targets["target_energy"])
        target_diffs.append((diff ** 2) * 1.5)
        weights.append(1.5)

    # Danceability target (weight 1.5)
    if audio_targets.get("target_danceability") is not None and row.get("danceability") is not None:
        diff = float(row["danceability"]) - float(audio_targets["target_danceability"])
        target_diffs.append((diff ** 2) * 1.5)
        weights.append(1.5)

    # Tempo target (normalized by 100 BPM, weight 1.0)
    if audio_targets.get("target_tempo") is not None and row.get("tempo") is not None:
        diff = (float(row["tempo"]) - float(audio_targets["target_tempo"])) / 100.0
        target_diffs.append((diff ** 2) * 1.0)
        weights.append(1.0)

    # Acousticness target (weight 1.0)
    if audio_targets.get("target_acousticness") is not None and row.get("acousticness") is not None:
        diff = float(row["acousticness"]) - float(audio_targets["target_acousticness"])
        target_diffs.append((diff ** 2) * 1.0)
        weights.append(1.0)

    # Speechiness target (weight 1.0)
    if audio_targets.get("target_speechiness") is not None and row.get("speechiness") is not None:
        diff = float(row["speechiness"]) - float(audio_targets["target_speechiness"])
        target_diffs.append((diff ** 2) * 1.0)
        weights.append(1.0)

    if not weights:
        return 1.0

    dist = (sum(target_diffs) / sum(weights)) ** 0.5
    return max(0.0, 1.0 - dist)


def _build_sql_params(
    query_vector: List[float],
    filters: Dict[str, Any],
    clean_excluded: Optional[List[str]],
    fetch_limit: int,
    af_filters: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "query_vector": query_vector,
        "excluded_track_ids": clean_excluded,
        "personal_feel": filters.get("personal_feel"),
        "release_year_before": filters.get("release_year_before"),
        "release_year_after": filters.get("release_year_after"),
        "album_name": filters.get("album_name"),
        "album_type": filters.get("album_type"),
        "min_valence": af_filters.get("min_valence"),
        "max_valence": af_filters.get("max_valence"),
        "min_energy": af_filters.get("min_energy"),
        "max_energy": af_filters.get("max_energy"),
        "min_danceability": af_filters.get("min_danceability"),
        "max_danceability": af_filters.get("max_danceability"),
        "min_tempo": af_filters.get("min_tempo"),
        "max_tempo": af_filters.get("max_tempo"),
        "min_acousticness": af_filters.get("min_acousticness"),
        "max_acousticness": af_filters.get("max_acousticness"),
        "mode": af_filters.get("mode"),
        "fetch_limit": fetch_limit
    }


def search_context_graph(
    query_vector: List[float],
    filters: Optional[Dict[str, Any]] = None,
    excluded_track_ids: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Executes parent-child hybrid context graph retrieval joining stanza, track, album,
    and audio features using pgvector cosine distance and SQL metadata/audio filters.
    Applies multi-modal weighted hybrid scoring and automatic filter relaxation
    if strict constraints yield fewer candidates than requested.
    """
    if filters is None:
        filters = {}

    limit = filters.get("limit") or 3
    limit = max(1, min(int(limit), 10))

    clean_excluded = list(excluded_track_ids) if excluded_track_ids else None

    # Parse audio filter dict
    raw_af = filters.get("audio_filters") or {}
    if hasattr(raw_af, "model_dump"):
        raw_af = raw_af.model_dump()

    # Parse audio target dict
    raw_targets = filters.get("audio_targets") or {}
    if hasattr(raw_targets, "model_dump"):
        raw_targets = raw_targets.model_dump()

    # Determine fetch multiplier for re-ranking pool
    has_audio_intent = bool(raw_af or raw_targets or filters.get("sort_by"))
    fetch_limit = max(limit * 5, 20) if has_audio_intent else limit

    query_params = _build_sql_params(
        query_vector=query_vector,
        filters=filters,
        clean_excluded=clean_excluded,
        fetch_limit=fetch_limit,
        af_filters=raw_af
    )

    rows: List[Dict[str, Any]] = []
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(CONTEXT_GRAPH_SQL, query_params)
            rows = cur.fetchall()

            # Automatic Filter Relaxation (Option A):
            # If hard filters produced fewer rows than requested limit, relax thresholds
            if len(rows) < limit and any(v is not None for v in raw_af.values()):
                logger.info("Hard audio filters matched only %d rows (< %d). Relaxing filter bounds...", len(rows), limit)
                relaxed_af = dict(raw_af)
                if relaxed_af.get("min_valence") is not None:
                    relaxed_af["min_valence"] = max(0.0, relaxed_af["min_valence"] - 0.15)
                if relaxed_af.get("max_valence") is not None:
                    relaxed_af["max_valence"] = min(1.0, relaxed_af["max_valence"] + 0.15)
                if relaxed_af.get("min_energy") is not None:
                    relaxed_af["min_energy"] = max(0.0, relaxed_af["min_energy"] - 0.15)
                if relaxed_af.get("max_energy") is not None:
                    relaxed_af["max_energy"] = min(1.0, relaxed_af["max_energy"] + 0.15)
                if relaxed_af.get("min_danceability") is not None:
                    relaxed_af["min_danceability"] = max(0.0, relaxed_af["min_danceability"] - 0.15)
                if relaxed_af.get("max_danceability") is not None:
                    relaxed_af["max_danceability"] = min(1.0, relaxed_af["max_danceability"] + 0.15)
                if relaxed_af.get("min_tempo") is not None:
                    relaxed_af["min_tempo"] = max(40.0, relaxed_af["min_tempo"] - 15.0)
                if relaxed_af.get("max_tempo") is not None:
                    relaxed_af["max_tempo"] = min(250.0, relaxed_af["max_tempo"] + 15.0)

                relaxed_params = _build_sql_params(
                    query_vector=query_vector,
                    filters=filters,
                    clean_excluded=clean_excluded,
                    fetch_limit=fetch_limit,
                    af_filters=relaxed_af
                )
                cur.execute(CONTEXT_GRAPH_SQL, relaxed_params)
                relaxed_rows = cur.fetchall()
                if len(relaxed_rows) >= len(rows):
                    rows = relaxed_rows

            # Second fallback if still empty: drop hard audio filters entirely
            # and rely on soft target hybrid scoring
            if len(rows) == 0 and any(v is not None for v in raw_af.values()):
                logger.info("Relaxed filters still empty. Dropping hard audio filters and using soft scoring.")
                fallback_params = _build_sql_params(
                    query_vector=query_vector,
                    filters=filters,
                    clean_excluded=clean_excluded,
                    fetch_limit=fetch_limit,
                    af_filters={}
                )
                cur.execute(CONTEXT_GRAPH_SQL, fallback_params)
                rows = cur.fetchall()

    # Determine alpha weighting for hybrid score
    # Default alpha = 0.60 semantic, 0.40 audio
    # If query is explicitly audio-dominated, alpha = 0.35 semantic, 0.65 audio
    alpha = 0.60
    sort_by = filters.get("sort_by")
    if sort_by in ("valence_asc", "valence_desc", "energy_desc", "tempo_asc", "danceability_desc"):
        alpha = 0.35

    results = []
    for r in rows:
        item = dict(r)
        if item.get("release_date") is not None:
            item["release_date"] = str(item["release_date"])
        
        sim_sem = float(item["similarity"]) if item.get("similarity") is not None else 0.0
        item["similarity"] = round(sim_sem, 4)

        # Convert audio features to floats if present
        for af_key in ("valence", "energy", "danceability", "tempo", "acousticness", "loudness", "speechiness"):
            if item.get(af_key) is not None:
                item[af_key] = round(float(item[af_key]), 4)
        if item.get("mode") is not None:
            item["mode"] = int(item["mode"])
        if item.get("key") is not None:
            item["key"] = int(item["key"])

        # Compute multi-modal hybrid score
        if raw_targets:
            sim_audio = _compute_audio_similarity(item, raw_targets)
            hybrid_score = (alpha * sim_sem) + ((1.0 - alpha) * sim_audio)
        else:
            hybrid_score = sim_sem

        item["hybrid_score"] = round(hybrid_score, 4)
        results.append(item)

    # Sort results
    if sort_by == "valence_asc":
        results.sort(key=lambda x: (x.get("valence") if x.get("valence") is not None else 1.0, -x["similarity"]))
    elif sort_by == "valence_desc":
        results.sort(key=lambda x: (-(x.get("valence") or 0.0), -x["similarity"]))
    elif sort_by == "energy_desc":
        results.sort(key=lambda x: (-(x.get("energy") or 0.0), -x["similarity"]))
    elif sort_by == "tempo_asc":
        results.sort(key=lambda x: (x.get("tempo") if x.get("tempo") is not None else 999.0, -x["similarity"]))
    elif raw_targets:
        results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    else:
        results.sort(key=lambda x: x["similarity"], reverse=True)

    return results[:limit]
