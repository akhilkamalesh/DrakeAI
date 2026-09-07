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
    a.images_url AS album_art_url
FROM stanza s
JOIN track t ON s.song_id = t.id
JOIN album a ON t.album_id = a.id
WHERE 1=1
  AND (%(excluded_track_ids)s IS NULL OR NOT (t.id = ANY(%(excluded_track_ids)s)))
  AND (%(personal_feel)s IS NULL OR s.personal_feel = %(personal_feel)s)
  AND (%(release_year_before)s IS NULL OR EXTRACT(YEAR FROM a.release_date) < %(release_year_before)s)
  AND (%(release_year_after)s IS NULL OR EXTRACT(YEAR FROM a.release_date) > %(release_year_after)s)
  AND (%(album_name)s IS NULL OR a.name ILIKE '%%' || %(album_name)s || '%%')
  AND (%(album_type)s IS NULL OR a.album_type ILIKE %(album_type)s)
ORDER BY s.embedding <=> %(query_vector)s::vector(1024) ASC
LIMIT %(limit)s;
"""


def search_context_graph(
    query_vector: List[float],
    filters: Optional[Dict[str, Any]] = None,
    excluded_track_ids: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Executes parent-child hybrid context graph retrieval joining stanza, track, and album
    using pgvector cosine distance and SQL metadata filters.
    Pulls full parent track (t.lyrics) along with child stanza chunks.
    Supports excluding rejected track IDs during vetting retry loops.
    """
    if filters is None:
        filters = {}

    limit = filters.get("limit") or 3
    # Clamp limit between 1 and 10
    limit = max(1, min(int(limit), 10))

    clean_excluded = list(excluded_track_ids) if excluded_track_ids else None

    query_params = {
        "query_vector": query_vector,
        "excluded_track_ids": clean_excluded,
        "personal_feel": filters.get("personal_feel"),
        "release_year_before": filters.get("release_year_before"),
        "release_year_after": filters.get("release_year_after"),
        "album_name": filters.get("album_name"),
        "album_type": filters.get("album_type"),
        "limit": limit
    }

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(CONTEXT_GRAPH_SQL, query_params)
            rows = cur.fetchall()

            # Normalize row dicts (e.g. serialize dates and parse URLs)
            results = []
            for r in rows:
                item = dict(r)
                # Ensure release_date is string
                if item.get("release_date") is not None:
                    item["release_date"] = str(item["release_date"])
                if item.get("similarity") is not None:
                    item["similarity"] = float(item["similarity"])
                results.append(item)
            return results
