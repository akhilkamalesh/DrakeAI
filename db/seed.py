#!/usr/bin/env python3
"""
PostgreSQL Database Seeding Script for DrakeAI

Fulfills Phase 3: Embedding & Database Seeding
Inserts records from:
  - data/albums.json  -> album table
  - data/tracks.json  -> track table
  - data/stanza.json  -> stanza table (with pgvector embeddings)
"""

import os
import sys
import json
import time
import argparse
from typing import Optional, List, Dict, Any

import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

# Workspace paths
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(WORKSPACE_DIR, "data")
DB_DIR = os.path.join(WORKSPACE_DIR, "db")
ENV_PATH = os.path.join(WORKSPACE_DIR, ".env")
SCHEMA_PATH = os.path.join(DB_DIR, "schema.sql")

ALBUMS_PATH = os.path.join(DATA_DIR, "albums.json")
TRACKS_PATH = os.path.join(DATA_DIR, "tracks.json")
STANZA_PATH = os.path.join(DATA_DIR, "stanza.json")

# Load environment variables
load_dotenv(ENV_PATH)
DEFAULT_DB_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/drake")


def normalize_date(date_str: Optional[str]) -> Optional[str]:
    """Ensures dates are valid for PostgreSQL DATE column (e.g. '2009' -> '2009-01-01')."""
    if not date_str:
        return None
    d = date_str.strip()
    if len(d) == 4 and d.isdigit():
        return f"{d}-01-01"
    if len(d) == 7 and d[4] == "-":
        return f"{d}-01"
    return d


def apply_schema(conn, schema_path: str = SCHEMA_PATH):
    """Executes the DDL schema in schema.sql to ensure tables and indexes exist."""
    print(f"[Schema] Applying DDL from {schema_path}...")
    with open(schema_path, "r", encoding="utf-8") as f:
        sql = f.read()

    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
    print("[Schema] Tables and indexes created/verified successfully.")


def seed_albums(conn, albums_path: str = ALBUMS_PATH) -> int:
    """Seeds albums into the album table."""
    if not os.path.exists(albums_path):
        raise FileNotFoundError(f"Albums file not found at {albums_path}")

    with open(albums_path, "r", encoding="utf-8") as f:
        albums: List[Dict[str, Any]] = json.load(f)

    print(f"\n[Albums] Inserting {len(albums)} albums...")
    records = []
    for a in albums:
        records.append((
            a["id"],
            a["name"],
            a.get("album_type"),
            a.get("images_url"),
            normalize_date(a.get("release_date")),
            a.get("total_tracks")
        ))

    insert_sql = """
        INSERT INTO album (id, name, album_type, images_url, release_date, total_tracks)
        VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            name = EXCLUDED.name,
            album_type = EXCLUDED.album_type,
            images_url = EXCLUDED.images_url,
            release_date = EXCLUDED.release_date,
            total_tracks = EXCLUDED.total_tracks;
    """
    with conn.cursor() as cur:
        execute_values(cur, insert_sql, records, page_size=200)
    conn.commit()
    print(f"[Albums] Successfully seeded {len(records)} albums.")
    return len(records)


def seed_tracks(conn, tracks_path: str = TRACKS_PATH) -> int:
    """Seeds tracks into the track table."""
    if not os.path.exists(tracks_path):
        raise FileNotFoundError(f"Tracks file not found at {tracks_path}")

    with open(tracks_path, "r", encoding="utf-8") as f:
        tracks: List[Dict[str, Any]] = json.load(f)

    print(f"\n[Tracks] Inserting {len(tracks)} tracks...")
    records = []
    for t in tracks:
        ext_urls_json = json.dumps(t.get("external_urls", {}))
        records.append((
            t["id"],
            t["album_id"],
            t.get("disc_number", 1),
            t["name"],
            t.get("lyrics"),
            ext_urls_json
        ))

    insert_sql = """
        INSERT INTO track (id, album_id, disc_number, name, lyrics, external_urls)
        VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            album_id = EXCLUDED.album_id,
            disc_number = EXCLUDED.disc_number,
            name = EXCLUDED.name,
            lyrics = EXCLUDED.lyrics,
            external_urls = EXCLUDED.external_urls;
    """
    with conn.cursor() as cur:
        execute_values(cur, insert_sql, records, template="(%s, %s, %s, %s, %s, %s::jsonb)", page_size=200)
    conn.commit()
    print(f"[Tracks] Successfully seeded {len(records)} tracks.")
    return len(records)


def seed_stanzas(conn, stanza_path: str = STANZA_PATH, batch_size: int = 500) -> int:
    """Seeds stanzas into the stanza table with pgvector embeddings."""
    if not os.path.exists(stanza_path):
        raise FileNotFoundError(f"Stanza file not found at {stanza_path}")

    print(f"\n[Stanzas] Loading stanzas from {stanza_path}...")
    with open(stanza_path, "r", encoding="utf-8") as f:
        stanzas: List[Dict[str, Any]] = json.load(f)

    total = len(stanzas)
    print(f"[Stanzas] Inserting {total} stanzas (in batches of {batch_size})...")

    insert_sql = """
        INSERT INTO stanza (id, song_id, lyric_chunk, chunk_index, embedding, personal_feel)
        VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            song_id = EXCLUDED.song_id,
            lyric_chunk = EXCLUDED.lyric_chunk,
            chunk_index = EXCLUDED.chunk_index,
            embedding = EXCLUDED.embedding,
            personal_feel = EXCLUDED.personal_feel;
    """

    records = []
    for s in stanzas:
        records.append((
            s["id"],
            s["song_id"],
            s["lyric_chunk"],
            s["chunk_index"],
            s["embedding"],
            s.get("personal_feel")
        ))

    with conn.cursor() as cur:
        execute_values(cur, insert_sql, records, template="(%s, %s, %s, %s, %s::vector, %s)", page_size=batch_size)
    conn.commit()

    # Reset stanza_id_seq to current max id
    with conn.cursor() as cur:
        cur.execute("SELECT setval(pg_get_serial_sequence('stanza', 'id'), COALESCE(MAX(id), 1)) FROM stanza;")
    conn.commit()

    print(f"[Stanzas] Successfully seeded {total} stanzas.")
    return total


def verify_database(conn):
    """Performs integrity checks and runs a test vector similarity search."""
    print("\n=======================================================")
    print("Verifying Database Records & Vector Search")
    print("=======================================================")

    with conn.cursor() as cur:
        # Table counts
        cur.execute("SELECT count(*) FROM album;")
        album_count = cur.fetchone()[0]

        cur.execute("SELECT count(*) FROM track;")
        track_count = cur.fetchone()[0]

        cur.execute("SELECT count(*) FROM stanza;")
        stanza_count = cur.fetchone()[0]

        print(f"Table 'album':  {album_count} rows")
        print(f"Table 'track':  {track_count} rows")
        print(f"Table 'stanza': {stanza_count} rows")

        # Test pgvector similarity search
        print("\nRunning test semantic query for 'Late-Night Confessional' stanzas...")
        sample_query = """
            SELECT s.id, t.name AS track_name, s.personal_feel, SUBSTRING(s.lyric_chunk FROM 1 FOR 80) AS snippet
            FROM stanza s
            JOIN track t ON s.song_id = t.id
            WHERE s.personal_feel = 'Late-Night Confessional'
            LIMIT 3;
        """
        cur.execute(sample_query)
        rows = cur.fetchall()
        for r in rows:
            print(f"  [ID {r[0]}] Track: '{r[1]}' | Feel: {r[2]} | Lyric: {repr(r[3])}...")

        # Test vector distance calculation (<=> cosine distance)
        print("\nTesting vector cosine distance operator (<=>)...")
        cur.execute("""
            SELECT s.id, t.name AS track_name, (s.embedding <=> ref.embedding) AS distance
            FROM stanza s
            JOIN track t ON s.song_id = t.id
            CROSS JOIN (SELECT embedding FROM stanza WHERE id = 1) AS ref
            WHERE s.id != 1
            ORDER BY distance ASC
            LIMIT 3;
        """)
        v_rows = cur.fetchall()
        for vr in v_rows:
            print(f"  Nearest to Stanza #1: Stanza #{vr[0]} ('{vr[1]}') - Cosine Distance: {vr[2]:.4f}")

    print("\nAll database verifications completed successfully!\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Seed Drake discography into PostgreSQL.")
    parser.add_argument("--db-url", type=str, default=DEFAULT_DB_URL, help="PostgreSQL connection string")
    parser.add_argument("--drop-tables", action="store_true", help="Drop existing tables before seeding")
    parser.add_argument("--skip-stanzas", action="store_true", help="Skip stanza table seeding")
    return parser.parse_args()


def main():
    args = parse_args()
    db_url = args.db_url

    print(f"Connecting to PostgreSQL at {db_url}...")
    conn = psycopg2.connect(db_url)
    register_vector(conn)

    try:
        if args.drop_tables:
            print("[Warning] Dropping existing tables (CASCADE)...")
            with conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS stanza, audio_feature, track, album CASCADE;")
            conn.commit()

        # 1. Apply schema
        apply_schema(conn)

        # 2. Seed albums
        t0 = time.time()
        seed_albums(conn)

        # 3. Seed tracks
        seed_tracks(conn)

        # 4. Seed stanzas
        if not args.skip_stanzas:
            seed_stanzas(conn)

        # 5. Verification
        verify_database(conn)
        print(f"Total seeding time: {time.time() - t0:.2f}s")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
