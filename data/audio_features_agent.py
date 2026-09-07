#!/usr/bin/env python3
"""
Drake Audio Features Generator Agent (data/audio_features_agent.py)

Generates Spotify-standard audio features (valence, energy, danceability,
tempo, key, loudness, mode, speechiness, acousticness, instrumentalness,
liveness, duration_ms, time_signature) for each Drake track based on the
track name using Google Gemini.

Saves results to data/track_audio_features.json and seeds records into the
PostgreSQL track_audio_feature table.
"""

import os
import sys
import json
import time
import argparse
from typing import List, Dict, Any, Optional

import requests
from dotenv import load_dotenv

# Root workspace directory
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(WORKSPACE_DIR, ".env"))

DATA_DIR = os.path.join(WORKSPACE_DIR, "data")
DB_DIR = os.path.join(WORKSPACE_DIR, "db")

TRACKS_PATH = os.path.join(DATA_DIR, "tracks.json")
ALBUMS_PATH = os.path.join(DATA_DIR, "albums.json")
RAW_CATALOG_PATH = os.path.join(DATA_DIR, "raw_spotify_catalog.json")
OUTPUT_JSON_PATH = os.path.join(DATA_DIR, "track_audio_features.json")
RAW_AUDIO_FEATURES_PATH = os.path.join(DATA_DIR, "raw_audio_features.json")
SCHEMA_SQL_PATH = os.path.join(DB_DIR, "schema.sql")

DEFAULT_DB_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/drake")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


def get_gemini_config():
    """Retrieves Gemini API key and model name from environment."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY must be set in .env")
    model = (os.getenv("GEMINI_MODEL") or "gemini-flash-latest").strip()
    if not model or "2.5" in model:
        model = "gemini-flash-latest"
    return api_key, model


def clamp(val: float, min_val: float, max_val: float) -> float:
    """Clamps float value within [min_val, max_val]."""
    return max(min_val, min(val, max_val))


def validate_and_normalize_feature(item: Dict[str, Any], known_duration: Optional[int] = None) -> Dict[str, Any]:
    """
    Validates and clamps audio feature fields to ensure compliance with
    Spotify audio feature specifications and PostgreSQL schema constraints.
    """
    track_id = str(item.get("track_id", "")).strip()

    danceability = round(clamp(float(item.get("danceability", 0.5)), 0.0, 1.0), 4)
    energy = round(clamp(float(item.get("energy", 0.5)), 0.0, 1.0), 4)
    key = int(clamp(int(item.get("key", 0)), 0, 11))
    loudness = round(clamp(float(item.get("loudness", -8.0)), -60.0, 0.0), 2)
    mode = 1 if int(item.get("mode", 0)) == 1 else 0
    speechiness = round(clamp(float(item.get("speechiness", 0.1)), 0.0, 1.0), 4)
    acousticness = round(clamp(float(item.get("acousticness", 0.2)), 0.0, 1.0), 4)
    instrumentalness = round(clamp(float(item.get("instrumentalness", 0.0001)), 0.0, 1.0), 6)
    liveness = round(clamp(float(item.get("liveness", 0.1)), 0.0, 1.0), 4)
    valence = round(clamp(float(item.get("valence", 0.5)), 0.0, 1.0), 4)
    tempo = round(clamp(float(item.get("tempo", 100.0)), 40.0, 250.0), 2)

    duration_ms = known_duration or item.get("duration_ms") or 210000
    try:
        duration_ms = int(duration_ms)
        if duration_ms <= 0:
            duration_ms = 210000
    except (ValueError, TypeError):
        duration_ms = 210000

    time_signature = int(item.get("time_signature", 4))
    if time_signature not in (3, 4, 5, 6, 7):
        time_signature = 4

    return {
        "track_id": track_id,
        "danceability": danceability,
        "energy": energy,
        "key": key,
        "loudness": loudness,
        "mode": mode,
        "speechiness": speechiness,
        "acousticness": acousticness,
        "instrumentalness": instrumentalness,
        "liveness": liveness,
        "valence": valence,
        "tempo": tempo,
        "type": "audio_features",
        "duration_ms": duration_ms,
        "time_signature": time_signature,
    }


class AudioFeaturesAgent:
    """
    Autonomous agent that generates Spotify-standard audio features for tracks
    using Google Gemini and manages JSON persistence and PostgreSQL seeding.
    """

    def __init__(
        self,
        tracks_path: str = TRACKS_PATH,
        output_path: str = OUTPUT_JSON_PATH,
        raw_catalog_path: str = RAW_CATALOG_PATH,
        db_url: str = DEFAULT_DB_URL,
        batch_size: int = 20,
    ):
        self.tracks_path = tracks_path
        self.output_path = output_path
        self.raw_catalog_path = raw_catalog_path
        self.db_url = db_url
        self.batch_size = batch_size

        self.api_key, self.model = get_gemini_config()
        self.duration_map = self._load_known_durations()
        self.cache: Dict[str, Dict[str, Any]] = self._load_existing_cache()

    def _load_known_durations(self) -> Dict[str, int]:
        """Loads track duration_ms from raw_spotify_catalog.json if present."""
        duration_map = {}
        if os.path.exists(self.raw_catalog_path):
            try:
                with open(self.raw_catalog_path, "r", encoding="utf-8") as f:
                    cat = json.load(f)
                    for t in cat.get("tracks", []):
                        if "id" in t and "duration_ms" in t:
                            duration_map[t["id"]] = int(t["duration_ms"])
            except Exception as e:
                print(f"[Agent] Notice: Could not parse durations from raw catalog: {e}")
        return duration_map

    def _load_existing_cache(self) -> Dict[str, Dict[str, Any]]:
        """Loads already generated features from output JSON file to support resumption."""
        cache = {}
        if os.path.exists(self.output_path):
            try:
                with open(self.output_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            tid = item.get("track_id")
                            if tid:
                                cache[tid] = item
                print(f"[Agent] Found existing cache with {len(cache)} tracks in {self.output_path}.")
            except Exception as e:
                print(f"[Agent] Notice: Could not read existing output file: {e}")
        return cache

    def _save_cache(self):
        """Saves current cache to output JSON file and raw_audio_features.json."""
        items = list(self.cache.values())
        temp_path = f"{self.output_path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2)
        os.replace(temp_path, self.output_path)

        # Also update raw_audio_features.json for compatibility
        try:
            raw_temp = f"{RAW_AUDIO_FEATURES_PATH}.tmp"
            with open(raw_temp, "w", encoding="utf-8") as f:
                json.dump({
                    "artist_id": "3TVXtAsR1Inumwj472S9r4",
                    "total_tracks": len(items),
                    "audio_features": items
                }, f, indent=2)
            os.replace(raw_temp, RAW_AUDIO_FEATURES_PATH)
        except Exception:
            pass

    def _call_gemini_batch(self, tracks_batch: List[Dict[str, Any]], max_retries: int = 3) -> List[Dict[str, Any]]:
        """Calls Gemini API with JSON mode to estimate audio features for a batch of tracks."""
        track_list_text = "\n".join([
            f"- Track ID: \"{t['id']}\", Title: \"{t['name']}\""
            for t in tracks_batch
        ])

        system_instruction = (
            "You are an expert musicologist, audio engineer, and data scientist specializing in Drake's discography. "
            "For each Drake track provided, generate accurate Spotify-standard audio features based on the track name "
            "and musical style.\n"
            "Key musical guidelines for Drake:\n"
            "- Late-night ambient/melodic R&B songs (e.g. Marvins Room, Jungle, Jaded) have low valence (0.15-0.35), "
            "low-to-mid energy (0.2-0.45), slower tempo (70-95 BPM), and higher acousticness.\n"
            "- Club & Dancehall hits (e.g. One Dance, Controlla, In My Feelings, Passionfruit) have high danceability (0.75-0.9), "
            "higher valence (0.5-0.85), mid-to-high energy (0.55-0.8), and tempos around 100-120 BPM.\n"
            "- Aggressive trap bangers (e.g. Energy, Mob Ties, Nonstop, Knife Talk) have high speechiness (0.15-0.35), "
            "high energy (0.65-0.85), minor keys (mode=0), low acousticness, and punchy loudness (-7 to -4 dB).\n"
            "- Pitch class keys range from 0 (C) to 11 (B).\n"
            "- Mode is 1 for Major, 0 for Minor.\n"
            "- Valence is musical positiveness/happiness from 0.0 (sad/depressed) to 1.0 (euphoric/cheerful)."
        )

        prompt = (
            f"Analyze the following {len(tracks_batch)} Drake tracks and generate audio features for each.\n\n"
            f"Tracks to analyze:\n{track_list_text}\n\n"
            "Return a JSON array of objects matching this exact schema for every track in the list:\n"
            "[\n"
            "  {\n"
            "    \"track_id\": \"<exact track_id from input>\",\n"
            "    \"danceability\": <float 0.0 to 1.0>,\n"
            "    \"energy\": <float 0.0 to 1.0>,\n"
            "    \"key\": <int 0 to 11>,\n"
            "    \"loudness\": <float -60.0 to 0.0, e.g. -7.5>,\n"
            "    \"mode\": <int 0 or 1>,\n"
            "    \"speechiness\": <float 0.0 to 1.0>,\n"
            "    \"acousticness\": <float 0.0 to 1.0>,\n"
            "    \"instrumentalness\": <float 0.0 to 1.0>,\n"
            "    \"liveness\": <float 0.0 to 1.0>,\n"
            "    \"valence\": <float 0.0 to 1.0>,\n"
            "    \"tempo\": <float 40.0 to 220.0 BPM>,\n"
            "    \"type\": \"audio_features\",\n"
            "    \"duration_ms\": <int>,\n"
            "    \"time_signature\": <int 3, 4, or 5>\n"
            "  }\n"
            "]"
        )

        url = f"{GEMINI_BASE_URL}/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json"
            }
        }

        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=45
                )
                if response.status_code == 200:
                    data = response.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        raw_text = candidates[0]["content"]["parts"][0]["text"]
                        parsed = json.loads(raw_text)
                        if isinstance(parsed, list):
                            return parsed
                        elif isinstance(parsed, dict) and "audio_features" in parsed:
                            return parsed["audio_features"]
                    raise RuntimeError("Gemini returned empty or unexpected response structure.")
                elif response.status_code == 429:
                    wait_time = attempt * 5
                    print(f"  [Rate Limit 429] Waiting {wait_time}s before retry (attempt {attempt}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    print(f"  [API Error {response.status_code}]: {response.text[:200]}")
                    time.sleep(attempt * 2)
            except Exception as e:
                print(f"  [Attempt {attempt} failed]: {e}")
                time.sleep(attempt * 2)

        print(f"  [Warning] Failed to generate features for batch of {len(tracks_batch)} tracks.")
        return []

    def generate(self, limit: Optional[int] = None, force: bool = False) -> List[Dict[str, Any]]:
        """
        Generates audio features for all tracks in tracks.json (or up to limit).
        Resumes from existing cache unless force is True.
        """
        if not os.path.exists(self.tracks_path):
            raise FileNotFoundError(f"Tracks file not found at {self.tracks_path}")

        with open(self.tracks_path, "r", encoding="utf-8") as f:
            all_tracks: List[Dict[str, Any]] = json.load(f)

        if limit is not None:
            all_tracks = all_tracks[:limit]

        total_tracks = len(all_tracks)
        print(f"\n[Agent] Starting audio features generation for {total_tracks} tracks...")

        # Determine tracks that need processing
        pending_tracks = []
        for t in all_tracks:
            tid = t["id"]
            if force or tid not in self.cache:
                pending_tracks.append(t)

        print(f"[Agent] {len(self.cache)} tracks cached. {len(pending_tracks)} pending generation.")

        if pending_tracks:
            num_batches = (len(pending_tracks) + self.batch_size - 1) // self.batch_size
            for b_idx in range(num_batches):
                batch = pending_tracks[b_idx * self.batch_size : (b_idx + 1) * self.batch_size]
                print(f"\n[Batch {b_idx + 1}/{num_batches}] Generating audio features for {len(batch)} tracks...")

                raw_results = self._call_gemini_batch(batch)
                results_by_id = {item.get("track_id"): item for item in raw_results if "track_id" in item}

                # Process and validate each track in the batch
                for t in batch:
                    tid = t["id"]
                    known_dur = self.duration_map.get(tid)
                    if tid in results_by_id:
                        normalized = validate_and_normalize_feature(results_by_id[tid], known_dur)
                    else:
                        # Fallback heuristic if Gemini missed a track in the batch
                        print(f"  [Fallback] Generating baseline features for '{t['name']}' ({tid})")
                        normalized = validate_and_normalize_feature({
                            "track_id": tid,
                            "danceability": 0.60,
                            "energy": 0.55,
                            "key": 1,
                            "loudness": -7.5,
                            "mode": 0,
                            "speechiness": 0.12,
                            "acousticness": 0.25,
                            "instrumentalness": 0.00001,
                            "liveness": 0.12,
                            "valence": 0.40,
                            "tempo": 115.0,
                            "duration_ms": known_dur or 210000,
                            "time_signature": 4
                        }, known_dur)

                    self.cache[tid] = normalized

                # Incremental persistence
                self._save_cache()
                print(f"  Saved progress: {len(self.cache)} total tracks saved to {self.output_path}")

        print(f"\n[Agent] Generation complete. Total tracks in catalog: {len(self.cache)}.")
        return list(self.cache.values())

    def seed_postgres(self, db_url: Optional[str] = None) -> int:
        """
        Seeds generated audio features from the cache/JSON directly into
        the PostgreSQL track_audio_feature table.
        """
        import psycopg2
        from psycopg2.extras import execute_values

        target_url = db_url or self.db_url
        print(f"\n[PostgreSQL] Connecting to {target_url}...")
        conn = psycopg2.connect(target_url)

        try:
            # Ensure table exists
            if os.path.exists(SCHEMA_SQL_PATH):
                with open(SCHEMA_SQL_PATH, "r", encoding="utf-8") as f:
                    schema_sql = f.read()
                with conn.cursor() as cur:
                    cur.execute(schema_sql)
                conn.commit()

            items = list(self.cache.values())
            if not items:
                print("[PostgreSQL] No audio features available to seed.")
                return 0

            records = []
            for af in items:
                records.append((
                    af["track_id"],
                    af["danceability"],
                    af["energy"],
                    af["key"],
                    af["loudness"],
                    af["mode"],
                    af["speechiness"],
                    af["acousticness"],
                    af["instrumentalness"],
                    af["liveness"],
                    af["valence"],
                    af["tempo"],
                    af.get("type", "audio_features"),
                    af.get("duration_ms"),
                    af.get("time_signature", 4),
                ))

            insert_sql = """
                INSERT INTO track_audio_feature (
                    track_id, danceability, energy, "key", loudness, "mode",
                    speechiness, acousticness, instrumentalness, liveness,
                    valence, tempo, type, duration_ms, time_signature
                )
                VALUES %s
                ON CONFLICT (track_id) DO UPDATE SET
                    danceability = EXCLUDED.danceability,
                    energy = EXCLUDED.energy,
                    "key" = EXCLUDED.key,
                    loudness = EXCLUDED.loudness,
                    "mode" = EXCLUDED.mode,
                    speechiness = EXCLUDED.speechiness,
                    acousticness = EXCLUDED.acousticness,
                    instrumentalness = EXCLUDED.instrumentalness,
                    liveness = EXCLUDED.liveness,
                    valence = EXCLUDED.valence,
                    tempo = EXCLUDED.tempo,
                    type = EXCLUDED.type,
                    duration_ms = EXCLUDED.duration_ms,
                    time_signature = EXCLUDED.time_signature;
            """
            with conn.cursor() as cur:
                execute_values(cur, insert_sql, records, page_size=200)
            conn.commit()

            # Verification count
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM track_audio_feature;")
                db_count = cur.fetchone()[0]

            print(f"[PostgreSQL] Successfully seeded {len(records)} records into track_audio_feature (Total rows: {db_count}).")
            return len(records)
        finally:
            conn.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Drake Track Audio Features Generator Agent")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size for Gemini requests (default: 20)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tracks to process (default: all)")
    parser.add_argument("--force", action="store_true", help="Force re-generation of features ignoring cache")
    parser.add_argument("--seed-only", action="store_true", help="Seed existing JSON to database without generating")
    parser.add_argument("--no-seed", action="store_true", help="Skip PostgreSQL database seeding")
    parser.add_argument("--db-url", type=str, default=DEFAULT_DB_URL, help="PostgreSQL connection URL")
    return parser.parse_args()


def main():
    args = parse_args()

    agent = AudioFeaturesAgent(
        batch_size=args.batch_size,
        db_url=args.db_url,
    )

    if not args.seed_only:
        agent.generate(limit=args.limit, force=args.force)

    if not args.no_seed:
        agent.seed_postgres()


if __name__ == "__main__":
    main()
