import os
import sys
import time
import json
import argparse
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv

# Base paths
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(WORKSPACE_DIR, ".env")
DATA_DIR = os.path.join(WORKSPACE_DIR, "data")
TRACKS_PATH = os.path.join(DATA_DIR, "tracks.json")
RAW_LYRICS_PATH = os.path.join(DATA_DIR, "raw_musixmatch_lyrics.json")
RAW_CATALOG_PATH = os.path.join(DATA_DIR, "raw_musix_match_catalog.json")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Load environment variables
load_dotenv(ENV_PATH)
MUSIXMATCH_API_KEY = os.getenv("MUSIXMATCH_API_KEY")

MUSIXMATCH_LYRICS_ENDPOINT = "https://api.musixmatch.com/ws/1.1/track.lyrics.get"

# Mock lyrics templates for simulated mode
MOCK_LYRIC_TEMPLATES = [
    (
        "Yeah, look\n"
        "I’ve been thinking 'bout the things that could’ve been\n"
        "Late nights in Toronto, talking to my closest kin\n"
        "Started with a dream, now we running through the city\n"
        "Every time I win, they act like it's a pity\n"
        "Champagne toasts for the ones who stayed true\n"
        "I do this for the six and everything we grew"
    ),
    (
        "Started from the bottom, look at how we living now\n"
        "They doubted every move, but we made them take a bow\n"
        "Cold winter nights, 40 got the beats right\n"
        "Working through the morning, turning off the street light\n"
        "Keeping my circle small, loyalty over everything\n"
        "Living out the legacy that destiny will bring"
    ),
    (
        "Ring ring on the cellular, heard you need a minute\n"
        "Talking 'bout the past and how we used to kick it\n"
        "Memories fade but the feelings always linger\n"
        "Diamonds on my wrist, another ring upon my finger\n"
        "Tryna find the balance in the middle of the noise\n"
        "Representing OVO, shout out to the boys"
    )
]


class QuotaExceededException(Exception):
    """Raised when Musixmatch API quota limit (status 402 / daily cap) is hit."""
    pass


class MusixmatchScraper:
    def __init__(self, api_key, delay=0.25, force=False):
        self.api_key = api_key
        self.delay = delay
        self.force = force
        self.cache = self.load_cache()
        self.session = requests.Session()
        self.api_calls_count = 0

    def load_cache(self):
        """Loads cached raw responses from disk to prevent redundant API queries."""
        for path in [RAW_LYRICS_PATH, RAW_CATALOG_PATH]:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, dict) and "tracks" in data:
                            print(f"[Cache] Loaded {len(data['tracks'])} cached responses from {os.path.basename(path)}.")
                            return data["tracks"]
                        elif isinstance(data, dict):
                            print(f"[Cache] Loaded {len(data)} cached responses from {os.path.basename(path)}.")
                            return data
                except Exception as e:
                    print(f"[Warning] Could not parse cache at {path}: {e}")
        return {}

    def save_cache(self):
        """Persists raw response cache to both specified raw output destinations."""
        payload = {
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "total_cached": len(self.cache),
            "tracks": self.cache
        }
        for path in [RAW_LYRICS_PATH, RAW_CATALOG_PATH]:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"[Warning] Failed to write cache to {path}: {e}")

    def get_lyrics(self, spotify_id, track_name="Unknown Track"):
        """
        Fetches lyrics for a given Spotify track ID.
        Checks cache first, then queries Musixmatch API with exponential backoff & rate-limit handling.
        """
        # Check cache unless force is specified
        if not self.force and spotify_id in self.cache:
            cached = self.cache[spotify_id]
            if isinstance(cached, dict):
                body = cached.get("message", {}).get("body", {})
                if isinstance(body, dict) and "lyrics" in body:
                    lyrics_body = body.get("lyrics", {}).get("lyrics_body")
                    return lyrics_body, True  # (lyrics, from_cache)
                elif cached.get("not_found"):
                    return None, True
            elif cached is None:
                return None, True

        if not self.api_key:
            raise ValueError("MUSIXMATCH_API_KEY is not set in environment or .env file.")

        url = MUSIXMATCH_LYRICS_ENDPOINT
        params = {
            "apikey": self.api_key,
            "track_spotify_id": spotify_id
        }

        max_retries = 5
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                self.api_calls_count += 1
                response = self.session.get(url, params=params, timeout=15)

                # HTTP 429: Rate Limit
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    sleep_seconds = int(retry_after) if (retry_after and retry_after.isdigit()) else retry_delay
                    print(f"\n[Rate Limit 429] Received HTTP 429. Sleeping for {sleep_seconds}s before retry...")
                    time.sleep(sleep_seconds)
                    retry_delay = min(retry_delay * 2, 60)
                    continue

                # Server errors (5xx)
                if response.status_code >= 500:
                    print(f"\n[Server Error {response.status_code}] Retrying in {retry_delay}s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 30)
                    continue

                # Bad response
                if response.status_code != 200:
                    print(f"\n[HTTP {response.status_code}] Unexpected status for track {spotify_id}: {response.text}")
                    response.raise_for_status()

                data = response.json()
                header = data.get("message", {}).get("header", {})
                status_code = header.get("status_code")

                # Successful lyrics retrieval
                if status_code == 200:
                    self.cache[spotify_id] = data
                    body = data.get("message", {}).get("body", {})
                    lyrics = body.get("lyrics", {}).get("lyrics_body") if isinstance(body, dict) else None
                    if self.delay > 0:
                        time.sleep(self.delay)
                    return lyrics, False

                # 404: Lyrics Not Found
                elif status_code == 404:
                    self.cache[spotify_id] = {"not_found": True, "message": data.get("message")}
                    if self.delay > 0:
                        time.sleep(self.delay)
                    return None, False

                # 402: Usage limit / Daily cap reached
                elif status_code in (402, 429):
                    raise QuotaExceededException(
                        f"Musixmatch daily quota limit or burst rate exceeded (status_code {status_code})."
                    )

                # 401 or 403: Auth error
                elif status_code in (401, 403):
                    raise PermissionError(
                        f"Musixmatch API authentication failure (status_code {status_code}). Please verify your MUSIXMATCH_API_KEY."
                    )

                else:
                    print(f"\n[Warning] Musixmatch returned status_code {status_code}: {header}")
                    self.cache[spotify_id] = {"error": status_code, "message": data.get("message")}
                    return None, False

            except requests.exceptions.RequestException as req_err:
                print(f"\n[Connection Issue] {req_err}. Retrying in {retry_delay}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 30)

        print(f"\n[Error] Failed to fetch lyrics for spotify_id {spotify_id} after {max_retries} attempts.")
        return None, False


def run_mock_scrape(tracks, limit=None):
    """Simulates Musixmatch scraping for testing without burning API requests."""
    print("[Simulate] Running scraper in MOCK mode (no API calls used)...")
    total = len(tracks) if limit is None else min(limit, len(tracks))
    found_count = 0

    for idx, track in enumerate(tracks[:total], 1):
        spotify_id = track.get("id")
        name = track.get("name", "Unknown")
        print(f"\r[Lyrics] Processing spotify_id {spotify_id} ({idx}/{total}): {name[:40]:<40}...", end="", flush=True)

        # Assign deterministic mock lyrics based on track id hash
        mock_idx = abs(hash(spotify_id)) % len(MOCK_LYRIC_TEMPLATES)
        lyrics = MOCK_LYRIC_TEMPLATES[mock_idx]
        track["lyrics"] = lyrics
        found_count += 1
        time.sleep(0.02)

    print(f"\n[Simulate] Completed mock extraction. Lyrics populated for {found_count}/{total} tracks.")
    return tracks


def main():
    parser = argparse.ArgumentParser(description="Musixmatch Lyric Scraper for Drake Discography")
    parser.add_argument("--mock", action="store_true", help="Run in mock/simulation mode without calling Musixmatch API")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tracks to process (e.g. for testing)")
    parser.add_argument("--force", action="store_true", help="Force re-fetching lyrics even if cached")
    parser.add_argument("--delay", type=float, default=0.25, help="Delay in seconds between API requests (default 0.25)")
    parser.add_argument("--dry-run", action="store_true", help="Fetch lyrics but do not save changes to tracks.json")
    args = parser.parse_args()

    print("=== Drake Musixmatch Lyrics Scraper ===")
    print(f"Target tracks file: {TRACKS_PATH}")
    print(f"Staging cache file: {RAW_LYRICS_PATH}")

    # Check tracks.json exists
    if not os.path.exists(TRACKS_PATH):
        print(f"[Fatal Error] {TRACKS_PATH} not found. Please run spotify_scraper.py first.")
        sys.exit(1)

    with open(TRACKS_PATH, "r", encoding="utf-8") as f:
        tracks = json.load(f)

    total_catalog = len(tracks)
    print(f"Loaded {total_catalog} tracks from catalog.")

    # In mock mode
    if args.mock:
        tracks = run_mock_scrape(tracks, limit=args.limit)
        if not args.dry_run:
            with open(TRACKS_PATH, "w", encoding="utf-8") as f:
                json.dump(tracks, f, indent=4, ensure_ascii=False)
            print(f"[Export] Saved mock lyrics to {TRACKS_PATH}.")
        return

    # In live API mode
    if not MUSIXMATCH_API_KEY:
        print("[Fatal Error] MUSIXMATCH_API_KEY is not defined in .env or environment.")
        print("Please configure MUSIXMATCH_API_KEY in .env or pass --mock to run in simulation mode.")
        sys.exit(1)

    scraper = MusixmatchScraper(
        api_key=MUSIXMATCH_API_KEY,
        delay=args.delay,
        force=args.force
    )

    total_to_process = total_catalog if args.limit is None else min(args.limit, total_catalog)
    print(f"Starting lyrics extraction for {total_to_process} tracks (Rate delay: {args.delay}s)...")
    print(f"Daily quota allowance: 500 requests (Drake catalog: {total_catalog} tracks)")

    success_count = 0
    not_found_count = 0
    cached_count = 0
    api_calls = 0

    try:
        for idx in range(total_to_process):
            track = tracks[idx]
            spotify_id = track.get("id")
            track_name = track.get("name", "Unknown")

            # Check if track already has lyrics in tracks.json and not forcing
            if not args.force and track.get("lyrics"):
                cached_count += 1
                success_count += 1
                print(f"[Lyrics] Track {idx+1}/{total_to_process}: {spotify_id} - {track_name[:35]} -> Already present in tracks.json")
                continue

            lyrics, from_cache = scraper.get_lyrics(spotify_id, track_name)

            if from_cache:
                cached_count += 1
                status_str = "Found (Cached)" if lyrics else "Not Found (Cached)"
            else:
                api_calls += 1
                status_str = "Found (API)" if lyrics else "Not Found (API)"

            if lyrics:
                track["lyrics"] = lyrics
                success_count += 1
            else:
                track["lyrics"] = None
                not_found_count += 1

            print(f"[Lyrics] Processing spotify_id {spotify_id} ({idx+1}/{total_to_process}): {track_name[:35]:<35} -> {status_str}")

            # Incremental checkpoint every 10 tracks or at end
            if (idx + 1) % 10 == 0:
                scraper.save_cache()
                if not args.dry_run:
                    with open(TRACKS_PATH, "w", encoding="utf-8") as f:
                        json.dump(tracks, f, indent=4, ensure_ascii=False)

    except QuotaExceededException as qe:
        print(f"\n[Quota Alert] {qe}")
        print("Checkpointing current progress and raw cache before stopping...")
    except KeyboardInterrupt:
        print("\n[Interrupt] Execution cancelled by user. Saving progress...")
    except Exception as e:
        print(f"\n[Error] Unexpected error during scraping: {e}")
        print("Saving progress before exit...")
    finally:
        # Final save of raw cache and clean data
        scraper.save_cache()
        if not args.dry_run:
            with open(TRACKS_PATH, "w", encoding="utf-8") as f:
                json.dump(tracks, f, indent=4, ensure_ascii=False)
            print(f"[Export] Saved updated tracks with lyrics to {TRACKS_PATH}.")
        else:
            print("[Dry Run] Skipped updating tracks.json.")

    print("\n=== Musixmatch Scraper Summary ===")
    print(f"Total Processed: {idx + 1 if 'idx' in locals() else 0}/{total_to_process}")
    print(f"Lyrics Found: {success_count}")
    print(f"Lyrics Not Found / Instrumental: {not_found_count}")
    print(f"Loaded from Cache: {cached_count}")
    print(f"Live API Requests Used: {scraper.api_calls_count}")
    print(f"Raw cache saved to: {RAW_LYRICS_PATH} and {RAW_CATALOG_PATH}")


if __name__ == "__main__":
    main()
