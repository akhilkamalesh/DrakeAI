# Spotify Scraper Specification

This document details the requirements and technical specifications for the Spotify extraction scraper. The scraper implements **Phase 1: Batch API Extraction** as described in the [PRD](../../documentation/drake_rag_agent_prd.md).

## 1. Objectives
* **Extraction:** Query the Spotify Web API to download the complete discography catalog of recording artist Drake (albums, singles, and all associated tracks).
* **Pagination:** Gracefully handle paginated results for albums and tracks.
* **Resilience:** Handle rate limits (HTTP 429) using response `Retry-After` headers and connection retry backoffs.
* **Staging:** Cache the raw API responses into `data/raw_spotify_catalog.json`.
* **Clean Data Output:** Export clean, deduplicated database-ready JSON records matching the PostgreSQL schema:
  - `data/albums.json`
  - `data/tracks.json`
* **Audio Features Note:** The Spotify `/v1/audio-features` endpoint is deprecated/restricted. Audio features extraction has been removed from this scraper. The database schema retains the `audio_feature` table structure so that an agent or audio analysis model can populate it in a subsequent phase.

## 2. API Endpoints & Parameters

### 2.1 Authentication
* **Endpoint:** `POST https://accounts.spotify.com/api/token`
* **Flow:** OAuth 2.0 Client Credentials Flow.
* **Parameters:** `grant_type=client_credentials`.
* **Headers:** `Authorization: Basic <base64(client_id:client_secret)>` or passed as request body params.

### 2.2 Artist Albums
* **Endpoint:** `GET https://api.spotify.com/v1/artists/{id}/albums`
* **Artist ID:** `3TVXtAsR1Inumwj472S9r4` (Drake)
* **Query Parameters:**
  - `include_groups=album,single` (filters to direct albums and singles, avoiding other compilations and appears_on features).

### 2.3 Album Tracks
* **Endpoint:** `GET https://api.spotify.com/v1/albums/{id}/tracks`
* **Query Parameters:**
  - `limit=50`

## 3. Data Transformations & DB Schema Alignment

The output schema must match the PostgreSQL database tables:

### 3.1 Albums
Map each album payload to:
* `id` (string, Primary Key)
* `name` (string)
* `album_type` (string, e.g. `'album'`, `'single'`)
* `images_url` (string, the URL of the first image returned in the `images` list, or null)
* `release_date` (string/date)
* `total_tracks` (integer)

### 3.2 Tracks
Map each track to:
* `id` (string, Primary Key)
* `album_id` (string, Foreign Key referencing `album.id`)
* `disc_number` (integer)
* `name` (string)
* `lyrics` (null/string - lyrics will be populated in a subsequent ingestion phase)
* `external_urls` (JSONB object, containing `{"spotify": "..."}`)

## 4. Operational Requirements

### 4.1 Rate Limit Handling (HTTP 429) & Connection Retries
The script must check response headers for status `429` and connection reset exceptions:
1. Extract the `Retry-After` header value (number of seconds to wait).
2. Pause execution (`time.sleep`) for the specified duration.
3. Retry the failed API call with exponential backoff.

### 4.2 Logging & Deduplication
* Print real-time execution logs (e.g. `Processing album 12/80: Scorpion...`).
* Dedup tracks and albums based on unique Spotify ID before exporting.

### 4.3 Output Files
* Raw response payload written to `data/raw_spotify_catalog.json`.
* Clean structured lists exported in pretty-printed JSON to `data/albums.json` and `data/tracks.json`.
