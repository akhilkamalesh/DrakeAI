# Musixmatch API scraper

This document details the requirements and technical specifications for the Musixmatch extraction scraper. The scraper implements **Phase 1: Batch API Extraction** as described in the [PRD](../../documentation/drake_rag_agent_prd.md).

## 1. Objectives
* **Extraction** Query the musixmatch track.lyrics.get api using the id of the track in tracks.json to download the lyric for each track
* **Pagination:** Gracefully handle paginated results for albums and tracks.
* **Resilience:** Handle rate limits (HTTP 429) using response `Retry-After` headers and connection retry backoffs.
* **Staging:** Cache the raw API responses into `data/raw_musixmatch_lyrics.json`.
* **Clean Data Output:** Attach the lyric found for each track to `data/tracks.json` in the lyrics field (use spotify_id to map)

## 2. API Endpoints

### 2.1 Authentication
* **Description** Add apikey to all requests
* **Endpoint** GET apikey=YOUR_API_KEY
* **Query Parameters:**
  - `apikey`: Your API key

### 2.2 Track Lyrics
* **Endpoint:** https://api.musixmatch.com/ws/1.1/track.lyrics.get?apikey=YOUR_API_KEY
* **Query Parameters:** 
  - `track_spotify_id`: The spotify ID of the track to get
  - `apikey`: Your API key

## 3. Data Transformations & DB Schema Alignment

The output schema must match the PostgreSQL database tables:
Map each lyric for spotify_id back to:
- track.lyrics

## 4. Operational Requirements

### 4.1 Rate Limit Handling (HTTP 429) & Connection Retries
The script must check response headers for status `429` and connection reset exceptions:
1. Extract the `Retry-After` header value (number of seconds to wait).
2. Pause execution (`time.sleep`) for the specified duration.
3. Retry the failed API call with exponential backoff.
4. There are only 500 requests per day, so it is imperative that we get all the lyrics for each drake song in the initial pull (since there less than 500 songs)

### 4.2 Logging & Deduplication
* Print real-time execution logs (e.g. `Processing spotify_id {spotify-id} 12/80: ...`).

### 4.3 Output Files
* Raw response payload written to `data/raw_musix_match_catalog.json`.
* Appended results based on spotify_id to `data/tracks.json`.
