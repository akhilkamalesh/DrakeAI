#!/usr/bin/env python3
"""
Stanza Creation Script (Phase 2: Structural Chunking & Embedding)

Fulfills product specification: product_specs/stanza_creation.md
Extracts lyrics from data/tracks.json, splits lyrics into cohesive stanzas
of 5-7 lines each, generates 1024-dimensional vector embeddings,
classifies personal feel using a local model, and outputs to data/stanza.json.
"""

import os
import sys
import json
import time
import re
import argparse
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional

import numpy as np
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Workspace paths
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(WORKSPACE_DIR, ".env"))
DATA_DIR = os.path.join(WORKSPACE_DIR, "data")
DEFAULT_TRACKS_PATH = os.path.join(DATA_DIR, "tracks.json")
DEFAULT_OUTPUT_PATH = os.path.join(DATA_DIR, "stanza.json")

# Categories defined in product_specs/stanza_creation.md
CATEGORIES = [
    "Late-Night Confessional",
    "Triumphant Flex",
    "Paranoid & Guarded",
    "Time-Stamp Introspection",
    "Toxic & Petty",
    "Global Groove / Island Infusion",
    "Pop Crossover / Radio R&B",
    "Hard-Hitting / Mob Tie",
    "Crew Loyalty & Brotherhood",
    "The Club Anthem",
]

CATEGORY_KEYWORDS = {
    "Late-Night Confessional": [
        "moody", "vulnerable", "regret", "longing", "miss", "late night", "crying",
        "alone", "heartbroken", "memories", "jungle", "marvins room", "feelings"
    ],
    "Triumphant Flex": [
        "trophy", "flex", "win", "rich", "money", "bottom", "started", "dominance",
        "boss", "billion", "plaque", "energy", "champions"
    ],
    "Paranoid & Guarded": [
        "fake", "snakes", "industry", "guarded", "trust", "paranoid", "enemies",
        "threat", "watching", "friends", "jealousy", "careful"
    ],
    "Time-Stamp Introspection": [
        "am", "pm", "toronto", "calabasas", "introspection", "career", "reflect",
        "journey", "trajectory", "story", "thoughts", "clock"
    ],
    "Toxic & Petty": [
        "toxic", "petty", "ex", "bitter", "vindictive", "fault", "jealous",
        "text", "phone", "curved", "options", "revenge", "lies"
    ],
    "Global Groove / Island Infusion": [
        "dance", "island", "afrobeats", "caribbean", "jamaica", "whine", "summer",
        "tempo", "rhythm", "blem", "controlla", "tropical", "sun"
    ],
    "Pop Crossover / Radio R&B": [
        "melodic", "sing", "hook", "radio", "pop", "hold on", "feel", "baby",
        "love", "sweet", "in my feelings", "passion"
    ],
    "Hard-Hitting / Mob Tie": [
        "mob", "ties", "knife", "street", "trap", "shoot", "gang", "demon",
        "heavy", "trigger", "war", "opps"
    ],
    "Crew Loyalty & Brotherhood": [
        "ovo", "crew", "brother", "brothers", "loyal", "loyalty", "fam", "team",
        "circle", "god's plan", "unison", "solid"
    ],
    "The Club Anthem": [
        "club", "anthem", "hype", "party", "vip", "bass", "shots", "turnt",
        "wild", "sexy", "flex", "drink", "bottles"
    ]
}


def partition_lines(n: int) -> List[int]:
    """
    Partitions n lines into chunks of 5-7 lines each (targeting 6 lines).
    For n < 10, handles edge cases gracefully:
      - n <= 7: single chunk of n lines
      - n == 8: [4, 4]
      - n == 9: [4, 5]
    For n >= 10: strictly uses chunk sizes from {5, 6, 7} minimizing deviation from 6.
    """
    if n <= 7:
        return [n]
    if n == 8:
        return [4, 4]
    if n == 9:
        return [4, 5]

    dp: Dict[int, List[int]] = {0: []}
    for i in range(1, n + 1):
        best = None
        for size in (6, 5, 7):
            prev = i - size
            if prev in dp:
                candidate = dp[prev] + [size]
                cost = sum(abs(s - 6) for s in candidate)
                if best is None or cost < sum(abs(s - 6) for s in best):
                    best = candidate
        if best is not None:
            dp[i] = best

    return dp.get(n, [n])


def chunk_lyrics(lyrics: str) -> List[str]:
    """
    Splits lyric text into cohesive stanzas of 5-7 lines.
    Preserves clean line formatting within each chunk.
    """
    if not lyrics or not lyrics.strip():
        return []

    raw_lines = [line.strip() for line in lyrics.split("\n") if line.strip()]
    if not raw_lines:
        return []

    partitions = partition_lines(len(raw_lines))
    chunks = []
    idx = 0
    for size in partitions:
        chunk_lines = raw_lines[idx : idx + size]
        chunks.append("\n".join(chunk_lines))
        idx += size

    return chunks


class StanzaEmbedder:
    """
    Generates 1024-dimensional dense vector embeddings using Sentence Transformers.
    Pads 'all-MiniLM-L6-v2' (384-d) with zeros to fulfill PostgreSQL vector(1024) schema.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", target_dim: int = 1024):
        self.model_name = model_name
        self.target_dim = target_dim
        print(f"[Embedder] Loading SentenceTransformer '{model_name}'...")
        self.model = SentenceTransformer(model_name)
        print(f"[Embedder] Model '{model_name}' loaded successfully.")

    def embed_batch(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        if not texts:
            return []

        raw_embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        padded_embeddings = []
        for emb in raw_embeddings:
            curr_dim = len(emb)
            if curr_dim < self.target_dim:
                padded = np.pad(emb, (0, self.target_dim - curr_dim), mode="constant")
            else:
                padded = emb[: self.target_dim]
            # Round floats to 6 decimal places to conserve space while maintaining accuracy
            padded_embeddings.append([round(float(x), 6) for x in padded])

        return padded_embeddings


class PersonalFeelClassifier:
    """
    Classifies Drake lyric stanzas into one of the 10 subjective categories
    using Google Gemini, local Ollama, or rule-based heuristics as fallback.
    """

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        gemini_model: Optional[str] = None,
        model_name: str = "gemma3:270m",
        ollama_url: str = "http://localhost:11434/api/generate",
        enabled: bool = True
    ):
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.gemini_model = gemini_model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.enabled = enabled
        self.cache: Dict[str, str] = {}
        self.provider = "heuristic"

        if self.enabled:
            if self.gemini_api_key:
                self.provider = "gemini"
                print(f"[Classifier] Google Gemini active with model '{self.gemini_model}'.")
            else:
                self._check_ollama_availability()

    def _check_ollama_availability(self):
        try:
            res = requests.post(
                self.ollama_url,
                json={
                    "model": self.model_name,
                    "prompt": "Test",
                    "stream": False,
                    "options": {"num_predict": 1},
                },
                timeout=3,
            )
            if res.status_code == 200:
                self.provider = "ollama"
                print(f"[Classifier] Ollama model '{self.model_name}' is online and active.")
            else:
                print(f"[Classifier] Ollama returned status {res.status_code}. Using heuristic fallback.")
        except Exception as e:
            print(f"[Classifier] Ollama connection unavailable ({e}). Using heuristic fallback.")

    def _normalize_category(self, raw_text: str) -> Optional[str]:
        raw_clean = raw_text.strip().strip('"\'*`').lower()

        # Exact match check
        for cat in CATEGORIES:
            if cat.lower() == raw_clean:
                return cat

        # Substring / partial match check
        for cat in CATEGORIES:
            if cat.lower() in raw_clean:
                return cat

        return None

    def _heuristic_classify(self, text: str) -> str:
        text_lower = text.lower()
        scores: Dict[str, int] = {cat: 0 for cat in CATEGORIES}

        # Keywords for category heuristic
        keywords = {
            "Late-Night Confessional": ["night", "call", "miss", "regret", "lonely", "alone", "3am", "4am", "drunk", "drinking"],
            "Triumphant Flex": ["top", "rich", "money", "boss", "champagne", "win", "winning", "billions", "first", "record"],
            "Paranoid & Guarded": ["fake", "snakes", "watch", "trust", "nobody", "opp", "enemy", "jealous", "envy", "lies"],
            "Time-Stamp Introspection": ["pm", "am", "toronto", "calabasas", "houston", "paris", "years", "remember", "time"],
            "Toxic & Petty": ["block", "ex", "side", "text", "bitch", "cheating", "heartless", "games", "never", "used to"],
            "Global Groove / Island Infusion": ["dance", "ting", "gyal", "rhythm", "afro", "vibes", "caribbean", "party", "tropical"],
            "Pop Crossover / Radio R&B": ["love", "baby", "feel", "hold", "dance", "touch", "song", "kiss", "heart"],
            "Hard-Hitting / Mob Tie": ["shoot", "gang", "mob", "gun", "drill", "slide", "war", "crews", "kill"],
            "Crew Loyalty & Brotherhood": ["ovo", "team", "crew", "brothers", "loyal", "family", "day one", "brotherhood"],
            "The Club Anthem": ["club", "dj", "shots", "bottles", "vip", "weekend", "lights", "bass", "crowd"]
        }

        for cat, words in keywords.items():
            for w in words:
                if w in text_lower:
                    scores[cat] += 1

        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "Late-Night Confessional"

    def classify(self, text: str) -> str:
        normalized_key = " ".join(text.strip().split())
        if normalized_key in self.cache:
            return self.cache[normalized_key]

        category = None
        prompt = (
            f"You are a music analyst classifying this Drake lyric stanza into EXACTLY ONE of these categories:\n"
            f"{', '.join(CATEGORIES)}\n\n"
            f"Lyric stanza:\n\"\"\"{text}\"\"\"\n\n"
            f"Respond with ONLY the exact category name."
        )

        if self.enabled and self.provider == "gemini":
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent?key={self.gemini_api_key}"
                res = requests.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json={
                        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                        "generationConfig": {"temperature": 0.0}
                    },
                    timeout=5
                )
                if res.status_code == 200:
                    data = res.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        resp_text = "".join(p.get("text", "") for p in parts)
                        category = self._normalize_category(resp_text)
            except Exception:
                pass

        elif self.enabled and self.provider == "ollama":
            try:
                res = requests.post(
                    self.ollama_url,
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.0, "num_predict": 20},
                    },
                    timeout=5,
                )
                if res.status_code == 200:
                    resp_text = res.json().get("response", "")
                    category = self._normalize_category(resp_text)
            except Exception:
                pass

        if not category:
            category = self._heuristic_classify(text)

        self.cache[normalized_key] = category
        return category

    def classify_single(self, text: str) -> str:
        return self.classify(text)

    def classify_batch(self, texts: List[str], max_workers: int = 4) -> List[str]:
        if not texts:
            return []
        if not self.enabled or self.provider == "heuristic":
            return [self.classify_single(t) for t in texts]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self.classify_single, texts))


class StanzaPipeline:
    """
    Coordinates chunking, embedding generation, classification,
    and output storage for all tracks in tracks.json.
    """

    def __init__(
        self,
        tracks_path: str = DEFAULT_TRACKS_PATH,
        output_path: str = DEFAULT_OUTPUT_PATH,
        embedding_model: str = "all-MiniLM-L6-v2",
        ollama_model: str = "gemma3:270m",
        batch_size: int = 64,
        workers: int = 4,
        skip_classification: bool = False,
        force: bool = False,
    ):
        self.tracks_path = tracks_path
        self.output_path = output_path
        self.batch_size = batch_size
        self.workers = workers
        self.force = force

        self.embedder = StanzaEmbedder(embedding_model)
        self.classifier = PersonalFeelClassifier(
            model_name=ollama_model,
            enabled=not skip_classification
        )

    def run(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        if not os.path.exists(self.tracks_path):
            raise FileNotFoundError(f"Tracks catalog not found at {self.tracks_path}")

        with open(self.tracks_path, "r", encoding="utf-8") as f:
            tracks = json.load(f)

        tracks_with_lyrics = [t for t in tracks if t.get("lyrics") and t["lyrics"].strip()]
        total_available = len(tracks_with_lyrics)

        if limit is not None and limit > 0:
            tracks_to_process = tracks_with_lyrics[:limit]
        else:
            tracks_to_process = tracks_with_lyrics

        print(f"\n=======================================================")
        print(f"Starting Stanza Creation Pipeline")
        print(f"Tracks with lyrics: {total_available}")
        print(f"Tracks to process:  {len(tracks_to_process)}")
        print(f"Output destination: {self.output_path}")
        print(f"=======================================================\n")

        all_stanzas: List[Dict[str, Any]] = []
        global_stanza_id = 1
        start_time = time.time()

        for idx, track in enumerate(tracks_to_process, start=1):
            spotify_id = track.get("id", "unknown")
            song_name = track.get("name", "Unknown Track")

            # 1. Chunk lyrics
            chunks = chunk_lyrics(track["lyrics"])
            num_chunks = len(chunks)

            # Log execution in real-time per spec Section 5.1
            print(
                f"Processing spotify_id {spotify_id} {idx}/{len(tracks_to_process)}: "
                f"'{song_name}' ({num_chunks} stanzas)"
            )

            if not chunks:
                continue

            # 2. Generate vector embeddings in batch
            embeddings = self.embedder.embed_batch(chunks, batch_size=self.batch_size)

            # 3. Classify personal feel in batch
            personal_feels = self.classifier.classify_batch(chunks, max_workers=self.workers)

            # 4. Assemble stanza records matching PostgreSQL schema
            for chunk_idx, (chunk, emb, feel) in enumerate(zip(chunks, embeddings, personal_feels)):
                stanza_record = {
                    "id": global_stanza_id,
                    "song_id": spotify_id,
                    "lyric_chunk": chunk,
                    "chunk_index": chunk_idx,
                    "embedding": emb,
                    "personal_feel": feel,
                }
                all_stanzas.append(stanza_record)
                global_stanza_id += 1

            # Checkpoint save every 25 tracks
            if idx % 25 == 0 or idx == len(tracks_to_process):
                self._save_output(all_stanzas)

        elapsed = time.time() - start_time
        print(f"\nPipeline finished in {elapsed:.2f}s.")
        print(f"Total stanzas generated: {len(all_stanzas)}")
        print(f"Output saved to: {self.output_path}\n")

        return all_stanzas

    def _save_output(self, stanzas: List[Dict[str, Any]]):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        temp_path = self.output_path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(stanzas, f, indent=2, ensure_ascii=False)
        os.replace(temp_path, self.output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Create stanzas and embeddings from Drake lyrics.")
    parser.add_argument("--tracks", type=str, default=DEFAULT_TRACKS_PATH, help="Path to tracks.json")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH, help="Path to output stanza.json")
    parser.add_argument("--limit", type=int, default=None, help="Process first N tracks (useful for testing)")
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model")
    parser.add_argument("--ollama-model", type=str, default="gemma3:270m", help="Local Ollama model name")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for embedding generation")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent workers for classification")
    parser.add_argument("--skip-classification", action="store_true", help="Skip LLM classification")
    parser.add_argument("--force", action="store_true", help="Overwrite existing stanza.json")
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline = StanzaPipeline(
        tracks_path=args.tracks,
        output_path=args.output,
        embedding_model=args.embedding_model,
        ollama_model=args.ollama_model,
        batch_size=args.batch_size,
        workers=args.workers,
        skip_classification=args.skip_classification,
        force=args.force,
    )
    pipeline.run(limit=args.limit)


if __name__ == "__main__":
    main()
