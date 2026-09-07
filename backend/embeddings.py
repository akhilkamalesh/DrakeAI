"""SentenceTransformer embedding generator with zero-padding to 1024 dimensions."""

import logging
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from backend.config import settings

logger = logging.getLogger(__name__)

_embedder_instance: Optional["SentenceTransformerEmbedder"] = None


class SentenceTransformerEmbedder:
    """
    Generates 1024-dimensional dense vector embeddings using Sentence Transformers.
    Pads 'all-MiniLM-L6-v2' (384-d) with zeros to match PostgreSQL stanza.embedding vector(1024).
    """

    def __init__(
        self,
        model_name: str = settings.EMBEDDING_MODEL_NAME,
        target_dim: int = settings.TARGET_EMBEDDING_DIM
    ):
        self.model_name = model_name
        self.target_dim = target_dim
        logger.info("Loading SentenceTransformer model '%s'...", model_name)
        self.model = SentenceTransformer(model_name)
        logger.info("SentenceTransformer model '%s' loaded successfully.", model_name)

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
            padded_embeddings.append([round(float(x), 6) for x in padded])

        return padded_embeddings

    def embed_query(self, text: str) -> List[float]:
        results = self.embed_batch([text])
        return results[0] if results else [0.0] * self.target_dim


def get_embedder() -> SentenceTransformerEmbedder:
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = SentenceTransformerEmbedder()
    return _embedder_instance
