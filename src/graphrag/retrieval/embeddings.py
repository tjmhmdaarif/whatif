"""Embedding generation using sentence-transformers."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from ..config import get_config
from ..logging_setup import get_logger

logger = get_logger(__name__)


class Embedder:
    """Wrapper for sentence-transformers embedding model."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        normalize_embeddings: bool | None = None,
    ):
        config = get_config()
        self.model_name = model_name or config.embedding.model_name
        self.device = device or config.embedding.device
        self.batch_size = batch_size or config.embedding.batch_size
        self.normalize_embeddings = (
            normalize_embeddings if normalize_embeddings is not None
            else config.embedding.normalize_embeddings
        )

        self._model: SentenceTransformer | None = None
        self._embedding_dim: int | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            # Use new method name (sentence-transformers >= 3.0)
            self._embedding_dim = getattr(self._model, 'get_embedding_dimension', 
                                         lambda: self._model.get_sentence_embedding_dimension())()
            logger.info(f"Model loaded. Embedding dimension: {self._embedding_dim}")
        return self._model

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        if self._embedding_dim is None:
            _ = self.model  # triggers loading
        return self._embedding_dim

    def encode(
        self,
        texts: list[str],
        batch_size: int | None = None,
        show_progress_bar: bool = False,
        normalize: bool | None = None,
    ) -> np.ndarray:
        """Encode a list of texts into embeddings."""
        batch_size = batch_size or self.batch_size
        normalize = normalize if normalize is not None else self.normalize_embeddings

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )

        return embeddings

    def encode_single(self, text: str, normalize: bool | None = None) -> np.ndarray:
        """Encode a single text."""
        return self.encode([text], normalize=normalize)[0]


# Global embedder instance
_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    """Get the global embedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder