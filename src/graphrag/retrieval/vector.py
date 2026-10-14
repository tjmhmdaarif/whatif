"""Vector store interface and FAISS implementation."""

from __future__ import annotations

import abc
import os
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from ..chunking import Chunk
from ..config import get_config
from ..logging_setup import get_logger

logger = get_logger(__name__)


class VectorStore(abc.ABC):
    """Abstract base class for vector stores."""

    @abc.abstractmethod
    def add_chunks(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Add chunks and their embeddings to the store."""
        pass

    @abc.abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[Chunk, float]]:
        """Search for similar chunks."""
        pass

    @abc.abstractmethod
    def save(self, path: str | Path) -> None:
        """Save the index to disk."""
        pass

    @abc.abstractmethod
    def load(self, path: str | Path) -> None:
        """Load the index from disk."""
        pass

    @abc.abstractmethod
    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Get a chunk by ID."""
        pass


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store."""

    def __init__(self, index_path: str | Path | None = None, index_type: str = "Flat"):
        config = get_config()
        self.index_path = Path(index_path or config.vector_store.faiss.index_path)
        self.index_type = index_type or config.vector_store.faiss.index_type

        self.index: faiss.Index | None = None
        self.chunks: dict[str, Chunk] = {}
        self.chunk_ids: list[str] = []
        self.dimension: int | None = None

    def add_chunks(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Add chunks and their embeddings to the FAISS index."""
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        if self.index is None:
            self.dimension = embeddings.shape[1]
            self.index = faiss.index_factory(self.dimension, self.index_type)
            logger.info(f"Created FAISS index: {self.index_type} with dimension {self.dimension}")

        # Normalize embeddings for inner product similarity
        faiss.normalize_L2(embeddings)

        # Add to index
        self.index.add(embeddings.astype(np.float32))

        # Store chunks
        for chunk, emb in zip(chunks, embeddings):
            self.chunks[chunk.chunk_id] = chunk
            self.chunk_ids.append(chunk.chunk_id)

        logger.info(f"Added {len(chunks)} chunks to FAISS index (total: {self.index.ntotal})")

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[Chunk, float]]:
        """Search for similar chunks using FAISS."""
        if self.index is None or self.index.ntotal == 0:
            return []

        # Normalize query
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunk_ids):
                chunk_id = self.chunk_ids[idx]
                chunk = self.chunks.get(chunk_id)
                if chunk:
                    results.append((chunk, float(score)))

        return results

    def save(self, path: str | Path | None = None) -> None:
        """Save the FAISS index and chunks to disk."""
        path = Path(path or self.index_path)
        path.mkdir(parents=True, exist_ok=True)

        if self.index is not None:
            faiss.write_index(self.index, str(path / "index.faiss"))

        # Save chunks and chunk_ids
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump({"chunks": self.chunks, "chunk_ids": self.chunk_ids}, f)

        logger.info(f"Saved FAISS index to {path}")

    def load(self, path: str | Path | None = None) -> None:
        """Load the FAISS index and chunks from disk."""
        path = Path(path or self.index_path)

        index_file = path / "index.faiss"
        chunks_file = path / "chunks.pkl"

        if not index_file.exists() or not chunks_file.exists():
            logger.warning(f"Index files not found at {path}")
            return

        self.index = faiss.read_index(str(index_file))

        with open(chunks_file, "rb") as f:
            data = pickle.load(f)
            self.chunks = data["chunks"]
            self.chunk_ids = data["chunk_ids"]

        self.dimension = self.index.d
        logger.info(f"Loaded FAISS index from {path} ({self.index.ntotal} vectors)")

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Get a chunk by ID."""
        return self.chunks.get(chunk_id)

    def __len__(self) -> int:
        return self.index.ntotal if self.index else 0


def create_vector_store() -> VectorStore:
    """Factory function to create a vector store from config."""
    config = get_config()
    if config.vector_store.type == "faiss":
        return FAISSVectorStore()
    elif config.vector_store.type == "chroma":
        from .chroma_store import ChromaVectorStore
        return ChromaVectorStore()
    else:
        raise ValueError(f"Unknown vector store type: {config.vector_store.type}")