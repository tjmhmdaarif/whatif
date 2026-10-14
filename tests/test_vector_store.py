"""Tests for vector store."""

import pytest
import numpy as np

from graphrag.chunking import Chunk
from graphrag.retrieval.vector import FAISSVectorStore


class TestFAISSVectorStore:
    """Tests for FAISSVectorStore."""

    def setup_method(self):
        """Create a fresh vector store for each test."""
        self.store = FAISSVectorStore(index_path="test_faiss_index")
        self.chunks = [
            Chunk(chunk_id="chunk_1", doc_id="doc1", text="First document", position=0),
            Chunk(chunk_id="chunk_2", doc_id="doc1", text="Second document", position=1),
            Chunk(chunk_id="chunk_3", doc_id="doc2", text="Third document", position=0),
        ]
        # Create dummy embeddings (dimension 384 for bge-small-en-v1.5)
        self.embeddings = np.random.rand(3, 384).astype(np.float32)

    def teardown_method(self):
        """Clean up test files."""
        import shutil
        import os
        if os.path.exists("test_faiss_index"):
            shutil.rmtree("test_faiss_index")

    def test_add_and_search(self):
        """Test adding chunks and searching."""
        self.store.add_chunks(self.chunks, self.embeddings)

        # Search with first chunk's embedding
        results = self.store.search(self.embeddings[0:1], top_k=2)

        assert len(results) == 2
        # All returned chunks should be from our stored chunks
        assert all(r[0].chunk_id in ["chunk_1", "chunk_2", "chunk_3"] for r in results)
        # Scores should be valid floats
        assert all(isinstance(r[1], float) for r in results)

    def test_get_chunk(self):
        """Test getting chunk by ID."""
        self.store.add_chunks(self.chunks, self.embeddings)

        chunk = self.store.get_chunk("chunk_1")
        assert chunk is not None
        assert chunk.chunk_id == "chunk_1"
        assert chunk.text == "First document"

        missing = self.store.get_chunk("nonexistent")
        assert missing is None

    def test_save_and_load(self):
        """Test saving and loading the index."""
        self.store.add_chunks(self.chunks, self.embeddings)
        self.store.save("test_faiss_index")

        # Create new store and load
        new_store = FAISSVectorStore(index_path="test_faiss_index")
        new_store.load("test_faiss_index")

        assert new_store.index.ntotal == 3
        assert len(new_store.chunks) == 3
        assert new_store.get_chunk("chunk_1").text == "First document"

    def test_empty_search(self):
        """Test searching empty index."""
        results = self.store.search(np.random.rand(1, 384).astype(np.float32), top_k=5)
        assert results == []

    def test_top_k_limit(self):
        """Test that top_k limits results."""
        self.store.add_chunks(self.chunks, self.embeddings)

        results = self.store.search(self.embeddings[0:1], top_k=1)
        assert len(results) == 1

        results = self.store.search(self.embeddings[0:1], top_k=10)
        assert len(results) == 3  # Only 3 chunks in index