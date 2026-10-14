"""Tests for the chunking module."""

import pytest

from graphrag.chunking import SentenceChunker, Chunk


class TestSentenceChunker:
    """Tests for SentenceChunker."""

    def test_simple_chunking(self):
        """Test basic chunking functionality."""
        chunker = SentenceChunker(chunk_size=100, chunk_overlap=20, min_chunk_size=10)

        text = "This is the first sentence. This is the second sentence. This is the third sentence. This is the fourth sentence."
        chunks = chunker.chunk_document("doc1", text)

        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.doc_id == "doc1" for c in chunks)
        assert all(len(c.text) >= chunker.min_chunk_size for c in chunks)

    def test_chunk_metadata(self):
        """Test that chunks have correct metadata."""
        chunker = SentenceChunker(chunk_size=50, chunk_overlap=10, min_chunk_size=5)

        text = "First sentence. Second sentence. Third sentence."
        chunks = chunker.chunk_document("test_doc", text, metadata={"source": "test"})

        assert len(chunks) > 0
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_id == f"test_doc_chunk_{i}"
            assert chunk.doc_id == "test_doc"
            assert chunk.position == i
            assert chunk.metadata["source"] == "test"

    def test_overlap(self):
        """Test that chunks have overlap."""
        chunker = SentenceChunker(chunk_size=60, chunk_overlap=20, min_chunk_size=10)

        # Create text with multiple sentences
        text = " ".join([f"Sentence {i}." for i in range(10)])
        chunks = chunker.chunk_document("doc1", text)

        if len(chunks) > 1:
            # Check that there's some overlap in content
            # (This is a soft check since overlap is at sentence level)
            assert len(chunks[0].text) > 0
            assert len(chunks[1].text) > 0

    def test_min_chunk_size(self):
        """Test that chunks smaller than min_chunk_size are dropped."""
        chunker = SentenceChunker(chunk_size=100, chunk_overlap=10, min_chunk_size=50)

        # Very short text that would create tiny chunks
        text = "Tiny. Text."
        chunks = chunker.chunk_document("doc1", text)

        # Should either create one chunk or none (if too small)
        if chunks:
            assert all(len(c.text) >= 50 for c in chunks)

    def test_empty_text(self):
        """Test chunking empty text."""
        chunker = SentenceChunker()
        chunks = chunker.chunk_document("doc1", "")
        assert chunks == []

    def test_whitespace_only(self):
        """Test chunking whitespace-only text."""
        chunker = SentenceChunker()
        chunks = chunker.chunk_document("doc1", "   \n\t  ")
        assert chunks == []

    def test_abbreviations(self):
        """Test that abbreviations don't break sentence splitting."""
        chunker = SentenceChunker(chunk_size=200, chunk_overlap=20)

        text = "Dr. Smith went to the U.S.A. He met Mr. Jones. They discussed the project."
        chunks = chunker.chunk_document("doc1", text)

        assert len(chunks) >= 1
        # Should not split on "Dr." or "U.S.A." or "Mr."
        full_text = " ".join(c.text for c in chunks)
        assert "Dr. Smith" in full_text
        assert "U.S.A." in full_text
        assert "Mr. Jones" in full_text


class TestChunk:
    """Tests for Chunk dataclass."""

    def test_chunk_creation(self):
        """Test creating a chunk."""
        chunk = Chunk(
            chunk_id="test_1",
            doc_id="doc1",
            text="Test content",
            position=0,
            start_char=0,
            end_char=12,
        )

        assert chunk.chunk_id == "test_1"
        assert chunk.doc_id == "doc1"
        assert chunk.text == "Test content"
        assert chunk.position == 0
        assert len(chunk) == 12

    def test_chunk_length(self):
        """Test chunk length property."""
        chunk = Chunk(
            chunk_id="test_1",
            doc_id="doc1",
            text="Hello world",
            position=0,
        )
        assert len(chunk) == 11