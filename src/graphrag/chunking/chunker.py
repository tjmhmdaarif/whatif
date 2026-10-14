"""Sentence-aware chunking with overlap."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..config import get_config
from ..logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class Chunk:
    """A text chunk with metadata."""
    chunk_id: str
    doc_id: str
    text: str
    position: int  # Position in the original document
    start_char: int = 0
    end_char: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.text)


class SentenceChunker:
    """Sentence-aware chunker with configurable overlap."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        min_chunk_size: int | None = None,
    ):
        config = get_config()
        self.chunk_size = chunk_size or config.chunking.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunking.chunk_overlap
        self.min_chunk_size = min_chunk_size or config.chunking.min_chunk_size

        # Sentence splitting regex
        self.sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

    def split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Handle common abbreviations to avoid over-splitting
        text = re.sub(r'\b(e\.g|i\.e|etc|Dr|Mr|Mrs|Ms|Prof|Sr|Jr)\.', r'\1<DOT>', text)
        sentences = self.sentence_endings.split(text)
        sentences = [s.replace('<DOT>', '.').strip() for s in sentences if s.strip()]
        return sentences

    def chunk_document(self, doc_id: str, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Chunk a document into overlapping chunks."""
        sentences = self.split_sentences(text)
        if not sentences:
            return []

        chunks = []
        current_chunk_sentences: list[str] = []
        current_length = 0
        position = 0
        char_offset = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # If adding this sentence would exceed chunk_size, finalize current chunk
            if current_length + sentence_length > self.chunk_size and current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences)
                chunk = Chunk(
                    chunk_id=f"{doc_id}_chunk_{position}",
                    doc_id=doc_id,
                    text=chunk_text,
                    position=position,
                    start_char=char_offset - current_length,
                    end_char=char_offset,
                    metadata=metadata or {},
                )
                chunks.append(chunk)
                position += 1

                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk_sentences)
                current_chunk_sentences = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk_sentences)
            else:
                current_chunk_sentences.append(sentence)
                current_length += sentence_length

            char_offset += sentence_length + 1  # +1 for space

        # Add final chunk if it meets minimum size
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            if len(chunk_text) >= self.min_chunk_size:
                chunk = Chunk(
                    chunk_id=f"{doc_id}_chunk_{position}",
                    doc_id=doc_id,
                    text=chunk_text,
                    position=position,
                    start_char=char_offset - current_length,
                    end_char=char_offset,
                    metadata=metadata or {},
                )
                chunks.append(chunk)

        logger.debug(f"Document {doc_id}: {len(chunks)} chunks created")
        return chunks

    def _get_overlap_sentences(self, sentences: list[str]) -> list[str]:
        """Get sentences for overlap from the end of the previous chunk."""
        if not sentences:
            return []

        overlap_text = ""
        overlap_sentences = []

        # Add sentences from the end until we reach overlap size
        for sentence in reversed(sentences):
            if len(overlap_text) + len(sentence) <= self.chunk_overlap:
                overlap_sentences.insert(0, sentence)
                overlap_text = sentence + " " + overlap_text
            else:
                break

        return overlap_sentences

    def chunk_corpus(self, corpus: Any) -> list[Chunk]:
        """Chunk all documents in a corpus."""
        all_chunks = []
        for doc in corpus.documents:
            chunks = self.chunk_document(doc.doc_id, doc.text, doc.metadata)
            all_chunks.extend(chunks)
        logger.info(f"Created {len(all_chunks)} chunks from {len(corpus.documents)} documents")
        return all_chunks


def create_chunker() -> SentenceChunker:
    """Factory function to create a chunker from config."""
    return SentenceChunker()