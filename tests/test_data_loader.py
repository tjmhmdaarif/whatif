"""Tests for data loading."""

import pytest

from graphrag.data.loader import Document, Corpus, HotpotQAExample


class TestDocument:
    """Tests for Document dataclass."""

    def test_document_creation(self):
        doc = Document(
            doc_id="test_1",
            title="Test Title",
            text="Test content",
            metadata={"source": "test"},
        )

        assert doc.doc_id == "test_1"
        assert doc.title == "Test Title"
        assert doc.text == "Test content"
        assert doc.metadata["source"] == "test"


class TestCorpus:
    """Tests for Corpus."""

    def test_add_and_get_document(self):
        corpus = Corpus()

        doc1 = Document(doc_id="doc1", title="Title 1", text="Content 1")
        doc2 = Document(doc_id="doc2", title="Title 2", text="Content 2")

        corpus.add_document(doc1)
        corpus.add_document(doc2)

        assert len(corpus.documents) == 2
        assert corpus.get_document("doc1") == doc1
        assert corpus.get_document("doc2") == doc2
        assert corpus.get_document("nonexistent") is None

    def test_duplicate_doc_id(self):
        corpus = Corpus()

        doc1 = Document(doc_id="doc1", title="Title 1", text="Content 1")
        doc2 = Document(doc_id="doc1", title="Title 2", text="Content 2")  # Same ID

        corpus.add_document(doc1)
        corpus.add_document(doc2)

        # Current implementation adds both but get_document returns the last one
        assert len(corpus.documents) == 2
        assert corpus.get_document("doc1") == doc2


class TestHotpotQAExample:
    """Tests for HotpotQAExample model."""

    def test_example_creation(self):
        example = HotpotQAExample(
            id="test_1",
            question="What is the capital of France?",
            answer="Paris",
            type="bridge",
            level="easy",
            supporting_facts=[["France", "0"], ["Paris", "1"]],
            context=[["France", ["France is a country.", "Its capital is Paris."]]],
        )

        assert example.id == "test_1"
        assert example.question == "What is the capital of France?"
        assert example.answer == "Paris"
        assert example.type == "bridge"
        assert len(example.supporting_facts) == 2
        assert len(example.context) == 1