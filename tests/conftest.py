"""Shared test fixtures and configuration."""

import pytest
import os

# Set test environment variables
os.environ["ANTHROPIC_API_KEY"] = "test_key"
os.environ["NEO4J_PASSWORD"] = "test_password"


@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Setup test environment."""
    yield
    # Cleanup if needed


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    from graphrag.chunking import Chunk

    return [
        Chunk(chunk_id="chunk_1", doc_id="doc1", text="This is the first document about Paris.", position=0),
        Chunk(chunk_id="chunk_2", doc_id="doc1", text="Paris is the capital of France.", position=1),
        Chunk(chunk_id="chunk_3", doc_id="doc2", text="France is a country in Europe.", position=0),
        Chunk(chunk_id="chunk_4", doc_id="doc3", text="The Eiffel Tower is in Paris.", position=0),
    ]


@pytest.fixture
def sample_corpus():
    """Create a sample corpus for testing."""
    from graphrag.data.loader import Document, Corpus

    corpus = Corpus()
    corpus.add_document(Document(doc_id="doc1", title="Paris", text="Paris is the capital of France. It is known for the Eiffel Tower."))
    corpus.add_document(Document(doc_id="doc2", title="France", text="France is a country in Europe. Its capital is Paris."))
    corpus.add_document(Document(doc_id="doc3", title="Eiffel Tower", text="The Eiffel Tower is a famous landmark in Paris, France."))
    return corpus


@pytest.fixture
def sample_hotpotqa_examples():
    """Create sample HotpotQA examples for testing."""
    from graphrag.data.loader import HotpotQAExample

    return [
        HotpotQAExample(
            id="q1",
            question="What is the capital of France?",
            answer="Paris",
            type="bridge",
            level="easy",
            supporting_facts=[["France", "0"], ["Paris", "0"]],
            context=[
                ["France", ["France is a country in Europe.", "Its capital is Paris."]],
                ["Paris", ["Paris is the capital of France.", "It is known for the Eiffel Tower."]],
            ],
        ),
        HotpotQAExample(
            id="q2",
            question="Where is the Eiffel Tower located?",
            answer="Paris, France",
            type="bridge",
            level="easy",
            supporting_facts=[["Eiffel Tower", "0"], ["Paris", "0"]],
            context=[
                ["Eiffel Tower", ["The Eiffel Tower is a famous landmark.", "It is located in Paris, France."]],
                ["Paris", ["Paris is the capital of France.", "The Eiffel Tower is in Paris."]],
            ],
        ),
    ]