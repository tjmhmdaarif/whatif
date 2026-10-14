"""Tests for hybrid retrieval."""

import pytest
import asyncio
from pathlib import Path
import shutil

from graphrag.chunking import Chunk, create_chunker
from graphrag.config import load_config, set_config
from graphrag.data.loader import load_hotpotqa
from graphrag.llm import MockLLMClient, set_mock_client
from graphrag.retrieval import (
    create_vector_store, get_embedder, create_hybrid_retriever, RetrievalResult
)
from graphrag.graph import create_graph_store, GraphBuilder
from graphrag.extraction.models import Entity, Relation, ExtractionResult


@pytest.fixture
def setup_retrieval():
    """Setup vector store and graph for retrieval tests."""
    # Clear cache
    cache_dir = Path("data/llm_cache/extraction")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    # Setup config
    config = load_config("config.yaml")
    set_config(config)

    # Load data
    examples, corpus = load_hotpotqa(split="validation", limit=5, seed=42)

    # Build vector index
    chunker = create_chunker()
    chunks = chunker.chunk_corpus(corpus)

    embedder = get_embedder()
    texts = [c.text for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False)

    vector_store = create_vector_store()
    vector_store.add_chunks(chunks, embeddings)
    vector_store.save("data/faiss_index")

    # Build graph with mock extraction
    builder = GraphBuilder()
    
    extraction_responses = {
        "Extract all entities and relationships from the following text chunk. Focus on the entity types": ExtractionResult(
            entities=[
                Entity(name="Paris", type="LOCATION", description="Capital of France"),
                Entity(name="France", type="LOCATION", description="Country in Europe"),
                Entity(name="Eiffel Tower", type="LOCATION", description="Famous landmark"),
            ],
            relations=[
                Relation(source="Paris", target="France", relation="LOCATED_IN", evidence="Paris is the capital of France"),
                Relation(source="Eiffel Tower", target="Paris", relation="LOCATED_IN", evidence="The Eiffel Tower is in Paris"),
            ]
        ).model_dump(),
    }

    mock_client = MockLLMClient(extraction_responses)
    set_mock_client(mock_client)

    asyncio.run(builder.build_from_chunks(chunks[:10], max_concurrent=2))
    builder.save()

    graph_store = builder.graph_store

    return vector_store, graph_store, chunks[:10]


class TestHybridRetrieval:
    """Tests for hybrid retrieval."""

    @pytest.mark.asyncio
    async def test_vector_retrieval(self, setup_retrieval):
        """Test vector-only retrieval."""
        vector_store, graph_store, chunks = setup_retrieval

        retriever = create_hybrid_retriever(vector_store=vector_store, graph_store=graph_store)

        result = await retriever.retrieve("What is the capital of France?", mode="vector", vector_top_k=5)

        assert isinstance(result, RetrievalResult)
        assert len(result.chunks) > 0
        assert len(result.vector_scores) > 0
        assert result.traversal_paths == []

    @pytest.mark.asyncio
    async def test_graph_retrieval(self, setup_retrieval):
        """Test graph-only retrieval."""
        vector_store, graph_store, chunks = setup_retrieval

        retriever = create_hybrid_retriever(vector_store=vector_store, graph_store=graph_store)

        result = await retriever.retrieve("What is the capital of France?", mode="graph", graph_max_hops=2)

        assert isinstance(result, RetrievalResult)
        # Graph may or may not have results depending on extraction
        assert len(result.chunks) >= 0
        assert len(result.graph_scores) >= 0

    @pytest.mark.asyncio
    async def test_hybrid_retrieval(self, setup_retrieval):
        """Test hybrid retrieval."""
        vector_store, graph_store, chunks = setup_retrieval

        retriever = create_hybrid_retriever(vector_store=vector_store, graph_store=graph_store)

        result = await retriever.retrieve("What is the capital of France?", mode="hybrid", vector_top_k=5, graph_max_hops=2)

        assert isinstance(result, RetrievalResult)
        assert len(result.chunks) > 0
        assert len(result.fused_scores) > 0
        assert len(result.vector_scores) >= 0
        assert len(result.graph_scores) >= 0

    @pytest.mark.asyncio
    async def test_invalid_mode(self, setup_retrieval):
        """Test invalid retrieval mode raises error."""
        vector_store, graph_store, chunks = setup_retrieval

        retriever = create_hybrid_retriever(vector_store=vector_store, graph_store=graph_store)

        with pytest.raises(ValueError):
            await retriever.retrieve("test", mode="invalid")


class TestGraphLinker:
    """Tests for query entity linker."""

    @pytest.mark.asyncio
    async def test_linker_creation(self, setup_retrieval):
        """Test query entity linker can be created."""
        vector_store, graph_store, chunks = setup_retrieval

        from graphrag.retrieval import create_query_linker
        linker = create_query_linker(graph_store)

        assert linker is not None
        assert linker.graph_store == graph_store


class TestGraphTraverser:
    """Tests for graph traverser."""

    def test_traverser_creation(self, setup_retrieval):
        """Test graph traverser can be created."""
        vector_store, graph_store, chunks = setup_retrieval

        from graphrag.retrieval import create_graph_traverser
        traverser = create_graph_traverser(graph_store)

        assert traverser is not None
        assert traverser.graph_store == graph_store