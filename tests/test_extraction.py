"""Tests for entity/relation extraction and graph building."""

import pytest

from graphrag.chunking import Chunk
from graphrag.extraction import (
    Entity,
    Relation,
    ExtractionResult,
    EntityRelationExtractor,
    EntityResolver,
)
from graphrag.graph import GraphBuilder
from graphrag.llm import MockLLMClient, set_mock_client


class TestExtractionModels:
    """Tests for extraction Pydantic models."""

    def test_entity_creation(self):
        entity = Entity(name="Paris", type="LOCATION", description="Capital of France")
        assert entity.name == "Paris"
        assert entity.type == "LOCATION"
        assert entity.description == "Capital of France"

    def test_relation_creation(self):
        relation = Relation(
            source="Paris",
            target="France",
            relation="LOCATED_IN",
            evidence="Paris is the capital of France"
        )
        assert relation.source == "Paris"
        assert relation.target == "France"
        assert relation.relation == "LOCATED_IN"

    def test_extraction_result(self):
        result = ExtractionResult(
            entities=[Entity(name="Paris", type="LOCATION")],
            relations=[Relation(source="Paris", target="France", relation="LOCATED_IN")]
        )
        assert len(result.entities) == 1
        assert len(result.relations) == 1


class TestEntityResolver:
    """Tests for entity resolution."""

    def setup_method(self):
        self.resolver = EntityResolver(
            embedding_similarity_threshold=0.85,
            string_similarity_threshold=0.9,
        )

    def test_normalize_name(self):
        assert self.resolver.normalize_name("Paris") == "paris"
        assert self.resolver.normalize_name("New York City") == "new york city"
        assert self.resolver.normalize_name("U.S.A.") == "usa"
        assert self.resolver.normalize_name("  Hello  World  ") == "hello world"

    def test_string_similarity(self):
        # Exact match
        assert self.resolver.compute_string_similarity("Paris", "Paris") == 1.0
        assert self.resolver.compute_string_similarity("paris", "Paris") == 1.0

        # Similar names
        sim = self.resolver.compute_string_similarity("New York", "New York City")
        assert sim > 0.8

        # Different names
        sim = self.resolver.compute_string_similarity("Paris", "London")
        assert sim < 0.5

    def test_build_clusters_simple(self):
        entities = [
            Entity(name="Paris", type="LOCATION", description="Capital of France"),
            Entity(name="Paris", type="LOCATION", description="City in France"),
            Entity(name="London", type="LOCATION", description="Capital of UK"),
        ]

        clusters = self.resolver.build_clusters(entities)

        # Should merge the two Paris entities
        assert len(clusters) == 2

        # Check cluster contents
        paris_cluster = next(c for c in clusters if c.canonical_name == "Paris")
        assert len(paris_cluster.members) == 2
        assert "Paris" in paris_cluster.aliases

        london_cluster = next(c for c in clusters if c.canonical_name == "London")
        assert len(london_cluster.members) == 1

    def test_build_clusters_different_types(self):
        entities = [
            Entity(name="Apple", type="ORG", description="Tech company"),
            Entity(name="Apple", type="PRODUCT", description="Fruit"),
        ]

        clusters = self.resolver.build_clusters(entities)

        # Should NOT merge different types
        assert len(clusters) == 2

    def test_create_alias_map(self):
        entities = [
            Entity(name="Paris", type="LOCATION"),
            Entity(name="Paris", type="LOCATION"),
            Entity(name="London", type="LOCATION"),
        ]
        clusters = self.resolver.build_clusters(entities)
        alias_map = self.resolver.create_alias_map(clusters)

        assert alias_map["Paris"] == "Paris"
        assert alias_map["London"] == "London"

    def test_resolve_relations(self):
        alias_map = {"NYC": "New York City", "NY": "New York City"}

        relations = [
            Relation(source="NYC", target="USA", relation="LOCATED_IN"),
            Relation(source="NY", target="USA", relation="LOCATED_IN"),
            Relation(source="London", target="UK", relation="LOCATED_IN"),
        ]

        resolved = self.resolver.resolve_relations(relations, alias_map)

        assert len(resolved) == 3
        assert resolved[0].source == "New York City"
        assert resolved[1].source == "New York City"
        assert resolved[2].source == "London"


class TestGraphBuilder:
    """Tests for graph builder with mocked LLM."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM client with predefined responses."""
        responses = {
            "Extract all entities and relationships": ExtractionResult(
                entities=[
                    Entity(name="Paris", type="LOCATION", description="Capital of France"),
                    Entity(name="France", type="LOCATION", description="Country in Europe"),
                    Entity(name="Eiffel Tower", type="LOCATION", description="Famous landmark"),
                ],
                relations=[
                    Relation(
                        source="Paris",
                        target="France",
                        relation="LOCATED_IN",
                        evidence="Paris is the capital of France"
                    ),
                    Relation(
                        source="Eiffel Tower",
                        target="Paris",
                        relation="LOCATED_IN",
                        evidence="The Eiffel Tower is in Paris"
                    ),
                ]
            ).model_dump(),
        }
        client = MockLLMClient(responses)
        set_mock_client(client)
        return client

    @pytest.mark.asyncio
    async def test_build_graph(self, mock_llm):
        """Test building graph from chunks."""
        chunks = [
            Chunk(
                chunk_id="chunk_1",
                doc_id="doc1",
                text="Paris is the capital of France. The Eiffel Tower is in Paris.",
                position=0,
            ),
        ]

        builder = GraphBuilder()
        result = await builder.build_from_chunks(chunks, max_concurrent=1)

        # Check stats
        stats = result["stats"]
        assert stats["num_nodes"] >= 3  # Paris, France, Eiffel Tower
        assert stats["num_edges"] >= 2  # Paris->France, Eiffel Tower->Paris

        # Check graph content
        graph_store = builder.graph_store
        paris = graph_store.get_entity("Paris")
        assert paris is not None
        assert paris["type"] == "LOCATION"

        france = graph_store.get_entity("France")
        assert france is not None

        eiffel = graph_store.get_entity("Eiffel Tower")
        assert eiffel is not None

        # Check neighbors
        neighbors = graph_store.get_neighbors("Paris", max_hops=1)
        assert len(neighbors) >= 2

        # Check chunk associations
        paris_chunks = graph_store.get_chunks_for_entity("Paris")
        assert "chunk_1" in paris_chunks

    @pytest.mark.asyncio
    async def test_entity_resolution_in_graph(self, mock_llm):
        """Test that entity resolution merges duplicates in graph."""
        # Two chunks mentioning the same entity
        chunks = [
            Chunk(chunk_id="chunk_1", doc_id="doc1", text="Paris is the capital of France.", position=0),
            Chunk(chunk_id="chunk_2", doc_id="doc2", text="Paris is known for the Eiffel Tower.", position=0),
        ]

        builder = GraphBuilder()
        result = await builder.build_from_chunks(chunks, max_concurrent=1)

        stats = result["stats"]
        # Should have merged Paris into one node
        paris = builder.graph_store.get_entity("Paris")
        assert paris is not None
        assert len(paris["chunk_ids"]) == 2  # Both chunks associated