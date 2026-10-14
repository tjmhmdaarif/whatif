"""Tests for the NetworkX graph store persistence and query linking."""

import asyncio
from pathlib import Path

import pytest

from graphrag.chunking import Chunk
from graphrag.config import get_config, load_config, set_config
from graphrag.extraction.models import Entity, Relation
from graphrag.graph import NetworkXGraphStore, create_graph_store
from graphrag.llm import MockLLMClient, set_mock_client
from graphrag.retrieval.graph_linker import QueryEntityLinker


@pytest.fixture
def populated_store(tmp_path: Path) -> NetworkXGraphStore:
    """Return a graph store containing entities and one relation."""
    config = load_config("config.yaml")
    set_config(config)

    store = NetworkXGraphStore(
        graph_path=tmp_path / "graph.graphml",
        alias_map_path=tmp_path / "aliases.json",
    )
    store.add_entity("Paris", "LOCATION", "Capital of France", ["doc1_chunk_0", "doc2_chunk_1"])
    store.add_entity("France", "LOCATION", "Country in Europe", ["doc2_chunk_1"])
    store.add_entity("Eiffel Tower", "LOCATION", "Famous landmark", ["doc3_chunk_0"])
    store.add_relation(
        "Eiffel Tower", "Paris", "LOCATED_IN",
        "The Eiffel Tower is in Paris", ["doc3_chunk_0"],
    )
    return store


class TestGraphPersistence:
    """GraphML cannot store list attributes; save/load must round-trip them."""

    def test_chunk_ids_survive_round_trip(self, populated_store: NetworkXGraphStore):
        """chunk_ids must come back as a list of strings, not a joined string."""
        populated_store.save()

        reloaded = NetworkXGraphStore(
            graph_path=populated_store.graph_path,
            alias_map_path=populated_store.alias_map_path,
        )
        reloaded.load()

        chunk_ids = reloaded.get_chunks_for_entity("Paris")

        assert isinstance(chunk_ids, list)
        assert set(chunk_ids) == {"doc1_chunk_0", "doc2_chunk_1"}
        # Guard against the pipe-delimited string bug, where len() counts characters.
        assert all(isinstance(c, str) for c in chunk_ids)

    def test_descriptions_survive_round_trip(self, populated_store: NetworkXGraphStore):
        """Entity descriptions must be restored as a list."""
        populated_store.save()

        reloaded = NetworkXGraphStore(
            graph_path=populated_store.graph_path,
            alias_map_path=populated_store.alias_map_path,
        )
        reloaded.load()

        entity = reloaded.get_entity("Paris")

        assert entity is not None
        assert isinstance(entity["descriptions"], list)
        assert "Capital of France" in entity["descriptions"]

    def test_relation_chunks_survive_round_trip(self, populated_store: NetworkXGraphStore):
        """Edge chunk_ids must be restored as a list."""
        populated_store.save()

        reloaded = NetworkXGraphStore(
            graph_path=populated_store.graph_path,
            alias_map_path=populated_store.alias_map_path,
        )
        reloaded.load()

        assert "doc3_chunk_0" in reloaded.get_chunks_for_relation("Eiffel Tower", "Paris")


class TestGraphStoreAutoLoad:
    """create_graph_store() must load a persisted graph when one exists."""

    @staticmethod
    def _patch_paths(monkeypatch: pytest.MonkeyPatch, store: NetworkXGraphStore) -> None:
        """Point the config-driven default paths at the temp-store paths."""
        config = get_config()
        monkeypatch.setattr(config.graph_store.networkx, "graph_path", str(store.graph_path))
        monkeypatch.setattr(config.graph_store.networkx, "alias_map_path", str(store.alias_map_path))

    def test_factory_loads_existing_graph(
        self, populated_store: NetworkXGraphStore, monkeypatch: pytest.MonkeyPatch
    ):
        """The factory should not hand back an empty graph when a file exists."""
        populated_store.save()
        self._patch_paths(monkeypatch, populated_store)

        store = create_graph_store()

        assert store.graph.number_of_nodes() == 3
        assert store.graph.number_of_edges() == 1

    def test_factory_can_skip_loading(
        self, populated_store: NetworkXGraphStore, monkeypatch: pytest.MonkeyPatch
    ):
        """load_if_exists=False must yield an empty store for rebuilding."""
        populated_store.save()
        self._patch_paths(monkeypatch, populated_store)

        store = create_graph_store(load_if_exists=False)

        assert store.graph.number_of_nodes() == 0


class TestQueryEntityLinking:
    """Linking must work even when the LLM extracts no entities."""

    @pytest.mark.asyncio
    async def test_links_graph_node_mentioned_in_question(
        self, populated_store: NetworkXGraphStore
    ):
        """With a mock LLM returning nothing, lexical matching should still link."""
        set_mock_client(MockLLMClient())

        linker = QueryEntityLinker(populated_store)

        linked = await linker.link_question("Where is the Eiffel Tower located?")

        node_names = {entity.graph_node for entity in linked}
        assert "Eiffel Tower" in node_names

    @pytest.mark.asyncio
    async def test_no_link_for_absent_entity(self, populated_store: NetworkXGraphStore):
        """Entities outside the graph must not be linked."""
        set_mock_client(MockLLMClient())

        linker = QueryEntityLinker(populated_store)

        linked = await linker.link_question("Who directed the film?")

        assert linked == []
