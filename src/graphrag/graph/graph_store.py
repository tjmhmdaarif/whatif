"""Graph store interface and NetworkX implementation."""

from __future__ import annotations

import abc
import json
from pathlib import Path
from typing import Any

import networkx as nx

from ..config import get_config
from ..logging_setup import get_logger

logger = get_logger(__name__)


class GraphStore(abc.ABC):
    """Abstract base class for graph stores."""

    @abc.abstractmethod
    def add_entity(
        self,
        name: str,
        entity_type: str,
        description: str = "",
        chunk_ids: list[str] | None = None,
    ) -> None:
        """Add or update an entity node."""
        pass

    @abc.abstractmethod
    def add_relation(
        self,
        source: str,
        target: str,
        relation: str,
        evidence: str = "",
        chunk_ids: list[str] | None = None,
    ) -> None:
        """Add or update a relation edge."""
        pass

    @abc.abstractmethod
    def get_entity(self, name: str) -> dict[str, Any] | None:
        """Get entity by name."""
        pass

    @abc.abstractmethod
    def get_neighbors(self, name: str, max_hops: int = 1) -> list[dict[str, Any]]:
        """Get neighboring entities within max_hops."""
        pass

    @abc.abstractmethod
    def get_chunks_for_entity(self, name: str) -> list[str]:
        """Get chunk IDs associated with an entity."""
        pass

    @abc.abstractmethod
    def get_chunks_for_relation(self, source: str, target: str) -> list[str]:
        """Get chunk IDs associated with a relation."""
        pass

    @abc.abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """Get graph statistics."""
        pass

    @abc.abstractmethod
    def save(self, path: str | Path) -> None:
        """Save graph to disk."""
        pass

    @abc.abstractmethod
    def load(self, path: str | Path) -> None:
        """Load graph from disk."""
        pass

    @abc.abstractmethod
    def export_pyvis(self, path: str | Path, max_nodes: int = 100) -> None:
        """Export subgraph for pyvis visualization."""
        pass


class NetworkXGraphStore(GraphStore):
    """NetworkX-based in-memory graph store."""

    def __init__(
        self,
        graph_path: str | Path | None = None,
        alias_map_path: str | Path | None = None,
    ):
        config = get_config()
        self.graph_path = Path(graph_path or config.graph_store.networkx.graph_path)
        self.alias_map_path = Path(alias_map_path or config.graph_store.networkx.alias_map_path)

        self.graph = nx.MultiDiGraph()
        self.alias_map: dict[str, str] = {}

    def _canonicalize(self, name: str) -> str:
        """Resolve entity name to canonical form."""
        return self.alias_map.get(name, name)

    def add_entity(
        self,
        name: str,
        entity_type: str,
        description: str = "",
        chunk_ids: list[str] | None = None,
    ) -> None:
        """Add or update an entity node."""
        canonical = self._canonicalize(name)

        if self.graph.has_node(canonical):
            # Update existing
            node_data = self.graph.nodes[canonical]
            if description and description not in node_data.get("descriptions", []):
                node_data.setdefault("descriptions", []).append(description)
            if chunk_ids:
                node_data.setdefault("chunk_ids", []).extend(chunk_ids)
                node_data["chunk_ids"] = list(set(node_data["chunk_ids"]))
        else:
            # Create new
            self.graph.add_node(
                canonical,
                name=canonical,
                type=entity_type,
                descriptions=[description] if description else [],
                chunk_ids=chunk_ids or [],
            )

    def add_relation(
        self,
        source: str,
        target: str,
        relation: str,
        evidence: str = "",
        chunk_ids: list[str] | None = None,
    ) -> None:
        """Add or update a relation edge."""
        canonical_source = self._canonicalize(source)
        canonical_target = self._canonicalize(target)

        if canonical_source == canonical_target:
            return

        # Ensure nodes exist
        if not self.graph.has_node(canonical_source):
            self.graph.add_node(canonical_source, name=canonical_source, type="UNKNOWN", descriptions=[], chunk_ids=[])
        if not self.graph.has_node(canonical_target):
            self.graph.add_node(canonical_target, name=canonical_target, type="UNKNOWN", descriptions=[], chunk_ids=[])

        # Add edge (MultiDiGraph allows multiple edges)
        edge_key = self.graph.add_edge(
            canonical_source,
            canonical_target,
            relation=relation,
            evidence=evidence,
            chunk_ids=chunk_ids or [],
        )

    def get_entity(self, name: str) -> dict[str, Any] | None:
        """Get entity by name."""
        canonical = self._canonicalize(name)
        if not self.graph.has_node(canonical):
            return None

        node_data = self.graph.nodes[canonical]
        return {
            "name": canonical,
            "type": node_data.get("type", ""),
            "descriptions": node_data.get("descriptions", []),
            "chunk_ids": node_data.get("chunk_ids", []),
        }

    def get_neighbors(self, name: str, max_hops: int = 1) -> list[dict[str, Any]]:
        """Get neighboring entities within max_hops."""
        canonical = self._canonicalize(name)
        if not self.graph.has_node(canonical):
            return []

        neighbors = []
        # BFS up to max_hops
        visited = {canonical}
        current_level = {canonical}

        for hop in range(1, max_hops + 1):
            next_level = set()
            for node in current_level:
                for succ in self.graph.successors(node):
                    if succ not in visited:
                        visited.add(succ)
                        next_level.add(succ)
                        for data in self._iter_edge_attributes(node, succ):
                            neighbors.append({
                                "source": node,
                                "target": succ,
                                "relation": data.get("relation", ""),
                                "evidence": data.get("evidence", ""),
                                "chunk_ids": data.get("chunk_ids", []),
                                "hop": hop,
                            })
                for pred in self.graph.predecessors(node):
                    if pred not in visited:
                        visited.add(pred)
                        next_level.add(pred)
                        for data in self._iter_edge_attributes(pred, node):
                            neighbors.append({
                                "source": pred,
                                "target": node,
                                "relation": data.get("relation", ""),
                                "evidence": data.get("evidence", ""),
                                "chunk_ids": data.get("chunk_ids", []),
                                "hop": hop,
                            })
            current_level = next_level
            if not current_level:
                break

        return neighbors

    def get_chunks_for_entity(self, name: str) -> list[str]:
        """Get chunk IDs associated with an entity."""
        canonical = self._canonicalize(name)
        if not self.graph.has_node(canonical):
            return []
        return self.graph.nodes[canonical].get("chunk_ids", [])

    def get_chunks_for_relation(self, source: str, target: str) -> list[str]:
        """Get chunk IDs associated with a relation."""
        canonical_source = self._canonicalize(source)
        canonical_target = self._canonicalize(target)

        if not self.graph.has_edge(canonical_source, canonical_target):
            return []

        chunks = []
        for data in self._iter_edge_attributes(canonical_source, canonical_target):
            chunks.extend(data.get("chunk_ids", []))
        return list(set(chunks))

    def get_stats(self) -> dict[str, Any]:
        """Get graph statistics."""
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()

        # Average degree
        degrees = [d for _, d in self.graph.degree()]
        avg_degree = sum(degrees) / num_nodes if num_nodes > 0 else 0

        # Connected components (undirected)
        undirected = self.graph.to_undirected()
        num_components = nx.number_connected_components(undirected)

        # Entity types
        type_counts: dict[str, int] = {}
        for _, data in self.graph.nodes(data=True):
            etype = data.get("type", "UNKNOWN")
            type_counts[etype] = type_counts.get(etype, 0) + 1

        # Relation types
        relation_counts: dict[str, int] = {}
        for _, _, data in self.graph.edges(data=True):
            rel = data.get("relation", "UNKNOWN")
            relation_counts[rel] = relation_counts.get(rel, 0) + 1

        return {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "avg_degree": round(avg_degree, 2),
            "num_components": num_components,
            "entity_types": type_counts,
            "relation_types": relation_counts,
        }

    def save(self, path: str | Path | None = None) -> None:
        """Save graph to GraphML and alias map to JSON."""
        path = Path(path or self.graph_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Create a copy of the graph with serialized attributes for GraphML
        graph_copy = self.graph.copy()

        # Convert list attributes to strings for GraphML compatibility
        for node, data in graph_copy.nodes(data=True):
            for key, value in data.items():
                if isinstance(value, list):
                    data[key] = "|".join(str(v) for v in value)
                elif not isinstance(value, (str, int, float, bool)):
                    data[key] = str(value)

        for u, v, data in graph_copy.edges(data=True):
            for key, value in data.items():
                if isinstance(value, list):
                    data[key] = "|".join(str(v) for v in value)
                elif not isinstance(value, (str, int, float, bool)):
                    data[key] = str(value)

        # Save graph as GraphML
        nx.write_graphml(graph_copy, path)
        logger.info(f"Saved graph to {path} ({self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges)")

        # Save alias map alongside the graph
        alias_path = self.alias_map_path
        if path is not None and Path(path).parent != self.graph_path.parent:
            alias_path = Path(path).with_suffix(".aliases.json")
        alias_path.parent.mkdir(parents=True, exist_ok=True)
        with open(alias_path, "w") as f:
            json.dump(self.alias_map, f, indent=2)
        logger.info(f"Saved alias map to {alias_path}")

    # Attributes that hold lists in-memory but must be scalar strings in GraphML.
    _LIST_ATTRIBUTE_KEYS = ("chunk_ids", "descriptions")

    def _deserialize_list_attributes(self, graph: nx.MultiDiGraph) -> None:
        """Convert pipe-delimited string attributes back into lists.

        GraphML only supports scalar attribute values, so save() encodes lists
        (``chunk_ids``, ``descriptions``) as ``"a|b|c"``. This reverses that so
        loaded graphs behave identically to freshly built ones.

        Only the keys in ``_LIST_ATTRIBUTE_KEYS`` are converted, so free-text
        attributes that legitimately contain ``|`` are left untouched.

        Args:
            graph: Graph whose node and edge attributes are fixed in place.
        """
        def _restore(data: dict[str, Any]) -> None:
            for key in self._LIST_ATTRIBUTE_KEYS:
                value = data.get(key)
                if isinstance(value, str):
                    data[key] = [part for part in value.split("|") if part]

        for _, data in graph.nodes(data=True):
            _restore(data)
        for _, _, data in graph.edges(data=True):
            _restore(data)

    def _iter_edge_attributes(self, source: str, target: str) -> list[dict[str, Any]]:
        """Return attribute dicts for all edges between two nodes.

        ``get_edge_data`` returns different shapes depending on graph class: a
        MultiGraph yields ``{edge_key: attrs}`` while a plain Graph yields the
        attribute dict itself. GraphML round-trips can collapse to a plain
        Graph, so both shapes must be handled.

        Args:
            source: Source node name.
            target: Target node name.

        Returns:
            List of edge attribute dictionaries (possibly empty).
        """
        edge_data = self.graph.get_edge_data(source, target)
        if not edge_data:
            return []

        if all(isinstance(value, dict) for value in edge_data.values()):
            return list(edge_data.values())
        return [edge_data]

    def load(self, path: str | Path | None = None) -> None:
        """Load graph from GraphML and alias map from JSON."""
        path = Path(path or self.graph_path)

        if not path.exists():
            logger.warning(f"Graph file not found at {path}")
            return

        self.graph = nx.read_graphml(path)

        # GraphML cannot store list attributes, so save() serializes them as
        # pipe-delimited strings. Restore them to lists here, otherwise
        # callers iterate over characters instead of chunk IDs / descriptions.
        self._deserialize_list_attributes(self.graph)

        logger.info(f"Loaded graph from {path} ({self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges)")

        alias_path = self.alias_map_path
        if alias_path.exists():
            with open(alias_path, "r") as f:
                self.alias_map = json.load(f)
            logger.info(f"Loaded alias map from {alias_path} ({len(self.alias_map)} entries)")

    def export_pyvis(self, path: str | Path, max_nodes: int = 100) -> None:
        """Export subgraph for pyvis visualization."""
        try:
            from pyvis.network import Network
        except ImportError:
            logger.warning("pyvis not installed, skipping visualization export")
            return

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Create subgraph with most connected nodes
        degrees = dict(self.graph.degree())
        top_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:max_nodes]
        subgraph = self.graph.subgraph(top_nodes)

        # Create pyvis network
        net = Network(
            height="800px",
            width="100%",
            directed=True,
            notebook=False,
            cdn_resources="remote",
        )

        # Add nodes
        for node in subgraph.nodes():
            node_data = subgraph.nodes[node]
            etype = node_data.get("type", "UNKNOWN")
            color = self._get_color_for_type(etype)
            title = f"{node} ({etype})"
            if node_data.get("descriptions"):
                title += f"\n{node_data['descriptions'][0][:200]}"

            net.add_node(node, label=node[:50], title=title, color=color)

        # Add edges
        for source, target, data in subgraph.edges(data=True):
            relation = data.get("relation", "")
            evidence = data.get("evidence", "")
            title = f"{relation}: {evidence[:200]}" if evidence else relation
            net.add_edge(source, target, title=title, label=relation)

        # Save
        net.save_graph(str(path))
        logger.info(f"Exported pyvis visualization to {path}")

    def _get_color_for_type(self, entity_type: str) -> str:
        """Get color for entity type."""
        colors = {
            "PERSON": "#FF6B6B",
            "ORG": "#4ECDC4",
            "LOCATION": "#45B7D1",
            "EVENT": "#FFBE0B",
            "DATE": "#FB5607",
            "WORK_OF_ART": "#8338EC",
            "PRODUCT": "#3A86FF",
            "CONCEPT": "#06D6A0",
            "UNKNOWN": "#999999",
        }
        return colors.get(entity_type, "#999999")


def create_graph_store(load_if_exists: bool = True) -> GraphStore:
    """Factory function to create graph store from config.

    Args:
        load_if_exists: If True (default), automatically load a previously
            persisted graph from the configured path when the file exists.
            Pass False to obtain an empty store (e.g. before building).
    """
    config = get_config()
    if config.graph_store.type == "networkx":
        store = NetworkXGraphStore()
    elif config.graph_store.type == "neo4j":
        from .neo4j_store import Neo4jGraphStore
        return Neo4jGraphStore()
    else:
        raise ValueError(f"Unknown graph store type: {config.graph_store.type}")

    if load_if_exists:
        store.load()
    return store