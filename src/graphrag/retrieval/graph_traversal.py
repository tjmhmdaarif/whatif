"""Graph traversal for retrieval - BFS and Personalized PageRank."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Protocol

import networkx as nx
import numpy as np

from ..config import get_config
from ..logging_setup import get_logger
from .graph_linker import LinkedEntity, GraphStoreProtocol

logger = get_logger(__name__)


@dataclass
class TraversalResult:
    """Result of graph traversal."""
    visited_nodes: list[str]
    traversal_paths: list[list[dict[str, Any]]]  # List of paths, each path is list of edges
    collected_chunk_ids: list[str]
    node_scores: dict[str, float]


class GraphTraverser:
    """Traverses knowledge graph from seed entities."""

    def __init__(
        self,
        graph_store: GraphStoreProtocol,
        max_hops: int | None = None,
        max_nodes_per_hop: int | None = None,
    ):
        config = get_config()
        self.graph_store = graph_store
        self.max_hops = max_hops or config.retrieval.graph_max_hops
        self.max_nodes_per_hop = max_nodes_per_hop or 50

        # Access underlying NetworkX graph
        self.graph = graph_store.graph if hasattr(graph_store, 'graph') else None

    def _get_nx_graph(self) -> nx.MultiDiGraph | None:
        """Get the underlying NetworkX graph."""
        if hasattr(self.graph_store, 'graph'):
            return self.graph_store.graph
        return None

    def bfs_traverse(
        self,
        seed_entities: list[LinkedEntity],
        max_hops: int | None = None,
    ) -> TraversalResult:
        """Breadth-first traversal from seed entities."""
        max_hops = max_hops or self.max_hops
        nx_graph = self._get_nx_graph()

        if not nx_graph:
            logger.warning("No underlying graph available for traversal")
            return TraversalResult([], [], [], {})

        visited = set()
        node_scores = {}
        traversal_paths = []
        collected_chunks = set()

        # Initialize queue with seed entities
        queue = deque()
        for linked in seed_entities:
            node = linked.graph_node
            if nx_graph.has_node(node):
                visited.add(node)
                node_scores[node] = linked.similarity
                queue.append((node, 0, []))  # (node, hop, path_so_far)

                # Collect chunks for seed node
                chunks = self.graph_store.get_chunks_for_entity(node)
                collected_chunks.update(chunks)

        while queue:
            current_node, hop, path = queue.popleft()

            if hop >= max_hops:
                continue

            # Get neighbors
            neighbors = []
            # Successors (outgoing edges)
            for succ in nx_graph.successors(current_node):
                edge_data = nx_graph.get_edge_data(current_node, succ)
                for key, data in edge_data.items():
                    neighbors.append((succ, data, "forward"))

            # Predecessors (incoming edges)
            for pred in nx_graph.predecessors(current_node):
                edge_data = nx_graph.get_edge_data(pred, current_node)
                for key, data in edge_data.items():
                    neighbors.append((pred, data, "backward"))

            # Sort by edge weight/importance if available, limit per hop
            neighbors.sort(key=lambda x: len(x[1].get("chunk_ids", [])), reverse=True)
            neighbors = neighbors[:self.max_nodes_per_hop]

            for neighbor, edge_data, direction in neighbors:
                relation = edge_data.get("relation", "")
                evidence = edge_data.get("evidence", "")
                edge_chunks = edge_data.get("chunk_ids", [])

                # Build edge info
                if direction == "forward":
                    edge_info = {
                        "source": current_node,
                        "target": neighbor,
                        "relation": relation,
                        "evidence": evidence,
                        "chunk_ids": edge_chunks,
                        "hop": hop + 1,
                    }
                else:
                    edge_info = {
                        "source": neighbor,
                        "target": current_node,
                        "relation": relation,
                        "evidence": evidence,
                        "chunk_ids": edge_chunks,
                        "hop": hop + 1,
                    }

                new_path = path + [edge_info]
                traversal_paths.append(new_path)

                # Collect chunks
                collected_chunks.update(edge_chunks)
                neighbor_chunks = self.graph_store.get_chunks_for_entity(neighbor)
                collected_chunks.update(neighbor_chunks)

                # Score propagation: decay by hop
                if neighbor not in visited:
                    visited.add(neighbor)
                    node_scores[neighbor] = node_scores.get(current_node, 0.5) * 0.7
                    queue.append((neighbor, hop + 1, new_path))
                else:
                    # Update score if better path found
                    new_score = node_scores.get(current_node, 0.5) * 0.7
                    if new_score > node_scores.get(neighbor, 0):
                        node_scores[neighbor] = new_score

        return TraversalResult(
            visited_nodes=list(visited),
            traversal_paths=traversal_paths,
            collected_chunk_ids=list(collected_chunks),
            node_scores=node_scores,
        )

    def ppr_traverse(
        self,
        seed_entities: list[LinkedEntity],
        alpha: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> TraversalResult:
        """Personalized PageRank traversal from seed entities."""
        nx_graph = self._get_nx_graph()

        if not nx_graph or nx_graph.number_of_nodes() == 0:
            logger.warning("No graph available for PPR")
            return TraversalResult([], [], [], {})

        # Build adjacency matrix
        nodes = list(nx_graph.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        n = len(nodes)

        if n == 0:
            return TraversalResult([], [], [], {})

        # Create transition matrix (row-stochastic)
        adj = np.zeros((n, n))
        for i, node in enumerate(nodes):
            out_edges = list(nx_graph.out_edges(node, data=True))
            if out_edges:
                for _, target, data in out_edges:
                    j = node_to_idx.get(target)
                    if j is not None:
                        # Weight by number of chunks or uniform
                        weight = len(data.get("chunk_ids", [])) + 1
                        adj[i, j] += weight

        # Normalize rows
        row_sums = adj.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        adj = adj / row_sums

        # Personalization vector
        personalization = np.zeros(n)
        for linked in seed_entities:
            idx = node_to_idx.get(linked.graph_node)
            if idx is not None:
                personalization[idx] = linked.similarity

        if personalization.sum() == 0:
            personalization = np.ones(n) / n
        else:
            personalization = personalization / personalization.sum()

        # Power iteration for PPR
        ppr = personalization.copy()
        for _ in range(max_iter):
            new_ppr = alpha * adj.T @ ppr + (1 - alpha) * personalization
            if np.linalg.norm(new_ppr - ppr, 1) < tol:
                ppr = new_ppr
                break
            ppr = new_ppr

        # Get top nodes
        top_indices = np.argsort(ppr)[::-1]
        visited_nodes = [nodes[i] for i in top_indices if ppr[i] > 0.001][:100]

        # Collect chunks for top nodes
        collected_chunks = set()
        for node in visited_nodes:
            chunks = self.graph_store.get_chunks_for_entity(node)
            collected_chunks.update(chunks)

        # Build paths between top nodes (simplified)
        traversal_paths = []
        for linked in seed_entities:
            if nx_graph.has_node(linked.graph_node):
                # Short paths from seed to other high-PPR nodes
                for node in visited_nodes[:20]:
                    if node != linked.graph_node:
                        try:
                            path = nx.shortest_path(nx_graph, linked.graph_node, node)
                            if len(path) <= 3:
                                edges_path = []
                                for i in range(len(path) - 1):
                                    edge_data = nx_graph.get_edge_data(path[i], path[i+1])
                                    if edge_data:
                                        for key, data in edge_data.items():
                                            edges_path.append({
                                                "source": path[i],
                                                "target": path[i+1],
                                                "relation": data.get("relation", ""),
                                                "evidence": data.get("evidence", ""),
                                                "chunk_ids": data.get("chunk_ids", []),
                                                "hop": i + 1,
                                            })
                                            collected_chunks.update(data.get("chunk_ids", []))
                                if edges_path:
                                    traversal_paths.append(edges_path)
                        except nx.NetworkXNoPath:
                            pass

        node_scores = {nodes[i]: float(ppr[i]) for i in range(n) if ppr[i] > 0}

        return TraversalResult(
            visited_nodes=visited_nodes,
            traversal_paths=traversal_paths,
            collected_chunk_ids=list(collected_chunks),
            node_scores=node_scores,
        )


def create_graph_traverser(graph_store: GraphStoreProtocol) -> GraphTraverser:
    """Factory function to create graph traverser from config."""
    config = get_config()
    return GraphTraverser(
        graph_store=graph_store,
        max_hops=config.retrieval.graph_max_hops,
    )