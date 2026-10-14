"""Hybrid retrieval fusion - combines vector and graph retrieval using RRF."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from ..chunking import Chunk
from ..config import get_config
from ..logging_setup import get_logger
from ..retrieval import create_vector_store, get_embedder
from .graph_linker import QueryEntityLinker, create_query_linker
from .graph_traversal import GraphTraverser, create_graph_traverser, TraversalResult
from .vector import VectorStore

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """Combined retrieval result."""
    chunks: list[Chunk]
    traversal_paths: list[list[dict[str, Any]]]
    vector_scores: dict[str, float]
    graph_scores: dict[str, float]
    fused_scores: dict[str, float]


class HybridRetriever:
    """Hybrid retriever combining vector search and graph traversal."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        graph_store: Any | None = None,
        vector_weight: float | None = None,
        graph_weight: float | None = None,
        rrf_k: int | None = None,
        max_context_tokens: int | None = None,
        deduplicate: bool | None = None,
    ):
        config = get_config()
        self.vector_store = vector_store if vector_store is not None else create_vector_store()
        self.graph_store = graph_store

        self.vector_weight = vector_weight or (1 - config.retrieval.hybrid_fusion_weight)
        self.graph_weight = graph_weight or config.retrieval.hybrid_fusion_weight
        self.rrf_k = rrf_k or config.retrieval.rrf_k
        self.max_context_tokens = max_context_tokens or config.retrieval.max_context_tokens
        self.graph_top_k = getattr(config.retrieval, "graph_top_k", 10)
        self.deduplicate = deduplicate if deduplicate is not None else config.retrieval.deduplicate_chunks

        self.embedder = get_embedder()

        # Initialize graph components if graph store available
        if self.graph_store:
            self.entity_linker = create_query_linker(self.graph_store)
            self.graph_traverser = create_graph_traverser(self.graph_store)
        else:
            self.entity_linker = None
            self.graph_traverser = None

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[tuple[Chunk, float]],
        graph_chunks: list[Chunk],
        graph_scores: dict[str, float],
    ) -> list[tuple[Chunk, float]]:
        """Apply Reciprocal Rank Fusion to combine vector and graph results."""
        # Rank vector results
        vector_ranks = {chunk.chunk_id: rank + 1 for rank, (chunk, _) in enumerate(vector_results)}

        # Rank graph results by score
        sorted_graph = sorted(graph_chunks, key=lambda c: graph_scores.get(c.chunk_id, 0), reverse=True)
        graph_ranks = {chunk.chunk_id: rank + 1 for rank, chunk in enumerate(sorted_graph)}

        # Compute RRF scores
        all_chunk_ids = set(vector_ranks.keys()) | set(graph_ranks.keys())
        rrf_scores = {}

        for chunk_id in all_chunk_ids:
            v_rank = vector_ranks.get(chunk_id, len(vector_results) + self.rrf_k)
            g_rank = graph_ranks.get(chunk_id, len(sorted_graph) + self.rrf_k)

            rrf_score = (
                self.vector_weight * (1 / (self.rrf_k + v_rank)) +
                self.graph_weight * (1 / (self.rrf_k + g_rank))
            )
            rrf_scores[chunk_id] = rrf_score

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # Build result list with chunks
        chunk_map = {}
        for chunk, _ in vector_results:
            chunk_map[chunk.chunk_id] = chunk
        for chunk in graph_chunks:
            chunk_map[chunk.chunk_id] = chunk

        results = []
        for chunk_id in sorted_ids:
            if chunk_id in chunk_map:
                results.append((chunk_map[chunk_id], rrf_scores[chunk_id]))

        return results

    def _deduplicate_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Deduplicate chunks by chunk_id, keeping first occurrence."""
        seen = set()
        unique = []
        for chunk in chunks:
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                unique.append(chunk)
        return unique

    def _apply_token_budget(self, chunks: list[Chunk]) -> list[Chunk]:
        """Apply token budget to chunks."""
        total_tokens = 0
        result = []

        for chunk in chunks:
            chunk_tokens = len(chunk.text) // 4  # Rough estimate
            if total_tokens + chunk_tokens > self.max_context_tokens:
                break
            result.append(chunk)
            total_tokens += chunk_tokens

        return result

    async def retrieve_vector(
        self,
        question: str,
        top_k: int | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Vector-only retrieval."""
        config = get_config()
        top_k = top_k or config.retrieval.vector_top_k

        query_embedding = self.embedder.encode_single(question, normalize=True)
        results = self.vector_store.search(query_embedding, top_k)
        return results

    async def retrieve_graph(
        self,
        question: str,
        max_hops: int | None = None,
    ) -> tuple[list[Chunk], TraversalResult]:
        """Graph-only retrieval."""
        if not self.graph_store or not self.entity_linker or not self.graph_traverser:
            logger.warning("Graph store not available, returning empty graph results")
            return [], TraversalResult([], [], [], {})

        # Link question entities to graph
        linked_entities = await self.entity_linker.link_question(question)

        if not linked_entities:
            logger.debug("No entities linked to graph")
            return [], TraversalResult([], [], [], {})

        # Traverse graph
        traversal_result = self.graph_traverser.bfs_traverse(linked_entities, max_hops)

        # Get chunks for visited nodes, keeping only the highest-scoring ones.
        # A 2-hop walk over a dense graph can touch hundreds of nodes, so the
        # result is ranked by node score and capped at graph_top_k.
        graph_top_k = self.graph_top_k
        ranked_nodes = sorted(
            traversal_result.node_scores.items(), key=lambda kv: kv[1], reverse=True
        )

        graph_chunks = []
        seen_chunks: set[str] = set()
        for node, _ in ranked_nodes:
            if len(graph_chunks) >= graph_top_k:
                break
            for chunk_id in self.graph_store.get_chunks_for_entity(node):
                if chunk_id in seen_chunks or len(graph_chunks) >= graph_top_k:
                    continue
                chunk = self.vector_store.get_chunk(chunk_id)
                if chunk:
                    seen_chunks.add(chunk_id)
                    graph_chunks.append(chunk)

        # Score chunks by node scores
        graph_scores = {}
        for chunk in graph_chunks:
            # Find which node this chunk belongs to
            best_score = 0.0
            for node, score in traversal_result.node_scores.items():
                node_chunks = self.graph_store.get_chunks_for_entity(node)
                if chunk.chunk_id in node_chunks:
                    best_score = max(best_score, score)
            graph_scores[chunk.chunk_id] = best_score or 0.1

        return graph_chunks, traversal_result

    async def retrieve_hybrid(
        self,
        question: str,
        vector_top_k: int | None = None,
        graph_max_hops: int | None = None,
    ) -> RetrievalResult:
        """Hybrid retrieval combining vector and graph."""
        # Vector retrieval
        vector_results = await self.retrieve_vector(question, vector_top_k)

        # Graph retrieval
        graph_chunks, traversal_result = await self.retrieve_graph(question, graph_max_hops)

        # Deduplicate
        all_chunks = [c for c, _ in vector_results] + graph_chunks
        if self.deduplicate:
            all_chunks = self._deduplicate_chunks(all_chunks)

        # Apply RRF fusion
        fused_results = self._reciprocal_rank_fusion(vector_results, graph_chunks, traversal_result.node_scores)

        # Apply token budget
        fused_chunks = [c for c, _ in fused_results]
        fused_chunks = self._apply_token_budget(fused_chunks)

        # Build score dicts
        vector_scores = {c.chunk_id: s for c, s in vector_results}
        graph_scores = traversal_result.node_scores
        fused_scores = {c.chunk_id: s for c, s in fused_results if c in fused_chunks}

        return RetrievalResult(
            chunks=fused_chunks,
            traversal_paths=traversal_result.traversal_paths,
            vector_scores=vector_scores,
            graph_scores=graph_scores,
            fused_scores=fused_scores,
        )

    async def retrieve(
        self,
        question: str,
        mode: str = "hybrid",
        vector_top_k: int | None = None,
        graph_max_hops: int | None = None,
    ) -> RetrievalResult:
        """Main retrieval entry point with mode selection."""
        if mode == "vector":
            vector_results = await self.retrieve_vector(question, vector_top_k)
            return RetrievalResult(
                chunks=[c for c, _ in vector_results],
                traversal_paths=[],
                vector_scores={c.chunk_id: s for c, s in vector_results},
                graph_scores={},
                fused_scores={c.chunk_id: s for c, s in vector_results},
            )
        elif mode == "graph":
            graph_chunks, traversal_result = await self.retrieve_graph(question, graph_max_hops)
            return RetrievalResult(
                chunks=graph_chunks,
                traversal_paths=traversal_result.traversal_paths,
                vector_scores={},
                graph_scores=traversal_result.node_scores,
                fused_scores=traversal_result.node_scores,
            )
        elif mode == "hybrid":
            return await self.retrieve_hybrid(question, vector_top_k, graph_max_hops)
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")


def create_hybrid_retriever(
    vector_store: VectorStore | None = None,
    graph_store: Any | None = None,
) -> HybridRetriever:
    """Factory function to create hybrid retriever from config."""
    return HybridRetriever(
        vector_store=vector_store,
        graph_store=graph_store,
    )