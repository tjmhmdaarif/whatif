"""Graph builder - orchestrates extraction, resolution, and graph construction."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

from ..chunking import Chunk
from ..config import get_config
from ..logging_setup import get_logger
from ..retrieval import get_embedder
from ..extraction import (
    Entity,
    Relation,
    EntityRelationExtractor,
    create_extractor,
    EntityResolver,
    create_resolver,
)
from .graph_store import GraphStore, create_graph_store

logger = get_logger(__name__)


class GraphBuilder:
    """Builds knowledge graph from document chunks."""

    def __init__(
        self,
        extractor: EntityRelationExtractor | None = None,
        resolver: EntityResolver | None = None,
        graph_store: GraphStore | None = None,
    ):
        self.extractor = extractor or create_extractor()
        self.resolver = resolver or create_resolver()
        self.graph_store = graph_store or create_graph_store(load_if_exists=False)

    async def build_from_chunks(
        self,
        chunks: list[Chunk],
        max_concurrent: int = 5,
    ) -> dict[str, Any]:
        """Build knowledge graph from chunks."""
        logger.info(f"Building graph from {len(chunks)} chunks...")

        # Step 1: Extract entities and relations from all chunks
        logger.info("Extracting entities and relations...")
        extraction_results = await self.extractor.extract_batch(chunks, max_concurrent)

        # Collect all entities and relations with source chunk tracking
        all_entities: list[Entity] = []
        all_relations: list[Relation] = []
        entity_chunk_map: dict[str, list[str]] = defaultdict(list)
        relation_chunk_map: dict[tuple[str, str, str], list[str]] = defaultdict(list)

        for chunk_id, result in extraction_results.items():
            for entity in result.entities:
                all_entities.append(entity)
                entity_chunk_map[entity.name].append(chunk_id)

            for relation in result.relations:
                all_relations.append(relation)
                key = (relation.source, relation.target, relation.relation)
                relation_chunk_map[key].append(chunk_id)

        logger.info(f"Extracted {len(all_entities)} entities, {len(all_relations)} relations (before resolution)")

        # Step 2: Entity resolution
        logger.info("Resolving entities...")
        clusters = self.resolver.build_clusters(all_entities)
        alias_map = self.resolver.create_alias_map(clusters)

        # Update graph store with alias map
        if hasattr(self.graph_store, 'alias_map'):
            self.graph_store.alias_map = alias_map

        logger.info(f"Resolved into {len(clusters)} unique entities")

        # Step 3: Add resolved entities to graph
        logger.info("Adding entities to graph...")
        for cluster in clusters:
            # Use canonical name and merge descriptions
            canonical_name = cluster.canonical_name
            entity_type = cluster.canonical_type
            descriptions = list(set(cluster.descriptions))
            description = " | ".join(descriptions) if descriptions else ""

            # Collect all chunk IDs from cluster members
            chunk_ids = []
            for member in cluster.members:
                chunk_ids.extend(entity_chunk_map.get(member.name, []))
            chunk_ids = list(set(chunk_ids))

            self.graph_store.add_entity(
                name=canonical_name,
                entity_type=entity_type,
                description=description,
                chunk_ids=chunk_ids,
            )

        # Step 4: Resolve and add relations
        logger.info("Adding relations to graph...")
        resolved_relations = self.resolver.resolve_relations(all_relations, alias_map)

        for relation in resolved_relations:
            key = (relation.source, relation.target, relation.relation)
            chunk_ids = list(set(relation_chunk_map.get(key, [])))

            self.graph_store.add_relation(
                source=relation.source,
                target=relation.target,
                relation=relation.relation,
                evidence=relation.evidence,
                chunk_ids=chunk_ids,
            )

        # Step 5: Save alias map
        if hasattr(self.resolver, 'save_alias_map'):
            self.resolver.save_alias_map(alias_map, self.graph_store.alias_map_path)

        # Get final stats
        stats = self.graph_store.get_stats()
        logger.info(f"Graph built: {stats['num_nodes']} nodes, {stats['num_edges']} edges")

        return {
            "stats": stats,
            "num_entities_before_resolution": len(all_entities),
            "num_relations_before_resolution": len(all_relations),
            "num_clusters": len(clusters),
            "alias_map_size": len(alias_map),
        }

    def save(self, path: str | Path | None = None) -> None:
        """Save graph to disk."""
        self.graph_store.save(path)

    def load(self, path: str | Path | None = None) -> None:
        """Load graph from disk."""
        self.graph_store.load(path)

    def get_stats(self) -> dict[str, Any]:
        """Get graph statistics."""
        return self.graph_store.get_stats()

    def export_pyvis(self, path: str | Path, max_nodes: int = 100) -> None:
        """Export graph visualization."""
        self.graph_store.export_pyvis(path, max_nodes)


async def build_graph_from_corpus(
    chunks: list[Chunk],
    max_concurrent: int = 5,
) -> GraphBuilder:
    """Convenience function to build graph from chunks."""
    builder = GraphBuilder()
    await builder.build_from_chunks(chunks, max_concurrent)
    return builder