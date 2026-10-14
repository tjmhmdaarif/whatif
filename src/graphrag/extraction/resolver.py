"""Entity resolution - merge duplicate entities using embeddings and string matching."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from rapidfuzz import fuzz

from ..config import get_config
from ..logging_setup import get_logger
from .models import Entity, Relation

logger = get_logger(__name__)


def _get_embedder():
    """Lazy import to avoid circular dependency."""
    from ..retrieval import get_embedder
    return get_embedder()


@dataclass
class EntityCluster:
    """Cluster of entities that refer to the same real-world entity."""
    canonical_name: str
    canonical_type: str
    members: list[Entity] = field(default_factory=list)
    aliases: set[str] = field(default_factory=set)
    descriptions: list[str] = field(default_factory=list)


class EntityResolver:
    """Resolves entity duplicates using embedding similarity and string matching."""

    def __init__(
        self,
        embedding_similarity_threshold: float | None = None,
        string_similarity_threshold: float | None = None,
        merge_aliases: bool | None = None,
    ):
        config = get_config()
        self.embedding_threshold = (
            embedding_similarity_threshold or config.entity_resolution.embedding_similarity_threshold
        )
        self.string_threshold = (
            string_similarity_threshold or config.entity_resolution.string_similarity_threshold
        )
        self.merge_aliases = merge_aliases or config.entity_resolution.merge_aliases

        self._embedder = None

    @property
    def embedder(self):
        """Lazy-loaded embedder."""
        if self._embedder is None:
            self._embedder = _get_embedder()
        return self._embedder

    def normalize_name(self, name: str) -> str:
        """Normalize entity name for comparison."""
        # Lowercase, remove punctuation, extra spaces
        import re
        normalized = name.lower().strip()
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized

    def compute_string_similarity(self, name1: str, name2: str) -> float:
        """Compute string similarity between two names.

        ``partial_ratio`` is deliberately not used: it scores any short name that
        appears as a substring of a longer one as a near-perfect match, which
        collapses unrelated entities ("Tivoli Gardens" vs "Radar Station")
        into a single cluster.

        Args:
            name1: First entity name.
            name2: Second entity name.

        Returns:
            Similarity in the range ``[0.0, 1.0]``.
        """
        norm1 = self.normalize_name(name1)
        norm2 = self.normalize_name(name2)

        if norm1 == norm2:
            return 1.0

        # Use order-insensitive whole-string measures only.
        ratio = fuzz.ratio(norm1, norm2) / 100.0
        token_sort = fuzz.token_sort_ratio(norm1, norm2) / 100.0

        # Token containment: high when one name's tokens are a subset of the
        # other's ("New York" within "New York City"), which is the genuine
        # abbreviation case, but low when the names merely share a token.
        tokens1 = set(norm1.split())
        tokens2 = set(norm2.split())
        if tokens1 and tokens2:
            containment = len(tokens1 & tokens2) / min(len(tokens1), len(tokens2))
        else:
            containment = 0.0

        return max(ratio, token_sort, containment)

    def compute_embedding_similarity(self, entities: list[Entity]) -> np.ndarray:
        """Compute pairwise embedding similarity matrix for entities."""
        if not entities:
            return np.array([])

        # Create texts for embedding: name + type + description
        texts = [
            f"{e.name} ({e.type}): {e.description}"
            for e in entities
        ]

        embeddings = self.embedder.encode(texts, normalize=True)

        # Compute cosine similarity matrix
        similarity = np.dot(embeddings, embeddings.T)
        return similarity

    def build_clusters(self, entities: list[Entity]) -> list[EntityCluster]:
        """Build entity clusters using embedding and string similarity."""
        if not entities:
            return []

        # Compute similarities
        embedding_sim = self.compute_embedding_similarity(entities)

        # Union-find for clustering
        n = len(entities)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[py] = px

        # Compare all pairs
        for i in range(n):
            for j in range(i + 1, n):
                # Check string similarity
                str_sim = self.compute_string_similarity(entities[i].name, entities[j].name)

                # Check embedding similarity
                emb_sim = embedding_sim[i, j] if embedding_sim.size > 0 else 0.0

                # Same type required
                type_match = entities[i].type == entities[j].type

                # Merge if either similarity exceeds threshold and types match
                if type_match and (str_sim >= self.string_threshold or emb_sim >= self.embedding_threshold):
                    union(i, j)

        # Build clusters
        clusters: dict[int, EntityCluster] = {}
        for i, entity in enumerate(entities):
            root = find(i)
            if root not in clusters:
                clusters[root] = EntityCluster(
                    canonical_name=entity.name,
                    canonical_type=entity.type,
                )
            cluster = clusters[root]
            cluster.members.append(entity)
            cluster.aliases.add(entity.name)
            if entity.description:
                cluster.descriptions.append(entity.description)

        return list(clusters.values())

    def create_alias_map(self, clusters: list[EntityCluster]) -> dict[str, str]:
        """Create mapping from alias to canonical name."""
        alias_map = {}
        for cluster in clusters:
            for alias in cluster.aliases:
                alias_map[alias] = cluster.canonical_name
        return alias_map

    def resolve_relations(
        self,
        relations: list[Relation],
        alias_map: dict[str, str],
    ) -> list[Relation]:
        """Resolve entity names in relations using alias map."""
        resolved = []
        for rel in relations:
            source = alias_map.get(rel.source, rel.source)
            target = alias_map.get(rel.target, rel.target)

            # Skip self-loops
            if source == target:
                continue

            resolved.append(Relation(
                source=source,
                target=target,
                relation=rel.relation,
                evidence=rel.evidence,
            ))
        return resolved

    def save_alias_map(self, alias_map: dict[str, str], path: str | Path) -> None:
        """Save alias map to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(alias_map, f, indent=2)
        logger.info(f"Saved alias map with {len(alias_map)} entries to {path}")

    def load_alias_map(self, path: str | Path) -> dict[str, str]:
        """Load alias map from JSON file."""
        path = Path(path)
        if not path.exists():
            return {}
        with open(path, "r") as f:
            return json.load(f)


def create_resolver() -> EntityResolver:
    """Factory function to create resolver from config."""
    return EntityResolver()