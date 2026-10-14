"""Query entity linking - extract entities from questions and link to graph nodes."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ..config import get_config
from ..llm import create_llm_client
from ..logging_setup import get_logger
from ..retrieval import get_embedder
from ..extraction.models import Entity

logger = get_logger(__name__)


class GraphStoreProtocol(Protocol):
    """Protocol for graph store - avoids circular imports."""
    def get_entity(self, name: str) -> dict[str, Any] | None: ...
    def get_chunks_for_entity(self, name: str) -> list[str]: ...
    @property
    def graph(self) -> Any: ...


@dataclass
class LinkedEntity:
    """Question entity linked to graph node."""
    question_entity: str
    graph_node: str
    similarity: float
    entity_type: str | None = None


class QueryEntityLinker:
    """Links entities extracted from questions to graph nodes."""

    def __init__(
        self,
        graph_store: GraphStoreProtocol,
        similarity_threshold: float = 0.7,
        max_candidates: int = 5,
    ):
        self.graph_store = graph_store
        self.similarity_threshold = similarity_threshold
        self.max_candidates = max_candidates
        self.embedder = get_embedder()
        self.client = create_llm_client()

        # Cache graph node embeddings
        self._node_embeddings: dict[str, np.ndarray] | None = None
        self._node_names: list[str] = []

    def _build_node_index(self) -> None:
        """Build embedding index for all graph nodes."""
        if self._node_embeddings is not None:
            return

        # Get all node names from graph
        self._node_names = list(self.graph_store.graph.nodes()) if hasattr(self.graph_store, 'graph') else []

        if not self._node_names:
            logger.warning("Graph is empty, cannot build node index")
            self._node_embeddings = {}
            return

        # Create texts for embedding: name + type + description
        node_texts = []
        for name in self._node_names:
            entity = self.graph_store.get_entity(name)
            if entity:
                text = f"{name} ({entity.get('type', '')}): {' '.join(entity.get('descriptions', [])[:2])}"
            else:
                text = name
            node_texts.append(text)

        # Generate embeddings
        embeddings = self.embedder.encode(node_texts, normalize=True)
        self._node_embeddings = {name: emb for name, emb in zip(self._node_names, embeddings)}
        logger.info(f"Built node embedding index for {len(self._node_names)} nodes")

    async def extract_question_entities(self, question: str) -> list[Entity]:
        """Extract entities from the question using LLM."""
        prompt = f"""Extract all named entities from the following question.
Return a JSON array of entities with name and type.

Question: {question}

Entity types to consider: PERSON, ORG, LOCATION, EVENT, DATE, WORK_OF_ART, PRODUCT, CONCEPT

Output format:
[
  {{"name": "Entity Name", "type": "ENTITY_TYPE"}},
  ...
]

Only include entities explicitly mentioned in the question. Return empty array if none found."""

        try:
            response = await self.client.complete_json(
                prompt,
                system_prompt="You are an expert at extracting named entities from questions. Output only valid JSON.",
                temperature=0.0,
            )

            entities = []
            for item in response if isinstance(response, list) else response.get("entities", []):
                entities.append(Entity(name=item["name"], type=item["type"]))

            logger.debug(f"Extracted {len(entities)} entities from question: {entities}")
            return entities

        except Exception as e:
            logger.error(f"Failed to extract question entities: {e}")
            return []

    def link_entities(self, question_entities: list[Entity]) -> list[LinkedEntity]:
        """Link question entities to graph nodes using embedding similarity."""
        if not self._node_names:
            self._build_node_index()

        if not self._node_names or not question_entities:
            return []

        linked = []

        for q_entity in question_entities:
            # Embed the question entity
            q_text = f"{q_entity.name} ({q_entity.type})"
            q_embedding = self.embedder.encode_single(q_text, normalize=True)

            # Compute similarities
            similarities = []
            for node_name in self._node_names:
                node_emb = self._node_embeddings.get(node_name)
                if node_emb is not None:
                    sim = float(np.dot(q_embedding, node_emb))
                    similarities.append((node_name, sim))

            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Take top candidates above threshold
            for node_name, sim in similarities[:self.max_candidates]:
                if sim >= self.similarity_threshold:
                    entity = self.graph_store.get_entity(node_name)
                    linked.append(LinkedEntity(
                        question_entity=q_entity.name,
                        graph_node=node_name,
                        similarity=sim,
                        entity_type=entity.get("type") if entity else None,
                    ))

        logger.debug(f"Linked {len(linked)} question entities to graph nodes")
        return linked

    def _link_by_string_match(self, question: str) -> list[LinkedEntity]:
        """Link graph nodes to the question by direct surface-form matching.

        Used as a fallback when LLM entity extraction yields nothing (e.g. the
        mock client). Matches each graph node name against the question using
        case-insensitive substring and token-boundary comparison.

        Args:
            question: Raw question text.

        Returns:
            Linked entities for every graph node mentioned in the question.
        """
        if not self._node_names:
            self._build_node_index()

        if not self._node_names:
            return []

        question_lower = question.lower()
        linked: list[LinkedEntity] = []

        for node_name in self._node_names:
            name_lower = node_name.lower()
            if not name_lower:
                continue

            # Substring match, requiring non-alphanumeric boundaries so that
            # "Paris" does not match inside "Comparison".
            start = question_lower.find(name_lower)
            while start != -1:
                before = question_lower[start - 1] if start > 0 else " "
                after_index = start + len(name_lower)
                after = question_lower[after_index] if after_index < len(question_lower) else " "

                if not before.isalnum() and not after.isalnum():
                    entity = self.graph_store.get_entity(node_name)
                    linked.append(LinkedEntity(
                        question_entity=node_name,
                        graph_node=node_name,
                        similarity=1.0,
                        entity_type=entity.get("type") if entity else None,
                    ))
                    break

                start = question_lower.find(name_lower, start + 1)

        logger.debug(f"String-matched {len(linked)} graph nodes in question")
        return linked

    async def link_question(self, question: str) -> list[LinkedEntity]:
        """Full pipeline: extract entities from question and link to graph.

        Falls back to lexical surface-form matching when LLM extraction returns
        no entities, so graph retrieval still works without a real LLM.

        Args:
            question: Question to link.

        Returns:
            Entities linked to graph nodes, or an empty list if none matched.
        """
        question_entities = await self.extract_question_entities(question)
        linked = self.link_entities(question_entities)

        if not linked:
            linked = self._link_by_string_match(question)

        return linked


def create_query_linker(graph_store: GraphStore) -> QueryEntityLinker:
    """Factory function to create query linker from config."""
    config = get_config()
    return QueryEntityLinker(
        graph_store=graph_store,
        similarity_threshold=config.retrieval.graph_max_hops * 0.1 + 0.6,  # heuristic
        max_candidates=5,
    )