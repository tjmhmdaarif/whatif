"""Entity and relation extraction using LLM."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from ..chunking import Chunk
from ..config import get_config
from ..llm import create_llm_client
from ..logging_setup import get_logger
from .models import Entity, ExtractionResult, Relation, EXTRACTION_SCHEMA

logger = get_logger(__name__)

# Extraction prompt template
EXTRACTION_PROMPT = """You are an expert at extracting entities and relationships from text.

Extract all entities and relationships from the following text chunk. Focus on the entity types and relation types listed below.

Entity Types: {entity_types}
Relation Types: {relation_types}

Text:
{text}

Output valid JSON only with this exact structure:
{{
  "entities": [
    {{"name": "Entity Name", "type": "ENTITY_TYPE", "description": "Brief description"}}
  ],
  "relations": [
    {{"source": "Source Entity", "target": "Target Entity", "relation": "RELATION_TYPE", "evidence": "Supporting text snippet"}}
  ]
}}

Rules:
1. Only extract entities and relations explicitly mentioned in the text
2. Use exact entity names as they appear in the text
3. Use only the provided entity and relation types
4. Include evidence snippets for each relation
5. Return empty arrays if nothing is found
6. Output ONLY valid JSON, no additional text
"""


class EntityRelationExtractor:
    """Extracts entities and relations from text chunks using LLM."""

    def __init__(
        self,
        entity_types: list[str] | None = None,
        relation_types: list[str] | None = None,
        max_entities_per_chunk: int | None = None,
        max_relations_per_chunk: int | None = None,
        retry_attempts: int | None = None,
        cache_dir: str | Path | None = None,
    ):
        config = get_config()
        self.entity_types = entity_types or config.extraction.entity_types
        self.relation_types = relation_types or config.extraction.relation_types
        self.max_entities = max_entities_per_chunk or config.extraction.max_entities_per_chunk
        self.max_relations = max_relations_per_chunk or config.extraction.max_relations_per_chunk
        self.retry_attempts = retry_attempts or config.extraction.retry_attempts

        # Setup cache
        self.cache_dir = Path(cache_dir or config.llm.cache.cache_dir) / "extraction"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.client = create_llm_client()

    def _make_cache_key(self, text: str) -> str:
        """Generate cache key from text content."""
        key_data = {
            "text": text,
            "entity_types": self.entity_types,
            "relation_types": self.relation_types,
            "model": self.client.model_name,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]

    def _get_cached(self, cache_key: str) -> ExtractionResult | None:
        """Get cached extraction result."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
            return ExtractionResult(**data)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None

    def _save_cache(self, cache_key: str, result: ExtractionResult) -> None:
        """Save extraction result to cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump(result.model_dump(), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _extract_with_retry(self, prompt: str) -> ExtractionResult:
        """Extract with retry logic."""
        response = await self.client.complete_json(
            prompt,
            system_prompt="You are an expert at extracting entities and relationships from text. Output only valid JSON.",
            temperature=0.0,
            schema=EXTRACTION_SCHEMA,
        )

        # Validate and truncate
        entities = [Entity(**e) for e in response.get("entities", [])][:self.max_entities]
        relations = [Relation(**r) for r in response.get("relations", [])][:self.max_relations]

        return ExtractionResult(entities=entities, relations=relations)

    async def extract(self, chunk: Chunk) -> ExtractionResult:
        """Extract entities and relations from a chunk."""
        cache_key = self._make_cache_key(chunk.text)

        # Check cache
        cached = self._get_cached(cache_key)
        if cached:
            logger.debug(f"Cache hit for chunk {chunk.chunk_id}")
            return cached

        # Build prompt
        prompt = EXTRACTION_PROMPT.format(
            entity_types=", ".join(self.entity_types),
            relation_types=", ".join(self.relation_types),
            text=chunk.text,
        )

        try:
            result = await self._extract_with_retry(prompt)

            # Save to cache
            self._save_cache(cache_key, result)

            logger.debug(
                f"Extracted {len(result.entities)} entities, "
                f"{len(result.relations)} relations from {chunk.chunk_id}"
            )
            return result

        except Exception as e:
            logger.error(f"Extraction failed for {chunk.chunk_id}: {e}")
            return ExtractionResult(entities=[], relations=[])

    async def extract_batch(self, chunks: list[Chunk], max_concurrent: int = 5) -> dict[str, ExtractionResult]:
        """Extract from multiple chunks with concurrency control."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def extract_one(chunk: Chunk) -> tuple[str, ExtractionResult]:
            async with semaphore:
                result = await self.extract(chunk)
                return chunk.chunk_id, result

        tasks = [extract_one(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output = {}
        for chunk_id, result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch extraction failed for {chunk_id}: {result}")
                output[chunk_id] = ExtractionResult(entities=[], relations=[])
            else:
                output[chunk_id] = result

        return output


def create_extractor() -> EntityRelationExtractor:
    """Factory function to create extractor from config."""
    return EntityRelationExtractor()