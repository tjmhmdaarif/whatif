"""Pydantic models for entity and relation extraction."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Entity(BaseModel):
    """Extracted entity."""
    name: str = Field(..., description="Entity name")
    type: str = Field(..., description="Entity type (PERSON, ORG, LOCATION, etc.)")
    description: str = Field(default="", description="Brief description of the entity")


class Relation(BaseModel):
    """Extracted relation between entities."""
    source: str = Field(..., description="Source entity name")
    target: str = Field(..., description="Target entity name")
    relation: str = Field(..., description="Relation type (FOUNDED, LOCATED_IN, etc.)")
    evidence: str = Field(default="", description="Text evidence from the chunk")


class ExtractionResult(BaseModel):
    """Result of entity/relation extraction from a chunk."""
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)


# JSON Schema for LLM output validation
EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "description": {"type": "string"}
                },
                "required": ["name", "type"]
            }
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "relation": {"type": "string"},
                    "evidence": {"type": "string"}
                },
                "required": ["source", "target", "relation"]
            }
        }
    },
    "required": ["entities", "relations"]
}