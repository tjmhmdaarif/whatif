"""Graph models - re-export from extraction for convenience."""

from ..extraction.models import Entity, Relation, ExtractionResult, EXTRACTION_SCHEMA

__all__ = ["Entity", "Relation", "ExtractionResult", "EXTRACTION_SCHEMA"]