"""Configuration management for GraphRAG-MultiHop."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingConfig(BaseModel):
    model_name: str = "BAAI/bge-small-en-v1.5"
    device: str = "cpu"
    batch_size: int = 32
    normalize_embeddings: bool = True


class FAISSConfig(BaseModel):
    index_path: str = "data/faiss_index"
    index_type: str = "IndexFlatIP"


class ChromaConfig(BaseModel):
    persist_directory: str = "data/chroma_db"
    collection_name: str = "graphrag_chunks"


class VectorStoreConfig(BaseModel):
    type: Literal["faiss", "chroma"] = "faiss"
    faiss: FAISSConfig = Field(default_factory=FAISSConfig)
    chroma: ChromaConfig = Field(default_factory=ChromaConfig)


class NetworkXConfig(BaseModel):
    graph_path: str = "data/knowledge_graph.graphml"
    alias_map_path: str = "data/alias_map.json"


class Neo4jConfig(BaseModel):
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = ""
    database: str = "neo4j"


class GraphStoreConfig(BaseModel):
    type: Literal["networkx", "neo4j"] = "networkx"
    networkx: NetworkXConfig = Field(default_factory=NetworkXConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)


class AnthropicConfig(BaseModel):
    model: str = "claude-3-haiku-20240307"
    api_key: str = ""
    max_tokens: int = 1024
    temperature: float = 0.0


class OllamaConfig(BaseModel):
    model: str = "llama3.1:8b"
    base_url: str = "http://localhost:11434"
    max_tokens: int = 1024
    temperature: float = 0.0


class LLMCacheConfig(BaseModel):
    enabled: bool = True
    cache_dir: str = "data/llm_cache"


class LLMConfig(BaseModel):
    provider: Literal["anthropic", "ollama", "extractive", "mock"] = "extractive"
    anthropic: AnthropicConfig = Field(default_factory=AnthropicConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    cache: LLMCacheConfig = Field(default_factory=LLMCacheConfig)


class ChunkingConfig(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 50


class RetrievalConfig(BaseModel):
    vector_top_k: int = 10
    graph_max_hops: int = 2
    graph_top_k: int = 10
    hybrid_fusion_weight: float = 0.5
    rrf_k: int = 60
    max_context_tokens: int = 4000
    deduplicate_chunks: bool = True


class ExtractionConfig(BaseModel):
    entity_types: list[str] = Field(
        default_factory=lambda: [
            "PERSON", "ORG", "LOCATION", "EVENT", "DATE",
            "WORK_OF_ART", "PRODUCT", "CONCEPT"
        ]
    )
    relation_types: list[str] = Field(
        default_factory=lambda: [
            "FOUNDED", "LOCATED_IN", "WORKS_FOR", "CREATED",
            "PART_OF", "RELATED_TO", "BORN_IN", "DIED_IN",
            "MARRIED_TO", "EDUCATED_AT"
        ]
    )
    max_entities_per_chunk: int = 20
    max_relations_per_chunk: int = 30
    retry_attempts: int = 3


class EntityResolutionConfig(BaseModel):
    embedding_similarity_threshold: float = 0.85
    string_similarity_threshold: float = 0.9
    merge_aliases: bool = True


class EvaluationConfig(BaseModel):
    dataset: Literal["hotpotqa", "musique", "2wikimultihopqa"] = "hotpotqa"
    split: str = "validation"
    limit: int = 300
    seed: int = 42
    metrics: list[str] = Field(
        default_factory=lambda: ["exact_match", "f1", "supporting_fact_recall", "llm_judge"]
    )


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True


class StreamlitConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8501


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/graphrag.log"


class Config(BaseModel):
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    graph_store: GraphStoreConfig = Field(default_factory=GraphStoreConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    entity_resolution: EntityResolutionConfig = Field(default_factory=EntityResolutionConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    streamlit: StreamlitConfig = Field(default_factory=StreamlitConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @field_validator("llm", mode="before")
    @classmethod
    def resolve_env_vars(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Resolve environment variables in LLM config."""
        if isinstance(v, dict):
            if "anthropic" in v and isinstance(v["anthropic"], dict):
                api_key = v["anthropic"].get("api_key", "")
                if api_key.startswith("${") and api_key.endswith("}"):
                    env_var = api_key[2:-1]
                    v["anthropic"]["api_key"] = os.getenv(env_var, "")
            if "ollama" in v and isinstance(v["ollama"], dict):
                pass  # No env vars for ollama currently
        return v


class Settings(BaseSettings):
    """Settings that can be overridden by environment variables."""
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: str = ""
    neo4j_password: str = ""


def load_config(config_path: str | Path = "config.yaml") -> Config:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        return Config()

    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}

    # Apply settings from environment
    settings = Settings()
    if settings.anthropic_api_key:
        data.setdefault("llm", {}).setdefault("anthropic", {})["api_key"] = settings.anthropic_api_key
    if settings.neo4j_password:
        data.setdefault("graph_store", {}).setdefault("neo4j", {})["password"] = settings.neo4j_password

    return Config(**data)


# Global config instance
_config: Config | None = None


def get_config() -> Config:
    """Get the global config instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: Config) -> None:
    """Set the global config instance."""
    global _config
    _config = config