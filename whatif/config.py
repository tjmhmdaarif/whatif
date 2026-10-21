"""Configuration loading for WhatIf.

Loads settings from whatif.toml with sensible defaults.
All paths are resolved relative to the vault root.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # Python < 3.11


@dataclass
class VaultConfig:
    """Vault directory configuration."""
    path: Path = Path(".")
    decisions_dir: str = "Decisions"
    briefs_dir: str = "Briefs"
    canvases_dir: str = "Canvases"
    templates_dir: str = "Templates"
    whatif_dir: str = ".whatif"

    def __post_init__(self) -> None:
        self.path = Path(self.path).expanduser().resolve()

    @property
    def decisions_path(self) -> Path:
        return self.path / self.decisions_dir

    @property
    def briefs_path(self) -> Path:
        return self.path / self.briefs_dir

    @property
    def canvases_path(self) -> Path:
        return self.path / self.canvases_dir

    @property
    def templates_path(self) -> Path:
        """Path to the decision note template file."""
        return self.path / self.templates_dir / "decision.md"

    @property
    def whatif_path(self) -> Path:
        return self.path / self.whatif_dir

    @property
    def db_path(self) -> Path:
        return self.whatif_path / "index.sqlite"


@dataclass
class ModelConfig:
    """LLM and embedding model configuration."""
    llm_model: str = "llama3.1"
    embed_model: str = "nomic-embed-text"


@dataclass
class OllamaConfig:
    """Ollama server configuration."""
    host: str = "http://localhost:11434"
    timeout: int = 120


@dataclass
class IndexConfig:
    """Indexing configuration."""
    content_hash_algo: str = "sha256"


@dataclass
class RetrievalConfig:
    """Retrieval configuration."""
    default_k: int = 5
    similarity_threshold: float = 0.3
    failed_assumption_threshold: float = 0.6


@dataclass
class BriefConfig:
    """Brief generation configuration."""
    max_retries: int = 1
    legend: str = ""


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    leave_one_out: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(levelname)s | %(name)s | %(message)s"


DECISION_TEMPLATE = """---
id: {{date}}-{{slug}}
type: decision
status: open
created: {{date}}
domain: other
options: []
chosen:
criteria:
  - {name: criterion1, weight: 1}
scores: {}
assumptions: []
risks: []
outcome:
  reviewed_on:
  result:
  failed_assumptions: []
tags: []
---
## Context

## Options

## Decision

## Outcome
"""


@dataclass
class Config:
    """Top-level configuration container."""
    vault: VaultConfig = field(default_factory=VaultConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    brief: BriefConfig = field(default_factory=BriefConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_file(cls, path: Path | str) -> Config:
        """Load configuration from a TOML file.

        A relative ``vault.path`` is resolved against the directory holding the
        config file, so a config can be used from any working directory.
        """
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open("rb") as f:
            data = tomllib.load(f)

        config = cls.from_dict(data)

        # Resolve a relative vault path against the config file's directory.
        raw_vault_path = Path(data.get("vault", {}).get("path", "."))
        if not raw_vault_path.is_absolute():
            config.vault.path = (path.parent / raw_vault_path).resolve()

        return config

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        """Create Config from a dictionary (parsed TOML)."""
        config = cls()

        if "vault" in data:
            config.vault = VaultConfig(**data["vault"])
        if "models" in data:
            config.models = ModelConfig(**data["models"])
        if "ollama" in data:
            config.ollama = OllamaConfig(**data["ollama"])
        if "index" in data:
            config.index = IndexConfig(**data["index"])
        if "retrieval" in data:
            config.retrieval = RetrievalConfig(**data["retrieval"])
        if "brief" in data:
            config.brief = BriefConfig(**data["brief"])
        if "eval" in data:
            config.eval = EvalConfig(**data["eval"])
        if "logging" in data:
            config.logging = LoggingConfig(**data["logging"])

        return config

    def setup_logging(self) -> None:
        """Configure logging based on config."""
        level = getattr(logging, self.logging.level.upper(), logging.INFO)
        logging.basicConfig(
            level=level,
            format=self.logging.format,
            force=True,
        )

    def ensure_directories(self) -> None:
        """Create all required directories."""
        self.vault.decisions_path.mkdir(parents=True, exist_ok=True)
        self.vault.briefs_path.mkdir(parents=True, exist_ok=True)
        self.vault.canvases_path.mkdir(parents=True, exist_ok=True)
        self.vault.whatif_path.mkdir(parents=True, exist_ok=True)
        self.vault.templates_path.parent.mkdir(parents=True, exist_ok=True)

    def ensure_template(self) -> bool:
        """Write the decision note template if it is missing.

        Returns:
            True if the template was created, False if it already existed.
        """
        template_path = self.vault.templates_path
        if template_path.exists():
            return False
        template_path.parent.mkdir(parents=True, exist_ok=True)
        template_path.write_text(DECISION_TEMPLATE, encoding="utf-8")
        return True


def load_config(config_path: Path | str | None = None, vault_path: Path | str | None = None) -> Config:
    """Load configuration with optional overrides.

    Args:
        config_path: Path to whatif.toml. If None, searches standard locations.
        vault_path: Override vault path from command line.

    Returns:
        Loaded and validated Config object.
    """
    if config_path is None:
        # Search for config file
        candidates = [
            Path.cwd() / "whatif.toml",
            Path.home() / ".config" / "whatif" / "whatif.toml",
        ]
        for c in candidates:
            if c.exists():
                config_path = c
                break
        else:
            config_path = Path.cwd() / "whatif.toml"

    config = Config.from_file(config_path)

    if vault_path is not None:
        config.vault.path = Path(vault_path).expanduser().resolve()

    config.ensure_directories()
    # Never overwrites an existing template; a vault pointed at for the first time
    # still gets a usable note template.
    config.ensure_template()
    config.setup_logging()

    return config


def create_default_config(vault_path: Path | str) -> Config:
    """Create a default configuration for a new vault."""
    config = Config()
    config.vault.path = Path(vault_path).expanduser().resolve()
    config.ensure_directories()
    config.ensure_template()
    config.setup_logging()
    return config


def write_default_config(path: Path | str, vault_path: Path | str) -> None:
    """Write a default whatif.toml to the given path."""
    path = Path(path).expanduser().resolve()
    vault_path = Path(vault_path).expanduser().resolve()

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Use forward slashes for TOML compatibility (backslashes are escape chars)
    vault_path_str = vault_path.as_posix()

    content = f"""# WhatIf configuration
# All paths are relative to the vault root unless absolute

[vault]
path = "{vault_path_str}"
decisions_dir = "Decisions"
briefs_dir = "Briefs"
canvases_dir = "Canvases"
templates_dir = "Templates"
whatif_dir = ".whatif"

[models]
llm_model = "llama3.1"
embed_model = "nomic-embed-text"

[ollama]
host = "http://localhost:11434"
timeout = 120

[index]
content_hash_algo = "sha256"

[retrieval]
default_k = 5
similarity_threshold = 0.3
failed_assumption_threshold = 0.6

[brief]
max_retries = 1
legend = \"\"\"Legend:
  [EVIDENCE: note-id] — Claim supported by a retrieved decision note
  [ESTIMATE]          — Reasoned judgment without direct evidence
  [UNVERIFIED]        — Claim that failed tag validation after retry
  COMPUTED            — Values calculated from your scores and weights (never LLM-generated)\"\"\"

[eval]
leave_one_out = true

[logging]
level = "INFO"
format = "%(levelname)s | %(name)s | %(message)s"
"""

    path.write_text(content, encoding="utf-8")