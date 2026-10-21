"""Shared test fixtures and configuration."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def vault_dir(temp_dir: Path) -> Path:
    """Create a vault directory structure."""
    vault = temp_dir / "test_vault"
    (vault / "Decisions").mkdir(parents=True)
    (vault / "Briefs").mkdir(parents=True)
    (vault / "Canvases").mkdir(parents=True)
    (vault / "Templates").mkdir(parents=True)
    (vault / ".whatif").mkdir(parents=True)
    return vault


@pytest.fixture
def config_path(vault_dir: Path) -> Path:
    """Create a default config file."""
    config_file = vault_dir / "whatif.toml"
    config_file.write_text(f"""[vault]
path = "{vault_dir}"
decisions_dir = "Decisions"
briefs_dir = "Briefs"
canvases_dir = "Canvases"
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
legend = "Legend"

[eval]
leave_one_out = true

[logging]
level = "DEBUG"
format = "%(levelname)s | %(name)s | %(message)s"
""")
    return config_file


@pytest.fixture
def sample_decision() -> str:
    """A sample decision note for testing."""
    return """---
id: 2026-01-15-test-decision
type: decision
status: decided
created: 2026-01-15
domain: study
options: [Option A, Option B]
chosen: Option A
criteria:
  - {name: cost, weight: 3}
  - {name: quality, weight: 2}
scores:
  Option A: {cost: 4, quality: 3}
  Option B: {cost: 2, quality: 5}
assumptions: [Assumption 1, Assumption 2]
risks: [Risk 1]
outcome:
  reviewed_on: 2026-06-15
  result: better
  failed_assumptions: [Assumption 1]
tags: [test]
---
## Context
Test context.

## Options
Option A details.
Option B details.

## Decision
Chose Option A.

## Outcome
Turned out well.
"""


@pytest.fixture
def sample_decision_open() -> str:
    """A sample open decision note for testing."""
    return """---
id: 2026-10-21-new-decision
type: decision
status: open
created: 2026-10-21
domain: tools
options: [Tool X, Tool Y]
chosen:
criteria:
  - {name: price, weight: 2}
  - {name: features, weight: 3}
scores:
  Tool X: {price: 3, features: 4}
  Tool Y: {price: 5, features: 2}
assumptions: [Prices will drop, Tool Y will add features]
risks: [Vendor lock-in]
outcome:
  reviewed_on:
  result:
  failed_assumptions: []
tags: [evaluation]
---
## Context
Need to choose a tool.

## Options
Tool X is cheaper but fewer features.
Tool Y is expensive but feature-rich.

## Decision

## Outcome
"""


# Pytest configuration
def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow tests")