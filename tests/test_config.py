"""Tests for config.py."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from whatif.config import (
    Config,
    VaultConfig,
    ModelConfig,
    OllamaConfig,
    IndexConfig,
    RetrievalConfig,
    BriefConfig,
    EvalConfig,
    LoggingConfig,
    load_config,
    create_default_config,
    write_default_config,
)


class TestVaultConfig:
    def test_defaults(self) -> None:
        vc = VaultConfig()
        assert vc.path == Path(".").resolve()
        assert vc.decisions_dir == "Decisions"
        assert vc.briefs_dir == "Briefs"
        assert vc.canvases_dir == "Canvases"
        assert vc.whatif_dir == ".whatif"

    def test_paths_resolved(self) -> None:
        vc = VaultConfig(path="/tmp/test")
        assert vc.path == Path("/tmp/test").resolve()
        assert vc.decisions_path == vc.path / "Decisions"
        assert vc.briefs_path == vc.path / "Briefs"
        assert vc.canvases_path == vc.path / "Canvases"
        assert vc.whatif_path == vc.path / ".whatif"
        assert vc.db_path == vc.path / ".whatif" / "index.sqlite"
        assert vc.templates_path == vc.path / "Templates" / "decision.md"


class TestModelConfig:
    def test_defaults(self) -> None:
        mc = ModelConfig()
        assert mc.llm_model == "llama3.1"
        assert mc.embed_model == "nomic-embed-text"


class TestOllamaConfig:
    def test_defaults(self) -> None:
        oc = OllamaConfig()
        assert oc.host == "http://localhost:11434"
        assert oc.timeout == 120


class TestRetrievalConfig:
    def test_defaults(self) -> None:
        rc = RetrievalConfig()
        assert rc.default_k == 5
        assert rc.similarity_threshold == 0.3
        assert rc.failed_assumption_threshold == 0.6


class TestConfigFromDict:
    def test_empty_dict(self) -> None:
        config = Config.from_dict({})
        assert isinstance(config.vault, VaultConfig)
        assert isinstance(config.models, ModelConfig)
        assert isinstance(config.ollama, OllamaConfig)
        assert isinstance(config.index, IndexConfig)
        assert isinstance(config.retrieval, RetrievalConfig)
        assert isinstance(config.brief, BriefConfig)
        assert isinstance(config.eval, EvalConfig)
        assert isinstance(config.logging, LoggingConfig)

    def test_partial_override(self) -> None:
        config = Config.from_dict({
            "vault": {"path": "/custom/vault", "decisions_dir": "MyDecisions"},
            "models": {"llm_model": "custom-model"},
        })
        assert config.vault.path == Path("/custom/vault").resolve()
        assert config.vault.decisions_dir == "MyDecisions"
        assert config.models.llm_model == "custom-model"
        assert config.models.embed_model == "nomic-embed-text"  # default preserved


class TestConfigFromFile:
    def test_loads_toml(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("""
[vault]
path = "/test/vault"
decisions_dir = "Decisions"

[models]
llm_model = "test-model"
""")
            f.flush()
            config = Config.from_file(f.name)

        assert config.vault.path == Path("/test/vault").resolve()
        assert config.models.llm_model == "test-model"

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            Config.from_file("/nonexistent/path.toml")


class TestCreateDefaultConfig:
    def test_creates_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = create_default_config(tmpdir)
            assert config.vault.path == Path(tmpdir).resolve()
            assert config.vault.decisions_path.exists()
            assert config.vault.briefs_path.exists()
            assert config.vault.canvases_path.exists()
            assert config.vault.whatif_path.exists()
            assert config.vault.templates_path.parent.exists()


class TestWriteDefaultConfig:
    def test_writes_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "whatif.toml"
            vault_path = Path(tmpdir) / "myvault"

            write_default_config(config_path, vault_path)

            assert config_path.exists()
            content = config_path.read_text()
            assert "path = " in content
            assert "llama3.1" in content
            assert "nomic-embed-text" in content


class TestLoadConfig:
    def test_loads_with_vault_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            config_path = tmpdir_path / "whatif.toml"
            vault_path = tmpdir_path / "myvault"
            vault_path.mkdir()

            # Write config with forward slashes for TOML compatibility
            config_content = f"""[vault]
path = "{vault_path.as_posix()}"
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
level = "INFO"
format = "%(levelname)s | %(name)s | %(message)s"
"""
            config_path.write_text(config_content)

            # Override vault path
            other_vault = tmpdir_path / "othervault"
            other_vault.mkdir()

            config = load_config(config_path, other_vault)
            assert config.vault.path == other_vault.resolve()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])