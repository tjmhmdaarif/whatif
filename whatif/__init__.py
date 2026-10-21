"""WhatIf - Local decision brief generator for Obsidian vaults."""

__version__ = "0.1.0"

from whatif.config import Config, load_config, create_default_config, write_default_config

__all__ = [
    "Config",
    "load_config",
    "create_default_config",
    "write_default_config",
]