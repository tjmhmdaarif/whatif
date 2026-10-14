"""GraphRAG-MultiHop: Graph-enhanced Retrieval-Augmented Generation for Multi-hop QA."""

from .config import Config, load_config, get_config, set_config
from .logging_setup import setup_logging, get_logger
from .cli import app
from .api import create_app

__version__ = "0.1.0"

__all__ = [
    "Config",
    "load_config",
    "get_config",
    "set_config",
    "setup_logging",
    "get_logger",
    "app",
    "create_app",
]