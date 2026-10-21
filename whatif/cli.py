"""Command-line interface for WhatIf."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from whatif.config import load_config, create_default_config, write_default_config

logger = logging.getLogger(__name__)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments to a subparser."""
    parser.add_argument(
        "--vault",
        type=Path,
        help="Path to Obsidian vault root (overrides config)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to whatif.toml (default: ./whatif.toml or ~/.config/whatif/whatif.toml)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable INFO logging",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG logging",
    )


def setup_logging_from_args(args: argparse.Namespace) -> None:
    """Configure logging based on CLI args."""
    if args.debug:
        level = logging.DEBUG
    elif args.verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(levelname)s | %(name)s | %(message)s",
        force=True,
    )


def cmd_init(args: argparse.Namespace) -> int:
    """Initialize a new WhatIf vault."""
    vault_path = args.vault or Path.cwd()
    vault_path = Path(vault_path).expanduser().resolve()

    config_path = vault_path / "whatif.toml"
    if config_path.exists() and not args.force:
        logger.error(f"Config already exists at {config_path}. Use --force to overwrite.")
        return 1

    write_default_config(config_path, vault_path)
    logger.info(f"Created {config_path}")

    # Create directory structure
    config = create_default_config(vault_path)
    logger.info(f"Created vault directories at {vault_path}")

    # Create template
    template_path = config.vault.templates_path
    if not template_path.exists():
        template_content = """---
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
        template_path.write_text(template_content, encoding="utf-8")
        logger.info(f"Created template at {template_path}")

    print(f"Initialized WhatIf vault at {vault_path}")
    print(f"Config: {config_path}")
    print(f"Template: {template_path}")
    print("\nNext steps:")
    print("  1. Add decision notes to Decisions/")
    print("  2. Run 'whatif index' to build the index")
    print("  3. Run 'whatif brief <note>' to generate a brief")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Run health checks on the WhatIf setup."""
    config = load_config(args.config, args.vault)

    checks = []

    # Check vault structure
    checks.append(("Vault root", config.vault.path.exists(), str(config.vault.path)))
    checks.append(("Decisions dir", config.vault.decisions_path.exists(), str(config.vault.decisions_path)))
    checks.append(("Briefs dir", config.vault.briefs_path.exists(), str(config.vault.briefs_path)))
    checks.append(("Canvases dir", config.vault.canvases_path.exists(), str(config.vault.canvases_path)))
    checks.append(("Template", config.vault.templates_path.exists(), str(config.vault.templates_path)))
    checks.append(("WhatIf dir", config.vault.whatif_path.exists(), str(config.vault.whatif_path)))

    # Check Ollama
    ollama_ok = False
    models_ok = False
    try:
        import urllib.request
        import json

        # Check Ollama server
        req = urllib.request.Request(f"{config.ollama.host}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.load(resp)
            models = {m["name"] for m in data.get("models", [])}
            ollama_ok = True
            models_ok = config.models.llm_model in models and config.models.embed_model in models
    except Exception as e:
        logger.debug(f"Ollama check failed: {e}")

    checks.append(("Ollama server", ollama_ok, config.ollama.host))
    checks.append((f"Model: {config.models.llm_model}", models_ok, ""))
    checks.append((f"Model: {config.models.embed_model}", models_ok, ""))

    # Check database
    db_exists = config.vault.db_path.exists()
    checks.append(("Index database", db_exists, str(config.vault.db_path)))

    # Print results
    all_ok = True
    for name, ok, detail in checks:
        status = "OK" if ok else "FAIL"
        if not ok:
            all_ok = False
        detail_str = f" ({detail})" if detail else ""
        print(f"  [{status}] {name}{detail_str}")

    if all_ok:
        print("\nAll checks passed.")
        return 0
    else:
        print("\nSome checks failed. Run with --debug for details.")
        return 1


def cmd_index(args: argparse.Namespace) -> int:
    """Index decision notes into SQLite."""
    config = load_config(args.config, args.vault)
    logger.info(f"Indexing decisions from {config.vault.decisions_path}")

    # TODO: Implement in M3
    logger.warning("Index command not yet implemented (M3)")
    return 0


def cmd_brief(args: argparse.Namespace) -> int:
    """Generate a decision brief for a note."""
    config = load_config(args.config, args.vault)

    note_path = Path(args.note).expanduser().resolve()
    if not note_path.exists():
        logger.error(f"Note not found: {note_path}")
        return 1

    logger.info(f"Generating brief for {note_path}")

    # TODO: Implement in M6
    logger.warning("Brief command not yet implemented (M6)")
    return 0


def cmd_canvas(args: argparse.Namespace) -> int:
    """Generate an Obsidian Canvas for a decision."""
    config = load_config(args.config, args.vault)

    note_path = Path(args.note).expanduser().resolve()
    if not note_path.exists():
        logger.error(f"Note not found: {note_path}")
        return 1

    logger.info(f"Generating canvas for {note_path}")

    # TODO: Implement in M7
    logger.warning("Canvas command not yet implemented (M7)")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Run leave-one-out evaluation."""
    config = load_config(args.config, args.vault)
    logger.info("Running leave-one-out evaluation")

    # TODO: Implement in M8
    logger.warning("Eval command not yet implemented (M8)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="whatif",
        description="WhatIf - Local decision brief generator for Obsidian vaults",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version="whatif 0.1.0")

    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    # init
    p_init = subparsers.add_parser("init", help="Initialize a new WhatIf vault")
    add_common_args(p_init)
    p_init.add_argument("--force", action="store_true", help="Overwrite existing config")
    p_init.set_defaults(func=cmd_init)

    # doctor
    p_doctor = subparsers.add_parser("doctor", help="Run health checks")
    add_common_args(p_doctor)
    p_doctor.set_defaults(func=cmd_doctor)

    # index
    p_index = subparsers.add_parser("index", help="Index decision notes")
    add_common_args(p_index)
    p_index.set_defaults(func=cmd_index)

    # brief
    p_brief = subparsers.add_parser("brief", help="Generate a decision brief")
    add_common_args(p_brief)
    p_brief.add_argument("note", help="Path to decision note")
    p_brief.add_argument("--k", type=int, default=5, help="Number of similar decisions to retrieve")
    p_brief.add_argument("--out", type=Path, help="Output path (default: Briefs/<id>.md)")
    p_brief.set_defaults(func=cmd_brief)

    # canvas
    p_canvas = subparsers.add_parser("canvas", help="Generate an Obsidian Canvas")
    add_common_args(p_canvas)
    p_canvas.add_argument("note", help="Path to decision note")
    p_canvas.set_defaults(func=cmd_canvas)

    # eval
    p_eval = subparsers.add_parser("eval", help="Run leave-one-out evaluation")
    add_common_args(p_eval)
    p_eval.set_defaults(func=cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Setup logging early for --debug/--verbose
    if hasattr(args, "debug") and args.debug:
        setup_logging_from_args(args)
    elif hasattr(args, "verbose") and args.verbose:
        setup_logging_from_args(args)

    try:
        return args.func(args)
    except KeyboardInterrupt:
        logger.info("Interrupted")
        return 130
    except Exception as e:
        logger.exception("Unhandled error: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())