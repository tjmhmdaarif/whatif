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
    if config.ensure_template():
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
    from whatif.embeddings import OllamaError, build_client

    ollama_ok = False
    models_ok = False
    try:
        client = build_client(config)
        available = client.list_models()
        ollama_ok = True
        models_ok = client.has_model(config.models.llm_model) and client.has_model(
            config.models.embed_model
        )
    except OllamaError as e:
        logger.debug("Ollama check failed: %s", e)

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
    logger.info("Indexing decisions from %s", config.vault.decisions_path)

    from whatif.embeddings import OllamaError
    from whatif.store import index_decisions

    try:
        stats = index_decisions(config)
    except OllamaError as exc:
        logger.error("Ollama unavailable: %s", exc)
        print(f"Ollama unavailable: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.error("Indexing failed: %s", exc)
        if getattr(args, "debug", False):
            raise
        print(f"Indexing failed: {exc}", file=sys.stderr)
        return 1

    for error in stats.errors:
        print(f"  FAILED: {error}", file=sys.stderr)

    print(f"Indexed: {stats.indexed}  Skipped: {stats.skipped}  Failed: {stats.failed}")
    return 0 if stats.failed == 0 else 1


def cmd_brief(args: argparse.Namespace) -> int:
    """Generate a decision brief for a note."""
    config = load_config(args.config, args.vault)

    from whatif.brief import BriefGenerator
    from whatif.embeddings import OllamaError, build_client
    from whatif.notes import parse_decision_file
    from whatif.retrieve import Retriever
    from whatif.store import DecisionStore

    note_path = _resolve_note_path(args.note, config)
    if note_path is None:
        return 1

    result = parse_decision_file(note_path)
    if result.decision is None:
        print(f"Could not parse the note: {result.error}", file=sys.stderr)
        return 1
    decision = result.decision

    if not config.vault.db_path.exists():
        print(
            f"No index found at {config.vault.db_path}. Run 'whatif index' first.",
            file=sys.stderr,
        )
        return 1

    store = DecisionStore(config.vault.db_path)
    try:
        client = build_client(config)
        embedder_client = build_client(config)
        embedder_client.model = config.models.embed_model
        retriever = Retriever(store, embedder=embedder_client.embed)

        generator = BriefGenerator(
            config, client, retriever, note_bodies=store.get_note_bodies()
        )
        brief = generator.generate(decision, k=args.k)
    except OllamaError as exc:
        print(f"Ollama unavailable: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.error("Brief generation failed: %s", exc)
        if getattr(args, "debug", False):
            raise
        print(f"Brief generation failed: {exc}", file=sys.stderr)
        return 1
    finally:
        store.close()

    out_path = args.out or (config.vault.briefs_path / f"{decision.id}.md")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(brief.content, encoding="utf-8")

    print(f"Brief written to {out_path}")
    print(
        f"  retrieved {len(brief.similar)} similar decisions, "
        f"{len(brief.failed_assumptions)} past assumption failures"
    )
    if brief.validation and not brief.validation.passed:
        print(
            f"  {len(brief.validation.issues)} claim(s) marked [UNVERIFIED] "
            f"after {brief.retries} retry attempt(s)"
        )
    return 0


def _resolve_note_path(note: str, config) -> Path | None:
    """Find a note by path, by id, or by id prefix inside the Decisions folder."""
    candidate = Path(note).expanduser()
    if candidate.exists() and candidate.is_file():
        return candidate.resolve()

    decisions_dir = config.vault.decisions_path
    if decisions_dir.exists():
        matches = sorted(decisions_dir.glob(f"{note}*.md"))
        if len(matches) == 1:
            return matches[0].resolve()
        if len(matches) > 1:
            print(
                f"More than one note matches '{note}':\n"
                + "\n".join(f"  {m.name}" for m in matches),
                file=sys.stderr,
            )
            return None

    print(
        f"Note not found: {note}\n"
        f"Looked for a file at {candidate} and for notes in {decisions_dir}",
        file=sys.stderr,
    )
    return None


def cmd_canvas(args: argparse.Namespace) -> int:
    """Generate an Obsidian Canvas for a decision."""
    config = load_config(args.config, args.vault)

    from whatif.canvas import write_canvas
    from whatif.notes import parse_decision_file

    note_path = _resolve_note_path(args.note, config)
    if note_path is None:
        return 1

    result = parse_decision_file(note_path)
    if result.decision is None:
        print(f"Could not parse the note: {result.error}", file=sys.stderr)
        return 1

    try:
        out_path = write_canvas(result.decision, config, args.out)
    except Exception as exc:  # noqa: BLE001
        logger.error("Canvas export failed: %s", exc)
        if getattr(args, "debug", False):
            raise
        print(f"Canvas export failed: {exc}", file=sys.stderr)
        return 1

    print(f"Canvas written to {out_path}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Run the leave-one-out evaluation."""
    config = load_config(args.config, args.vault)

    from whatif.embeddings import OllamaError, build_client
    from whatif.eval import check_model_consistency, format_report, run_eval, write_report
    from whatif.store import DecisionStore

    if not config.vault.db_path.exists():
        print(
            f"No index found at {config.vault.db_path}. Run 'whatif index' first.",
            file=sys.stderr,
        )
        return 1

    store = DecisionStore(config.vault.db_path)
    try:
        client = build_client(config)
        if not client.is_reachable():
            raise OllamaError(
                f"Cannot reach Ollama at {config.ollama.host}. "
                "Start the server with 'ollama serve' and re-run."
            )

        report = run_eval(
            config,
            store,
            embedder=client.embed,
            threshold=args.threshold,
        )
        report.model_warning = check_model_consistency(store, config)
    except OllamaError as exc:
        print(f"Ollama unavailable: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.error("Evaluation failed: %s", exc)
        if getattr(args, "debug", False):
            raise
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1
    finally:
        store.close()

    print(format_report(report))

    out_path = args.out or (config.vault.whatif_path / "eval-results.json")
    write_report(report, out_path)
    print(f"Results written to {out_path}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Run the local web interface."""
    from whatif.webapp import serve

    return serve(
        config_path=args.config,
        vault_path=args.vault,
        host=args.host,
        port=args.port,
    )


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
    p_canvas.add_argument("note", help="Path to decision note or its id")
    p_canvas.add_argument("--out", type=Path, help="Output path (default: Canvases/<id>.canvas)")
    p_canvas.set_defaults(func=cmd_canvas)

    # eval
    p_eval = subparsers.add_parser("eval", help="Run leave-one-out evaluation")
    add_common_args(p_eval)
    p_eval.add_argument(
        "--threshold",
        type=float,
        help="Similarity a hit must reach (default: from whatif.toml)",
    )
    p_eval.add_argument(
        "--out", type=Path, help="Output path (default: .whatif/eval-results.json)"
    )
    p_eval.set_defaults(func=cmd_eval)

    # serve
    p_serve = subparsers.add_parser("serve", help="Run the local web interface")
    add_common_args(p_serve)
    p_serve.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=8765, help="Port to listen on")
    p_serve.set_defaults(func=cmd_serve)

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