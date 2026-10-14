"""CLI for GraphRAG-MultiHop."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .config import load_config, set_config
from .data.loader import load_hotpotqa
from .eval.runner import UnifiedRAGPipeline, main as run_eval_main, run_all_modes
from .eval.report import generate_report
from .graph import create_graph_store
from .logging_setup import setup_logging

app = typer.Typer(name="graphrag", help="GraphRAG-MultiHop CLI")
console = Console()


@app.command()
def eval(
    method: Annotated[str, typer.Option("--method", "-m", help="Retrieval method: vector, graph, hybrid, all")] = "vector",
    dataset: Annotated[str, typer.Option("--dataset", "-d", help="Dataset to evaluate on")] = "hotpotqa",
    split: Annotated[str, typer.Option("--split", "-s", help="Dataset split")] = "validation",
    limit: Annotated[int, typer.Option("--limit", "-n", help="Number of questions to evaluate")] = 50,
    seed: Annotated[int, typer.Option("--seed", help="Random seed")] = 42,
    output: Annotated[str, typer.Option("--output", "-o", help="Output directory for results")] = "results",
    rebuild: Annotated[bool, typer.Option("--rebuild/--no-rebuild", help="Rebuild index")] = True,
    config_file: Annotated[str, typer.Option("--config", "-c", help="Config file path")] = "config.yaml",
):
    """Run evaluation."""
    config = load_config(config_file)
    set_config(config)
    setup_logging()

    console.print(f"[bold]Running {method} evaluation on {dataset} ({split})[/bold]")
    console.print(f"Limit: {limit}, Seed: {seed}")

    if method == "all":
        asyncio.run(run_all_modes(
            dataset=dataset,
            split=split,
            limit=limit,
            seed=seed,
            output_dir=output,
            rebuild=rebuild,
        ))
    elif method in ("vector", "graph", "hybrid"):
        asyncio.run(run_eval_main(
            dataset=dataset,
            split=split,
            limit=limit,
            seed=seed,
            output_dir=output,
            rebuild=rebuild,
            mode=method,
        ))
    else:
        console.print(f"[red]Method '{method}' not supported[/red]")
        sys.exit(1)


@app.command()
def index(
    dataset: Annotated[str, typer.Option("--dataset", "-d")] = "hotpotqa",
    split: Annotated[str, typer.Option("--split", "-s")] = "validation",
    limit: Annotated[int, typer.Option("--limit", "-n")] = 300,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    output: Annotated[str, typer.Option("--output", "-o")] = "data/faiss_index",
    config_file: Annotated[str, typer.Option("--config", "-c")] = "config.yaml",
):
    """Build vector index from dataset."""
    config = load_config(config_file)
    set_config(config)
    setup_logging()

    console.print(f"[bold]Building index from {dataset} ({split})[/bold]")

    examples, corpus = load_hotpotqa(split=split, limit=limit, seed=seed)

    pipeline = UnifiedRAGPipeline(mode="vector")
    asyncio.run(pipeline.build_index(corpus))
    pipeline.save_index(output)

    console.print(f"[green]Index saved to {output}[/green]")


@app.command()
def query(
    question: Annotated[str, typer.Argument(help="Question to answer")],
    method: Annotated[str, typer.Option("--method", "-m")] = "vector",
    top_k: Annotated[int, typer.Option("--top-k", "-k")] = 10,
    index_path: Annotated[str, typer.Option("--index", "-i")] = "data/faiss_index",
    config_file: Annotated[str, typer.Option("--config", "-c")] = "config.yaml",
):
    """Query the system with a single question."""
    config = load_config(config_file)
    set_config(config)
    setup_logging()

    console.print(f"[bold]Query:[/bold] {question}")

    graph_store = create_graph_store() if method in ("graph", "hybrid") else None
    pipeline = UnifiedRAGPipeline(mode=method, graph_store=graph_store)
    pipeline.load_index(index_path)

    answer, chunks, latency, calls = asyncio.run(pipeline.query(question, top_k))

    console.print(f"\n[bold]Answer:[/bold] {answer}")
    console.print(f"[dim]Retrieved {len(chunks)} chunks in {latency:.1f}ms ({calls} LLM calls)[/dim]")


@app.command()
def report(
    results_dir: Annotated[str, typer.Option("--results-dir", "-r")] = "results",
    output: Annotated[str, typer.Option("--output", "-o")] = "results/REPORT.md",
    config_file: Annotated[str, typer.Option("--config", "-c")] = "config.yaml",
):
    """Generate evaluation report."""
    config = load_config(config_file)
    set_config(config)
    setup_logging()

    console.print("[bold]Generating report...[/bold]")
    generate_report(results_dir, output)
    console.print(f"[green]Report saved to {output}[/green]")


@app.command()
def serve(
    host: Annotated[str, typer.Option("--host")] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port")] = 8000,
    reload: Annotated[bool, typer.Option("--reload")] = False,
    config_file: Annotated[str, typer.Option("--config", "-c")] = "config.yaml",
):
    """Start the FastAPI server."""
    config = load_config(config_file)
    set_config(config)
    setup_logging()

    import uvicorn

    console.print(f"[bold]Starting API server on {host}:{port}[/bold]")
    uvicorn.run("src.graphrag.api:create_app", host=host, port=port, reload=reload, factory=True)


@app.command()
def demo(
    host: Annotated[str, typer.Option("--host")] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port")] = 8501,
    config_file: Annotated[str, typer.Option("--config", "-c")] = "config.yaml",
):
    """Start the Streamlit demo."""
    config = load_config(config_file)
    set_config(config)
    setup_logging()

    console.print(f"[bold]Starting Streamlit demo on {host}:{port}[/bold]")
    import subprocess
    import os

    # Get the path to the demo script
    demo_path = Path(__file__).parent.parent.parent / "streamlit_demo.py"
    if not demo_path.exists():
        console.print(f"[red]Demo script not found at {demo_path}[/red]")
        sys.exit(1)

    env = os.environ.copy()
    env["GRAPHRAG_CONFIG"] = config_file
    subprocess.run(["streamlit", "run", str(demo_path), "--server.address", host, "--server.port", str(port)], env=env)


if __name__ == "__main__":
    app()