"""Ablation studies for GraphRAG."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..config import get_config
from ..data.loader import load_hotpotqa, Corpus
from ..eval.runner import UnifiedRAGPipeline, run_evaluation
from ..eval.metrics import (
    EvaluationResult,
    AggregateMetrics,
    aggregate_results,
    bootstrap_confidence_interval,
)
from ..graph import create_graph_store
from ..logging_setup import get_logger, setup_logging

logger = get_logger(__name__)


async def run_ablation(
    name: str,
    examples: list[Any],
    corpus: Corpus,
    pipeline_config: dict[str, Any],
    output_dir: Path,
    rebuild: bool = True,
) -> tuple[list[EvaluationResult], AggregateMetrics]:
    """Run a single ablation configuration."""
    logger.info(f"Running ablation: {name}")

    graph_store = create_graph_store() if pipeline_config.get("mode") in ("graph", "hybrid") else None
    pipeline = UnifiedRAGPipeline(graph_store=graph_store, **pipeline_config)

    if rebuild:
        pipeline.build_index(corpus)
        pipeline.save_index("data/faiss_index")
    else:
        pipeline.load_index("data/faiss_index")

    output_path = output_dir / f"ablation_{name}.json"
    results, aggregate = await run_evaluation(pipeline, examples, output_path=output_path)

    return results, aggregate


async def ablation_hops(
    examples: list[Any],
    corpus: Corpus,
    output_dir: Path,
    rebuild: bool = True,
) -> dict[str, AggregateMetrics]:
    """Ablation: number of graph hops."""
    logger.info("Running hops ablation...")

    hops_list = [1, 2, 3]
    results = {}

    for hops in hops_list:
        name = f"hops_{hops}"
        config = {"mode": "hybrid", "max_hops": hops}
        _, agg = await run_ablation(name, examples, corpus, config, output_dir, rebuild)
        results[name] = agg
        rebuild = False  # Only rebuild once

    return results


async def ablation_top_k(
    examples: list[Any],
    corpus: Corpus,
    output_dir: Path,
    rebuild: bool = True,
) -> dict[str, AggregateMetrics]:
    """Ablation: vector top-k."""
    logger.info("Running top-k ablation...")

    k_list = [3, 5, 10]
    results = {}

    for k in k_list:
        name = f"topk_{k}"
        config = {"mode": "hybrid", "top_k": k}
        _, agg = await run_ablation(name, examples, corpus, config, output_dir, rebuild)
        results[name] = agg
        rebuild = False

    return results


async def ablation_fusion_weight(
    examples: list[Any],
    corpus: Corpus,
    output_dir: Path,
    rebuild: bool = True,
) -> dict[str, AggregateMetrics]:
    """Ablation: fusion weight between vector and graph."""
    logger.info("Running fusion weight ablation...")

    weights = [0.0, 0.25, 0.5, 0.75, 1.0]  # 0=vector only, 1=graph only
    results = {}

    for weight in weights:
        name = f"fusion_{weight:.2f}"
        config = {"mode": "hybrid", "vector_weight": 1 - weight, "graph_weight": weight}
        _, agg = await run_ablation(name, examples, corpus, config, output_dir, rebuild)
        results[name] = agg
        rebuild = False

    return results


async def ablation_entity_resolution(
    examples: list[Any],
    corpus: Corpus,
    output_dir: Path,
    rebuild: bool = True,
) -> dict[str, AggregateMetrics]:
    """Ablation: with and without entity resolution."""
    logger.info("Running entity resolution ablation...")

    # This would require modifying the GraphBuilder to skip resolution
    # For now, we'll just run hybrid mode
    config = {"mode": "hybrid"}
    _, agg = await run_ablation("with_er", examples, corpus, config, output_dir, rebuild)

    # Without entity resolution would need a custom GraphBuilder
    # We'll skip for now as it requires more infrastructure

    return {"with_er": agg}


async def run_all_ablations(
    dataset: str = "hotpotqa",
    split: str = "validation",
    limit: int = 100,
    seed: int = 42,
    output_dir: str = "results/ablations",
) -> dict[str, Any]:
    """Run all ablation studies."""
    setup_logging()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load dataset
    if dataset == "hotpotqa":
        examples, corpus = load_hotpotqa(split=split, limit=limit, seed=seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    all_results = {}

    # Run ablations
    all_results["hops"] = await ablation_hops(examples, corpus, output_path, rebuild=True)
    all_results["top_k"] = await ablation_top_k(examples, corpus, output_path, rebuild=False)
    all_results["fusion_weight"] = await ablation_fusion_weight(examples, corpus, output_path, rebuild=False)
    all_results["entity_resolution"] = await ablation_entity_resolution(examples, corpus, output_path, rebuild=False)

    # Save summary
    summary = {}
    for category, results in all_results.items():
        summary[category] = {k: {
            "exact_match": v.exact_match,
            "f1": v.f1,
            "supporting_fact_recall": v.supporting_fact_recall,
            "avg_latency_ms": v.avg_latency_ms,
        } for k, v in results.items()}

    with open(output_path / "ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("All ablations complete!")
    return all_results


def generate_ablation_report(results: dict[str, Any], output_path: Path) -> None:
    """Generate markdown report for ablations."""
    lines = [
        "# Ablation Study Results",
        "",
        "## Hops Ablation",
        "",
        "| Hops | EM | F1 | SF Recall | Latency (ms) |",
        "|------|-----|-----|-----------|--------------|",
    ]

    for name, agg in results.get("hops", {}).items():
        lines.append(f"| {name} | {agg.exact_match:.4f} | {agg.f1:.4f} | {agg.supporting_fact_recall:.4f} | {agg.avg_latency_ms:.1f} |")

    lines.extend([
        "",
        "## Top-K Ablation",
        "",
        "| Top-K | EM | F1 | SF Recall | Latency (ms) |",
        "|-------|-----|-----|-----------|--------------|",
    ])

    for name, agg in results.get("top_k", {}).items():
        lines.append(f"| {name} | {agg.exact_match:.4f} | {agg.f1:.4f} | {agg.supporting_fact_recall:.4f} | {agg.avg_latency_ms:.1f} |")

    lines.extend([
        "",
        "## Fusion Weight Ablation",
        "",
        "| Weight (Graph) | EM | F1 | SF Recall | Latency (ms) |",
        "|----------------|-----|-----|-----------|--------------|",
    ])

    for name, agg in results.get("fusion_weight", {}).items():
        weight = name.split("_")[1]
        lines.append(f"| {weight} | {agg.exact_match:.4f} | {agg.f1:.4f} | {agg.supporting_fact_recall:.4f} | {agg.avg_latency_ms:.1f} |")

    output_path.write_text("\n".join(lines))
    logger.info(f"Ablation report saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(run_all_ablations())