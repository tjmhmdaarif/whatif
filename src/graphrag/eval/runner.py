"""Unified evaluation runner for all GraphRAG retrieval modes."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..chunking import create_chunker
from ..config import get_config
from ..data.loader import load_hotpotqa, Corpus
from ..generation import generate_answer, generate_answer_with_graph_context
from ..graph import create_graph_store, GraphBuilder
from ..llm import create_llm_client
from ..logging_setup import get_logger, setup_logging
from ..retrieval import (
    create_vector_store, get_embedder, create_hybrid_retriever,
    RetrievalResult
)
from .metrics import (
    EvaluationResult,
    AggregateMetrics,
    aggregate_results,
    bootstrap_confidence_interval,
    evaluate_single,
)

logger = get_logger(__name__)


class UnifiedRAGPipeline:
    """Unified pipeline supporting vector, graph, and hybrid retrieval."""

    def __init__(
        self,
        mode: Literal["vector", "graph", "hybrid"] = "hybrid",
        vector_store=None,
        graph_store=None,
        embedder=None,
        chunker=None,
    ):
        self.mode = mode
        self.vector_store = vector_store if vector_store is not None else create_vector_store()
        self.graph_store = graph_store
        self.embedder = embedder or get_embedder()
        self.chunker = chunker or create_chunker()
        self.client = create_llm_client()

        # Build chunk_id to title mapping
        self.chunk_id_to_title: dict[str, str] = {}

        # Initialize retriever
        self.retriever = create_hybrid_retriever(
            vector_store=self.vector_store,
            graph_store=self.graph_store,
        )

    async def build_index(self, corpus: Corpus) -> None:
        """Build both vector and graph indices from corpus."""
        logger.info(f"Building indices for {self.mode} mode...")

        # Chunk all documents
        all_chunks = self.chunker.chunk_corpus(corpus)

        # Build chunk_id to title mapping
        self.chunk_id_to_title = {chunk.chunk_id: chunk.doc_id for chunk in all_chunks}

        # Generate embeddings
        texts = [chunk.text for chunk in all_chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=True)

        # Add to vector store
        self.vector_store.add_chunks(all_chunks, embeddings)
        logger.info(f"Vector index built with {len(all_chunks)} chunks")

        # Build graph if needed
        if self.mode in ("graph", "hybrid") and self.graph_store:
            logger.info("Building knowledge graph...")
            builder = GraphBuilder(graph_store=self.graph_store)
            await builder.build_from_chunks(all_chunks, max_concurrent=5)
            builder.save()
            stats = builder.get_stats()
            logger.info(f"Graph built: {stats['num_nodes']} nodes, {stats['num_edges']} edges")

    def save_index(self, path: str | Path) -> None:
        """Save the vector index."""
        self.vector_store.save(path)

    def load_index(self, path: str | Path) -> None:
        """Load the vector index and rebuild chunk_id_to_title mapping."""
        self.vector_store.load(path)
        # Rebuild chunk_id_to_title from loaded chunks
        self.chunk_id_to_title = {chunk_id: chunk.doc_id for chunk_id, chunk in self.vector_store.chunks.items()}
        logger.info(f"Loaded index with {len(self.chunk_id_to_title)} chunks")

    async def query(
        self,
        question: str,
        top_k: int | None = None,
        max_hops: int | None = None,
        mode: str | None = None,
    ) -> tuple[str, list[str], float, int]:
        """Answer a question using the configured retrieval mode."""
        config = get_config()
        top_k = top_k or config.retrieval.vector_top_k
        max_hops = max_hops or config.retrieval.graph_max_hops
        max_context_tokens = config.retrieval.max_context_tokens
        query_mode = mode or self.mode

        start_time = time.perf_counter()

        # Retrieve using hybrid retriever
        retrieval_result: RetrievalResult = await self.retriever.retrieve(
            question,
            mode=query_mode,
            vector_top_k=top_k,
            graph_max_hops=max_hops,
        )

        # Generate answer
        if query_mode == "vector":
            answer_result = await generate_answer(
                question,
                retrieval_result.chunks,
                max_context_tokens,
            )
        else:
            answer_result = await generate_answer_with_graph_context(
                question,
                retrieval_result,
                max_context_tokens,
            )

        latency_ms = (time.perf_counter() - start_time) * 1000

        return (
            answer_result.answer,
            [c.chunk_id for c in retrieval_result.chunks],
            latency_ms,
            answer_result.prompt_tokens + answer_result.completion_tokens,
        )


async def run_evaluation(
    pipeline: UnifiedRAGPipeline,
    examples: list[Any],
    limit: int | None = None,
    output_path: str | Path | None = None,
) -> tuple[list[EvaluationResult], AggregateMetrics]:
    """Run evaluation on a set of examples."""
    if limit:
        examples = examples[:limit]

    logger.info(f"Running {pipeline.mode} evaluation on {len(examples)} examples...")

    results = []
    for i, example in enumerate(examples):
        logger.info(f"Evaluating {i+1}/{len(examples)}: {example.id}")

        try:
            answer, retrieved_chunk_ids, latency_ms, token_count = await pipeline.query(example.question)

            result = evaluate_single(
                example, answer, retrieved_chunk_ids,
                pipeline.chunk_id_to_title, latency_ms, 1
            )

            results.append(result)

        except Exception as e:
            logger.error(f"Error evaluating {example.id}: {e}")
            results.append(EvaluationResult(
                question_id=example.id,
                question=example.question,
                predicted_answer="ERROR",
                gold_answer=example.answer,
                exact_match=False,
                f1_score=0.0,
                supporting_fact_recall=0.0,
                retrieved_chunk_ids=[],
                gold_supporting_fact_titles=[f[0] for f in example.supporting_facts],
                latency_ms=0.0,
                llm_calls=0,
            ))

    # Aggregate
    aggregate = aggregate_results(results)

    # Compute confidence intervals
    em_ci = bootstrap_confidence_interval(results, lambda r: np.mean([x.exact_match for x in r]))
    f1_ci = bootstrap_confidence_interval(results, lambda r: np.mean([x.f1_score for x in r]))
    sf_ci = bootstrap_confidence_interval(results, lambda r: np.mean([x.supporting_fact_recall for x in r]))

    aggregate.em_ci = em_ci
    aggregate.f1_ci = f1_ci
    aggregate.sf_ci = sf_ci

    # Save results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "mode": pipeline.mode,
            "aggregate": {
                "exact_match": aggregate.exact_match,
                "f1": aggregate.f1,
                "supporting_fact_recall": aggregate.supporting_fact_recall,
                "avg_latency_ms": aggregate.avg_latency_ms,
                "avg_llm_calls": aggregate.avg_llm_calls,
                "num_questions": aggregate.num_questions,
                "em_ci": list(aggregate.em_ci) if hasattr(aggregate, 'em_ci') else None,
                "f1_ci": list(aggregate.f1_ci) if hasattr(aggregate, 'f1_ci') else None,
                "sf_ci": list(aggregate.sf_ci) if hasattr(aggregate, 'sf_ci') else None,
            },
            "results": [asdict(r) for r in results],
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Results saved to {output_path}")

    return results, aggregate


async def run_all_modes(
    dataset: str = "hotpotqa",
    split: str = "validation",
    limit: int = 50,
    seed: int = 42,
    output_dir: str = "results",
    rebuild: bool = True,
) -> dict[str, tuple[list[EvaluationResult], AggregateMetrics]]:
    """Run evaluation for all three modes."""
    setup_logging()

    # Load dataset
    if dataset == "hotpotqa":
        examples, corpus = load_hotpotqa(split=split, limit=limit, seed=seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_by_mode = {}

    # Vector mode
    logger.info("=" * 50)
    logger.info("Running VECTOR mode evaluation")
    logger.info("=" * 50)
    pipeline = UnifiedRAGPipeline(mode="vector")
    if rebuild:
        await pipeline.build_index(corpus)
        pipeline.save_index("data/faiss_index")
    else:
        pipeline.load_index("data/faiss_index")

    results, aggregate = await run_evaluation(
        pipeline, examples, output_path=output_path / "vector_results.json"
    )
    results_by_mode["vector"] = (results, aggregate)

    # Graph mode
    logger.info("=" * 50)
    logger.info("Running GRAPH mode evaluation")
    logger.info("=" * 50)
    graph_store = create_graph_store()
    pipeline = UnifiedRAGPipeline(mode="graph", graph_store=graph_store)
    if rebuild:
        await pipeline.build_index(corpus)
        pipeline.save_index("data/faiss_index")
    else:
        pipeline.load_index("data/faiss_index")

    results, aggregate = await run_evaluation(
        pipeline, examples, output_path=output_path / "graph_results.json"
    )
    results_by_mode["graph"] = (results, aggregate)

    # Hybrid mode
    logger.info("=" * 50)
    logger.info("Running HYBRID mode evaluation")
    logger.info("=" * 50)
    graph_store = create_graph_store()
    pipeline = UnifiedRAGPipeline(mode="hybrid", graph_store=graph_store)
    if rebuild:
        await pipeline.build_index(corpus)
        pipeline.save_index("data/faiss_index")
    else:
        pipeline.load_index("data/faiss_index")

    results, aggregate = await run_evaluation(
        pipeline, examples, output_path=output_path / "hybrid_results.json"
    )
    results_by_mode["hybrid"] = (results, aggregate)

    # Print comparison
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS COMPARISON")
    print("=" * 70)
    print(f"{'Mode':<10} {'EM':<10} {'F1':<10} {'SF Recall':<12} {'Latency':<10} {'LLM Calls':<10}")
    print("-" * 70)
    for mode, (_, agg) in results_by_mode.items():
        print(f"{mode:<10} {agg.exact_match:<10.4f} {agg.f1:<10.4f} {agg.supporting_fact_recall:<12.4f} {agg.avg_latency_ms:<10.1f} {agg.avg_llm_calls:<10.1f}")

    # Compute statistical significance (paired bootstrap)
    if "vector" in results_by_mode and "hybrid" in results_by_mode:
        vector_results = results_by_mode["vector"][0]
        hybrid_results = results_by_mode["hybrid"][0]
        diff_em = bootstrap_confidence_interval(
            list(zip(vector_results, hybrid_results)),
            lambda r: np.mean([h.exact_match - v.exact_match for v, h in r])
        )
        print(f"\nHybrid - Vector EM difference 95% CI: [{diff_em[0]:.4f}, {diff_em[1]:.4f}]")
        if diff_em[0] > 0:
            print(">>> Hybrid significantly outperforms Vector!")
        elif diff_em[1] < 0:
            print(">>> Vector significantly outperforms Hybrid!")
        else:
            print(">>> No significant difference between Hybrid and Vector")

    return results_by_mode


async def main(
    dataset: str = "hotpotqa",
    split: str = "validation",
    limit: int = 50,
    seed: int = 42,
    output_dir: str = "results",
    rebuild: bool = True,
    mode: str = "all",
):
    """Main evaluation entry point."""
    if mode == "all":
        return await run_all_modes(dataset, split, limit, seed, output_dir, rebuild)
    else:
        examples, corpus = load_hotpotqa(split=split, limit=limit, seed=seed)
        graph_store = create_graph_store() if mode in ("graph", "hybrid") else None
        pipeline = UnifiedRAGPipeline(mode=mode, graph_store=graph_store)
        if rebuild:
            await pipeline.build_index(corpus)
            pipeline.save_index("data/faiss_index")
        else:
            pipeline.load_index("data/faiss_index")

        output_path = Path(output_dir) / f"{mode}_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results, aggregate = await run_evaluation(pipeline, examples, output_path=output_path)

        print(f"\n{'=' * 50}")
        print(f"EVALUATION RESULTS ({mode.upper()})")
        print(f"{'=' * 50}")
        print(f"Questions evaluated: {aggregate.num_questions}")
        print(f"Exact Match: {aggregate.exact_match:.4f}")
        print(f"F1 Score: {aggregate.f1:.4f}")
        print(f"Supporting Fact Recall: {aggregate.supporting_fact_recall:.4f}")
        print(f"Avg Latency: {aggregate.avg_latency_ms:.1f} ms")
        print(f"Avg LLM Calls: {aggregate.avg_llm_calls:.1f}")

        if hasattr(aggregate, 'em_ci'):
            print(f"EM 95% CI: [{aggregate.em_ci[0]:.4f}, {aggregate.em_ci[1]:.4f}]")
            print(f"F1 95% CI: [{aggregate.f1_ci[0]:.4f}, {aggregate.f1_ci[1]:.4f}]")
            print(f"SF Recall 95% CI: [{aggregate.sf_ci[0]:.4f}, {aggregate.sf_ci[1]:.4f}]")

        return {mode: (results, aggregate)}


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["vector", "graph", "hybrid", "all"], default="all")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument("--no-rebuild", action="store_true")
    args = parser.parse_args()

    asyncio.run(main(
        mode=args.mode,
        limit=args.limit,
        seed=args.seed,
        output_dir=args.output,
        rebuild=not args.no_rebuild,
    ))