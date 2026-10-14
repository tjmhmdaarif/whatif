"""Report generation for GraphRAG evaluation results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_results(results_dir: Path) -> dict[str, Any]:
    """Load all result files from a directory."""
    results = {}
    for file in results_dir.glob("*_results.json"):
        mode = file.stem.replace("_results", "")
        with open(file, "r") as f:
            results[mode] = json.load(f)
    return results


def format_metric_table(results: dict[str, Any]) -> str:
    """Format aggregate metrics as markdown table."""
    lines = [
        "## Aggregate Metrics Comparison",
        "",
        "| Mode | Exact Match | F1 | Supporting Fact Recall | Avg Latency (ms) | Avg LLM Calls |",
        "|------|-------------|-----|------------------------|------------------|---------------|",
    ]

    for mode, data in sorted(results.items()):
        agg = data.get("aggregate", {})
        lines.append(
            f"| {mode} | "
            f"{agg.get('exact_match', 0):.4f} | "
            f"{agg.get('f1', 0):.4f} | "
            f"{agg.get('supporting_fact_recall', 0):.4f} | "
            f"{agg.get('avg_latency_ms', 0):.1f} | "
            f"{agg.get('avg_llm_calls', 0):.1f} |"
        )

    return "\n".join(lines)


def format_confidence_intervals(results: dict[str, Any]) -> str:
    """Format confidence intervals as markdown table."""
    lines = [
        "",
        "## 95% Confidence Intervals (Paired Bootstrap)",
        "",
        "| Mode | EM CI | F1 CI | SF Recall CI |",
        "|------|-------|-------|--------------|",
    ]

    for mode, data in sorted(results.items()):
        agg = data.get("aggregate", {})
        em_ci = agg.get("em_ci", [0, 0])
        f1_ci = agg.get("f1_ci", [0, 0])
        sf_ci = agg.get("sf_ci", [0, 0])
        lines.append(
            f"| {mode} | [{em_ci[0]:.4f}, {em_ci[1]:.4f}] | "
            f"[{f1_ci[0]:.4f}, {f1_ci[1]:.4f}] | "
            f"[{sf_ci[0]:.4f}, {sf_ci[1]:.4f}] |"
        )

    return "\n".join(lines)


def compute_statistical_significance(results: dict[str, Any]) -> str:
    """Compute and format statistical significance tests."""
    if "vector" not in results or "hybrid" not in results:
        return "\n## Statistical Significance\n\nNot enough data for comparison.\n"

    vector_data = results["vector"]["results"]
    hybrid_data = results["hybrid"]["results"]

    # Pair results by question_id
    vector_by_id = {r["question_id"]: r for r in vector_data}
    hybrid_by_id = {r["question_id"]: r for r in hybrid_data}

    common_ids = set(vector_by_id.keys()) & set(hybrid_by_id.keys())
    if not common_ids:
        return "\n## Statistical Significance\n\nNo common questions for paired comparison.\n"

    # Paired bootstrap for EM difference
    n = len(common_ids)
    differences = []
    for qid in common_ids:
        v_em = vector_by_id[qid]["exact_match"]
        h_em = hybrid_by_id[qid]["exact_match"]
        differences.append(h_em - v_em)

    # Bootstrap CI
    bootstrap_diffs = []
    for _ in range(1000):
        sample = np.random.choice(differences, n, replace=True)
        bootstrap_diffs.append(np.mean(sample))

    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)
    mean_diff = np.mean(differences)

    lines = [
        "",
        "## Statistical Significance (Hybrid vs Vector)",
        "",
        f"- Mean EM difference (Hybrid - Vector): {mean_diff:.4f}",
        f"- 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]",
        "",
    ]

    if ci_lower > 0:
        lines.append("✅ **Hybrid significantly outperforms Vector** (p < 0.05)")
    elif ci_upper < 0:
        lines.append("❌ **Vector significantly outperforms Hybrid** (p < 0.05)")
    else:
        lines.append("⚠️ **No significant difference** between Hybrid and Vector (p > 0.05)")

    return "\n".join(lines)


def find_qualitative_examples(results: dict[str, Any], n_win: int = 5, n_lose: int = 3) -> str:
    """Find qualitative examples where hybrid wins/loses vs vector."""
    if "vector" not in results or "hybrid" not in results:
        return ""

    vector_data = results["vector"]["results"]
    hybrid_data = results["hybrid"]["results"]

    vector_by_id = {r["question_id"]: r for r in vector_data}
    hybrid_by_id = {r["question_id"]: r for r in hybrid_data}

    common_ids = set(vector_by_id.keys()) & set(hybrid_by_id.keys())

    wins = []
    losses = []
    ties = []

    for qid in common_ids:
        v_em = vector_by_id[qid]["exact_match"]
        h_em = hybrid_by_id[qid]["exact_match"]

        if h_em and not v_em:
            wins.append((qid, vector_by_id[qid], hybrid_by_id[qid]))
        elif v_em and not h_em:
            losses.append((qid, vector_by_id[qid], hybrid_by_id[qid]))
        else:
            ties.append((qid, vector_by_id[qid], hybrid_by_id[qid]))

    lines = [
        "",
        "## Qualitative Analysis",
        "",
        f"Total questions: {len(common_ids)}",
        f"Hybrid wins: {len(wins)}",
        f"Vector wins: {len(losses)}",
        f"Ties: {len(ties)}",
        "",
        "### Hybrid Wins (Vector fails, Hybrid succeeds)",
        "",
    ]

    for qid, v_res, h_res in wins[:n_win]:
        lines.extend([
            f"**Q: {qid}**",
            f"- Question: {v_res['question'][:200]}",
            f"- Gold: {v_res['gold_answer']}",
            f"- Vector: {v_res['predicted_answer'][:200]} ❌",
            f"- Hybrid: {h_res['predicted_answer'][:200]} ✅",
            f"- Retrieved (Vector): {v_res['retrieved_chunk_ids'][:5]}",
            f"- Retrieved (Hybrid): {h_res['retrieved_chunk_ids'][:5]}",
            "",
        ])

    lines.append("### Hybrid Losses (Hybrid fails, Vector succeeds)")
    lines.append("")

    for qid, v_res, h_res in losses[:n_lose]:
        lines.extend([
            f"**Q: {qid}**",
            f"- Question: {v_res['question'][:200]}",
            f"- Gold: {v_res['gold_answer']}",
            f"- Vector: {v_res['predicted_answer'][:200]} ✅",
            f"- Hybrid: {h_res['predicted_answer'][:200]} ❌",
            f"- Retrieved (Vector): {v_res['retrieved_chunk_ids'][:5]}",
            f"- Retrieved (Hybrid): {h_res['retrieved_chunk_ids'][:5]}",
            "",
        ])

    return "\n".join(lines)


def generate_report(
    results_dir: str = "results",
    output_path: str = "results/REPORT.md",
) -> None:
    """Generate full evaluation report."""
    results_path = Path(results_dir)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    results = load_results(results_path)

    if not results:
        print("No results found!")
        return

    lines = [
        "# GraphRAG-MultiHop Evaluation Report",
        "",
        f"Generated from: {results_dir}",
        f"Modes evaluated: {', '.join(sorted(results.keys()))}",
        "",
    ]

    # Aggregate metrics
    lines.append(format_metric_table(results))
    lines.append(format_confidence_intervals(results))

    # Statistical significance
    lines.append(compute_statistical_significance(results))

    # Ablation results
    ablation_file = results_path / "ablations" / "ablation_summary.json"
    if ablation_file.exists():
        with open(ablation_file, "r") as f:
            ablation_data = json.load(f)

        lines.append("")
        lines.append("# Ablation Studies")
        lines.append("")

        for category, cat_results in ablation_data.items():
            lines.append(f"## {category.replace('_', ' ').title()}")
            lines.append("")
            lines.append("| Config | EM | F1 | SF Recall | Latency (ms) |")
            lines.append("|--------|-----|-----|-----------|--------------|")
            for config, metrics in cat_results.items():
                lines.append(
                    f"| {config} | {metrics['exact_match']:.4f} | "
                    f"{metrics['f1']:.4f} | {metrics['supporting_fact_recall']:.4f} | "
                    f"{metrics['avg_latency_ms']:.1f} |"
                )
            lines.append("")

    # Qualitative examples
    lines.append(find_qualitative_examples(results))

    # Design decisions
    lines.extend([
        "",
        "# Design Decisions & Trade-offs",
        "",
        "## Architecture",
        "- **Modular design**: Interfaces for LLM, Vector Store, Graph Store enable swapping components",
        "- **Pluggable LLM providers**: Anthropic, Ollama, Mock (for testing)",
        "- **Vector store abstraction**: FAISS default, Chroma supported",
        "- **Graph store abstraction**: NetworkX in-memory, Neo4j ready",
        "",
        "## Retrieval",
        "- **Hybrid fusion**: Reciprocal Rank Fusion (RRF) with configurable weights",
        "- **Graph traversal**: BFS (default) and Personalized PageRank options",
        "- **Query entity linking**: LLM-based extraction + embedding similarity",
        "- **Token budgeting**: Automatic context truncation",
        "",
        "## Knowledge Graph Construction",
        "- **LLM extraction**: Structured JSON output with Pydantic validation",
        "- **Entity resolution**: Embedding + string similarity clustering",
        "- **Caching**: Disk-based LLM and embedding cache for reproducibility",
        "",
        "# Limitations",
        "",
        "1. **Extraction cost**: LLM-based extraction is slow and expensive",
        "2. **Error propagation**: Bad extraction → bad graph → bad retrieval",
        "3. **Entity resolution mistakes**: Can merge distinct entities or miss aliases",
        "4. **Scalability**: In-memory NetworkX graph doesn't scale to millions of nodes",
        "5. **Mock LLM limitations**: Evaluation with mock LLM doesn't reflect real performance",
        "",
        "# Future Work",
        "",
        "1. **Neo4j backend**: For production-scale graphs",
        "2. **Community summaries**: Hierarchical graph summarization (GraphRAG style)",
        "3. **Incremental updates**: Add documents without full rebuild",
        "4. **Better entity resolution**: Cross-document coreference resolution",
        "5. **Learned fusion**: Train fusion weights instead of fixed RRF",
        "6. **Real LLM evaluation**: Run with Anthropic/Ollama for true metrics",
    ])

    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report generated: {output}")


if __name__ == "__main__":
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "results/REPORT.md"
    generate_report(results_dir, output_path)