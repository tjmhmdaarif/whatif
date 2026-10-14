"""Evaluation metrics for GraphRAG."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..data.loader import HotpotQAExample
from ..logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """Results for a single question."""
    question_id: str
    question: str
    predicted_answer: str
    gold_answer: str
    exact_match: bool
    f1_score: float
    supporting_fact_recall: float
    retrieved_chunk_ids: list[str]
    gold_supporting_fact_titles: list[str]
    latency_ms: float = 0.0
    llm_calls: int = 0


@dataclass
class AggregateMetrics:
    """Aggregated metrics across all questions."""
    exact_match: float
    f1: float
    supporting_fact_recall: float
    avg_latency_ms: float
    avg_llm_calls: float
    num_questions: int
    by_type: dict[str, dict[str, float]] = None
    by_hops: dict[int, dict[str, float]] = None


def normalize_answer(text: str) -> str:
    """Normalize answer for comparison (lowercase, remove punctuation/articles)."""
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    # Remove punctuation but keep alphanumeric and whitespace
    text = re.sub(r"[^\w\s]", '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def compute_exact_match(prediction: str, gold: str) -> bool:
    """Compute exact match after normalization."""
    return normalize_answer(prediction) == normalize_answer(gold)


def compute_f1(prediction: str, gold: str) -> float:
    """Compute token-level F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_supporting_fact_recall(
    retrieved_chunk_ids: list[str],
    gold_supporting_facts: list[list[str]],
    chunk_id_to_title: dict[str, str],
) -> float:
    """Compute recall of gold supporting fact titles in retrieved chunks.

    Args:
        retrieved_chunk_ids: List of retrieved chunk IDs
        gold_supporting_facts: List of [title, sentence_id] pairs
        chunk_id_to_title: Mapping from chunk_id to document title

    Returns:
        Recall score (0-1)
    """
    if not gold_supporting_facts:
        return 1.0

    gold_titles = set(fact[0] for fact in gold_supporting_facts)
    retrieved_titles = set(chunk_id_to_title.get(cid, "") for cid in retrieved_chunk_ids)

    if not gold_titles:
        return 1.0

    matched = gold_titles & retrieved_titles
    return len(matched) / len(gold_titles)


async def llm_judge_accuracy(
    question: str,
    prediction: str,
    gold: str,
    context: str,
) -> bool:
    """Use LLM as judge to evaluate answer correctness (optional).

    This is a placeholder - in practice you'd call an LLM to judge.
    """
    # Placeholder - would call an LLM to judge correctness
    return compute_exact_match(prediction, gold)


def evaluate_single(
    example: HotpotQAExample,
    predicted_answer: str,
    retrieved_chunk_ids: list[str],
    chunk_id_to_title: dict[str, str],
    latency_ms: float = 0.0,
    llm_calls: int = 0,
) -> EvaluationResult:
    """Evaluate a single prediction."""
    exact_match = compute_exact_match(predicted_answer, example.answer)
    f1 = compute_f1(predicted_answer, example.answer)
    sf_recall = compute_supporting_fact_recall(
        retrieved_chunk_ids,
        example.supporting_facts,
        chunk_id_to_title,
    )

    return EvaluationResult(
        question_id=example.id,
        question=example.question,
        predicted_answer=predicted_answer,
        gold_answer=example.answer,
        exact_match=exact_match,
        f1_score=f1,
        supporting_fact_recall=sf_recall,
        retrieved_chunk_ids=retrieved_chunk_ids,
        gold_supporting_fact_titles=[f[0] for f in example.supporting_facts],
        latency_ms=latency_ms,
        llm_calls=llm_calls,
    )


def aggregate_results(results: list[EvaluationResult]) -> AggregateMetrics:
    """Aggregate individual results into summary metrics."""
    if not results:
        return AggregateMetrics(0, 0, 0, 0, 0, 0)

    em_scores = [r.exact_match for r in results]
    f1_scores = [r.f1_score for r in results]
    sf_recalls = [r.supporting_fact_recall for r in results]
    latencies = [r.latency_ms for r in results]
    llm_calls = [r.llm_calls for r in results]

    # Group by question type if available
    by_type = {}
    for r in results:
        # We'd need to store type in EvaluationResult
        pass

    return AggregateMetrics(
        exact_match=np.mean(em_scores),
        f1=np.mean(f1_scores),
        supporting_fact_recall=np.mean(sf_recalls),
        avg_latency_ms=np.mean(latencies),
        avg_llm_calls=np.mean(llm_calls),
        num_questions=len(results),
    )


def bootstrap_confidence_interval(
    results: list[EvaluationResult],
    metric_fn: callable,
    n_resamples: int = 1000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for a metric.

    Args:
        results: List of evaluation results
        metric_fn: Function that takes results and returns a scalar metric
        n_resamples: Number of bootstrap resamples
        confidence: Confidence level (e.g., 0.95 for 95%)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    n = len(results)
    if n == 0:
        return (0.0, 0.0)

    original_metric = metric_fn(results)
    bootstrap_metrics = []

    for _ in range(n_resamples):
        # Sample with replacement
        indices = np.random.choice(n, n, replace=True)
        resampled = [results[i] for i in indices]
        bootstrap_metrics.append(metric_fn(resampled))

    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_metrics, alpha * 100)
    upper = np.percentile(bootstrap_metrics, (1 - alpha) * 100)

    return (float(lower), float(upper))