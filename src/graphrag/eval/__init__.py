"""Evaluation module exports."""

from .metrics import (
    EvaluationResult,
    AggregateMetrics,
    compute_exact_match,
    compute_f1,
    compute_supporting_fact_recall,
    evaluate_single,
    aggregate_results,
    bootstrap_confidence_interval,
)
from .runner import UnifiedRAGPipeline, run_evaluation, run_all_modes, main
from .ablations import run_all_ablations, generate_ablation_report
from .report import generate_report

__all__ = [
    "EvaluationResult",
    "AggregateMetrics",
    "compute_exact_match",
    "compute_f1",
    "compute_supporting_fact_recall",
    "evaluate_single",
    "aggregate_results",
    "bootstrap_confidence_interval",
    "UnifiedRAGPipeline",
    "run_evaluation",
    "run_all_modes",
    "main",
    "run_all_ablations",
    "generate_ablation_report",
    "generate_report",
]