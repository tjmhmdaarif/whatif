"""Tests for evaluation metrics."""

import pytest

from graphrag.eval.metrics import (
    normalize_answer,
    compute_exact_match,
    compute_f1,
    compute_supporting_fact_recall,
)


class TestNormalizeAnswer:
    """Tests for answer normalization."""

    def test_lowercase(self):
        assert normalize_answer("Hello World") == "hello world"

    def test_remove_articles(self):
        assert normalize_answer("The cat") == "cat"
        assert normalize_answer("A dog") == "dog"
        assert normalize_answer("An apple") == "apple"

    def test_remove_punctuation(self):
        assert normalize_answer("Hello, world!") == "hello world"
        assert normalize_answer("What's up?") == "whats up"

    def test_extra_whitespace(self):
        assert normalize_answer("  Hello   world  ") == "hello world"


class TestComputeExactMatch:
    """Tests for exact match computation."""

    def test_exact_match(self):
        assert compute_exact_match("hello world", "hello world") is True
        assert compute_exact_match("Hello World", "hello world") is True

    def test_no_match(self):
        assert compute_exact_match("hello world", "goodbye world") is False
        assert compute_exact_match("cat", "dog") is False

    def test_articles_ignored(self):
        assert compute_exact_match("the cat", "a cat") is True
        assert compute_exact_match("an apple", "the apple") is True


class TestComputeF1:
    """Tests for F1 computation."""

    def test_perfect_match(self):
        assert compute_f1("hello world", "hello world") == 1.0

    def test_partial_match(self):
        # "hello world" vs "hello there" -> common: "hello"
        f1 = compute_f1("hello world", "hello there")
        assert 0 < f1 < 1

    def test_no_overlap(self):
        assert compute_f1("cat dog", "bird fish") == 0.0

    def test_empty_strings(self):
        assert compute_f1("", "") == 1.0
        assert compute_f1("hello", "") == 0.0
        assert compute_f1("", "hello") == 0.0

    def test_case_insensitive(self):
        assert compute_f1("Hello World", "hello world") == 1.0


class TestComputeSupportingFactRecall:
    """Tests for supporting fact recall."""

    def test_perfect_recall(self):
        chunk_id_to_title = {"chunk_1": "Title A", "chunk_2": "Title B"}
        retrieved = ["chunk_1", "chunk_2"]
        gold = [["Title A", "0"], ["Title B", "1"]]

        recall = compute_supporting_fact_recall(retrieved, gold, chunk_id_to_title)
        assert recall == 1.0

    def test_partial_recall(self):
        chunk_id_to_title = {"chunk_1": "Title A", "chunk_2": "Title B", "chunk_3": "Title C"}
        retrieved = ["chunk_1", "chunk_3"]
        gold = [["Title A", "0"], ["Title B", "1"]]

        recall = compute_supporting_fact_recall(retrieved, gold, chunk_id_to_title)
        assert recall == 0.5

    def test_zero_recall(self):
        chunk_id_to_title = {"chunk_1": "Title A"}
        retrieved = ["chunk_1"]
        gold = [["Title B", "0"]]

        recall = compute_supporting_fact_recall(retrieved, gold, chunk_id_to_title)
        assert recall == 0.0

    def test_empty_gold(self):
        chunk_id_to_title = {"chunk_1": "Title A"}
        retrieved = ["chunk_1"]
        gold = []

        recall = compute_supporting_fact_recall(retrieved, gold, chunk_id_to_title)
        assert recall == 1.0

    def test_missing_chunk_ids(self):
        chunk_id_to_title = {"chunk_1": "Title A"}
        retrieved = ["chunk_999"]  # Doesn't exist
        gold = [["Title A", "0"]]

        recall = compute_supporting_fact_recall(retrieved, gold, chunk_id_to_title)
        assert recall == 0.0