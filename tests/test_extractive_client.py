"""Tests for the offline extractive answer provider."""

import pytest

from graphrag.config import load_config, set_config
from graphrag.generation.answer import ANSWER_PROMPT, GRAPH_ANSWER_PROMPT, format_context
from graphrag.llm.extractive_client import (
    ExtractiveClient,
    _extract_question,
    _split_chunks,
)
from graphrag.chunking import Chunk


@pytest.fixture
def client():
    """Return an extractive client."""
    return ExtractiveClient()


def _prompt(question: str, chunks_text: list[tuple[str, str]], graph: bool = False) -> str:
    """Build a real prompt from chunk text using the production template."""
    chunks = [
        Chunk(chunk_id=cid, doc_id="doc", text=text, position=i)
        for i, (cid, text) in enumerate(chunks_text)
    ]
    context, _ = format_context(chunks)

    if graph:
        return GRAPH_ANSWER_PROMPT.format(
            vector_context=context,
            graph_context="[Graph Path 1] A --[LOCATED_IN]--> B",
            question=question,
        )
    return ANSWER_PROMPT.format(context=context, question=question)


class TestPromptParsing:
    """The client must recover question and context from real prompts."""

    def test_extracts_question_from_vector_prompt(self):
        """Question is recovered from the vector answer template."""
        prompt = _prompt("Where is the Eiffel Tower located?", [("c1", "Paris is big.")])

        assert _extract_question(prompt) == "Where is the Eiffel Tower located?"

    def test_extracts_question_from_graph_prompt(self):
        """The graph template repeats the word 'Question', last match must win."""
        prompt = _prompt(
            "Where is the Eiffel Tower located?", [("c1", "Paris is big.")], graph=True
        )

        assert _extract_question(prompt) == "Where is the Eiffel Tower located?"

    def test_chunk_bodies_exclude_trailing_instructions(self):
        """Prompt prose must not leak into the chunk body."""
        prompt = _prompt("Where is the Eiffel Tower located?", [("c1", "Paris is big.")])

        chunks = _split_chunks(prompt)

        assert [cid for cid, _ in chunks] == ["c1"]
        assert "Instructions" not in chunks[0][1]
        assert "Answer:" not in chunks[0][1]

    def test_graph_section_is_excluded_from_chunks(self):
        """Graph traversal paths are not quotable prose."""
        prompt = _prompt("Where is the Eiffel Tower located?", [("c1", "Paris is big.")], graph=True)

        chunks = _split_chunks(prompt)

        assert "Graph Traversal Paths" not in "".join(body for _, body in chunks)


class TestAnswerExtraction:
    """Answers must be spans drawn from the supplied context."""

    @pytest.mark.asyncio
    async def test_answers_from_context_not_stub(self, client):
        """A populated context yields a real extracted span, not a placeholder."""
        prompt = _prompt(
            "What is the capital of France?",
            [("c1", "France is a country in Europe. The capital of France is Paris.")],
        )

        response = await client.complete(prompt)

        assert response.text != "Mock answer"
        assert "Paris" in response.text
        assert "c1" in response.text

    @pytest.mark.asyncio
    async def test_when_question_yields_a_date(self, client):
        """Temporal questions should compress to a date span."""
        prompt = _prompt(
            "When did the park open?",
            [("c1", "Tivoli Gardens opened on 15 August 1843 and is very popular.")],
        )

        response = await client.complete(prompt)

        assert "1843" in response.text

    @pytest.mark.asyncio
    async def test_no_overlap_returns_unanswerable(self, client):
        """Context with no lexical overlap must not fabricate an answer."""
        prompt = _prompt("Who directed the film?", [("c1", "Zebras graze quietly.")])

        response = await client.complete(prompt)

        assert "don't have enough information" in response.text

    @pytest.mark.asyncio
    async def test_graph_prompt_answers_without_leaking_paths(self, client):
        """Graph-aware prompts must not echo traversal paths as the answer."""
        prompt = _prompt(
            "What is the capital of France?",
            [("c1", "The capital of France is Paris.")],
            graph=True,
        )

        response = await client.complete(prompt)

        assert "Graph Path" not in response.text
        assert "--[" not in response.text
        assert "Paris" in response.text

    @pytest.mark.asyncio
    async def test_tokens_are_reported(self, client):
        """Token counts stay populated for downstream accounting."""
        prompt = _prompt("What is the capital of France?", [("c1", "Paris is the capital.")])

        response = await client.complete(prompt)

        assert response.prompt_tokens > 0
        assert response.total_tokens >= response.completion_tokens
        assert response.model == "extractive-v1"


class TestStructureExtraction:
    """complete_json must yield usable entities and relations for the graph."""

    @pytest.mark.asyncio
    async def test_extracts_entities_and_relations(self, client):
        """Capitalized phrases become entities; co-occurrence becomes relations."""
        prompt = (
            "You are an expert at extracting entities and relationships from text.\n\n"
            "Entity Types: PERSON, LOCATION\nRelation Types: LOCATED_IN\n\n"
            "Text:\nThe Eiffel Tower is located in Paris.\n\n"
            "Output valid JSON only"
        )

        result = await client.complete_json(prompt)

        names = {e["name"] for e in result["entities"]}
        assert "Eiffel Tower" in names
        assert "Paris" in names
        assert all(e["type"] and e["name"] for e in result["entities"])
        assert len(result["relations"]) >= 1
        assert result["relations"][0]["relation"] == "RELATED_TO"
