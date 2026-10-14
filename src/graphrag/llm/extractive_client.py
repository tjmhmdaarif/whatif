"""Extractive answer provider.

Produces real answers by selecting and compressing sentences from the retrieved
context instead of calling an external LLM. This keeps the pipeline fully
functional offline: answers are genuinely derived from the corpus, so EM/F1
metrics measure retrieval and ranking quality rather than a stub.
"""

from __future__ import annotations

import re
from typing import Any

from .base import LLMClient, LLMResponse

# Common English words that carry no answer signal.
STOPWORDS = frozenset(
    """a an the and or but if then than that this these those of in on at to for from by
    with without about into over after before between during is are was were be been being
    am do does did doing have has had having will would shall should can could may might must
    it its as so such not no nor only own same too very just don now i you he she they we
    me him her them us my your his their our which who whom whose what when where why how""".split()
)

# Question words that hint at the expected answer type.
_WHEN = re.compile(r"\b(when|what year|what date|which year)\b", re.I)
_WHERE = re.compile(r"\b(where|which country|which city|what place)\b", re.I)
_WHO = re.compile(r"\b(who|whose|whom)\b", re.I)

# Answer-type specific patterns applied to candidate sentences.
_DATE = re.compile(
    r"\b(\d{1,2}\s+\w+\s+\d{4}|\w+\s+\d{1,2},?\s+\d{4}|\d{4}|"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"(?:\s+\d{1,2})?(?:,?\s+\d{4})?)"
)


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens with punctuation stripped."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _content_words(text: str) -> set[str]:
    """Content-bearing tokens, excluding stopwords.

    Tokens are lightly stemmed (plural and common verb endings removed) so that
    "open" matches "opened" and "towers" matches "tower".

    Args:
        text: Text to tokenize.

    Returns:
        Set of stemmed content tokens.
    """
    return {_stem(t) for t in _tokenize(text) if t not in STOPWORDS and len(t) > 1}


def _stem(token: str) -> str:
    """Crude suffix stripping to unify common inflections.

    Args:
        token: Lowercase word token.

    Returns:
        The stemmed token.
    """
    if len(token) > 4 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _sentences(text: str) -> list[str]:
    """Split text into sentences on terminal punctuation."""
    parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    return [p.strip() for p in parts if len(p.strip()) > 1]


def _vector_section(prompt: str) -> str:
    """Isolate the vector-retrieved portion of a graph-aware prompt.

    The graph-aware template places graph paths after the vector passages. Those
    paths are not quotable prose, so they are excluded from answer extraction.

    Args:
        prompt: Full prompt text.

    Returns:
        The prompt text limited to the vector context, or the whole prompt when
        no graph section is present.
    """
    boundary = re.search(r"Graph Traversal Paths:", prompt)
    return prompt[: boundary.start()] if boundary else prompt


def _split_chunks(context: str) -> list[tuple[str, str]]:
    """Parse ``[Chunk: id]`` blocks out of a formatted context string.

    Args:
        context: Context produced by ``format_context``.

    Returns:
        List of ``(chunk_id, body_text)`` pairs.
    """
    context = _vector_section(context)

    # Chunk bodies must stop before the trailing question/instructions prose.
    context = re.split(r"\n\s*\nQuestion:", context)[0]

    chunks: list[tuple[str, str]] = []
    pattern = re.compile(r"\[Chunk:\s*([^\]]+)\]\s*\n?(.*?)(?=\[Chunk:|$)", re.S)
    for match in pattern.finditer(context):
        chunk_id = match.group(1).strip()
        body = match.group(2).strip()
        if body:
            chunks.append((chunk_id, body))
    return chunks


def _extract_question(prompt: str) -> str:
    """Pull the question out of an answer prompt.

    Both answer templates embed a ``Question:`` line, but the graph-aware
    template also mentions the word earlier in its instruction prose, so the
    *last* ``Question:`` line is the real one.

    Args:
        prompt: Full prompt text.

    Returns:
        The question string, or an empty string if not found.
    """
    matches = re.findall(r"Question:\s*(.+?)(?:\n\s*\n|Instructions:|Answer:|$)", prompt, re.S)
    return matches[-1].strip() if matches else ""


def _compress(sentence: str, question_type: str) -> str:
    """Reduce a sentence to the most answer-like span.

    HotpotQA gold answers are short spans (a name, a date, a noun phrase), not
    whole sentences, so this trims the selected sentence toward the answer.

    Args:
        sentence: Candidate sentence.
        question_type: One of ``when``, ``where``, ``who``, or ``what``.

    Returns:
        A shortened answer span.
    """
    if question_type == "when":
        date_match = _DATE.search(sentence)
        if date_match:
            return date_match.group(1).strip(" ,.")

    if question_type == "who":
        # Prefer a proper noun phrase following a copula or preposition.
        match = re.search(
            r"\b(?:by|is|was|were|are)\s+([A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*)*)", sentence
        )
        if match:
            return match.group(1).strip()

    # Trim trailing subordinate clauses and leading discourse markers so the
    # span stays close to the words that actually overlap the question.
    span = re.sub(
        r"^(?:and|but|however|also|it|they|there|he|she)\s+", "", sentence, flags=re.I
    )
    span = re.split(
        r"\s(?:which|who|that|because|although|whereas|while|when|where)\s", span
    )[0]
    span = re.split(r",\s*(?:which|who|and|but)\s", span)[0]

    return span.strip().rstrip(".").strip()


class ExtractiveClient(LLMClient):
    """Answer provider that extracts answers from supplied context."""

    def __init__(self, model: str = "extractive-v1") -> None:
        self.model = model
        self._calls = 0

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self.model

    @property
    def call_count(self) -> int:
        """Number of completions served so far."""
        return self._calls

    def _classify(self, question: str) -> str:
        """Infer the expected answer type from question wording."""
        if _WHEN.search(question):
            return "when"
        if _WHERE.search(question):
            return "where"
        if _WHO.search(question):
            return "who"
        return "what"

    def _rank(self, question: str, chunks: list[tuple[str, str]]) -> tuple[str, str] | None:
        """Choose the sentence best answering the question.

        Scoring rewards content-word overlap while penalising words the sentence
        merely repeats from the question, since a good answer sentence states
        something *new* about the asked entity.

        Args:
            question: The question text.
            chunks: ``(chunk_id, text)`` pairs from the context.

        Returns:
            ``(answer, chunk_id)`` or ``None`` when no sentence overlaps.
        """
        question_type = self._classify(question)
        q_words = _content_words(question)

        if not q_words:
            return None

        best: tuple[float, str, str] | None = None

        for chunk_id, body in chunks:
            for sentence in _sentences(body):
                s_words = _content_words(sentence)
                if not s_words:
                    continue

                overlap = len(q_words & s_words)
                if overlap == 0:
                    continue

                # Coverage: how much of the question this sentence addresses.
                coverage = overlap / len(q_words)

                # Novelty: how much of the sentence is NOT in the question.
                # The answer usually lives in the novel words.
                novelty = 1.0 - (overlap / len(s_words))

                # Coverage dominates: a sentence must actually be about the
                # asked entity before novelty can help pick the right span.
                score = (coverage * 0.85 + novelty * 0.15) * (overlap ** 0.5)

                # Reward sentences containing the expected answer type.
                if question_type == "when" and _DATE.search(sentence):
                    score *= 1.5
                elif question_type == "where" and re.search(
                    r"\b(?:in|near|located|capital of)\b", sentence, re.I
                ):
                    score *= 1.25

                if best is None or score > best[0]:
                    best = (score, sentence, chunk_id)

        if best is None:
            return None

        _, sentence, chunk_id = best
        answer = _compress(sentence, question_type)
        return (answer, chunk_id) if answer else None

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Extract an answer from the context embedded in the prompt.

        Args:
            prompt: Prompt containing ``[Chunk: ...]`` blocks and a question.
            system_prompt: Ignored; kept for interface compatibility.
            temperature: Ignored; extraction is deterministic.
            max_tokens: Ignored.
            **kwargs: Ignored.

        Returns:
            An ``LLMResponse`` whose text is the extracted answer.
        """
        self._calls += 1

        question = _extract_question(prompt)
        chunks = _split_chunks(prompt)

        if not question or not chunks:
            answer = "I don't have enough information to answer this question."
        else:
            ranked = self._rank(question, chunks)
            answer = (
                f"{ranked[0]} [{ranked[1]}]"
                if ranked
                else "I don't have enough information to answer this question."
            )

        return LLMResponse(
            text=answer,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(answer.split()),
            total_tokens=len(prompt.split()) + len(answer.split()),
            model=self.model,
            cached=False,
        )

    async def complete_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Extract entities and relations from the text in the prompt.

        Uses capitalization heuristics plus sentence-level co-occurrence, which
        is enough to build a usable knowledge graph without an external LLM.

        Args:
            prompt: Prompt containing the text to analyse.
            system_prompt: Ignored.
            temperature: Ignored.
            max_tokens: Ignored.
            schema: Ignored.
            **kwargs: Ignored.

        Returns:
            A dict with ``entities`` and ``relations`` lists.
        """
        self._calls += 1

        text_match = re.search(r"Text:\s*\n(.+?)(?:\n\s*\n|Output valid JSON|$)", prompt, re.S)
        text = text_match.group(1).strip() if text_match else prompt

        return self._extract_structure(text)

    @staticmethod
    def _extract_structure(text: str) -> dict[str, Any]:
        """Derive entities and relations from text using surface heuristics.

        Args:
            text: Passage to analyse.

        Returns:
            Dict with ``entities`` and ``relations`` keys.
        """
        entities: list[dict[str, str]] = []
        seen: set[str] = set()

        # Capitalized multi-word phrases, skipping sentence-initial words.
        phrase = re.compile(
            r"\b([A-Z][a-z]+(?:\s+(?:of|the|de|van)\s+[A-Z][a-z]+|\s+[A-Z][a-z]+)*)\b"
        )
        for match in phrase.finditer(text):
            # Drop a leading article so "The Eiffel Tower" stores as "Eiffel Tower".
            name = re.sub(
                r"^(?:The|A|An)\s+", "", match.group(1).strip(), flags=re.I
            ).strip()
            if len(name) < 3 or name in seen:
                continue
            # Skip a phrase that only looks capitalized because it starts a sentence.
            prefix = text[max(0, match.start() - 2):match.start()]
            if prefix.endswith(". "):
                continue
            seen.add(name)
            entities.append({"name": name, "type": "CONCEPT", "description": name})
            if len(entities) >= 20:
                break

        # Relations from co-occurrence of two entities in one sentence.
        relations: list[dict[str, str]] = []
        names = [e["name"] for e in entities]
        for sentence in _sentences(text):
            present = [n for n in names if n in sentence]
            for i, source in enumerate(present):
                for target in present[i + 1:]:
                    relations.append({
                        "source": source,
                        "target": target,
                        "relation": "RELATED_TO",
                        "evidence": sentence[:200],
                    })
            if len(relations) >= 30:
                break

        return {"entities": entities, "relations": relations[:30]}


def create_extractive_client(model: str = "extractive-v1") -> ExtractiveClient:
    """Create an extractive answer client.

    Args:
        model: Model label reported in responses.

    Returns:
        A configured ``ExtractiveClient``.
    """
    return ExtractiveClient(model)

