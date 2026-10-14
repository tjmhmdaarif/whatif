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

# Common words that are capitalized only because they begin a sentence. They are
# rejected as entity candidates.
_SENTENCE_STARTERS = frozenset(
    """he she it they we i you the a an this that these those there here his her its
    their our my your his after before during when where which who whom whose what
    while although however therefore thus also both each some many most other another
    several such as at by for from in of on to with without about into over under
    between among within upon despite according following born based known located
    founded established created opened closed built designed used made called named
    born died married educated worked served represented included featured
    """.split()
)

# Single words that are far more likely to be ordinary prose than entity names.
# Multi-word phrases are not filtered against this, since names like
# "On Chesil Beach" are only distinguishable by their proper-noun shape.
_COMMON_WORDS = frozenset(
    """american british english french german italian spanish russian chinese japanese
    indian pakistani european african australian canadian irish scottish welsh
    city town village county state country region district province area island
    river mountain lake sea ocean north south east west northern southern central
    first second third last next new old early late modern ancient royal national
    international public private general major minor local rural urban
    film album song single series show game book novel magazine newspaper
    school college university church cathedral castle palace museum theatre theater
    company group band team club society association organisation organization
    people person man woman child family member king queen president minister
    war battle treaty election revolution century decade year month week day
    population area name type form part side end top bottom front back
    """.split()
)

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


_PERSON_HINT = re.compile(
    r"\b(?:born|died|actor|actress|singer|author|politician|player|director|producer|"
    r"admiral|general|president|minister|writer|artist|photographer|journalist)\b", re.I
)
_LOCATION_HINT = re.compile(
    r"\b(?:city|town|village|county|country|island|river|mountain|located|capital|"
    r"province|district|region|state|peninsula)\b", re.I
)
_DATE_HINT = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December|\b(?:1[5-9]|20)\d{2}\b)\b"
)
_ORG_HINT = re.compile(
    r"\b(?:company|band|group|club|school|university|party|organisation|organization|"
    r"association|studio|label|agency|force|army|navy|squadron|station)\b", re.I
)
_WORK_HINT = re.compile(
    r"\b(?:film|album|song|single|novel|book|series|show|magazine|newspaper|"
    r"musical|opera|play)\b", re.I
)


def _guess_type(name: str, context: str) -> str:
    """Infer a coarse entity type from surrounding text.

    Assigning a realistic type matters because the resolver only merges entities
    of the same type; labelling everything ``CONCEPT`` lets unrelated names merge.

    Args:
        name: The entity name.
        context: Text the entity was found in.

    Returns:
        One of ``PERSON``, ``LOCATION``, ``ORG``, ``DATE``, ``WORK_OF_ART`` or
        ``CONCEPT``.
    """
    window = context[max(0, context.find(name) - 120): context.find(name) + 160]

    if _DATE_HINT.search(name) or (name.isdigit() and len(name) == 4):
        return "DATE"
    if _LOCATION_HINT.search(window):
        return "LOCATION"
    if _PERSON_HINT.search(window):
        return "PERSON"
    if _WORK_HINT.search(window):
        return "WORK_OF_ART"
    if _ORG_HINT.search(window):
        return "ORG"

    # Multi-word capitalized phrases are most often works or organisations.
    return "WORK_OF_ART" if " " in name else "CONCEPT"


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

        # Capitalized phrases, allowing initials ("Iqbal F. Qadir"), apostrophes,
        # digits and internal connectors ("On Chesil Beach (film)" style titles).
        phrase = re.compile(
            r"\b([A-Z][\w.'-]*(?:\s+(?:of|the|de|van|and)\s+[A-Z][\w.'-]*|\s+[A-Z][\w.'-]*)*)\b"
        )
        for match in phrase.finditer(text):
            # Drop a leading article so "The Eiffel Tower" stores as "Eiffel Tower".
            name = re.sub(
                r"^(?:The|A|An)\s+", "", match.group(1).strip(), flags=re.I
            ).strip()

            # Reject sentence-initial common words, which are capitalized by
            # position rather than because they name something.
            if name.lower() in _SENTENCE_STARTERS:
                continue
            # A phrase must not span a sentence boundary ("Denmark. It").
            if re.search(r"[.!?;:]", name):
                continue
            if len(name) < 3 or name in seen:
                continue
            # A bare single word is only a candidate when it is not a common
            # English word, so plain nouns and verbs are filtered out.
            if " " not in name and "." not in name and name.lower() in _COMMON_WORDS:
                continue

            seen.add(name)
            # The description is left empty rather than echoing the name: the
            # resolver embeds "name (type): description", and identical
            # descriptions across every entity push all embedding similarities
            # toward 1.0, collapsing unrelated entities into one cluster.
            entities.append({"name": name, "type": _guess_type(name, text), "description": ""})
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

