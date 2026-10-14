"""Answer generation with citations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..chunking import Chunk
from ..llm import create_llm_client
from ..logging_setup import get_logger
from ..retrieval.hybrid import RetrievalResult

logger = get_logger(__name__)


@dataclass
class GeneratedAnswer:
    """Generated answer with citations."""
    answer: str
    citations: list[dict[str, Any]]  # List of {chunk_id, doc_id, text_snippet}
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    traversal_paths: list[list[dict[str, Any]]] = None


# Answer generation prompt template
ANSWER_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Your task is to answer the question using only the information from the context.

Context:
{context}

Question: {question}

Instructions:
1. Answer the question directly and concisely.
2. Cite your sources using [chunk_id] format at the end of each sentence.
3. If the context doesn't contain enough information, say "I don't have enough information to answer this question."
4. Do not include any information not present in the context.

Answer:"""

# Graph-aware answer prompt
GRAPH_ANSWER_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Your task is to answer the question using information from both retrieved text passages and knowledge graph traversal paths.

Vector Retrieved Passages:
{vector_context}

Graph Traversal Paths:
{graph_context}

Question: {question}

Instructions:
1. Answer the question directly and concisely.
2. Use the graph paths to reason across multiple hops when needed.
3. Cite your sources using [chunk_id] format at the end of each sentence.
4. Reference graph paths as [Path: A --[relation]--> B --[relation]--> C] when using them.
5. If the context doesn't contain enough information, say "I don't have enough information to answer this question."
6. Do not include any information not present in the context.

Answer:"""


def format_context(chunks: list[Chunk], max_tokens: int = 4000) -> tuple[str, list[dict[str, Any]]]:
    """Format chunks into context string with citations."""
    from ..retrieval.embeddings import get_embedder

    embedder = get_embedder()
    context_parts = []
    citations = []
    total_tokens = 0

    for chunk in chunks:
        # Estimate tokens (rough: 1 token ≈ 4 chars)
        chunk_tokens = len(chunk.text) // 4
        if total_tokens + chunk_tokens > max_tokens:
            break

        context_parts.append(f"[Chunk: {chunk.chunk_id}]\n{chunk.text}\n")
        citations.append({
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "text_snippet": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
        })
        total_tokens += chunk_tokens

    return "\n".join(context_parts), citations


def format_graph_paths(paths: list[list[dict[str, Any]]]) -> tuple[str, list[dict[str, Any]]]:
    """Format graph traversal paths into context string with citations."""
    context_parts = []
    citations = []
    path_id = 0

    for path in paths:
        if not path:
            continue
        path_id += 1
        path_str = " -> ".join(
            f"{edge['source']} --[{edge['relation']}]--> {edge['target']}"
            for edge in path
        )
        context_parts.append(f"[Graph Path {path_id}] {path_str}")

        # Add evidence and chunk citations from path edges
        for edge in path:
            if edge.get("evidence"):
                context_parts.append(f"  Evidence: {edge['evidence'][:300]}")
            for chunk_id in edge.get("chunk_ids", []):
                citations.append({
                    "chunk_id": chunk_id,
                    "type": "graph_edge",
                    "path_id": path_id,
                    "relation": edge.get("relation", ""),
                })

    return "\n".join(context_parts), citations


async def generate_answer(
    question: str,
    chunks: list[Chunk],
    max_context_tokens: int = 4000,
) -> GeneratedAnswer:
    """Generate an answer from retrieved chunks."""
    client = create_llm_client()

    context_str, citations = format_context(chunks, max_context_tokens)
    prompt = ANSWER_PROMPT.format(context=context_str, question=question)

    response = await client.complete(prompt, temperature=0.0)

    return GeneratedAnswer(
        answer=response.text.strip(),
        citations=citations,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        total_tokens=response.total_tokens,
        model=client.model_name,
    )


async def generate_answer_with_graph_context(
    question: str,
    retrieval_result: RetrievalResult,
    max_context_tokens: int = 4000,
) -> GeneratedAnswer:
    """Generate an answer using hybrid retrieval result with graph context."""
    client = create_llm_client()

    # Split token budget
    vector_budget = max_context_tokens // 2
    graph_budget = max_context_tokens - vector_budget

    # Format vector context
    vector_context, vector_citations = format_context(retrieval_result.chunks, vector_budget)

    # Format graph paths
    graph_context, graph_citations = format_graph_paths(retrieval_result.traversal_paths)

    # Combine
    combined_context = f"Vector Retrieved Passages:\n{vector_context}\n\nGraph Traversal Paths:\n{graph_context}"

    prompt = GRAPH_ANSWER_PROMPT.format(
        vector_context=vector_context,
        graph_context=graph_context,
        question=question,
    )

    response = await client.complete(prompt, temperature=0.0)

    # Combine citations
    all_citations = vector_citations + graph_citations

    return GeneratedAnswer(
        answer=response.text.strip(),
        citations=all_citations,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        total_tokens=response.total_tokens,
        model=client.model_name,
        traversal_paths=retrieval_result.traversal_paths,
    )