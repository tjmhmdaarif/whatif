"""Data loading and corpus building for GraphRAG."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from datasets import load_dataset
from pydantic import BaseModel

from ..config import get_config
from ..logging_setup import get_logger

logger = get_logger(__name__)


class HotpotQAExample(BaseModel):
    """A single HotpotQA example."""
    id: str
    question: str
    answer: str
    type: str  # "bridge" or "comparison"
    level: str  # "easy", "medium", "hard"
    supporting_facts: list[list[str | int]]  # [[title, sentence_id], ...]
    context: list[list[Any]]  # [[title, [sentences]], ...]


@dataclass
class Document:
    """A document with title and content."""
    doc_id: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Corpus:
    """Collection of documents."""
    documents: list[Document] = field(default_factory=list)
    doc_id_to_index: dict[str, int] = field(default_factory=dict)

    def add_document(self, doc: Document) -> None:
        self.doc_id_to_index[doc.doc_id] = len(self.documents)
        self.documents.append(doc)

    def get_document(self, doc_id: str) -> Document | None:
        idx = self.doc_id_to_index.get(doc_id)
        if idx is not None:
            return self.documents[idx]
        return None


def load_hotpotqa(
    split: str = "validation",
    limit: int | None = None,
    seed: int = 42,
) -> tuple[list[HotpotQAExample], Corpus]:
    """Load HotpotQA dataset and build corpus from context paragraphs.

    Args:
        split: Dataset split to load (train, validation, test)
        limit: Maximum number of examples to load (None for all)
        seed: Random seed for sampling

    Returns:
        Tuple of (examples, corpus)
    """
    logger.info(f"Loading HotpotQA {split} split...")
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)

    examples = []
    for item in dataset:
        # Handle new dataset format where supporting_facts and context are dicts
        sf = item["supporting_facts"]
        if isinstance(sf, dict):
            supporting_facts = list(zip(sf["title"], sf["sent_id"]))
        else:
            supporting_facts = sf

        ctx = item["context"]
        if isinstance(ctx, dict):
            context = list(zip(ctx["title"], ctx["sentences"]))
        else:
            context = ctx

        example = HotpotQAExample(
            id=item["id"],
            question=item["question"],
            answer=item["answer"],
            type=item["type"],
            level=item["level"],
            supporting_facts=supporting_facts,
            context=context,
        )
        examples.append(example)

    # Shuffle and limit
    random.seed(seed)
    random.shuffle(examples)
    if limit:
        examples = examples[:limit]

    logger.info(f"Loaded {len(examples)} examples")

    # Build corpus from context paragraphs
    corpus = Corpus()
    seen_titles = set()

    for example in examples:
        for title, sentences in example.context:
            if title in seen_titles:
                continue
            seen_titles.add(title)

            text = " ".join(sentences)
            doc = Document(
                doc_id=title,
                title=title,
                text=text,
                metadata={"source": "hotpotqa"},
            )
            corpus.add_document(doc)

    logger.info(f"Built corpus with {len(corpus.documents)} unique documents")
    return examples, corpus


def load_musique(
    split: str = "validation",
    limit: int | None = None,
    seed: int = 42,
) -> tuple[list[dict], Corpus]:
    """Load MuSiQue dataset (placeholder for future implementation)."""
    raise NotImplementedError("MuSiQue loading not yet implemented")


def load_2wikimultihopqa(
    split: str = "validation",
    limit: int | None = None,
    seed: int = 42,
) -> tuple[list[dict], Corpus]:
    """Load 2WikiMultiHopQA dataset (placeholder for future implementation)."""
    raise NotImplementedError("2WikiMultiHopQA loading not yet implemented")


def save_corpus(corpus: Corpus, path: str | Path) -> None:
    """Save corpus to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "documents": [
            {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "text": doc.text,
                "metadata": doc.metadata,
            }
            for doc in corpus.documents
        ]
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved corpus to {path}")


def load_corpus(path: str | Path) -> Corpus:
    """Load corpus from JSON file."""
    path = Path(path)
    with open(path, "r") as f:
        data = json.load(f)

    corpus = Corpus()
    for doc_data in data["documents"]:
        doc = Document(**doc_data)
        corpus.add_document(doc)
    logger.info(f"Loaded corpus with {len(corpus.documents)} documents from {path}")
    return corpus


def get_dataset_loader(dataset_name: str):
    """Get the appropriate dataset loader function."""
    loaders = {
        "hotpotqa": load_hotpotqa,
        "musique": load_musique,
        "2wikimultihopqa": load_2wikimultihopqa,
    }
    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return loaders[dataset_name]