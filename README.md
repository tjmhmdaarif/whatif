# GraphRAG-MultiHop

Graph-enhanced Retrieval-Augmented Generation for multi-hop question answering.

The system combines dense vector retrieval over a document corpus with traversal
of an extracted knowledge graph, fusing both rankings with Reciprocal Rank
Fusion (RRF). It is designed so any component — LLM provider, vector store, graph
store — can be swapped without touching the retrieval or evaluation logic.

## Status

| Check | Result |
|---|---|
| Tests | 69 passing |
| TypeScript (`tsc --noEmit`) | 0 errors |
| Frontend production build | succeeds |
| API `/health` | healthy, index loaded |

**Answer quality note:** the default provider is `extractive`, a deterministic
offline reader that selects answer spans from the retrieved context. It produces
genuine corpus-derived answers and lets the pipeline run with no API keys, but it
is far weaker than a real LLM. Set `llm.provider` to `anthropic` or `ollama` for
substantially better answer quality.

## Architecture

```
src/graphrag/
├── chunking/      # Document → overlapping chunks
├── retrieval/     # Vector search, graph traversal, RRF hybrid fusion
│   ├── embeddings.py      # sentence-transformers wrapper
│   ├── vector.py          # FAISS store
│   ├── graph_linker.py    # question → graph entity linking
│   ├── graph_traversal.py # BFS traversal from seed entities
│   └── hybrid.py          # RRF fusion
├── graph/         # Knowledge graph store + builder
│   ├── graph_store.py     # NetworkX store, GraphML persistence
│   └── builder.py         # extraction → resolution → graph
├── extraction/    # Entity/relation extraction and resolution
├── generation/    # Answer synthesis with citations
├── llm/           # Provider clients
│   ├── anthropic_client.py
│   ├── ollama_client.py
│   ├── extractive_client.py  # offline span extraction (default)
│   └── mock_client.py        # test fixture
├── eval/          # Metrics, runner, ablation, report
├── data/          # HotpotQA loader
├── api.py         # FastAPI server
└── cli.py         # Typer CLI
```

### Retrieval modes

| Mode | Description |
|---|---|
| `vector` | Dense similarity search over chunk embeddings (FAISS inner product) |
| `graph` | Link question entities to graph nodes, BFS outward, collect their chunks |
| `hybrid` | Both rankings fused with RRF — usually the strongest mode |

## Installation

Requires Python 3.11+.

```bash
pip install -e ".[dev]"
```

## Building an index

```bash
# Vector index
python -m src.graphrag.cli index --limit 150 --split validation

# Knowledge graph (slower: extracts + resolves entities for every chunk)
python -m src.graphrag.cli eval --method graph --limit 150
```

## Running the services

Three independent front doors over the same pipeline:

```bash
python -m src.graphrag.cli serve              # FastAPI  → http://localhost:8000
python -m src.graphrag.cli demo               # Streamlit → http://localhost:8501
cd frontend && npm install && npm run dev     # Next.js  → http://localhost:3000
```

| URL | What it is |
|---|---|
| http://localhost:3000 | Next.js UI (primary interface) |
| http://localhost:8000/docs | Interactive API docs |
| http://localhost:8501 | Streamlit demo |

## CLI

```bash
python -m src.graphrag.cli index   # Build the vector index
python -m src.graphrag.cli query "Who founded Microsoft?" --method hybrid
python -m src.graphrag.cli eval --method all --limit 50   # Evaluate all modes
python -m src.graphrag.cli report                          # Generate results/REPORT.md
python -m src.graphrag.cli serve                           # FastAPI server
python -m src.graphrag.cli demo                            # Streamlit demo
```

## API

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"Who founded Microsoft?","method":"hybrid","top_k":10}'
```

Response:

```json
{
  "answer": "...",
  "chunks": ["Microsoft_chunk_0", "..."],
  "latency_ms": 42.1,
  "llm_calls": 1,
  "citations": []
}
```

## Evaluation

```bash
python -m src.graphrag.cli eval --method all --limit 50 --seed 42
python -m src.graphrag.cli report
```

Reports exact match, token-level F1, supporting-fact recall, latency, LLM call
count, and 95% paired-bootstrap confidence intervals, plus a significance test
for hybrid vs. vector.

## Configuration

Everything is driven by [`config.yaml`](config.yaml); secrets come from `.env`
(copy [`.env.example`](.env.example)). Key settings:

```yaml
llm:
  provider: "extractive"   # anthropic | ollama | extractive | mock

retrieval:
  vector_top_k: 10
  graph_max_hops: 2
  graph_top_k: 10
  hybrid_fusion_weight: 0.5
  rrf_k: 60
```

## Testing

```bash
python -m pytest tests          # 69 tests
ruff check src tests            # lint
mypy src                        # types
```

> **Note:** the test suite writes fixtures into the shared `data/` directory, so
> it overwrites `data/faiss_index` and `data/knowledge_graph.graphml`. Rebuild
> both after running tests.

## Known limitations

1. **Extractive answering is weak.** Span selection without a language model
   returns overly broad spans; EM/F1 are correspondingly low. Use `anthropic` or
   `ollama` for real quality.
2. **Index and graph are sample-sized.** The committed artifacts cover ~150
   HotpotQA questions; rebuild at a larger `--limit` for meaningful results.
3. **NetworkX graph is in-memory.** It does not scale to millions of nodes —
   the `neo4j` backend is the intended path for production scale.
4. **Extraction cost.** LLM-based entity/relation extraction is the slowest and
   most expensive stage; entity resolution over ~17k entities dominates build time.
5. **Error propagation.** Poor extraction degrades the graph, which degrades
   graph and hybrid retrieval.

## Tech stack

Python 3.11+ · FastAPI · Streamlit · FAISS · NetworkX · sentence-transformers ·
Typer · Pydantic — with a Next.js 14 + TypeScript + Tailwind frontend in `frontend/`.
