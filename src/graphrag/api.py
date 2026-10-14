"""FastAPI server for GraphRAG-MultiHop."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import load_config, set_config, get_config
from .logging_setup import setup_logging, get_logger
from .eval.runner import UnifiedRAGPipeline
from .graph import create_graph_store

logger = get_logger(__name__)

# Global pipeline instance
_pipeline: UnifiedRAGPipeline | None = None


class QueryRequest(BaseModel):
    """Query request model."""
    question: str
    method: str = "vector"
    top_k: int = 10
    max_hops: int = 2


class QueryResponse(BaseModel):
    """Query response model."""
    answer: str
    chunks: list[str]
    latency_ms: float
    llm_calls: int
    citations: list[dict[str, Any]] = []


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    mode: str
    index_loaded: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global _pipeline
    
    # Load config
    config = load_config()
    set_config(config)
    setup_logging()
    
    logger.info("Starting GraphRAG-MultiHop API server...")
    
    # Try to load existing index
    index_path = Path(config.vector_store.faiss.index_path)
    graph_store = create_graph_store()
    
    _pipeline = UnifiedRAGPipeline(mode="hybrid", graph_store=graph_store)
    
    if index_path.exists():
        try:
            _pipeline.load_index(index_path)
            logger.info(f"Loaded index from {index_path}")
        except Exception as e:
            logger.warning(f"Failed to load index: {e}")
    else:
        logger.warning(f"Index not found at {index_path}, server will need manual index building")
    
    yield
    
    logger.info("Shutting down GraphRAG-MultiHop API server...")


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(
        title="GraphRAG-MultiHop API",
        description="Graph-enhanced Retrieval-Augmented Generation for Multi-hop QA",
        version="0.1.0",
        lifespan=lifespan,
    )
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        global _pipeline
        return HealthResponse(
            status="healthy",
            mode=_pipeline.mode if _pipeline else "unknown",
            index_loaded=_pipeline is not None and _pipeline.vector_store.index is not None,
        )
    
    @app.post("/query", response_model=QueryResponse)
    async def query(request: QueryRequest):
        """Query the GraphRAG system."""
        global _pipeline
        
        if _pipeline is None:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        if _pipeline.vector_store.index is None:
            raise HTTPException(status_code=503, detail="Index not loaded. Please build index first.")
        
        try:
            answer, chunks, latency, calls = await _pipeline.query(
                request.question,
                top_k=request.top_k,
                max_hops=request.max_hops,
                mode=request.method,
            )
            
            return QueryResponse(
                answer=answer,
                chunks=chunks,
                latency_ms=latency,
                llm_calls=calls,
            )
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/rebuild")
    async def rebuild_index(
        dataset: str = "hotpotqa",
        split: str = "validation",
        limit: int = 300,
        seed: int = 42,
        method: str = "hybrid",
    ):
        """Rebuild the index from dataset."""
        global _pipeline
        
        if _pipeline is None:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        try:
            from .data.loader import load_hotpotqa
            
            examples, corpus = load_hotpotqa(split=split, limit=limit, seed=seed)
            
            # Create new pipeline with requested mode
            graph_store = create_graph_store() if method in ("graph", "hybrid") else None
            _pipeline = UnifiedRAGPipeline(mode=method, graph_store=graph_store)
            
            await _pipeline.build_index(corpus)
            _pipeline.save_index(get_config().vector_store.faiss.index_path)
            
            return {"status": "success", "chunks": len(_pipeline.chunk_id_to_title)}
        except Exception as e:
            logger.error(f"Index rebuild failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/stats")
    async def get_stats():
        """Get system statistics."""
        global _pipeline
        
        if _pipeline is None:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
        stats = {
            "mode": _pipeline.mode,
            "index_loaded": _pipeline.vector_store.index is not None,
            "num_chunks": len(_pipeline.chunk_id_to_title) if _pipeline.chunk_id_to_title else 0,
        }
        
        if _pipeline.graph_store:
            stats["graph"] = _pipeline.graph_store.get_stats()
        
        return stats
    
    return app


if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)