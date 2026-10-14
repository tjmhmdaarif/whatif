"""Streamlit demo for GraphRAG-MultiHop."""

import asyncio
import json
import os
import sys
from pathlib import Path

import streamlit as st

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.graphrag.config import load_config, set_config
from src.graphrag.logging_setup import setup_logging, get_logger
from src.graphrag.eval.runner import UnifiedRAGPipeline
from src.graphrag.graph import create_graph_store

logger = get_logger(__name__)

# Page config
st.set_page_config(
    page_title="GraphRAG-MultiHop Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_pipeline(method: str = "hybrid"):
    """Get or create the pipeline."""
    config = load_config()
    set_config(config)
    setup_logging()

    graph_store = create_graph_store() if method in ("graph", "hybrid") else None
    pipeline = UnifiedRAGPipeline(mode=method, graph_store=graph_store)

    index_path = Path(config.vector_store.faiss.index_path)
    if index_path.exists():
        pipeline.load_index(index_path)
        st.session_state.index_loaded = True
    else:
        st.session_state.index_loaded = False

    return pipeline


def main():
    """Main Streamlit app."""
    st.title("🔍 GraphRAG-MultiHop Demo")
    st.caption("Graph-enhanced Retrieval-Augmented Generation for Multi-hop QA")

    # Sidebar
    with st.sidebar:
        st.header("Settings")

        method = st.selectbox(
            "Retrieval Method",
            ["vector", "graph", "hybrid"],
            index=2,
            help="Vector: semantic search only. Graph: knowledge graph traversal. Hybrid: both combined with RRF.",
        )

        top_k = st.slider("Top-K Chunks", 1, 20, 10)
        max_hops = st.slider("Max Graph Hops", 1, 3, 2)

        st.divider()

        # Index status
        st.subheader("Index Status")
        if st.session_state.get("index_loaded", False):
            st.success("✅ Index loaded")
        else:
            st.warning("⚠️ Index not loaded")

        # Rebuild index button
        if st.button("Rebuild Index", type="primary"):
            with st.spinner("Rebuilding index..."):
                pipeline = get_pipeline(method)
                pipeline.build_index()
                st.success("Index rebuilt!")

        st.divider()

        # System info
        st.subheader("System Info")
        config = load_config()
        st.json({
            "embedding_model": config.embedding.model_name,
            "vector_store": config.vector_store.type,
            "graph_store": config.graph_store.type,
            "llm_provider": config.llm.provider,
        })

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Ask a Question")

        # Question input
        question = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="e.g., What is the capital of France?",
        )

        # Example questions
        st.caption("Example questions:")
        example_cols = st.columns(3)
        examples = [
            "Who founded Microsoft?",
            "Where is the Eiffel Tower located?",
            "When was Barack Obama born?",
        ]
        for i, ex in enumerate(examples):
            if example_cols[i].button(ex, key=f"ex_{i}"):
                st.session_state.question = ex
                st.rerun()

        # Query button
        if st.button("Query", type="primary", disabled=not question.strip()):
            if not st.session_state.get("index_loaded", False):
                st.error("Index not loaded. Please rebuild the index first.")
            else:
                with st.spinner("Querying..."):
                    pipeline = get_pipeline(method)
                    answer, chunks, latency, calls = asyncio.run(
                        pipeline.query(question, top_k=top_k, max_hops=max_hops)
                    )

                    st.session_state.last_answer = answer
                    st.session_state.last_chunks = chunks
                    st.session_state.last_latency = latency
                    st.session_state.last_calls = calls

        # Display answer
        if "last_answer" in st.session_state:
            st.subheader("Answer")
            st.write(st.session_state.last_answer)

            st.caption(f"Latency: {st.session_state.last_latency:.1f}ms | LLM calls: {st.session_state.last_calls} | Chunks retrieved: {len(st.session_state.last_chunks)}")

    with col2:
        st.subheader("Retrieved Chunks")
        if "last_chunks" in st.session_state and st.session_state.last_chunks:
            for i, chunk_id in enumerate(st.session_state.last_chunks):
                with st.expander(f"Chunk {i+1}: {chunk_id}"):
                    st.code(chunk_id)
        else:
            st.info("No chunks retrieved yet")

    # Footer
    st.divider()
    st.caption("GraphRAG-MultiHop v0.1.0 | Built with FastAPI, Streamlit, FAISS, NetworkX, and sentence-transformers")


if __name__ == "__main__":
    main()