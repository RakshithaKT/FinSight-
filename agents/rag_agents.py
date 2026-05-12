"""
agents/rag_agent.py
The RAG Agent node for LangGraph.

What it does:
  - Receives the current AgentState (which contains the user query)
  - Queries the FAISS vector store for the top-k most relevant document chunks
  - Writes those chunks into state["documents"]
  - Returns the updated state

The orchestrator calls this node when it needs context from uploaded PDFs.
"""

from core.ingestor import load_vectorstore
from core.state import AgentState

# How many chunks to retrieve — 6 gives good coverage without overloading the LLM
TOP_K = 6

# Load vectorstore once at import time (avoid re-loading on every call)
_vectorstore = None


def _get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = load_vectorstore()
    return _vectorstore


def rag_agent_node(state: AgentState) -> AgentState:
    """
    LangGraph node: retrieves relevant document chunks for the current query.

    Args:
        state: The current AgentState flowing through the graph.

    Returns:
        Updated AgentState with `documents` populated.
    """
    query = state["query"]
    print(f"\n[RAG Agent] Searching vector store for: '{query}'")

    try:
        vs = _get_vectorstore()
        results = vs.similarity_search(query, k=TOP_K)

        # Format each chunk with its source file for citation
        formatted_docs = []
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source_file", "unknown")
            page = doc.metadata.get("page", "?")
            formatted_docs.append(
                f"[Source {i}: {source}, page {page}]\n{doc.page_content}"
            )

        print(f"[RAG Agent] Retrieved {len(formatted_docs)} chunks")
        return {**state, "documents": formatted_docs, "error": None}

    except FileNotFoundError as e:
        error_msg = f"RAG Agent error: {e}"
        print(f"[RAG Agent] ⚠️  {error_msg}")
        return {**state, "documents": [], "error": error_msg}

    except Exception as e:
        error_msg = f"RAG Agent unexpected error: {e}"
        print(f"[RAG Agent] ⚠️  {error_msg}")
        return {**state, "documents": [], "error": error_msg}