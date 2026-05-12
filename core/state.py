"""
core/state.py
Defines the shared state that flows between all agents in the LangGraph graph.
Think of this as the "memory" that every node reads from and writes to.
"""

from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    The single state object that travels through every node in the graph.

    - messages      : Full conversation + tool call history (auto-merged by add_messages)
    - query         : The user's original question / company to analyze
    - documents     : Chunks retrieved from FAISS (populated by RAG agent)
    - web_results   : Live news / data fetched by the search agent
    - report        : Final structured analyst report (populated by analyst agent)
    - next          : Routing signal used by the orchestrator's conditional edge
    - error         : Any error string; lets the orchestrator decide to retry
    - retry_count   : How many times we've retried a failing step (self-healing logic)
    """
    messages: Annotated[list, add_messages]
    query: str
    documents: list[str]
    web_results: list[str]
    report: Optional[str]
    next: Optional[str]
    error: Optional[str]
    retry_count: int