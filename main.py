"""
main.py
The Orchestrator — builds the LangGraph state machine and runs the full agent loop.

Agent flow:
    START
      │
      ▼
  [rag_agent]       ← retrieves relevant chunks from FAISS
      │
      ▼
  [search_agent]    ← fetches live news via Tavily
      │
      ▼
  [analyst_agent]   ← synthesises report via Groq LLM
      │
      ├─ next="retry" AND retry_count < 2 ──► [analyst_agent] (retry loop)
      │
      └─ next="END"  ──► END

This is the self-correcting retry logic mentioned in your CV.

Usage:
    python main.py
or import run_analysis() from app.py (Streamlit UI).
"""

from langgraph.graph import StateGraph, END
from core.state import AgentState
from agents.rag_agents import rag_agent_node
from agents.search_agent import search_agent_node
from agents.analyst_agent import analyst_agent_node


# ── Build the graph ────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Constructs and compiles the LangGraph state machine.
    Returns a compiled graph ready to invoke.
    """
    graph = StateGraph(AgentState)

    # Register nodes (each node = a function that receives state and returns state)
    graph.add_node("rag_agent", rag_agent_node)
    graph.add_node("search_agent", search_agent_node)
    graph.add_node("analyst_agent", analyst_agent_node)

    # Linear edges: START → RAG → SEARCH → ANALYST
    graph.set_entry_point("rag_agent")
    graph.add_edge("rag_agent", "search_agent")
    graph.add_edge("search_agent", "analyst_agent")

    # Conditional edge on analyst output: retry OR end
    def route_after_analyst(state: AgentState) -> str:
        """
        Conditional edge function.
        Reads state["next"] to decide where to route after the analyst runs.
        This is the self-correcting retry logic.
        """
        if state.get("next") == "retry" and state.get("retry_count", 0) < 2:
            print(f"[Orchestrator] Retrying analyst (attempt {state['retry_count']})...")
            return "analyst_agent"   # loop back
        return END                    # done

    graph.add_conditional_edges(
        "analyst_agent",
        route_after_analyst,
        {
            "analyst_agent": "analyst_agent",
            END: END,
        },
    )

    return graph.compile()


# ── Public API ─────────────────────────────────────────────────────────────────

def run_analysis(query: str) -> dict:
    """
    Run the full FinSight agent pipeline for a given query.

    Args:
        query: Company name or financial question, e.g. "Apple Q3 2024 earnings"

    Returns:
        Final AgentState dict containing 'report', 'documents', 'web_results', 'error'
    """
    print(f"\n{'='*60}")
    print(f"FinSight Agent — Analyzing: {query}")
    print(f"{'='*60}")

    # Initial state — all fields set to defaults
    initial_state: AgentState = {
        "messages": [],
        "query": query,
        "documents": [],
        "web_results": [],
        "report": None,
        "next": None,
        "error": None,
        "retry_count": 0,
    }

    graph = build_graph()
    final_state = graph.invoke(initial_state)

    print(f"\n{'='*60}")
    if final_state.get("report"):
        print("✅ Report generated successfully.")
    else:
        print("⚠️  Report generation failed — check errors above.")
    print(f"{'='*60}\n")

    return final_state


# ── CLI entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Apple Inc annual report 2023"
    result = run_analysis(query)

    if result.get("report"):
        print("\n" + result["report"])
    else:
        print(f"\nError: {result.get('error', 'Unknown error')}")