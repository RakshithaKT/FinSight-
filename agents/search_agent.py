"""
agents/search_agent.py
The Web Search Agent node for LangGraph.

What it does:
  - Takes the query from AgentState
  - Calls Tavily's search API to fetch live news and market data
  - Writes results into state["web_results"]
  - Falls back gracefully if Tavily API key is missing

Tavily is purpose-built for LLM agents — it returns clean, relevant text
instead of raw HTML. Sign up free at: https://tavily.com
"""

import os
from dotenv import load_dotenv
from core.state import AgentState

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


def search_agent_node(state: AgentState) -> AgentState:
    """
    LangGraph node: fetches live news and market data for the query.

    Args:
        state: Current AgentState.

    Returns:
        Updated state with `web_results` populated.
    """
    query = state["query"]
    print(f"\n[Search Agent] Searching web for: '{query}'")

    if not TAVILY_API_KEY or TAVILY_API_KEY == "your_tavily_api_key_here":
        print("[Search Agent] ⚠️  No Tavily API key — skipping web search.")
        return {
            **state,
            "web_results": [
                "Web search skipped: no Tavily API key configured. "
                "Add TAVILY_API_KEY to your .env file (free at tavily.com)."
            ],
        }

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=TAVILY_API_KEY)

        # Search for recent financial news about the company/topic
        search_query = f"{query} financial results earnings revenue 2024 2025"
        response = client.search(
            query=search_query,
            search_depth="basic",
            max_results=5,
            include_answer=True,  # Tavily synthesises a short answer too
        )

        results = []

        # Include Tavily's synthesised answer if available
        if response.get("answer"):
            results.append(f"[Web Summary]\n{response['answer']}")

        # Include individual search results
        for i, result in enumerate(response.get("results", []), 1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            content = result.get("content", "")
            results.append(f"[Web Result {i}: {title}]\nURL: {url}\n{content}")

        print(f"[Search Agent] Retrieved {len(results)} web results")
        return {**state, "web_results": results, "error": None}

    except Exception as e:
        error_msg = f"Search Agent error: {e}"
        print(f"[Search Agent] ⚠️  {error_msg}")
        return {
            **state,
            "web_results": [f"Web search failed: {error_msg}"],
            "error": error_msg,
        }