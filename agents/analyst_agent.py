"""
agents/analyst_agent.py
The Analyst Agent — the "brain" of FinSight.

What it does:
  - Takes the retrieved documents (from RAG agent) and web results (from search agent)
  - Constructs a detailed prompt for the LLM
  - Calls Groq's Llama 3.x to generate a structured analyst report
  - Writes the report into state["report"]

This is where all the retrieved context is synthesised into something useful.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from core.state import AgentState

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# We use llama-3.3-70b-versatile — best free model on Groq for reasoning tasks.
# Fallback: llama3-8b-8192 (faster, smaller, still good)
GROQ_MODEL = "llama-3.3-70b-versatile"


def _build_llm() -> ChatGroq:
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=0.2,       # low temp = factual, consistent outputs
        max_tokens=2048,
    )


SYSTEM_PROMPT = """You are FinSight, an expert financial analyst AI. 
Your job is to synthesise information from financial reports and live market data 
into a structured, professional analyst report.

Always structure your report with these exact sections:
1. Executive Summary (3-4 sentences)
2. Financial Performance (revenue, profit, margins, YoY changes)
3. Key Business Highlights (major events, product launches, acquisitions)
4. Risks and Concerns (be honest about negatives)
5. Market Context (how does live news / market data contextualise this?)
6. Analyst Verdict (buy/hold/watch — with clear reasoning)

For each specific claim, reference which source it came from (e.g. [Source 2]).
If data is unavailable or uncertain, say so clearly — never hallucinate numbers.
Use professional financial language but keep it readable."""


def analyst_agent_node(state: AgentState) -> AgentState:
    """
    LangGraph node: generates the structured analyst report using Groq LLM.

    Args:
        state: Current AgentState (must have documents and web_results populated).

    Returns:
        Updated state with `report` populated.
    """
    query = state["query"]
    documents = state.get("documents", [])
    web_results = state.get("web_results", [])

    print(f"\n[Analyst Agent] Generating report for: '{query}'")

    # ── Build context string ──────────────────────────────────────────────────
    doc_context = "\n\n".join(documents) if documents else "No PDF documents retrieved."
    web_context = "\n\n".join(web_results) if web_results else "No web data available."

    user_prompt = f"""Analyze the following company/topic: {query}

=== FINANCIAL DOCUMENT EXCERPTS (from SEC filings / annual reports) ===
{doc_context}

=== LIVE MARKET & NEWS DATA ===
{web_context}

Generate a complete structured analyst report following the format in your instructions.
Be specific, cite your sources, and flag any gaps in the data."""

    # ── Call Groq ─────────────────────────────────────────────────────────────
    try:
        if not GROQ_API_KEY or GROQ_API_KEY == "gsk":
            raise ValueError(
                "GROQ_API_KEY not set. Add it to your .env file. "
                "Get a free key at: https://console.groq.com"
            )

        llm = _build_llm()
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        print("[Analyst Agent] Calling Groq API...")
        response = llm.invoke(messages)
        report = response.content

        print(f"[Analyst Agent] Report generated ({len(report)} chars)")
        return {**state, "report": report, "error": None, "next": "END"}

    except Exception as e:
        error_msg = f"Analyst Agent error: {e}"
        print(f"[Analyst Agent] ⚠️  {error_msg}")

        # If we've retried fewer than 2 times, signal orchestrator to retry
        retry_count = state.get("retry_count", 0)
        if retry_count < 2:
            return {
                **state,
                "report": None,
                "error": error_msg,
                "next": "retry",
                "retry_count": retry_count + 1,
            }
        else:
            # Give up after 2 retries — return partial report
            fallback = (
                f"⚠️ Report generation failed after {retry_count} retries.\n"
                f"Last error: {error_msg}\n\n"
                "Partial context was retrieved — please check your GROQ_API_KEY "
                "and try again."
            )
            return {**state, "report": fallback, "error": error_msg, "next": "END"}