"""LangGraph StateGraph compiler and workflow definition for DrakeAI."""

import logging
from typing import Literal

from langgraph.graph import END, StateGraph

from backend.graph.nodes import (
    guardrail_intent_node,
    hybrid_retrieval_node,
    out_of_scope_node,
    reasoning_agent_node,
    response_formatter_node,
    vet_track_node,
)
from backend.graph.state import AgentState

logger = logging.getLogger(__name__)

MAX_RETRIES = 2


def route_guardrail(state: AgentState) -> Literal["hybrid_retrieval", "out_of_scope"]:
    """Conditional router based on relevance determination."""
    if state.get("is_relevant", False):
        return "hybrid_retrieval"
    return "out_of_scope"


def route_vetting(state: AgentState) -> Literal["hybrid_retrieval", "reasoning_agent"]:
    """
    Conditional router checking candidate track vetting.
    If vetted successfully, continues to reasoning_agent.
    If candidate failed vetting, retries retrieval up to MAX_RETRIES times.
    """
    if state.get("is_vetted", False):
        return "reasoning_agent"

    retry_count = state.get("retry_count", 0)
    if retry_count <= MAX_RETRIES:
        logger.info("Vetting failed; retrying retrieval (attempt %d/%d)...", retry_count, MAX_RETRIES)
        return "hybrid_retrieval"

    logger.warning("Max retrieval retries (%d) reached. Proceeding with available context.", MAX_RETRIES)
    return "reasoning_agent"


def create_chat_graph():
    """Builds and compiles the LangGraph State machine for the RAG pipeline."""
    workflow = StateGraph(AgentState)

    # 1. Add nodes
    workflow.add_node("guardrail_intent", guardrail_intent_node)
    workflow.add_node("hybrid_retrieval", hybrid_retrieval_node)
    workflow.add_node("vet_track", vet_track_node)
    workflow.add_node("reasoning_agent", reasoning_agent_node)
    workflow.add_node("response_formatter", response_formatter_node)
    workflow.add_node("out_of_scope", out_of_scope_node)

    # 2. Set Entry Point
    workflow.set_entry_point("guardrail_intent")

    # 3. Add Conditional Edge for Intent Router
    workflow.add_conditional_edges(
        "guardrail_intent",
        route_guardrail,
        {
            "hybrid_retrieval": "hybrid_retrieval",
            "out_of_scope": "out_of_scope"
        }
    )

    # 4. In-Scope Pipeline edges: retrieval -> vet_track
    workflow.add_edge("hybrid_retrieval", "vet_track")

    # 5. Conditional Edge for Vetting & Retry loop
    workflow.add_conditional_edges(
        "vet_track",
        route_vetting,
        {
            "hybrid_retrieval": "hybrid_retrieval",
            "reasoning_agent": "reasoning_agent"
        }
    )

    # 6. Post-vetting pipeline edges
    workflow.add_edge("reasoning_agent", "response_formatter")
    workflow.add_edge("response_formatter", END)

    # 7. Out-of-Scope Exit edge
    workflow.add_edge("out_of_scope", END)

    return workflow.compile()


# Precompiled singleton graph instance
chat_graph = create_chat_graph()
