"""
LangGraph StateGraph definition.

Graph topology:
    intent_node → tool_node → synthesis_node → END

All three nodes run sequentially. Error states are carried in AgentState
fields (intent_error, tool_error) rather than exceptions, so the graph
always reaches synthesis and returns a graceful answer.
"""

import logging
from langgraph.graph import StateGraph, END

from app.agent.state import AgentState
from app.agent.nodes import intent_node, tool_node, synthesis_node

logger = logging.getLogger(__name__)


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("intent", intent_node)
    graph.add_node("tool", tool_node)
    graph.add_node("synthesis", synthesis_node)

    # Linear edges
    graph.set_entry_point("intent")
    graph.add_edge("intent", "tool")
    graph.add_edge("tool", "synthesis")
    graph.add_edge("synthesis", END)

    return graph.compile()


# Module-level compiled graph — built once at import time, reused per request
agent_graph = build_graph()