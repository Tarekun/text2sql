from langgraph.graph import StateGraph, START
from langchain.messages import HumanMessage
from graph import *


def compile():
    # Build workflow
    agent_builder = StateGraph(state_schema=MessagesState)
    # Add nodes
    agent_builder.add_node("llm_call", llm_call)  # type:ignore
    agent_builder.add_node("tool_node", tool_node)  # type:ignore
    # Add edges to connect nodes
    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    agent_builder.add_edge("tool_node", "llm_call")

    # Compile the agent
    agent = agent_builder.compile()

    return agent


def call(agent, message: str):
    messages = agent.invoke({"messages": [HumanMessage(content=message)]})
    return messages["messages"][-1]
