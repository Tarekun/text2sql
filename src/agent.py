from langgraph.graph import StateGraph, START
from langchain.messages import HumanMessage
from src.graph import *
from src.llm_backend import instantiate_llm


def compile():
    instantiate_llm()
    # Build workflow
    agent_builder = StateGraph(state_schema=MessagesState)
    # Add nodes
    agent_builder.add_node("generate_sql", node_generate_sql)  # type:ignore
    agent_builder.add_node("execute_sql", node_execute_sql)  # type:ignore
    agent_builder.add_node("answer", node_final_answer)  # type:ignore
    # Add edges to connect nodes
    agent_builder.add_edge(START, "generate_sql")
    agent_builder.add_edge("generate_sql", "execute_sql")
    agent_builder.add_edge("execute_sql", "answer")
    agent_builder.add_edge("answer", END)
    # agent_builder.add_conditional_edges(
    #     "generate_sql", should_continue, ["tool_node", END]
    # )

    # Compile the agent
    agent = agent_builder.compile()

    return agent


def call(agent, message: str):
    messages = agent.invoke({"messages": [HumanMessage(content=message)]})
    return messages["messages"][-1]
