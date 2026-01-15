from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langchain.messages import HumanMessage
from src.graph import *
from src.llm_backend import instantiate_llm


def compile() -> CompiledStateGraph:
    instantiate_llm()
    # Build workflow
    agent_builder = StateGraph(state_schema=MessagesState)
    # Add nodes
    agent_builder.add_node(NODE_GENERATE_NAME, node_generate_sql)  # type:ignore
    agent_builder.add_node(NODE_EXECUTE_NAME, node_execute_sql)  # type:ignore
    agent_builder.add_node(NODE_ANSWER_NAME, node_final_answer)  # type:ignore
    # Add edges to connect nodes
    agent_builder.add_edge(START, NODE_GENERATE_NAME)
    agent_builder.add_edge(NODE_GENERATE_NAME, NODE_EXECUTE_NAME)
    agent_builder.add_conditional_edges(
        NODE_EXECUTE_NAME,
        edge_execution_success_check,
        [NODE_ANSWER_NAME, NODE_GENERATE_NAME],
    )
    agent_builder.add_edge(NODE_EXECUTE_NAME, NODE_ANSWER_NAME)
    agent_builder.add_edge(NODE_ANSWER_NAME, END)
    # agent_builder.add_conditional_edges(
    #     NODE_GENERATE_NAME, should_continue, ["tool_node", END]
    # )

    # Compile the agent
    agent = agent_builder.compile()

    return agent


def call(agent: CompiledStateGraph, message: str):
    messages = agent.invoke({"messages": [HumanMessage(content=message)]})
    return content_as_string(messages["messages"][-1])
