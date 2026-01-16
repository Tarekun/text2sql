from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langchain.messages import (
    ToolMessage,
    SystemMessage,
    AnyMessage,
    AIMessage,
    HumanMessage,
)
from langgraph.prebuilt import ToolNode
import operator
from typing_extensions import TypedDict, Annotated
from src.agent.llm_backend import get_llm, instantiate_llm
from src.agent.tools import tool_list
from src.config import Config
from src.db import get_table_metadata, run_sql_query
from src.prompts import prompts, Prompts
from src.utils import get_user_question, content_as_string
from src.logger import logger


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    metadata: str
    fetched_data: str
    retry_count: int
    sufficient_context: bool


NODE_GENERATE_NAME = "generate_sql"
NODE_EXECUTE_NAME = "execute_sql"
NODE_ANSWER_NAME = "answer"
NODE_TOOLS_NAME = "tools"
NODE_POST_TOOL_NAME = "tool_state_mngt"
NODE_SUFFEVAL_NAME = "context_eval"

EXECUTION_ERROR_PREFIX = "SQL execution error:"
max_retries: int = 5
local_prompts: Prompts = prompts["en"]


################################### GRAPH DEFINITION
def compile(config: Config) -> CompiledStateGraph:
    global max_retries

    instantiate_llm(config)
    _set_prompt_language(config)
    max_retries = config.max_retries

    # Build workflow
    agent_builder = StateGraph(state_schema=MessagesState)
    # Add nodes
    agent_builder.add_node(NODE_GENERATE_NAME, _node_generate_sql)
    agent_builder.add_node(NODE_TOOLS_NAME, ToolNode(tool_list))
    agent_builder.add_node(NODE_POST_TOOL_NAME, _node_post_tool)
    agent_builder.add_node(NODE_ANSWER_NAME, _node_final_answer)
    agent_builder.add_node(NODE_SUFFEVAL_NAME, _node_sufficiency_evaluation)
    # Add edges to connect nodes
    agent_builder.add_edge(START, NODE_GENERATE_NAME)
    agent_builder.add_conditional_edges(
        NODE_GENERATE_NAME, _edge_skip_execution, [NODE_TOOLS_NAME, NODE_ANSWER_NAME]
    )
    agent_builder.add_edge(NODE_TOOLS_NAME, NODE_POST_TOOL_NAME)
    agent_builder.add_edge(NODE_POST_TOOL_NAME, NODE_SUFFEVAL_NAME)
    agent_builder.add_conditional_edges(
        NODE_SUFFEVAL_NAME,
        _edge_sufficiency_evaluation,
        [NODE_GENERATE_NAME, NODE_ANSWER_NAME],
    )
    # agent_builder.add_edge(NODE_TOOLS_NAME, NODE_POST_TOOL_NAME)
    # agent_builder.add_conditional_edges(
    #     NODE_POST_TOOL_NAME,
    #     _edge_execution_success_check,
    #     [NODE_GENERATE_NAME, NODE_ANSWER_NAME],
    # )
    agent_builder.add_edge(NODE_ANSWER_NAME, END)

    # Compile the agent
    agent = agent_builder.compile()

    return agent


def call(agent: CompiledStateGraph, message: str):
    messages = agent.invoke({"messages": [HumanMessage(content=message)]})
    return content_as_string(messages["messages"][-1])


################################### GRAPH DEFINITION


################################### GRAPH COMPONENTS
def _node_generate_sql(state: MessagesState):
    logger.debug("node: main control node")
    if state.get("retry_count", 0) > max_retries:
        logger.warning("Query generation failed too many times. Skipping")
        return {
            "messages": [
                HumanMessage(content="Query generation failed too many times. Skipping")
            ],
        }

    llm = get_llm()
    user_query = get_user_question(state)
    metadata = state.get("metadata", "No metadata fetched yet")
    data = state.get("fetched_data", "No rows fetched yet")
    system_prompt = local_prompts.sql_generation.format(
        metadata=metadata,
        data=data,
        db_kind="BigQuery",
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query),
    ]
    if _did_last_execution_fail(state):
        last_msg = state["messages"][-1]
        # add to context the failed query
        messages.append(state["messages"][-2])
        # add to context the error message
        messages.append(last_msg)
        messages.append(
            HumanMessage(
                content="The above SQL query failed. Please analyze the error and generate a corrected query."
            )
        )

    response = llm.invoke(messages)

    return {
        "messages": [response],
    }


def _node_post_tool(state: MessagesState):
    logger.debug("node: post tool state management")
    retry = 0
    if _did_last_execution_fail(state):
        logger.warning("SQL execution failed, retrying")
        retry = state.get("retry_count", 0) + 1

    return {
        "retry_count": retry,
        "metadata": _get_fetched_metadata(state),
        "fetched_data": _get_fetched_data(state),
    }


def _node_final_answer(state: MessagesState):
    logger.debug("node: final answer")
    llm = get_llm(with_tools=False)
    user_query = get_user_question(state)
    metadata = state.get("metadata", "No metadata fetched yet")
    sql_result = state.get("fetched_data", "No rows fetched yet")

    if sql_result is None:
        logger.debug("Final answer has no SQL data available")
    if metadata is None:
        logger.debug("Final answer has not metadata available")
    system_prompt = local_prompts.final_answer.format(
        data=sql_result,
        metadata=metadata,
    )
    messages = [("system", system_prompt), ("human", user_query)]

    response = llm.invoke(messages)
    return {"messages": [response]}


def _node_sufficiency_evaluation(state: MessagesState):
    logger.debug("node: context evaluation")
    llm = get_llm()
    user_query = get_user_question(state)
    metadata = state.get("metadata", "No metadata fetched yet")
    data = state.get("fetched_data", "No rows fetched yet")
    system_prompt = local_prompts.evaluate_context.format(
        metadata=metadata, data=data, user_query=user_query
    )

    response = llm.invoke(
        [
            # gemini api always wants a system message AND a human message
            SystemMessage(
                content="You are a strict evaluator. Respond only with DATA IS EXAUSTIVE or MISSING DATA."
            ),
            HumanMessage(content=system_prompt),
        ]
    )
    response = content_as_string(response)
    return {"sufficient_context": "DATA IS EXAUSTIVE" in response.upper()}


def _edge_skip_execution(state: MessagesState) -> str:
    """Routes to tool sql execution or final answer generation depending on
    if the model produced a sql query tool call in the previous message"""
    last_message = state["messages"][-1]
    logger.debug("edge: skip tools to final execution")

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:  # type:ignore
        logger.debug("going to tool calls")
        return NODE_TOOLS_NAME
    else:
        logger.debug("going to final generation")
        return NODE_ANSWER_NAME


def _edge_sufficiency_evaluation(state: MessagesState) -> str:
    logger.debug("edge: sufficient context branching")
    if state["sufficient_context"]:
        logger.debug("proceeding to answer")
        return NODE_ANSWER_NAME
    else:
        logger.debug("looping again")
        return NODE_GENERATE_NAME


################################### GRAPH COMPONENTS


################################### HELPER FUNCTIONS
def _did_last_execution_fail(state: MessagesState) -> bool:
    last_message = content_as_string(state["messages"][-1])
    return EXECUTION_ERROR_PREFIX in last_message


def _set_prompt_language(config: Config):
    global local_prompts

    try:
        local_prompts = prompts[config.language]
    except KeyError:
        raise ValueError(f"Prompt language {config.language} not currently supported")


def _get_fetched_metadata(state: MessagesState) -> str | None:
    for msg in reversed(state["messages"]):
        if msg.type == "tool" and msg.name == "fetch_metadata":
            return content_as_string(msg)
    return None


def _get_fetched_data(state: MessagesState) -> str | None:
    for msg in reversed(state["messages"]):
        if msg.type == "tool" and msg.name == "execute_sql":
            return content_as_string(msg)
    return None


################################### HELPER FUNCTIONS
