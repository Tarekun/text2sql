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


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    retry_count: int


NODE_GENERATE_NAME = "generate_sql"
NODE_EXECUTE_NAME = "execute_sql"
NODE_ANSWER_NAME = "answer"
NODE_TOOLS_NAME = "tools"
NODE_RETRYMNG_NAME = "retry_management"

EXECUTION_ERROR_PREFIX = "SQL execution error:"
max_retries: int = 5
local_prompts: Prompts = prompts["it"]


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
    agent_builder.add_node(NODE_RETRYMNG_NAME, _node_check_tool_result)
    agent_builder.add_node(NODE_TOOLS_NAME, ToolNode(tool_list))
    agent_builder.add_node(NODE_ANSWER_NAME, _node_final_answer)
    # Add edges to connect nodes
    agent_builder.add_edge(START, NODE_GENERATE_NAME)
    agent_builder.add_conditional_edges(
        NODE_GENERATE_NAME, _edge_skip_execution, [NODE_TOOLS_NAME, NODE_ANSWER_NAME]
    )
    agent_builder.add_edge(NODE_TOOLS_NAME, NODE_RETRYMNG_NAME)
    agent_builder.add_conditional_edges(
        NODE_RETRYMNG_NAME,
        _edge_execution_success_check,
        [NODE_GENERATE_NAME, NODE_ANSWER_NAME],
    )
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
    print("node generate")
    llm = get_llm()
    user_query = get_user_question(state)
    schema_context = get_table_metadata(user_query)
    system_prompt = local_prompts.sql_generation.format(schema=schema_context)

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


def _node_check_tool_result(state: MessagesState):
    print("node check tool result")
    if _did_last_execution_fail(state):
        print("\tfailed, increase retry")
        return {"retry_count": state.get("retry_count", 0) + 1}
    else:
        print("\tsuccess go forward")
        return {"retry_count": 0}


def _node_final_answer(state: MessagesState):
    print("node final answer")
    llm = get_llm()
    user_query = get_user_question(state)
    sql_result = None
    for msg in reversed(state["messages"]):
        if msg.type == "tool":
            sql_result = content_as_string(msg)
            break

    if sql_result is None:
        print("MISSING DATA")
    system_prompt = local_prompts.final_answer.format(
        data=(
            sql_result
            if sql_result is not None
            else "Data is missing for unknown reason. Notify user"
        )
    )

    messages = [("system", system_prompt), ("human", user_query)]

    response = llm.invoke(messages).content
    return {"messages": [AIMessage(content=response)]}


def _edge_skip_execution(state: MessagesState) -> str:
    """Routes to tool sql execution or final answer generation depending on
    if the model produced a sql query tool call in the previous message"""
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:  # type:ignore
        print("edge go to tool execution")
        return NODE_TOOLS_NAME
    else:
        print("edge skip tool execution")
        return NODE_ANSWER_NAME


def _edge_execution_success_check(state: MessagesState) -> str:
    current_retries = state.get("retry_count", 0)
    if _did_last_execution_fail(state) and current_retries < max_retries:
        print("edge retry generation")
        return NODE_GENERATE_NAME
    else:
        print("edge goto final answer")
        return NODE_ANSWER_NAME


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


################################### HELPER FUNCTIONS
