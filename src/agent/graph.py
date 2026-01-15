from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langchain.messages import (
    ToolMessage,
    SystemMessage,
    AnyMessage,
    AIMessage,
    HumanMessage,
)
import operator
from typing_extensions import TypedDict, Annotated
from src.agent.llm_backend import get_llm, instantiate_llm
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
    agent_builder.add_node(NODE_EXECUTE_NAME, _node_execute_sql)
    agent_builder.add_node(NODE_ANSWER_NAME, _node_final_answer)
    # Add edges to connect nodes
    agent_builder.add_edge(START, NODE_GENERATE_NAME)
    agent_builder.add_edge(NODE_GENERATE_NAME, NODE_EXECUTE_NAME)
    agent_builder.add_conditional_edges(
        NODE_EXECUTE_NAME,
        _edge_execution_success_check,
        [NODE_ANSWER_NAME, NODE_GENERATE_NAME],
    )
    agent_builder.add_edge(NODE_EXECUTE_NAME, NODE_ANSWER_NAME)
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
        print("appending error context for retry")
        # add to context the failed query
        messages.append(AIMessage(content=state["messages"][-2].content))
        # add to context the error message
        messages.append(last_msg)
        messages.append(
            HumanMessage(
                content="The above SQL query failed. Please analyze the error and generate a corrected query."
            )
        )
        retry_count = state.get("retry_count", 0) + 1
    else:
        retry_count = 0

    content = llm.invoke(messages).content[0]["text"]  # type:ignore
    if isinstance(content, list):
        sql_text = "".join(block for block in content if isinstance(block, str))
    else:
        sql_text = str(content)

    return {
        "messages": [
            AIMessage(
                content=sql_text.strip(),
            )
        ],
        "retry_count": retry_count,
    }


def _node_execute_sql(state: MessagesState):
    last_message = state["messages"][-1]
    if last_message.type != "ai":
        raise ValueError("Expected last message to be from AI")
    sql_text = last_message.content
    if isinstance(sql_text, list):
        sql_text = "".join(block for block in sql_text if isinstance(block, str))

    try:
        rows, schema = run_sql_query(sql_text)
        column_names = [col.name for col in schema]
        header = "\t".join(column_names)
        values = ""
        for row in rows:
            values += "\t".join([str(cell) for cell in row])
            values += "\n"
        query_result_string = f"{header}\n{values}"

        return {
            "messages": [
                ToolMessage(content=query_result_string, tool_call_id="sql_execution")
            ]
        }

    except Exception as e:
        return {
            "messages": [
                ToolMessage(
                    content=f"{EXECUTION_ERROR_PREFIX} {str(e)}",
                    tool_call_id="sql_execution",
                )
            ]
        }


def _node_final_answer(state: MessagesState):
    llm = get_llm()
    user_query = get_user_question(state)
    sql_result = None
    for msg in reversed(state["messages"]):
        if msg.type == "tool" and msg.tool_call_id == "sql_execution":
            sql_result = msg.content
            break
    system_prompt = local_prompts.final_answer.format(data=sql_result)

    messages = [("system", system_prompt), ("human", user_query)]

    response = llm.invoke(messages).content
    return {"messages": [AIMessage(content=response)]}


def _edge_execution_success_check(state: MessagesState) -> str:
    current_retries = state.get("retry_count", 0)
    if _did_last_execution_fail(state) and current_retries < max_retries:
        return NODE_GENERATE_NAME
    else:
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
