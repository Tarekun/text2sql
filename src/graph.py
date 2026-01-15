from langchain.messages import (
    ToolMessage,
    SystemMessage,
    AnyMessage,
    AIMessage,
    HumanMessage,
)
import operator
from typing_extensions import TypedDict, Annotated
from src.llm_backend import get_llm
from src.utils import get_user_question, content_as_string
from src.db import get_table_metadata, run_sql_query
from src.prompts import prompts, Prompts


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    retry_count: int


NODE_GENERATE_NAME = "generate_sql"
NODE_EXECUTE_NAME = "execute_sql"
NODE_ANSWER_NAME = "answer"

EXECUTION_ERROR_PREFIX = "SQL execution error:"
MAX_RETRIES = 5
local_prompts: Prompts = prompts["it"]


def node_generate_sql(state: MessagesState):
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


def node_execute_sql(state: MessagesState):
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


def node_final_answer(state: MessagesState):
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


def edge_execution_success_check(state: MessagesState) -> str:
    current_retries = state.get("retry_count", 0)
    if _did_last_execution_fail(state) and current_retries < MAX_RETRIES:
        return NODE_GENERATE_NAME
    else:
        return NODE_ANSWER_NAME


def _did_last_execution_fail(state: MessagesState) -> bool:
    last_message = content_as_string(state["messages"][-1])
    return EXECUTION_ERROR_PREFIX in last_message
