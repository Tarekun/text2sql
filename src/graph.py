from langchain.messages import (
    ToolMessage,
    SystemMessage,
    AnyMessage,
    AIMessage,
    HumanMessage,
)
from langgraph.graph import END
import operator
import re
from typing import Literal
from typing_extensions import TypedDict, Annotated
from src.tools import tools_dict
from src.llm_backend import get_llm
from src.utils import get_user_question, content_as_string
from src.db import get_table_metadata, run_sql_query


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


SQL_GENERATION_SYSTEM_PROMPT = """You are a database expert. Generate a valid SQL query to fetch data useful to answer the original user question.
- Use only tables and columns from the schema below.
- Do not use CREATE, DROP, INSERT, UPDATE, DELETE, or any statement with side effects.
- Only output the SQL query. No explanations, no markdown, no comments.
- Always include a LIMIT 100 clause unless aggregation makes it unnecessary.

Schema:
{schema}
"""

FINAL_ANSWER_PROMPT = """
You are a domain expert of data supporting exploratory investigations on databases.
You are provided some prefetched data from the underlying database containing info to answer the user question.
Stay focused on the user question and answer in a way that is grounded on the data available

Fetched data:
{data}
"""

NODE_GENERATE_NAME = "generate_sql"
NODE_EXECUTE_NAME = "execute_sql"
NODE_ANSWER_NAME = "answer"


def node_generate_sql(state: MessagesState):
    llm = get_llm()
    user_query = get_user_question(state)
    schema_context = get_table_metadata(user_query)
    system_prompt = SQL_GENERATION_SYSTEM_PROMPT.format(schema=schema_context)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query),
    ]
    last_msg = state["messages"][-1]
    if isinstance(last_msg, ToolMessage) and "SQL execution error" in last_msg.content:
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
        ]
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
                    content=f"SQL execution error: {str(e)}",
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
            print(msg.content)
            print()
            sql_result = msg.content
            break
    system_prompt = FINAL_ANSWER_PROMPT.format(data=sql_result)

    messages = [("system", system_prompt), ("human", user_query)]

    response = llm.invoke(messages).content
    return {"messages": [AIMessage(content=response)]}


def edge_execution_success_check(state: MessagesState) -> str:
    last_message = content_as_string(state["messages"][-1])
    print(f"last message is\n{last_message}")
    if "SQL execution error" in last_message:
        print("regenerating")
        return NODE_GENERATE_NAME
    else:
        print("going to final answer")
        return NODE_ANSWER_NAME


def tool_node(state: dict):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_dict[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


def should_continue(state: MessagesState) -> Literal["tool_node", END]:  # type:ignore
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:  # type:ignore
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END
