from langchain.messages import (
    AnyMessage,
)
import operator
from typing_extensions import TypedDict, Annotated
from src.agent.tools import *
from src.utils import content_as_string


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    metadata: str
    fetched_data: str
    retry_count: int
    sufficient_context: bool
    python_output: str


def did_last_sql_run_fail(state: MessagesState) -> bool:
    """Returns `True` iff the last message was from the sql execution tool and failed"""
    last_message = content_as_string(state["messages"][-1])
    return SQL_EXECUTION_ERROR_PREFIX in last_message


def did_last_python_run_fail(state: MessagesState) -> bool:
    """Returns `True` iff the last message was from the python execution tool and failed"""
    last_message = content_as_string(state["messages"][-1])
    return PYTHON_EXECUTION_ERROR_PREFIX in last_message


def get_fetched_metadata(state: MessagesState) -> str | None:
    """Returns the string output of the latest `fetch_metadata` tool if it was run before,
    `None` otherwise"""
    return _get_tool_output_as_string(state, "fetch_metadata")


def get_fetched_data(state: MessagesState) -> str | None:
    """Returns the string output of the latest `execute_sql` tool if it was run before,
    `None` otherwise"""
    return _get_tool_output_as_string(state, "execute_sql")


def get_python_output(state: MessagesState) -> str | None:
    """Returns the string collected from std output of the latest execution of python code
    by the `python_interpreter` tool if it was run before, `None` otherwise"""
    return _get_tool_output_as_string(state, "python_interpreter")


def _get_tool_output_as_string(state: MessagesState, tool_name: str) -> str | None:
    for msg in reversed(state["messages"]):
        if msg.type == "tool" and msg.name == tool_name:
            return content_as_string(msg)
    return None
