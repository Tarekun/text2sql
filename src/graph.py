from langchain.messages import ToolMessage, SystemMessage, AnyMessage
from langgraph.graph import END
from langchain_google_genai import ChatGoogleGenerativeAI
import operator
from typing import Literal
from typing_extensions import TypedDict, Annotated
from tools import *


model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",  # or "gemini-1.5-flash", etc.
    temperature=0,
    project="your-gcp-project-id",  # optional if set via env/GCP config
    location="us-central1",  # or your preferred region
)


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


tools = [add, multiply]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)


def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def tool_node(state: dict):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
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
