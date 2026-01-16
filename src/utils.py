from langchain.messages import AnyMessage
from langgraph.graph.state import CompiledStateGraph
from src.logger import logger


def get_user_question(state) -> str:
    user_query = None
    for msg in reversed(state["messages"]):
        if msg.type == "human":
            user_query = msg.content
            break
    if user_query is None:
        raise ValueError("No human message found in state")

    return user_query


def content_as_string(message: AnyMessage) -> str:
    """Takes a langchain message and returns the content string"""

    content = message.content
    if isinstance(content, list):
        content = ""
        for block in content:
            if isinstance(block, dict):
                content += block["text"]
            elif isinstance(block, str):
                content += block
            else:
                logger.error(
                    f"Couldn't handle message content block of type {type(block)}. Value is {block}"
                )

    elif isinstance(content, dict):
        content = message.content["text"]  # type:ignore

    return content


def print_graph(compiled_graph: CompiledStateGraph):
    png_bytes = compiled_graph.get_graph(xray=True).draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_bytes)
