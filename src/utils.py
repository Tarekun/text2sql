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


def print_graph(
    compiled_graph: CompiledStateGraph,
    tool_nodes=[],
    llm_control_nodes=[],
    llm_nodes=[],
):
    # i dont like these kinds of imports but at this stage who cares tbh
    import requests
    import base64

    graph = compiled_graph.get_graph(xray=True)
    # default graph styling
    mermaid_code = graph.draw_mermaid()

    custom_styles = []
    for node in llm_nodes:
        # LLM nodes colored in red
        custom_styles.append(f"style {node} fill:#b5f5c8,color:#333")
    for node in llm_control_nodes:
        custom_styles.append(f"style {node} fill:#2ecc71,color:#000")
    for node in tool_nodes:
        # tool nodes colored in blue
        custom_styles.append(f"style {node} fill:#3498db,color:#fff")

    styled_mermaid = mermaid_code + "\n" + "\n".join(custom_styles)
    response = requests.get(
        "https://mermaid.ink/img/"
        + base64.urlsafe_b64encode(styled_mermaid.encode()).decode()
    )
    with open("graph.png", "wb") as f:
        f.write(response.content)
