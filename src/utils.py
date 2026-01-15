def get_user_question(state) -> str:
    user_query = None
    for msg in reversed(state["messages"]):
        if msg.type == "human":
            user_query = msg.content
            break
    if user_query is None:
        raise ValueError("No human message found in state")

    return user_query


def content_as_string(message) -> str:
    """Takes a langchain message and returns the content string"""

    content = message.content
    if isinstance(content, list):
        content = "".join(block["text"] for block in content)
    elif isinstance(content, dict):
        content = message.content["text"]

    return content


def print_graph(compiled_graph):
    png_bytes = compiled_graph.get_graph(xray=True).draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_bytes)
