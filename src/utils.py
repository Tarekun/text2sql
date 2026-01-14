from google.cloud import bigquery
from google.cloud.bigquery.table import Row
import re
import yaml


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
    content = message.content
    if isinstance(content, list):
        content = "".join(block for block in content if isinstance(block, str))

    return content
