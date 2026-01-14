from google.cloud import bigquery
from google.cloud.bigquery.table import Row
import re


def get_user_question(state) -> str:
    user_query = None
    for msg in reversed(state["messages"]):
        if msg.type == "human":
            user_query = msg.content
            break
    if user_query is None:
        raise ValueError("No human message found in state")

    return user_query


def run_sql_query(query: str) -> list[Row]:
    if not isinstance(query, str):
        raise TypeError(f"Expected SQL as string, got {type(query)}")
    query = query.strip().rstrip(";")

    # # Block non-SELECT queries
    # if not sql.upper().lstrip().startswith("SELECT"):
    #     return {
    #         "messages": [
    #             ToolMessage(
    #                 content="Error: Only SELECT queries are allowed.",
    #                 tool_call_id="sql_execution"
    #             )
    #         ]
    #     }
    # # Block dangerous keywords
    # if DANGEROUS_PATTERNS.search(sql):
    #     return {
    #         "messages": [
    #             ToolMessage(
    #                 content="Error: Query contains forbidden operations.",
    #                 tool_call_id="sql_execution"
    #             )
    #         ]
    #     }

    # Enforce LIMIT 100 if not present (simple heuristic)
    if not re.search(r"\bLIMIT\s+\d+", query, re.IGNORECASE):
        query += " LIMIT 100"

    try:
        job_config = bigquery.QueryJobConfig(
            use_query_cache=True,
            maximum_bytes_billed=100 * 1024 * 1024,  # 100 MB cap
        )
        client = bigquery.Client(project="soges-group-data-platform")
        # client = bigquery.Client(project="formazione-danieletarek-iaisy")
        query_job = client.query(query, job_config=job_config, timeout=30.0)
        print(f"running sql query {query}")
        result = query_job.result()

        return list(result), result.schema  # type:ignore
    except Exception as e:
        print("failed to fetch")
        return [], []  # type:ignore


schema = """
table: `gold.indirizzi`
description: contains records of all the facilities owned by the company
schema:
- ID:INTEGER numerical id of rows
- STRUTTURA:STRING name of the facility
- INDIRIZZO:STRING address of the facility
- LOCALITA:STRING name of the town where the facility is located
- PROVINCIA:STRING name of the big city closest to the town where the facility is located    
"""


def get_relevant_table_schemas(asd):
    return schema
