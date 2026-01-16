from langchain.tools import tool
from src.db import run_sql_query, get_table_metadata
from src.logger import logger

EXECUTION_ERROR_PREFIX = "SQL execution error:"


@tool
def execute_sql(query: str) -> str:
    """Execute a SQL query against the database and return results.
    If the query fails, returns an error message that should be used to fix and retry the query.
    """

    try:
        logger.debug("tool: execute sql")
        rows, schema = run_sql_query(query)
        column_names = [col.name for col in schema]
        header = "\t".join(column_names)
        values = ""
        for row in rows:
            values += "\t".join([str(cell) for cell in row])
            values += "\n"
        return f"{header}\n{values}"
    except Exception as e:
        return f"{EXECUTION_ERROR_PREFIX} {str(e)}"


@tool
def fetch_metadata(user_question: str) -> str:
    """Fetch metadata about possibly relevant tables to the `user_question`"""
    logger.debug("tool: fetch metadata")
    raw_metadata = get_table_metadata(user_question)
    return raw_metadata


tool_list = [execute_sql, fetch_metadata]
tools_dict = {tool.name: tool for tool in tool_list}
