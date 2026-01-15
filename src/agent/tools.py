from langchain.tools import tool
from src.db import run_sql_query

EXECUTION_ERROR_PREFIX = "SQL execution error:"


@tool
def execute_sql(query: str) -> str:
    """Execute a SQL query against the database and return results.
    If the query fails, returns an error message that should be used to fix and retry the query.
    """

    try:
        print("TOOL EXECUTION")
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


tool_list = [execute_sql]
tools_dict = {tool.name: tool for tool in tool_list}
