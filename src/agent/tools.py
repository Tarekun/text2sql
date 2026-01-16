from langchain.messages import (
    ToolMessage,
    SystemMessage,
    AnyMessage,
    AIMessage,
    HumanMessage,
)
from langchain.tools import tool
from src.agent.llm_backend import instantiate_llm
from src.config import read_config
from src.db import run_sql_query, get_table_metadata
from src.prompts.en import metadata_extraction
from src.logger import logger
from src.utils import content_as_string


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
    metadata = get_table_metadata()
    if len(metadata) > 5000:
        logger.debug("raw metadata too long, calling LLM to parse it")
        config = read_config("./config.yml")
        llm = instantiate_llm(config)
        response = llm.invoke(
            [
                SystemMessage(content=metadata_extraction),
                HumanMessage(
                    content=f"Original user question:\n{user_question}\n\nFull metadata fetched:\n{metadata}"
                ),
            ]
        )
        metadata = content_as_string(response)

    return metadata


tool_list = [execute_sql, fetch_metadata]
tools_dict = {tool.name: tool for tool in tool_list}
