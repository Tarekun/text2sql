import csv
from langchain.messages import (
    SystemMessage,
    HumanMessage,
)
from langchain.tools import tool
import os
import re
import subprocess
import sys
from src.agent.llm_backend import instantiate_llm
from src.config import read_config
from src.db.bigquery import run_sql_query, get_table_metadata
from src.prompts.en import metadata_extraction
from src.logger import logger
from src.utils import content_as_string
from src.cache.query_embedding import top_k_lookup


SQL_EXECUTION_ERROR_PREFIX = "SQL execution error:"
PYTHON_EXECUTION_ERROR_PREFIX = "Python execution error"
QUERY_RESULT_DIRECTORY = "./query_results"
os.makedirs(QUERY_RESULT_DIRECTORY, exist_ok=True)


@tool
def execute_sql(query: str, meaningful_filename: str) -> str:
    """Execute the SQL `query` against the database and return results.
    If the query fails, returns an error message that should be used to fix and retry the query.
    Requires also `meaningful_filename` the file name that will be used to store the result as a csv file
    """

    try:
        logger.debug("tool: execute sql")
        save_code(query, extension="sql", custom_name=meaningful_filename)
        rows, schema = run_sql_query(query)
        column_names = [col.name for col in schema]
        if not meaningful_filename.endswith(".csv"):
            meaningful_filename += ".csv"

        with open(
            f"{QUERY_RESULT_DIRECTORY}/{meaningful_filename}",
            mode="w",
            newline="",
            encoding="utf-8",
        ) as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(column_names)
            writer.writerows(rows)  # type:ignore

        header = "\t".join(column_names)
        values = ""
        for row in rows[:30]:  # TODO: this should be configurable from the config
            values += "\t".join([str(cell) for cell in row])
            values += "\n"
        return f"(The full query result is available at the path {QUERY_RESULT_DIRECTORY}/{meaningful_filename})\n{header}\n{values}"
    except Exception as e:
        return f"{SQL_EXECUTION_ERROR_PREFIX} {str(e)}"


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


@tool
def python_interpreter(code: str) -> str:
    """Execute arbitrary Python code in the current environment and return stdout + repr of last expression (if any).
    Use this to analyze data, create visualizations (e.g., matplotlib), or process files.
    The code runs in the same Python process as the agent — use with caution.
    """

    logger.debug("tool: python interpreter")
    code = code.replace("\\\\", "\\")
    script_path = save_code(code, extension="py")

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = result.stdout
        errors = result.stderr

        if result.returncode != 0:
            return f"{PYTHON_EXECUTION_ERROR_PREFIX}: {errors}"

        if errors:
            return f"{output}\nStderr: {errors}"

        return output if output else "Code executed successfully (no output)"

    except subprocess.TimeoutExpired:
        return f"{PYTHON_EXECUTION_ERROR_PREFIX}: Execution timed out"
    except Exception as e:
        return f"{PYTHON_EXECUTION_ERROR_PREFIX}: {e}"


@tool
def queries_rag_lookup(user_question: str, k: int = 5) -> str:
    """Perform a top-k neighbours lookup on the QueryEmbeddings table to find relevant queries for the user question.
    Returns a formatted string with the most similar queries and their descriptions."""
    cached_queries = top_k_lookup(user_question, 5)
    result = ""
    for query in cached_queries:
        result += f"Query name: {query.name}\n"
        result += f"Description: {query.description}\n"
        result += f"Code:\n```sql{query.query}```\n"
        result += "\n\n"

    return result


def save_code(code: str, extension: str, custom_name="generated") -> str:
    directory = "generated_code"
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, f"{custom_name}.{extension}")

    if os.path.exists(file_path):
        # Find all existing generated-N.py files
        pattern = re.compile(rf"^{custom_name}-(\d+)\.{re.escape(extension)}$")
        max_num = 0

        for filename in os.listdir(directory):
            match = pattern.match(filename)
            if match:
                num = int(match.group(1))
                max_num = max(max_num, num)

        new_filename = f"{custom_name}-{max_num + 1}.{extension}"
        file_path = os.path.join(directory, new_filename)

    with open(file_path, "w") as f:
        f.write(code)
    return file_path


tool_list = [execute_sql, fetch_metadata, queries_rag_lookup]
tools_dict = {tool.name: tool for tool in tool_list}
