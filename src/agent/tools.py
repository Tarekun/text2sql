import csv
from langchain.messages import (
    ToolMessage,
    SystemMessage,
    AnyMessage,
    AIMessage,
    HumanMessage,
)
from langchain.tools import tool
from os import makedirs
from src.agent.llm_backend import instantiate_llm
from src.config import read_config
from src.db import run_sql_query, get_table_metadata
from src.prompts.en import metadata_extraction
from src.logger import logger
from src.utils import content_as_string
import sys
import io

EXECUTION_ERROR_PREFIX = "SQL execution error:"
QUERY_RESULT_DIRECTORY = "./query_results"
makedirs(QUERY_RESULT_DIRECTORY, exist_ok=True)


@tool
def execute_sql(query: str, meaningful_filename: str) -> str:
    """Execute the SQL `query` against the database and return results.
    If the query fails, returns an error message that should be used to fix and retry the query.
    Requires also `meaningful_filename` the file name that will be used to store the result as a csv file
    """

    try:
        logger.debug("tool: execute sql")
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


@tool
def python_interpreter(code: str) -> str:
    """Execute arbitrary Python code in the current environment and return stdout + repr of last expression (if any).
    Use this to analyze data, create visualizations (e.g., matplotlib), or process files.
    The code runs in the same Python process as the agent — use with caution.
    """
    print("tool: python interpreter")
    old_stdout = sys.stdout
    try:
        # Capture stdout
        # old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()

        # Prepare a namespace (you can pre-load useful modules if desired)
        global_namespace = {
            "__builtins__": __builtins__,
            # Optionally pre-import common libs:
            "pd": __import__("pandas"),
            "plt": __import__("matplotlib.pyplot"),
            "np": __import__("numpy"),
        }
        local_namespace = {}

        # Execute the code
        try:
            # Try to compile as an expression first (to capture return value)
            compiled = compile(code.strip(), "<string>", "eval")
            result = eval(compiled, global_namespace, local_namespace)
            output = captured_output.getvalue()
            if output:
                result_str = f"{output}{repr(result)}"
            else:
                result_str = repr(result)
        except SyntaxError:
            # Fall back to exec for statements
            exec(code, global_namespace, local_namespace)
            result_str = captured_output.getvalue() or "(no output)"

        print(f"returning {result_str}")
        return result_str

    except Exception as e:
        return f"Python execution error: {e}"
    finally:
        sys.stdout = old_stdout


tool_list = [execute_sql, fetch_metadata]
tools_dict = {tool.name: tool for tool in tool_list}
